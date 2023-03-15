#!/usr/bin/env/python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import os
import math
import networkx as nx
from utils.hgnn_utils import one_hot_vec
from utils.data_utils import process
from utils.hyperbolicity import hyperbolicity_sample

from hyperbolic_learning_master.utils.embed import train_embeddings, load_embeddings,load_embeddings_npy, evaluate_model,mean_average_precision
from collections import defaultdict
import json
import networkx as nx
from datetime import datetime
import scipy.sparse as sp
import pandas as pd
import copy


### this will take a list of edges/ list of nodes and return the adj_mat for hgcn
class GraphDataset(Dataset):

	# def __init__(self, args, logger, split):
	def __init__(self, args, split):
		### add virtual node arguement
		self.args = args
		# self.logger = logger
		print(split,'SPLIT')
		print((hasattr(self.args,'train_only')),'HAS TRAIN ONLY')
		print(self.args.train_only,'USES TRAIN ONLY')

		if split=='special':
			self.dataset = json.load(open(self.args.special_file))
		if hasattr(self.args,'idxs_dict'):
			full_dataset=json.load(open(self.args.all_file))
			# full_dataset=json.load(open(self.args.all_file))
			indx_dict=self.args.idxs_dict  ### if no indx_dict, read indx file

			# print(indx_dict)

			if (split=='train') and (hasattr(self.args,'train_only')) and (self.args.train_only):
				self.indices=indx_dict['all']
				print('TRAINING ON ALL DATA \n\n')
				print("ALERT ALERT \n")
				print("TRAINING ON ALL DATA")
			else:
				if split=='dev':
					self.indices=list(indx_dict['valid']) ## gotta fix inconsistencies
				else:
					self.indices=list(indx_dict[split])


			# indices.sort()
			print(self.indices,'SELF INDICES')
			print([g['graph_id'] for g in full_dataset],'dta set indices')
			self.dataset= [
			g for g in full_dataset if int(g['graph_id']) in self.indices
			]
			if (split=='train') and (hasattr(self.args,'train_noise_level')) and (self.args.train_noise_level)>0 and (self.args.train_noise_num>0):
				print('using noisy training')
				self.use_noise=True
				self.noise_prob=self.args.train_noise_prob
			else:
				self.use_noise=False



		elif split == 'train':
			# print(self.args.train_file,'trained')
			# setattr(self.args,'frech_B_dict','')
			# print(open(self.args.train_file),'lets see?')
			# print(self.args,'ALL ARGS')

			self.dataset = json.load(open(self.args.train_file))
		elif split == 'dev':
			self.dataset = json.load(open(self.args.dev_file))
		elif split == 'test':
			self.dataset = json.load(open(self.args.test_file))


		self.shape=self.shape()
		self.use_virtual=True if (hasattr(self.args, 'use_virtual') and (self.args.use_virtual>0)) else False

		# print(data)

	def __len__(self):
		return len(self.dataset)

	def shape(self):
		### should use max of dataset (?)
		example_graph = self.dataset[0]
		# for d in self.dataset[0]
		return len(self.dataset), len(example_graph['node_features']), len(example_graph['node_features'])

	def node_num(self,idx):

		graph=self[idx]
		# print(graph,'GRAPH')
		node_num = len(graph['features']) if not self.use_virtual else len(graph['features'])-1

		# pruin
		# for f in graph['node_features']:
			# print(len(graph['node_features']),len(f))
		# print(graph['node_features'].shape,'SHAPE UP')
		# print(graph['node_features'])

		return node_num


	def __getitem__(self, idx): ### make recreate graph func in utls?

		#### HOW TO RANDOMIZE EDGES??? ----- create randoms in self.dataset, then if train and add noise, you just call those!!!!!

		graph_og = self.dataset[idx]  ## here would be a good place to use graph id?
		# return transform_input(graph_og,self.args,use_noise=self.use_noise)

		start=datetime.now()
		graph=copy.deepcopy(graph_og)

		if self.use_noise:
			draw_p=np.random.uniform(0,1)  
			if draw_p<self.noise_prob:  ### if self.noise_prib<0, add og to noise data and draw from that ?
				noise_data=graph['noise_data']
				draw_graph=np.random.choice(np.arange(len(noise_data)))
				# print(draw_graph,'DRAW GRAPH')
				graph=noise_data[draw_graph]
			else:
				# print('no draw')
				pass
			# print(no_draw)


		copytime = datetime.now()
		use_virtual = self.use_virtual
		# print(use_virtual,'USE VIRTUAL')

		adj_prob=torch.tensor(graph['adj_prob'])
		# print(adj_prob.shape,'ADJ SHAPE')
		# eee
		# use_virtual=False

		# graph_id = graph['graph_id'] if 'graph_id' in graph.keys() else idx 
		if 'graph_id' not in graph.keys():
			raise Exception('hold up there buster youre out of data')
		graph_id = graph['graph_id']
		nx_graph =nx.Graph()
		node_num = len(graph['node_features'])
		nx_graph.add_nodes_from([i for i in range(node_num)])
		if True:
		# if hasattr(self.args,'use_weights') and not self.args.use_weights:
			for e in graph['edges']:
				# print(e,'edge!!!')
				if e[0]!=e[1]:
					nx_graph.add_edge(e[0],e[1],weight=1)
					# print()
					# print(nx_graph.has_edge(e[1],e[0]),'both ways?')
				else:
					print(' self loop')
		else:
			nx_graph.add_edges_from(graph['edges'])
			# for e in nx_graph.edges:
				# print(e,'edge')
				# print(nx_graph.has_edge(e[1],e[0]),'both ways?')
		nx_old=nx_graph

		adj_mat = nx.adjacency_matrix(nx_graph)
		# adj_mat = nx.adjacency_matrix(nx_old,'old') 
		spr_mat=sp.csr_matrix(adj_mat)
		x, y = sp.triu(adj_mat,k=1).nonzero() ## k=1 ignores the diagonal/ self edges, double check tomorrow
		pos_edges = np.array(list(zip(x, y))).astype(int)
		np.random.shuffle(pos_edges)
		pos_edges = torch.tensor(pos_edges).long()
		x, y = sp.triu(sp.csr_matrix(1. - adj_mat.toarray()),k=1).nonzero()
		neg_edges = np.array(list(zip(x, y))).astype(int)
		np.random.shuffle(neg_edges)
		neg_edges = torch.tensor(neg_edges).long()


		# adj_mat_weight=np.array(graph['adj_prob'])
		# full_weighted_adj=False
		# weighted_adj=False
		# add_to_existing_graph=False
		# continuous_thresh=True
		# if not weighted_adj:
		# 	full_weighted_adj=False

		# use_virtual= False if full_weighted_adj else True

		# print(np.percentile(adj_mat_weight,78),'PERCENTILE??')
		# print(np.percentile(adj_mat_weight,90),'PERCENTILE??')
		# adj_mat = nx.adjacency_matrix(nx_old,'old') 
		# if weighted_adj:
		# 	if full_weighted_adj:
		# 		nx_graph_new = nx.convert_matrix.from_numpy_matrix(adj_mat_weight)
		# 	# el
		# 	else:
		# 		low_thresh =5000
		# 		high_thresh = .7
		# 		low_mat = np.where(adj_mat_weight>low_thresh,.5,0)
		# 		high_mat = np.where(adj_mat_weight>high_thresh,1+adj_mat_weight,0)
		# 		# high_mat = np.where(adj_mat_weight>high_thresh,2,0)
		# 		full_mat = high_mat+low_mat ## should never overlap 
		# 		nx_graph_new = nx.convert_matrix.from_numpy_matrix(full_mat)
		# # print([n for n in nx_old],'old nodes first')
		# if add_to_existing_graph:
		# 	# print('add_to_existing_graph')
		# 	try:
		# 		nx_graph_new
		# 	except:
		# 		print('no new grah')
		# 	for e in nx_graph_new.edges():
		# 		weight = nx_graph_new.get_edge_data(e[0], e[1], default=None)['weight']
		# 		if nx_graph.has_edge(e[0],e[1]):
		# 			nx_graph.remove_edge(e[0],e[1])
		# 		nx_graph.add_edge(e[0],e[1],weight=weight)
		# 		# nx_graph.add_edge(e[0],e[1])
		# 		# nx_graph.add_edge(e)
		# print('a')
		if (hasattr(self.args, 'continuous_thresh') and self.args.continuous_thresh>0):
			# print('continously')
			for e in nx_graph.edges():
				# weight = nx_graph_new.get_edge_data(e[0], e[1], default=None)['weight']
				# if nx_graph.has_edge(e[0],e[1]):
				nx_graph.remove_edge(e[0],e[1])
				weight=adj_prob[e[0],e[1]]
				# print(weight,'WEIGHT')
				nx_graph.add_edge(e[0],e[1],weight=adj_prob[e[0],e[1]])
		# else:
			# fuck
		# # adj_mat = nx.adjacency_matrix(nx_old)
		# print(adj_mat,'ADJ MAT') 
		# adj_mat = nx.adjacency_matrix(nx_graph) 
		# print(adj_mat.shape,'ADJ MAT NEW')
		# adj_mat=adj_mat/2
		# ssss
		# print(nx_graph,'graphy graphy what are you? have you changed',nx_old)
		# print([n for n in nx_old],'old nodes mid')



		# print(use_virtual,'USE VIRTUAL')
		if use_virtual:
			for i in range(node_num+1):
				# print("EFFF")
				# print()
				nx_graph.add_edge(i,node_num,weight=1) ## adds virtual node
				# nx_graph.add_edge(i,i,weight=1)  ## adds self node
				# nx_graph.add_edge(i,node_num,weight=.5) ## adds virtual node
				nx_graph.add_edge(i,i)  ## adds self node
		else:
			for i in range(node_num):
				# nx_graph.add_edge(i,node_num) ## adds virtual node
				nx_graph.add_edge(i,i)  ## adds self node	
		# print([e for e in nx_graph.edges()],'EDGE ME')		

		# add self connection and a virtual node
		virtual_weight = self.args.edge_type - 1 if (hasattr(self.args, 'edge_type') and self.args.edge_type>1) else 1
		# adj_mat = [[i, node_num] for i in range(node_num)]
		weight = [[1, virtual_weight] for _ in range(node_num)]

		# print("MIDWAY")


		if self.args.normalization:
			normalize_weight(adj_mat, weight)

		node_feature = graph['node_features']
		if isinstance(node_feature[0], float): ## so ugly
			# print('changed')
			node_feature=np.array(node_feature,dtype=np.integer)
		if isinstance(node_feature[0], int): ## change to onehot.--- yo we gotta get this fixed ### no reason not to have in get_data
		# if True:
			# print(node_feature)
			new_node_feature = np.zeros((len(node_feature), self.args.num_feature))
			for i in range(len(node_feature)):
				# print(node_feature[i],'NODES')
				new_node_feature[i][node_feature[i]] = 1
			node_feature = new_node_feature.tolist() ### 
		if use_virtual: ## for now not going to add extra
			for i in range(len(node_feature)):
				node_feature[i]+=[0]
			node_feature.append(one_hot_vec(len(node_feature[0]),-1))
		if len(node_feature[0]) < self.args.num_feature: ## if we want to add more stuff.
			zeros = np.zeros((len(node_feature), self.args.num_feature - len(node_feature[0])))
			node_feature = np.concatenate((node_feature, zeros), axis=1).tolist()

		# print([n for n in nx_old],'old nodes last')

		# print(adj_mat,'adaz')
		# print(adj_mat.shape,'shape getting unqaure?')
		# print('does it keep weight?')
		adj_mat = nx.adjacency_matrix(nx_graph) 

		# print(adj_mat,'ADJ MAT NEW')
		spr_mat=sp.csr_matrix(adj_mat)
		# print(spr_mat,'SPR MAT')
		adj_mat,node_feature = process(spr_mat, node_feature, normalize_adj=False, normalize_feats=False)

		# weight_adj_mat = 

		# print(adh_mat,'ADJ MAT')

		# print(adj_mat.shape,'adj shape',node_feature,'node feature')

		labels = torch.tensor([graph['targets'][0] for i in range(node_num)]).long()

		# print("GRAAAAAAPH")
		# print('graph_id',graph_id)

		# print(datetime.now()-copytime,'THE REST')

		# print(graph['targets'])
		# print(pos_edges.cpu().detach().numpy(),'positive edges')
		# sdd

		# for e in pos_edges:
			# if e == torch.tensor([4,4]):
			
				# print(e,'EEEEEEEEEE')
			# if e[0]==0:
				# print(e,'EEEEEEEEEE')
		# assert 0 in pos_edges or 4 in pos_edges or 10 in pos_edges ### sloppy check to make sure this assertion has meaning
		assert node_num==90
		assert node_num not in pos_edges ### making sure virtual node not included
		assert node_num not in neg_edges ### " "  

		# for i in num_nodes:
			# assert

		# print(pos_edges,)
		for e in pos_edges:
			if e[0]==e[1]:

				print(e)
				raise Exception("SELF EDGE IN POS EDGES ALERT ALERT ALERT ")
			if (e[0]==node_num) or (e[1]==node_num):
				print(e)
				raise Exception("VIRTUAL EDGE IN POS EDGES ALERT ALERT ALERT ")

		# for e in neg_edges:
		# 	if e[0]==e[1]:

		# 		print(e)
		# 		raise Exception("SELF EDGE IN POS EDGES ALERT ALERT ALERT ")
		# 	if (e[0]==node_num) or (e[1]==node_num):
		# 		print(e)
		# 		raise Exception("VIRTUAL EDGE IN POS EDGES ALERT ALERT ALERT ")
		# print(node_feature.shape,'Node Feat')
		# print(adj_mat.shape,'ADJ MAT SHAPE')
		# print(pos_edges.shape,'POS EDGE')
		# print(neg_edges.shape,'Neg EDGE')

		# print(adj_prob.shape,'ADJ PROB\n\n\n')
		# print(np.percentile(adj_prob,20),20)
		# print(np.percentile(adj_prob,80),'80')
		# 20_pct =.057
		# 80_pct = .45
		# adj_conn = adj_prob<
		# print(adj_mat,'ADJ MAT')
		# print(adj_mat.shape,'91x91 right?')
		# print(np.isnan(adj_mat),'INPUT NANS')
		# print(adj_prob)
		return  {
		          'features': node_feature,
		          'adj_mat': adj_mat,
		          # 'weight': weight,
		          'edges':pos_edges,  ### can slice down to len 2 here
		          'edges_false':neg_edges, ### do sample or calculate all
		          'labels': labels,
		          'graph_id':graph_id,
		          'adj_prob':adj_prob,
		          # 'len_edges':pos_edges.shape[0],
		          # 'len_edges_false':neg_edges.shape[0]}
		          }
		# try:
		# 	return  {
		# 	          'features': node_feature,
		# 	          'adj_mat': adj_mat,
		# 	          # 'weight': weight,
		# 	          'edges':pos_edges,  ### can slice down to len 2 here
		# 	          'edges_false':neg_edges, ### do sample or calculate all
		# 	          'labels': labels,
		# 	          'graph_id':graph_id,
		# 	          'adj_prob':adj_prob
		# 	        }
		# except:
		# 	return  {
  #         'features': node_feature,
  #         'adj_mat': adj_mat,
  #         # 'weight': weight,
  #         'edges':pos_edges,  ### can slice down to len 2 here
  #         'edges_false':neg_edges, ### do sample or calculate all
  #         'labels': labels,
  #         'graph_id':graph_id
  #       }


	# def create_edge_list():
	def save_all_edge_dfs(self,output_rel_dir,add_self=True):
		### why not save all graph information??? could easily be used for plotting?

		raise Exception("NOT COMPLETE")
		save_dir = os.path.join(root_dir,str('relations'))
		save_dir =output_rel_dir
		if not os.path.exists(save_dir):
			os.mkdir(save_dir)


		for i in range(len(self.dataset)):
			graph = self[i]
			# print(graph.keys(),'GRAPH')
			graph_id = graph['graph_id'] ## no need to save other clinical data, we can retrieve as long as we have graph
			# print(graph_id,'GRAPH ID')
			graph_save_dir = os.path.join(save_dir,str(graph_id)+'.csv')

			
			if add_self:  ### bru need to figure out a way for this
				self_edges = np.array([[i,i] for i in range(num_nodes)])
				# print(self_edges.shape,'selfie should be 90')
				# print(graph['edges'].cpu().detach().numpy().shape,'edges should be >90')
				edges = np.concatenate([graph['edges'].cpu().detach().numpy(),self_edges])
			else:
				edges=graph['edges'].cpu().detach().numpy()
			# print(full_edges.shape,'HEY full shape')


			edge_df = pd.DataFrame(graph['edges'].cpu().detach().numpy())
			edge_df.to_csv(graph_save_dir,index=False)

			# train_embeddings(relations_path,delimiter,output_path)
	def stack_all(self):

		stacked_graphs = np.zeros(self.shape) ### right now will only work for meg or data with set num nodes ## will need to adjust self.shape to deal with max. either way shouldn't be used with larger sets
		stacked_labels = np.zeros((len(self.dataset)))
		stacked_graphids = np.zeros((len(self.dataset)))
		graph_ids =set([])

		if self.shape[0]*self.shape[1]>1000000:
			print("CAREFUL THIS COULD BE UUUUUGE")


		for i in range(len(self.dataset)):
		# for graph in self:
			graph = self[i] 
			graph_id = graph['graph_id']

			label = graph['labels'][0]   ## only get label for first node bc all are the same

			if int(float(graph_id)*10) != int(graph_id)*10:
				print(graph_id,' NOT WHOLE NUMBER')

			graph_id=int(graph_id)
			# nx_Graph = 
			# print(graph_id)
			nx_graph =nx.Graph()

			# print(graph['features'].shape,'huh')
			node_num = self.node_num(i)
			# print(node_num,'????????')

			nx_graph.add_nodes_from([i for i in range(node_num)])
			# print(graph['edges'])
			# print(nx_graph.number_of_nodes(),'nodes before>????')
			for e in graph['edges']:
				# if e[]
				nx_graph.add_edge(int(e[0]),int(e[1]))
			# nx_graph.add_edges_from(graph['edges'])

			# print(nx_graph.number_of_nodes(),'nodes')
			# print(nx_graph.number_of_edges(),'edges')
			# sp.csr_matrix(adj_mat)
			# adj_mat =sp.csr_matrix(nx.adjacency_matrix(nx_graph)).to_dense()
			adj_mat =nx.adjacency_matrix(nx_graph).todense()
			for z in range(adj_mat.shape[0]): ### assert balance
				for j in range(adj_mat.shape[1]):
					if adj_mat[z,j]>0:
						assert adj_mat[j,z]==adj_mat[z,j]

			if graph_id in graph_ids:
				raise Exception("DOUBLE GRAPH ID")

			stacked_graphs[i]=adj_mat
			stacked_labels[i]=label
			stacked_graphids[i]=graph_id
		print(stacked_graphs,stacked_labels,'final!!')
		# saas
		return stacked_graphs,stacked_labels,stacked_graphids


	# def stack_all(self):

	# 	stacked_graphs = np.zeros(self.shape) ### right now will only work for meg or data with set num nodes ## will need to adjust self.shape to deal with max. either way shouldn't be used with larger sets
	# 	stacked_labels = np.zeros((len(self.dataset)))
	# 	stacked_graphids = np.zeros((len(self.dataset)))
	# 	graph_ids =set([])

	# 	if self.shape[0]*self.shape[1]>1000000:
	# 		print("CAREFUL THIS COULD BE UUUUUGE")


	# 	for i in range(len(self.dataset)):
	# 		graph = self[i] 
	# 		graph_id = graph['graph_id']

	# 		label = graph['labels'][0]   ## only get label for first node bc all are the same

	# 		if int(float(graph_id)*10) != int(graph_id)*10:
	# 			print(graph_id,' NOT WHOLE NUMBER')

	# 		graph_id=int(graph_id)
	# 		# nx_Graph = 
	# 		# print(graph_id)
	# 		nx_graph =nx.Graph()

	# 		# print(graph['features'].shape,'huh')
	# 		node_num = self.node_num(i)
	# 		# print(node_num,'????????')

	# 		nx_graph.add_nodes_from([i for i in range(node_num)])
	# 		# print(graph['edges'])
	# 		# print(nx_graph.number_of_nodes(),'nodes before>????')
	# 		for e in graph['edges']:
	# 			# if e[]
	# 			nx_graph.add_edge(int(e[0]),int(e[1]))
	# 		# nx_graph.add_edges_from(graph['edges'])

	# 		# print(nx_graph.number_of_nodes(),'nodes')
	# 		# print(nx_graph.number_of_edges(),'edges')
	# 		# sp.csr_matrix(adj_mat)
	# 		# adj_mat =sp.csr_matrix(nx.adjacency_matrix(nx_graph)).to_dense()
	# 		adj_mat =nx.adjacency_matrix(nx_graph).todense()
	# 		for z in range(adj_mat.shape[0]): ### assert balance
	# 			for j in range(adj_mat.shape[1]):
	# 				if adj_mat[z,j]>0:
	# 					assert adj_mat[j,z]==adj_mat[z,j]

	# 		if graph_id in graph_ids:
	# 			raise Exception("DOUBLE GRAPH ID")
	# 		stacked_graphs[i]=adj_mat
	# 		stacked_labels[i]=label
	# 		stacked_graphids[i]=graph_id
	# 	print(stacked_graphs,stacked_labels,'final!!')
	# 	# saas
	# 	return stacked_graphs,stacked_labels,stacked_graphids
	# 					# break
			# print(adj_mat.shape)
			# print(adj_mat,'adj mat')
			# sddkgs

	def poincare_embeddings_all(self,output_rel_dir,output_emb_dir,add_self=True):  ## really no reason for this to be a dataset function? but it will keep things organized
		self.stack_all()
		delimiter=','
		relations_dir = output_rel_dir
		# embeddings_dir = os.path.join(root_dir,str('embeddings'))
		embeddings_dir = output_emb_dir
		if not os.path.exists(embeddings_dir):
			os.makedirs(embeddings_dir)

		print(output_rel_dir,"OUTPUT RELATIONS")


		if not os.path.exists(output_rel_dir):
			print('NO RELATIONS!, sending to save_all_edge_dfs')
			self.save_all_edge_dfs(output_rel_dir,add_self=False) ### we can do both? shouldn't change anything at all.

		for i in range(len(self.dataset)):
			# print("VERY START")
			graph = self[i]
			# print(graph.keys(),'GRAPH')
			graph_id = graph['graph_id'] ## no need to save other clinical data, we can retrieve as long as we have graph
			relations_path = os.path.join(relations_dir,str(graph_id)+'_relations.csv')
			# graph_save_dir
			if not os.path.exists(relations_path):
				edge_df = pd.DataFrame(graph['edges'].cpu().detach().numpy())
				edge_df.to_csv(relations_path,index=False)


			output_path = os.path.join(embeddings_dir,str(graph_id)+'_embeddings.tsv')
			# print("BEFOREEEE")
			train_embeddings(relations_path,delimiter,output_path)	
			# print("AFTERRR ")
			# break ## for now to test	

	def analyze_hyperbolicity(self,graph_samples=10,node_samples=20000):
		running_total = 0 
		# print(node_samples,'NODE SAMPLES')

		if not node_samples:
			node_samples = 10000
		if not graph_samples:
			graph_samples= len(self)
		graph_samples = min(graph_samples,len(self))
		# print('a','')

		running_prop = 0
		possible_nodes = ((90-1)*90)//2 ### won't always be
		group_hyp = {'all':[],'scd':[],'hc':[]}
		group_prop = {'all':[],'scd':[],'hc':[]}
		group_comp = {'all':[],'scd':[],'hc':[]}

		for i in range(graph_samples):
			graph = self.dataset[i]
			# print(graph.keys(),'GRAPH')
			# sdd
			node_num = len(graph['node_features']) ### make node feature
			edge_num = len(graph['edges'])
			print(edge_num,'EDGE NUM')
			# grou
			
			nx_graph =nx.Graph()
			nx_graph.add_nodes_from([i for i in range(node_num)])
			for e in graph['edges']:
				# print(e,'edge')
				# print(nx_graph[e[0]][e[1]])
				if e[0]!=e[1]:
					nx_graph.add_edge(e[0],e[1],weight=e[2]['weight'])
					# print(nx_graph.edges(data=True))
				else:
					print(' self loop')

			prop= edge_num/possible_nodes
			# comp= nx.number_connected_components(nx_graph)
			comp = len(max(nx.connected_components(nx_graph), key=len))
			# print(comp,'connected comps')

			hyp = hyperbolicity_sample(nx_graph,node_samples)

			group_prop['all'].append(prop)
			group_hyp['all'].append(hyp)
			group_comp['all'].append(comp)

			if graph['targets'][0]==1:
				group_prop['hc'].append(prop)
				group_hyp['hc'].append(hyp)
				group_comp['hc'].append(comp)
			elif graph['targets'][0]==2:
				group_prop['scd'].append(prop)
				group_hyp['scd'].append(hyp)
				group_comp['scd'].append(comp)


		final_avg= float(running_total)/float(graph_samples)
		final_prop= float(running_prop)/float(graph_samples)
		# final_avg=0
		# print(final_avg,'finals')

		group_results = {'all':{},'scd':{},'hc':{}}
		for g in group_hyp.keys():
			props= np.array(group_prop[g]) 
			hyps= np.array(group_hyp[g]) 
			comps= np.array(group_comp[g]) 
			

			std = np.std(hyps)
			mean = np.mean(hyps)
			props =  np.mean(props)
			comps =  np.mean(comps)
			# print(hyps,mean,'CHECK')
			# sdd
			group_results[g]['hyp_std']=std
			group_results[g]['hyp_mean']=mean
			group_results[g]['edge_prop']=props
			# group_results[g]['seperate_components']=comps
			group_results[g]['largest_comp']=comps

		print('Final hyperbolicity for {} graph samples with {}node samples: {}'.format(graph_samples,node_samples,final_avg))
		return group_results


def transform_input(graph_og,args,use_noise=False):
	graph=copy.deepcopy(graph_og)

	if use_noise:
		draw_p=np.random.uniform(0,1)  
		if draw_p<args.noise_prob:  ### if self.noise_prib<0, add og to noise data and draw from that ?
			noise_data=graph['noise_data']
			draw_graph=np.random.choice(np.arange(len(noise_data)))
			# print(draw_graph,'DRAW GRAPH')
			graph=noise_data[draw_graph]
		else:
			# print('no draw')
			pass
	use_virtual=args.use_virtual
	adj_prob=torch.tensor(graph['adj_prob'])
	if 'graph_id' not in graph.keys():
		raise Exception('hold up there buster youre out of data')
	graph_id = graph['graph_id']
	nx_graph =nx.Graph()
	node_num = len(graph['node_features'])
	nx_graph.add_nodes_from([i for i in range(node_num)])
	if True:
	# if hasattr(self.args,'use_weights') and not self.args.use_weights:
		for e in graph['edges']:
			# print(e,'edge!!!')
			if e[0]!=e[1]:
				nx_graph.add_edge(e[0],e[1],weight=1)
				# print()
				# print(nx_graph.has_edge(e[1],e[0]),'both ways?')
			else:
				print(' self loop')
	else:
		nx_graph.add_edges_from(graph['edges'])
		# for e in nx_graph.edges:
			# print(e,'edge')
			# print(nx_graph.has_edge(e[1],e[0]),'both ways?')
	nx_old=nx_graph

	adj_mat = nx.adjacency_matrix(nx_graph)
	# adj_mat = nx.adjacency_matrix(nx_old,'old') 
	spr_mat=sp.csr_matrix(adj_mat)
	x, y = sp.triu(adj_mat,k=1).nonzero() ## k=1 ignores the diagonal/ self edges, double check tomorrow
	pos_edges = np.array(list(zip(x, y))).astype(int)
	np.random.shuffle(pos_edges)
	pos_edges = torch.tensor(pos_edges).long()
	x, y = sp.triu(sp.csr_matrix(1. - adj_mat.toarray()),k=1).nonzero()
	neg_edges = np.array(list(zip(x, y))).astype(int)
	np.random.shuffle(neg_edges)
	neg_edges = torch.tensor(neg_edges).long()
	
	if use_virtual:
		for i in range(node_num+1):
			# print("EFFF")
			# print()
			nx_graph.add_edge(i,node_num,weight=1) ## adds virtual node
			# nx_graph.add_edge(i,i,weight=1)  ## adds self node
			# nx_graph.add_edge(i,node_num,weight=.5) ## adds virtual node
			nx_graph.add_edge(i,i)  ## adds self node
	else:
		for i in range(node_num):
			# nx_graph.add_edge(i,node_num) ## adds virtual node
			nx_graph.add_edge(i,i)  ## adds self node	

	node_feature = graph['node_features']
	if isinstance(node_feature[0], float): ## so ugly
		node_feature=np.array(node_feature,dtype=np.integer)
	if isinstance(node_feature[0], int): ## change to onehot.--- yo we gotta get this fixed ### no reason not to have in get_data
		new_node_feature = np.zeros((len(node_feature), self.args.num_feature))
		for i in range(len(node_feature)):
			new_node_feature[i][node_feature[i]] = 1
		node_feature = new_node_feature.tolist() ### 

	if use_virtual: ## for now not going to add extra
		for i in range(len(node_feature)):
			node_feature[i]+=[0]
		node_feature.append(one_hot_vec(len(node_feature[0]),-1))
	if len(node_feature[0]) < args.num_feature: ## if we want to add more stuff.
		zeros = np.zeros((len(node_feature), args.num_feature - len(node_feature[0])))
		node_feature = np.concatenate((node_feature, zeros), axis=1).tolist()



	adj_mat = nx.adjacency_matrix(nx_graph) 
	spr_mat=sp.csr_matrix(adj_mat)
	adj_mat,node_feature = process(spr_mat, node_feature, normalize_adj=False, normalize_feats=False)
	labels = torch.tensor([graph['targets'][0] for i in range(node_num)]).long()

	# assert 0 in pos_edges or 4 in pos_edges or 10 in pos_edges ### sloppy check to make sure this assertion has meaning
	assert node_num==90
	assert node_num not in pos_edges ### making sure virtual node not included
	assert node_num not in neg_edges ### " "  

	for e in pos_edges:
		if e[0]==e[1]:

			print(e)
			raise Exception("SELF EDGE IN POS EDGES ALERT ALERT ALERT ")
		if (e[0]==node_num) or (e[1]==node_num):
			print(e)
			raise Exception("VIRTUAL EDGE IN POS EDGES ALERT ALERT ALERT ")

	return  {
	          'features': node_feature,
	          'adj_mat': adj_mat,
	          # 'weight': weight,
	          'edges':pos_edges,  ### can slice down to len 2 here
	          'edges_false':neg_edges, ### do sample or calculate all
	          'labels': labels,
	          'graph_id':graph_id,
	          'adj_prob':adj_prob,
	          # 'len_edges':pos_edges.shape[0],
	          # 'len_edges_false':neg_edges.shape[0]}
	          }
		
