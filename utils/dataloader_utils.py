# from dataset.SyntheticDataset import SyntheticDataset
from dataset.GraphDataset import GraphDataset
# from dataset.LinkNodeDataset import LinkNodeDataset
from data.MEG.get_data import preprocess, get_average_data
# from data.MEG.get_data_linknode import preprocess as preprocess_linknode
from torch.utils.data import Dataset, DataLoader
from hyperbolic_learning_master.utils.embed import train_embeddings, load_embeddings,load_embeddings_npy, evaluate_model,mean_average_precision
import os
import pandas as pd
import numpy as np
import networkx as nx
import torch
from torch.utils.data.dataloader import default_collate
from datetime import datetime

def collate_fn(batch):
    ## turns adj_mat into a proper adjacency matrix instead of an adjacency list??
    ## problem is this can slow down if small graph paired with bigone ?

    # print(batch,'BATCH')
    # print(batch[0])
    # print(batch[0].keys(),'keys me')
    # return batch
    max_num_edges = -1
    max_num_edges_false = -1
    start=datetime.now()
    for data in batch:
    #     for row in data['edges']:
        num_edges=data['edges'].shape[0]
        num_edges_false=data['edges_false'].shape[0]

        max_num_edges_false = max(max_num_edges_false,num_edges_false)
        max_num_edges = max(max_num_edges,num_edges)
        # print(num_edges,'edge')
        # print(num_edges_false,'edge False')

    # print(max_num_edges,'edge max')
    # print(max_num_edges_false,'edge False max')



    for data in batch:
        # pad the adjacency list
        ##

        # extra_rows = max_neighbor_num-len(data['adj_mat'])



        data['edges_false'] = pad_sequence(data['edges_false'], maxlen=max_num_edges_false)
        data['edges'] = pad_sequence(data['edges'], maxlen=max_num_edges)
        # data['weight'] = pad_sequence(data['weight'], maxlen=max_neighbor_num)


        # data['node'] = np.array(data['node']).astype(np.float32)
        # node_size = data['node'].shape[-1]
        # data['adj_mat'] = np.array(data['adj_mat']).astype(np.int32)

        # data['weight'] = np.array(data['weight']).astype(np.float32)
        # data['label'] = np.array(data['label'])
        # ### this is adj matrix so must be square
        # ### in og, there were no extra rows so it wasn't an issue.
        # if extra_rows>0:

        #     data['adj_mat'] = np.append(data['adj_mat'],np.zeros((extra_rows,max_neighbor_num)),axis=0) ## padding row count, could make using batches a mistake
        #     data['weight'] = np.append(data['weight'],np.zeros((extra_rows,max_neighbor_num)),axis=0) ## padding row count
        #     data['node'] = np.append(data['node'],np.zeros((extra_rows,node_size)),axis=0) ## padding row count

    mid=datetime.now()
    # print(mid-start,'our time')
    output=default_collate(batch)
    end=datetime.now()
    # print(end-mid,'default time')
    
    # print(end-start,'total time')
    

    return output

def pad_sequence(seq,maxlen):
    oglen=seq.shape[0]
    remlen=maxlen-oglen
    # print(seq.shape[1:])
    remdims=tuple([remlen]+[seq.shape[1]])
    # if remlen==0:
        # return seq
    addseq=torch.full(remdims,fill_value=-1)
    newseq=torch.cat([seq,addseq])
    # print(newseq)
    return newseq

def load_data_graphalg(args):
    # return load_dataset(SyntheticDataset, collate_fn)  
    return load_dataset(args,SyntheticDataset)  

# def load_data(args):
#     # return load_dataset(SyntheticDataset, collate_fn)  
#     return load_dataset(args,SyntheticDataset) 

def load_meg_averages(args,split_cols): 
    return get_average_data(args,split_cols)

def load_dataset(args,dataset_name, distributed=True):
    # train_dataset = dataset_class(args, logger, split='train')
    # dev_dataset = dataset_class(args, logger, split='dev')
    # test_dataset = dataset_class(args, logger, split='test')
    # if hasattr(args,'avg_scans') and args.avg_scans:
    #     dataset_class= GraphDataset
    #     preprocess_func=preprocess
    if dataset_name=='synthetic':
        dataset_class= SyntheticDataset
        preprocess_func=preprocess
    elif dataset_name=='meg_ln':
        print('lets get it')
        dataset_class=LinkNodeDataset
        preprocess_func=preprocess_linknode

    else:
        print('in here ')
        print(dataset_name,'DATA NAME')
        dataset_class= GraphDataset
        preprocess_func=preprocess

    if hasattr(args,'refresh_data') and args.refresh_data>0:
        train_file,dev_file,test_file,all_file,idxs_dict,indx_file = preprocess_func(args)
        args.train_file=train_file
        args.dev_file=dev_file
        args.test_file=test_file
        args.all_file=all_file
        args.idxs_dict=idxs_dict   ### important bc this won't change for any model
        args.indx_file=indx_file ## this will be constantly overwritten

    train_dataset = dataset_class(args, split='train')
    dev_dataset = dataset_class(args, split='dev')
    test_dataset = dataset_class(args, split='test')

    train_sampler, dev_sampler, test_sampler = None, None, None
    # setattr(args,'use_batch',1)
    # setattr(args,'batch_size',4)

    batch_size=args.batch_size if args.use_batch else 1 ### can't just use default 1 bc we want to keep original 'no batch' way

    shuffle = True

    no_test=True
    if hasattr(args,'train_only') and args.train_only:
        if not (hasattr(args,'criteria_dict') and args.criteria_dict):

            assert (len(train_dataset))==180
    elif no_test:
        print(len(train_dataset),len(dev_dataset))
        assert (len(train_dataset)+len(dev_dataset))==180
    else:
        assert (len(train_dataset)+len(test_dataset)+len(dev_dataset))==180 
    # else:
        # raise 
    print(len(train_dataset),len(test_dataset),len(dev_dataset))
    # sss
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size,  ### original location
                                     num_workers=0, sampler=train_sampler,shuffle=shuffle,collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=1,
                                     num_workers=0, sampler=dev_sampler,shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=1,
                                     num_workers=0, sampler=test_sampler,shuffle=shuffle)
    return train_loader, dev_loader, test_loader

def load_dataset_linknode(args,dataset_name, collate_fn=None, distributed=True):
    # train_dataset = dataset_class(args, logger, split='train')
    # dev_dataset = dataset_class(args, logger, split='dev')
    # test_dataset = dataset_class(args, logger, split='test')
    if dataset_name=='synthetic':
        dataset_class= SyntheticDataset
    elif dataset_name=='meg_ln':
        print('lets get it')
        dataset_class=LinkNodeDataset
        preprocess_func=preprocess_linknode
    else:
        print('in here ')
        print(dataset_name,'DATA NAME')
        dataset_class= GraphDataset
        preprocess_func=preprocess



    if hasattr(args,'refresh_data') and args.refresh_data>0:
        train_file,dev_file,test_file = preprocess_func(args)
        args.train_file=train_file
        args.dev_file=dev_file
        args.test_file=test_file
    train_dataset = dataset_class(args, split='train')
    dev_dataset = dataset_class(args, split='dev')
    test_dataset = dataset_class(args, split='test')

    train_sampler, dev_sampler, test_sampler = None, None, None
    batch_size=args.batch_size if args.use_batch else 1 ### can't just use default 1 bc we want to keep original 'no batch' way
    shuffle = True
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size,  ### original location
                                     num_workers=0, sampler=train_sampler,shuffle=shuffle)
    dev_loader = DataLoader(dev_dataset, batch_size=1,
                                     num_workers=0, sampler=dev_sampler,shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=1,
                                     num_workers=0, sampler=test_sampler,shuffle=shuffle)
    return train_loader, dev_loader, test_loader




def analyze_dataset_hyperbolicity(args,dataset_name,graph_samples=None,node_samples=None):
    if dataset_name=='synthetic':
        dataset_class= SyntheticDataset
    else:
        dataset_class= GraphDataset


    train_dataset = dataset_class(args, split='train')
    # dev_dataset = dataset_class(args, split='dev')
    # test_dataset = dataset_class(args, split='test')


    if node_samples:
        res = train_dataset.analyze_hyperbolicity(graph_samples,node_samples)
    else:
        res = train_dataset.analyze_hyperbolicity(graph_samples)
    return res

def save_all_edge_dfs_handler(args,dataset_name):
    if dataset_name=='synthetic':
        dataset_class= SyntheticDataset
    else:
        dataset_class= GraphDataset

    train_dataset = dataset_class(args, split='train')
    dev_dataset = dataset_class(args, split='dev')
    test_dataset = dataset_class(args, split='test') 

    root_dir = args.data_root+"\\"+str(args.adj_threshold)[2:]   ### to get rid of decimal
    print(root_dir,'ROOT DIRECTORY')
    train_dataset.save_all_edge_dfs(root_dir)
    dev_dataset.save_all_edge_dfs(root_dir)
    test_dataset.save_all_edge_dfs(root_dir)

def poincare_embeddings_all_handler(args,dataset_name,output_rel_dir,output_emb_dir):
    if dataset_name=='synthetic':
        dataset_class= SyntheticDataset
    else:
        dataset_class= GraphDataset

    train_dataset = dataset_class(args, split='train')
    dev_dataset = dataset_class(args, split='dev')
    test_dataset = dataset_class(args, split='test') 

    root_dir = args.data_root+"\\"+str(args.adj_threshold)[2:]   ### to get rid of decimal
    print(root_dir,'ROOT DIRECTORY')
    train_dataset.poincare_embeddings_all(output_rel_dir,output_emb_dir)
    dev_dataset.poincare_embeddings_all(output_rel_dir,output_emb_dir)
    test_dataset.poincare_embeddings_all(output_rel_dir,output_emb_dir)    
# def create_


def poincare_embeddings_all(relations_dir,output_dir,repeat=0):
    ### no need to make csv only?
    relation_files = [f for f in os.listdir(relations_dir) if ('csv' in f) and ('relation' in f)]
    relation_files = [f for f in os.listdir(relations_dir) if ('csv' in f)]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) ### holey shite thats convenient
        # os.mkdir(output_dir)

    for f in relation_files:
        graph_id = f[:-4]
        # print(graph_id,'should be all before csv')
        ### should go all the way to relations
        # relations_path = os.path.join(relations_dir,graph_id+'_relations.tsv')
        relations_path = os.path.join(relations_dir,graph_id+'.csv')
        output_path = os.path.join(output_dir,graph_id+'_embeddings.tsv')
        poincare_embeddings(relations_path,output_path)
        for r in range(repeat):
            output_path = os.path.join(output_dir,graph_id+'_embeddings_copy_{}.tsv'.format(str(r)))
            poincare_embeddings(relations_path,output_path)

def poincare_embeddings(relations_path, output_path):  ## really no reason for this to be a dataset function? but it will keep things organized
    delimiter=','
    train_embeddings(relations_path,delimiter,output_path)  


def relations_to_degree_analysis(relations_dir,output_dir=''):

    columns=['degseq_file','num_edges', 'Weighted', 'Directed',
                                     'Bipartite', 'Multigraph', 'Multiplex',
                                     'fp_gml', 'n', 'alpha', 'xmin','ntail',
                                     'Lpl', 'ppl', 'dexp', 'dln', 'dstrexp',
                                     'dplwc', 'meandeg']
    analysis = pd.DataFrame(columns=columns)

    rel_files= [os.path.join(relations_dir,f) for f in os.listdir(relations_dir)]

    rel_files=[f for f in os.listdir(relations_dir) if 'relations.csv' in f ]

    degree_seq_output=os.path.join(output_dir,'degseq')
    if not os.path.exists(degree_seq_output):
        os.makedirs(degree_seq_output)

    for f in rel_files:
        fn = os.path.join(relations_dir,f)

        file_base=f[:-4] ## should get 'csv'
        print(file_base)
        new_file=file_base+'_degseq.csv'  ## csv for now
        print(new_file,'NEW FILE')
        degseq_file= os.path.join(degree_seq_output,new_file)
        print(degseq_file,'DEG SEQ')
        nx_graph =nx.Graph()
        edges = np.array(pd.read_csv(fn))
        print(edges,'EDGES')
        for e in edges:

            nx_graph.add_edge(e[0],e[1])

        degree_dict={}
        for n,d in nx_graph.degree():
            if d in degree_dict:
                degree_dict[d]+=1
            else:
                degree_dict[d]=1
        degree_sequence=[]
        for i in range(max(degree_dict.keys())):
            if i in degree_dict:
                degree_sequence.append([i,degree_dict[i]])
        # degree_sequence = sorted((d for n, d in nx_graph.degree()), reverse=True)
        print(degree_sequence)

        degree_df = pd.DataFrame(data=degree_sequence,columns=['xvalue','counts'])
        print(degree_df)
        degrees = np.array([nx_graph.degree(node) for node in nx_graph.nodes()])

        # print
        num_edges = len([e for e in nx_graph.edges()])

        # print
        mean_degree=np.mean(degrees)
        print(mean_degree)
        gsize=len([n for n in nx_graph.nodes()])
        domain=''
        subdomain=''
        fp=fn ## put something

        if degseq_file not in analysis.index:

            analysis.loc[degseq_file] = ''
        analysis.loc[degseq_file]['Domain'] = degseq_file
        analysis.loc[degseq_file]['degseq_file'] = degseq_file
        analysis.loc[degseq_file]['Subdomain'] = subdomain
        analysis.loc[degseq_file]['fp_gml'] = fp
        analysis.loc[degseq_file]['Graph_order'] = gsize
        analysis.loc[degseq_file]['num_edges'] = num_edges
        analysis.loc[degseq_file]['meandeg'] = mean_degree
        analysis.loc[degseq_file]['Weighted'] = False
        analysis.loc[degseq_file]['Directed'] = False
        analysis.loc[degseq_file]['Bipartite'] = False
        analysis.loc[degseq_file]['Multigraph'] = False
        analysis.loc[degseq_file]['Multiplex'] = False

        # degree_df.replace
        degree_df.to_csv(degseq_file, index=False)

        # break


    csvfile=os.path.join(output_dir,"degree_sequence.csv")
    # print(csv)
    print(analysis,'ANALUSIS')
    analysis.to_csv(csvfile,index=False)
    # df.to_csv(csvfile, index=False)




