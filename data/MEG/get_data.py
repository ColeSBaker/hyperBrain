#!/usr/bin/env/python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys, os
# sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
# from rdkit import Chem
# from rdkit.Chem import rdmolops
# from rdkit.Chem import QED
from copy import deepcopy
import glob
import csv, json
import numpy as np
# from utils import bond_dict, dataset_info, need_kekulize, to_graph
# import utils
import pickle
import random
# import wget
import pandas as pd 
import networkx as nx
from random import randint
from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.metrics import jaccard_score
from utils.hgnn_utils import one_hot_vec
from utils.math_utils import tanimoto_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
dataset_str = "MEG"
all_values = []
label_names = ['diagnosis']

BAND_TO_INDEX = {'theta':0,'alpha':1,'beta1':2,'beta2':3,'beta':4,'gamma':5}
METRIC_TO_INDEX = {'plv':0,'ciplv':1}

# if 'gdrive/MyDrive' in args.raw_clinical_file:
default_clinical=os.path.join(os.getcwd(),'data/MEG/MEG.clinical.csv')
default_MEG=os.path.join(os.getcwd(),'data/MEG/MEG.ROIs.npy')
default_temp=os.path.join(os.getcwd(),'data/MEG/AALtemplate.csv')
default_dataroot=os.path.join(os.getcwd(),'data/MEG')
def preprocess(args):
    # mean and std
    # mean = np.mean(all_values, axis=0).tolist()
    # std = np.std(all_values, axis=0).tolist()
    # mean = np.array(mean)
    # std = np.array(std)

    # dataset = np.load(os.path.join(data_path, "{}.ROIs.npy".format(dataset_str)))
    # clinical = pd.read_csv(os.path.join(data_path, "{}.clinical.csv".format(dataset_str))) ## should happen earlier?
    if os.getcwd() not in args.raw_clinical_file:
        args.raw_clinical_file=default_clinical
    if os.getcwd() not in args.raw_atlas_file:
        args.raw_atlas_file=default_temp
    if os.getcwd() not in args.raw_scan_file:
        args.raw_scan_file=default_MEG
    if os.getcwd() not in args.data_root:
        args.data_root=default_dataroot
    # else:
        # download_MEG = args.raw_scan_file

    atlas_file = args.raw_atlas_file
    download_MEG = args.raw_scan_file
    download_clinical=args.raw_clinical_file
    root=args.data_root

    dataset_str = "MEG"
    all_values = []
    label_names = ['diagnosis']
    atlas = download_atlas(atlas_file)
    print(args,'ARGS')

    val_subgroup = 1 if hasattr(args,'val_sub') and args.val_sub==1 else 0
    criteria_dict= args.criteria_dict if  hasattr(args,'criteria_dict') else {}
    val_pct = .95 if args.train_only else getattr(args,'val_prop')
    # use_exclude=False if ((args.train_only) or (val_subgroup>0)) else True
    use_exclude=False
    raw_data,idxs_dict = dataset_split(download_clinical,download_MEG,
         getattr(args,'val_prop'),getattr(args,'test_prop'),val_subgroup,use_exclude,criteria_dict=criteria_dict)

    if (hasattr(args,'train_noise_level')) and (args.train_noise_level)>0:
        train_noise_level=(args.train_noise_level)
        train_noise_num=args.train_noise_num
    else:
        train_noise_level=0
        train_noise_num=0

    print(idxs_dict,'index me')
    norm_functions=create_norm_functions(args)
    print(norm_functions.keys(),'keys')

    print('parsing scans as graphs...')
    processed_data = {'train': [], 'valid': [], 'test': [],'all':[]}
    file_count = 0
    all_idtest=set([])
    for section in ['train', 'valid', 'test']:
        # for i, (smiles, prop) in enumerate([(mol['smiles'], mol['prop'])  ### at some point we might make actual ROI feats (ie. location in brain?)
        #                                   for mol in raw_data[section]]):

        for i, (scan, clinical) in enumerate([(scan['scan'], scan['clinical'])  ### you have two scan variables!!! wft??
                                          for scan in raw_data[section]]):

            # print(section,i)
            if clinical['Scan Index']in all_idtest:
                raise Exception('Already Seen Index:',clinical['Scan Index'])
            all_idtest.add(str(clinical['Scan Index']))

            data_dict = graph_data_dict_recursive(args,scan,clinical,label_names,
                atlas=atlas,norm_functions=norm_functions,noise_num=train_noise_num,noise_level=train_noise_level)

            # edges,feats,labels,adj_prob,adj_noise = to_graph(args,scan,clinical,label_names,atlas=atlas,norm_functions=norm_functions)

            # labels = np.array(labels).astype(float)
            # feats = np.array(feats).astype(float)

            print(len(data_dict['edges']))

            # data_dict= {
            #     'targets': labels.tolist(),
            #     'edges': edges,
            #     'node_features': feats.tolist(),
            #     'graph_id': str(clinical['Scan Index']),
            #     'adj_prob': adj_prob.tolist(),
            #     'noise_data':[]}

            # print(data_dict_test,'testerr')
            # print(data_dict,'old')
            # print(data_dict_test['noise_data'][0]['edges']==data_dict_test['edges'])
            # print([len(data_dict_test['noise_data'][i]['edges'])  for i in range(train_noise_num)] ,'before')
            # print(len(data_dict_test['edges']),'before')
            # # print(len(data_dict_test['noise_data']))
            # print(data_dict['node_features'])
            num_feats=len(data_dict['node_features'][0])+1  ### to set args below
            # print(num_feats)  ## probably should just do this at start sum up numbers for each use__ and then add firt
            processed_data[section].append(data_dict)
            processed_data['all'].append(data_dict)
            
            # print(args.num_features,'new')



            # dddd
            if file_count % 40 == 0:
                print('finished processing: %d' % file_count, end='\r')
            file_count += 1
        print('%s: 100 %%      ' % (section))
        # save the dataset
        # print(processed_data[section].keys())
        save_file = os.path.join(root,'meg_%s_%s_%s.json' % (section, dataset_str,args.adj_threshold))
        # with open(save_file, 'w') as f:
            # json.dump(processed_data[section], f)
        if section=='train':
            train_file=save_file
        if section=='test':
            test_file=save_file
        if section=='valid':
            valid_file=save_file

    if args.num_feature!=num_feats:
        print("ADJUSTING NUM FEATS")
        print(args.num_feature,' to ',num_feats)
        setattr(args,'num_feature',num_feats)

    all_file = os.path.join(root,'meg_%s_%s_%s.json' % ('all', dataset_str,args.adj_threshold))
    indx_file = os.path.join(root,'meg_%s_%s_latestindx.json' % ( dataset_str,args.adj_threshold))

    with open(all_file, 'w') as f:
        json.dump(processed_data['all'], f)

    # print(idxs_dict)
    with open(indx_file, 'w') as f:
        json.dump(idxs_dict, f)

    print(len(all_idtest),'unique ids')
    print(file_count,'Final count')
    # pri
    # cant we get rid of these files??
    return None,None,None,all_file,idxs_dict,indx_file

def graph_data_dict_recursive(args,scan,clinical,label_names,atlas,norm_functions={},noise_num=0,noise_level=0,is_root=True):
    # print(scam)
    if is_root: ### base graph, means we 
        edges,feats,labels,adj_prob,_,adj_true  = to_graph(args,scan,clinical,label_names,atlas=atlas,norm_functions=norm_functions)
    else:
        edges,feats,labels,adj_prob,_,adj_true  = to_graph(args,scan,clinical,label_names,atlas=atlas,norm_functions=norm_functions,add_noise=noise_level)

    if len(edges) <= 0:
        print('NO EDGES???')
    # print(len(edges))
    base_data_dict= {
            'targets': labels.tolist(), #from graph
            'edges': edges, # from graph
            'node_features': feats.tolist(),# from graph
            'graph_id': str(clinical['Scan Index']), # from graph
            'adj_prob': adj_prob.tolist(), # from graph
            'adj_original':adj_true.tolist(),
            'noise_data':[]} # from graph make
    if noise_level<=0 or (noise_num==0) or (not is_root):  ### could do not "is root" bc we should do more than one layer..
    # if noise_level<=0 or (noise_num==0) or (not is_root):  ### could do not "is root" bc we should do more than one layer..
        return base_data_dict

    for n in range(noise_num):
        # data_dict_n=graph_data_dict_recursive(args,scan,clinical,label_names,atlas=atlas,
            # norm_functions=norm_functions,noise_num=0,noise_level=0,is_root=False)
        data_dict_n=graph_data_dict_recursive(args,scan,clinical,label_names,atlas=atlas,
            norm_functions=norm_functions,noise_num=0,noise_level=noise_level,is_root=False)
        base_data_dict['noise_data'].append(data_dict_n)
    return base_data_dict



def to_graph(args,scan,clinical,label_names,atlas,norm_functions={},return_label=False,add_noise=0.):
    # num_brains,NUM_BANDS,Num_Metric,NUM_ROIS,NUM_ROIS
    # 
    scan=deepcopy(scan)
    block = False  ## incase we want one big adj matrix
    adj_threshold = args.adj_threshold
    # print(scan.shape,'SCAN SHAPE!')

    # edges,sim = to_link_graph(args,scan)

    # print(edges,sim,'edges and simmy shimmy')
    # use_ciplv 
    # print("THRESHOLD: ",adj_threshold)
    max_degree=30
    min_degree = 10
    tot_degree = max_degree-min_degree

    band_adj = BAND_TO_INDEX[args.band]
    metric_adj = METRIC_TO_INDEX[args.metric]

    num_rois = scan.shape[3]  ### already narrowed down to one
    single_scan = scan[band_adj,metric_adj]  ## will take more work to do multi-edges
    # print(add_noise,'ADD NOISE')
    # print(single_scan[0,:],'FIRST')
    # sdd
    labels = clinical[label_names]
    # print(single_scan.mean(),'mean before')
    # print(single_scan.shape,'singlit')
    # print(single_scan,'scan it')
    
    # mean= ratio/add_noise
    # -.002
    # # mean = np.random.uniform(-0.002,.001),  ### if we wanted to add noise in positive way
    # mean=-.0015
    # if add_noise>0:
    #     print(mean/add_noise,'ratio')
    # # if add_noise>0:/
        # noise = np.random.normal(mean,single_scan.std()*add_noise,size=single_scan.shape)

    if add_noise>0:
        ratio=-.07
        mean_det= ratio*add_noise
        # print(mean_det,'MEAN')
        mean = np.random.uniform(mean_det,-mean_det/6)
        single_scan=norm_functions['add_noise'](single_scan,rel_noise=add_noise,mean=mean)
        # print(single_scan[0],'after example')
        adj_noise=single_scan
    else:
        single_scan=single_scan
        adj_noise=single_scan

    # print(adj_noise.std(),'STANDARD NEW',adj_noise.mean(),'mean')
    # adj = np.where(single_scan>adj_threshold,1,0)  ## we can do weighted homie
    adj = np.where(single_scan>adj_threshold,1,0)  ## we can do weighted homie
    # print(adj_threshold,'ADJ THRESHOLD,!!!')



    # adj_inv = np.where(single_scan<adj_threshold,1,0)  ## we can do weighted homie
    # print(adj.mean(),'mean after',adj_threshold)
    # print(adj_inv.mean())
    # print('ADJJJJ')
    # new_adjusted = (adj_flat-np.min(adj_flat))/(np.max(adj_flat)-np.min(adj_flat))
    # new_adjusted_clipped = (adj_flat-np.percentile(adj_flat,5))/(np.percentile(adj_flat,95)-np.percentile(adj_flat,5))
    # new_adjusted_clipped=np.clip(new_adjusted_clipped,0,1)
    adj_flat=[]
    for j in range(single_scan.shape[0]):
        for k in range(j+1, single_scan.shape[0]):
            adj_flat.append(single_scan[j, k])
    # print(norm_functions.keys(),'final')
    # print(norm_functions['mm_strech'])


    if hasattr(args,'stretch_sigmoid') and args.stretch_sigmoid:
        # adj_prob=norm_functions['ms_sigmoid'](single_scan,.36,.04,identity_treatment='1')
        adj_prob=norm_functions['ms_sigmoid'](single_scan,args.stretch_r,args.stretch_t,identity_treatment='1')
        plv_loss_func='sig'

    else:
        if hasattr(args,'stretch_loss') and args.stretch_loss:
            # print('STRECH LOS')
            stretch_loss=args.stretch_loss
        else:
            args.stretch_loss=95
            stretch_loss=95
        plv_loss_func='stretch'
        # # adj_prob=norm_functions['ms_sigmoid'](single_scan,.35,.04,identity_treatment='1')
        adj_prob=norm_functions['mm_strech'](single_scan,mm_min_pct=100-stretch_loss,mm_max_pct=stretch_loss,mm_clip_data=True,identity_treatment='Raw') ## seems like set to max is already cov
        # adj_prob=norm_functions['mm_strech'](single_scan,mm_min_pct=100-stretch_loss,mm_max_pct=stretch_loss,mm_clip_data=True,identity_treatment='set_to_max') ## seems like set to max is already covered

    # print(add_noise,'how much noise?')
    # print(adj_prob[0],'prob 1')
            

            # adj_prob=(single_scan-np.min(adj_flat))/(np.max(adj_flat)-np.min(adj_flat))
            # adj_flat_adj=(adj_flat-np.min(adj_flat))/(np.max(adj_flat)-np.min(adj_flat))
        
    # for i in range(adj.shape[0]):  ## self loop!!
        # adj[i,i]=1
    assert atlas.iloc[4]['Index']==4
    assert atlas.iloc[55]['Index']==55


    G = nx.convert_matrix.from_numpy_matrix(adj)
    adj = nx.adjacency_matrix(G)

    use_ciplv=args.use_ciplv
    use_plv=args.use_plv
    use_beta=args.use_beta
    use_identity=args.use_identity
    try:
        use_identity_sim=args.use_identity_sim
    except:
        use_identity_sim=False

    use_coords =args.use_coords
    use_degree=args.use_degree
    first=True
    ##put some thought into the args and sustainability!!!
    # need to have some indicators of what goes where.. pass in as pandas list???

    feat_list=[]  ### Work in progress for more seamless concating
    if use_ciplv:
        band_feat = BAND_TO_INDEX['alpha']
        metric_feat = METRIC_TO_INDEX['ciplv']
        feats =scan[band_feat,metric_feat]  ### need to do feats here?
        feats=feats.astype(float)
        feat_list.append(feats)
        first=False

    if use_plv:
        band_feat = BAND_TO_INDEX['alpha']
        metric_feat = METRIC_TO_INDEX['plv']
        # feats =deepcopy(scan[band_feat,metric_feat])  ### need to do feats here?
        feats=single_scan ## needs to be the same if adding noise?s

        if hasattr(args,'plv_pca_dim') and args.plv_pca_dim>0:
            feats=norm_functions['pca_transform'](single_scan)

        elif hasattr(args,'match_plv_inp_loss') and args.match_plv_inp_loss:
            # plt.hist(feats.flatten(),bins=30)
            # plt.show()
            if plv_loss_func=='sig':
                feats=norm_functions['ms_sigmoid'](feats,args.stretch_r,args.stretch_t,identity_treatment='1')

            elif plv_loss_func=='stretch':
                # same as in input func except id_treatment (which is irrelevant for loss func, because we ignore self-loss.)
                feats=norm_functions['mm_strech'](feats,mm_min_pct=5,mm_max_pct=100,mm_clip_data=False,identity_treatment='set_to_max') ## seems like set to max is already covered

            else:
                raise Exception('cannot mimic {}'.format(plv_loss_func))

            plt.hist(single_scan.flatten(),bins=50)
            plt.hist(feats.flatten(),bins=50)
            # plt.hist(oldway.flatten(),bins=30)
            plt.show()
        elif hasattr(args,'plv_inp_raw') and args.plv_inp_raw:
            feats=norm_functions['identity'](feats,identity_treatment='1')
            # plt.hist(single_scan.flatten(),bins=50)
            # plt.show()
            # plt.hist(feats.flatten(),bins=50)
            # # plt.hist(oldway.flatten(),bins=30)
            # plt.show()

        else:
            if hasattr(args,'stretch_pct'):
                stretch_pct=args.stretch_pct
            else:
                stretch_pct=95
            if hasattr(args,'plv_norm_w_id'):
                include_id_std=args.plv_norm_w_id
            else:
                include_id_std=False

            feats=norm_functions['ms_norm'](feats,mm_clip_data=False,identity_treatment='set_to_pct',percentile_to_set=stretch_pct,include_id_std=include_id_std)
            # plt.hist(feats.flatten(),bins=50)
            # plt.hist(single_scan.flatten(),bins=50)
            # plt.show()
        feats=feats.astype(float)
        feat_list.append(feats)
        print(feats.std(),'new standard')
        print(feats.mean(),'new mean')
        first=False
    if use_beta:
        band_feat = BAND_TO_INDEX['beta']
        metric_feat = METRIC_TO_INDEX['plv']
        if first:
            feats =scan[band_feat,metric_feat]  ### need to do feats here?
            first=False
        else:
            feats = np.concatenate((feats, scan[band_feat,metric_feat]), axis=1)
        feats=feats.astype(float)
        feat_list.append(feats)
        # print(feats.shape,'FEATS1')
    if use_identity:
        one_hot = np.identity(num_rois)
        if first:
            feats =one_hot  ### need to do feats here?
            first=False
        else:
            feats = np.concatenate((feats, one_hot), axis=1)
        feat_list.append(feats)
        # feats=one_hot

        feats=feats.astype(float)
    if use_identity_sim:
        id_sim=np.zeros((num_rois,(num_rois//2)+2))
        # print(id_sim.shape,'SHAPELY')
        for i in range(num_rois):
            roi_num=i//2
            l_sob_num=i%2
            r_sob_num=(i+1)%2

            id_sim[i,roi_num]=1
            id_sim[i,(num_rois//2)]=l_sob_num
            id_sim[i,(num_rois//2)+1]=r_sob_num
            # print(i,'THIS ROI')
        if first:
            feats =id_sim  ### need to do feats here?
            first=False
        else:
            feats = np.concatenate((feats, id_sim), axis=1)
        feats=feats.astype(float)
    # node_features = []
    if use_coords:
        coords = atlas[['X','Y','Z']]
        if first:
            feats =coords  ### need to do feats here?
            first=False
        else:
            feats = np.concatenate((feats, coords), axis=1)
        feat_list.append(feats)
        # print(feats.shape,'FEATS2')
    if use_degree: ##make one hot degree
        degs=[]
        for n in G.nodes:
            d = G.degree(n)
            # print(one_hot_vec(max_degree,d),'degre earlier')
            if d> max_degree:
                d=max_degree
            elif d<min_degree:
                d=min_degree
            d = d-min_degree
            # print(d,max_degree,'max')
            # print(one_hot_vec(max_degree,d),)
            degs.append(one_hot_vec(max_degree,d))
            # print(np.array(one_hot_vec(max_degree,d)).shape,'how much?')
            # print(one_hot_vec(max_degree,-1),)

            # if d < max_degree:
            #     print(max_degree,'max degree')
            #     print(one_hot_vec(max_degree,-1),)
            #     degs.append(one_hot_vec(max_degree,d))
            # else:
            #     print(max_degree,'max degree')
            #     print(one_hot_vec(max_degree,-1),)
            #     degs.append(one_hot_vec(max_degree,-1))
        degs=np.array(degs)
        # print(degs.shape,'deg shape')
        # print(feats.shape,'BEFORE ON GOING BB')


        if first:
            feats =degs  ### need to do feats here?
            first=False
        else:
            feats = np.concatenate((feats, degs), axis=1)
    # print(feats.shape,'FEAT SHAP')
    # print(add_noise,'noisey')
    # print(feats,'FEATS')
    print(feats.shape,'FEATS SHOULD BE HIGH')
    # sss
    return [e for e in G.edges(data=True)], feats,labels.astype(float)-1,adj_prob,adj_noise,single_scan
    # return get_edges(G),feats.astype(float),labels.astype(float)

def get_average_data(args,subject_cols):
    if os.getcwd() not in args.raw_clinical_file:
        args.raw_clinical_file=default_clinical
    if os.getcwd() not in args.raw_atlas_file:
        args.raw_atlas_file=default_temp
    if os.getcwd() not in args.raw_scan_file:
        args.raw_scan_file=default_MEG
    if os.getcwd() not in args.data_root:
        args.data_root=default_dataroot
    ## for now, just on subject column
    download_clinical = args.raw_clinical_file
    atlas_file = args.raw_atlas_file
    download_MEG = args.raw_scan_file
    root=args.data_root

    norm_functions=create_norm_functions(args)

    atlas = download_atlas(atlas_file)
    scan_data = np.load(download_MEG)
    raw_data = dataset_split(download_clinical,download_MEG)
    clinical_data = pd.read_csv(download_clinical)
    print(clinical_data.columns)
    print(clinical_data[subject_cols],'WHAT WE GOT?')
    clinical_groups = clinical_data[subject_cols].drop_duplicates()
    print(clinical_groups)
    label_names = ['diagnosis']
    # print(save_dir,'SAVE DIR')
    average_dict=[]
    # for g in clinical_groups:
    for i in range(clinical_groups.shape[0]):
        g=clinical_groups.iloc[i].values
        print(g,'G')
        print(subject_cols,'SUB COLS')
        # print(clinical_data[subject_cols].shape,'outshape')
        # print((clinical_data[subject_cols]==g).all(axis=1),'subs')
        truth=(clinical_data[subject_cols]==g).all(axis=1)
        scan = scan_data[truth].mean(axis=0)
        print(scan.shape,'scan shape??')
        clinical = clinical_data[clinical_data[subject_cols]==g].iloc[0] ### just one example patient bc label names is all we use it for 
        ## this will cause error, we've changed to_Graph
        data_dict = graph_data_dict_recursive(args,scan,clinical,label_names,
            atlas=atlas,norm_functions=norm_functions,noise_num=0,noise_level=0)

        full_dict={'data_dict':data_dict,'splits':{}}
        data_dict['splits']={}
        for i in range(len(subject_cols)):
            data_dict['splits'][subject_cols[i]]=g[i]
        print(data_dict.keys())
        average_dict.append(data_dict)

    return average_dict

def average_brain_edges(args,subject_cols,save_dir=''):
    ## for now, just on subject column
    download_clinical = args.raw_clinical_file
    atlas_file = args.raw_atlas_file
    download_MEG = args.raw_scan_file
    root=args.data_root

    atlas = download_atlas(atlas_file)
    scan_data = np.load(download_MEG)
    raw_data = dataset_split(download_clinical,download_MEG)
    clinical_data = pd.read_csv(download_clinical)
    label_names = ['diagnosis']

    clinical_groups = clinical_data[subject_cols].unique()
    
    print(save_dir,'SAVE DIR')
    # if not os.path.exists(os.path.join(save_dir,str(args.adj_threshold)[2:])):
    #     print('no exists')
    #     os.mkdir(os.path.join(save_dir,str(args.adj_threshold)[2:]))
    # if not os.path.exists(os.path.join(save_dir,str(args.adj_threshold)[2:],'relations')):
    #     print('no exists')
    #     os.mkdir(os.path.join(save_dir,str(args.adj_threshold)[2:],'relations'))
    # save_dir= os.path.join(save_dir,str(args.adj_threshold)[2:],'relations','special')
    if save_dir!='' and not os.path.exists(save_dir):
        print('no exists')
        os.makedirs(save_dir)
    # sdd
    
    for g in clinical_groups:
        average_dict[g]={}
        scan = scan_data[clinical_data[subject_cols]==g].mean(axis=0)
        # print(scan.shape,'meaned')

        clinical = clinical_data[clinical_data[subject_cols]==g].iloc[0] ### just one example patient bc label names is all we use it for 
        ## this will cause error, we've changed to_Graph
        edges,feats,_ = to_graph(args,scan,clinical,subject_cols,atlas)
        edges = np.array([[e[0],e[1]] for e in edges])
        # edges = pd.
        if type(subject_cols)==list:
            subject_col_str = subject_cols[0]
        else:
            subject_col_str = subject_cols
        graph_id = "AVG"+str(subject_col_str)+"_"+str(g)## no need to save other clinical data, we can retrieve as long as we have graph
        # print(edges)
        graph_save_dir = os.path.join(save_dir,str(graph_id)+'.csv')
        edge_df = pd.DataFrame(edges)
        if save_dir!='':
            edge_df.to_csv(graph_save_dir,index=False)
            print(scan.shape)

        average_dict[g]={'edge_df':edge_df,'scan':scan}

    return average_dict



def get_edges(G,bi_directional=False):
    ## these can go one way
    ## did most of this to get into format for HGnn
    one_way = [(e[0],G.get_edge_data(e[0],e[1])['weight'],e[1]) for e in G.edges]
    if bi_directional:
        other_way = [(e[1],G.get_edge_data(e[0],e[1])['weight'],e[0]) for e in G.edges]
        edges = one_way+other_way
    else:
        edges = one_way
    
    return edges

# def 
def get_scan_index_split_bygroup(clinical_data,in_group):
    print(in_group)
    criteria = (clinical_data['Scan Index'].isin(in_group))
    clinical_val=clinical_data[~criteria]
    clinical_train=clinical_data[criteria]

    print(clinical_val.shape)
    print(clinical_train.shape)

    # sjskg
    val_idx=clinical_val['Scan Index'].values[2:].astype(int).tolist()
    test_idx=clinical_train['Scan Index'].values[0:2].astype(int).tolist()
    return val_idx,test_idx


    # clinical_train=clinical_data[~criteria]

def get_scan_index_patient_split(clinical_data,val_pct,test_pct,pat_col = 'ID',use_exclude=True):
    ### returns scan ids that are partitioned by patient

    ### special condidtion make it so we can change
    criteria = (clinical_data['CogTr']==1)

    pats_full = clinical_data[pat_col].unique()

    print(use_exclude,'EXCLUSION')
    if use_exclude:
        pats= clinical_data[~criteria][pat_col].unique()
    else:
        pats=pats_full

    print(len(pats),'HOW MANY DID WE KEEP SHOULD BE 45')
  

    valid_ids= set()
    valid_idx = []

    test_ids= set()
    test_idx = []

    n_val = len(pats_full)*val_pct*2
    n_tests = len(pats_full)*test_pct*2
    print(n_val,'NUM VAL')
    print(len(pats))
    if n_val+n_tests>len(pats_full)*2:
        raise Exception('not enough to go around')
    while len(valid_idx)<n_val:
        r = randint(0,len(pats)-1)
        if r in valid_ids:
            continue
        valid_ids.add(r)
        pat = pats[r]
        rows = clinical_data[clinical_data[pat_col]==pat]
        # print(rows,"ROWS LET IT ROW!")
        valid_idx.extend(rows['Scan Index'])
        print(len(valid_idx))
        # valid_idx.extend(rows.index())

    print('now test')
    while len(test_idx)<n_tests:
        r = randint(0,len(pats)-1)
        if r in valid_ids or r in test_ids:
            print('whoopsie')
            continue
        test_ids.add(r)
        pat = pats[r]
        rows = clinical_data[clinical_data[pat_col]==pat]
        print(rows.shape,'matching rows')
        test_idx.extend(rows['Scan Index']) ##feels like this should be linked to the row index, bc scan refers to the index in numpy array
        ### this only works if they stay coupled
        # test_idx.extend(rows.index())

    # print()
    # print(valid_idx,;validsations)
    return valid_idx,test_idx

def download_atlas(atlas_file):
    atlas=pd.read_csv(atlas_file)
    print(atlas.iloc[0])
    print(atlas[['Index','X','Y','Z']])
    # asser
    # ss
    return atlas
def dataset_split(download_clinical,download_MEG,val_pct=.2,test_pct=.1,val_subgroup=False,use_exclude=False,criteria_dict={}):
    print('reading data...')
    clinical_data = pd.read_csv(download_clinical)
    scan_data = np.load(download_MEG)  ### here is where we find the norm.. return norm function?
    # load validation dataset
    # valid_idx,test_idx =get_scan_index_patient_split(clinical_data,val_pct,test_pct)
    # print(valid_idx,'VALID')
    print(use_exclude,'USE EXCLUDE')
    print(val_subgroup,'USE Val')
    # assert False
    clinical_data_old=clinical_data
    for criteria,val in criteria_dict.items():
        clinical_data=clinical_data[clinical_data[criteria]==val]


    # if not val_subgroup:
    #     for criteria,val in criteria_dict.items():
    #         clinical_data=clinical_data[clinical_data[criteria]==val]
    # print(val_subgroup,'VAL SUBGROUP??')
    # assert False

    if val_subgroup:
        assert criteria_dict
        print('SUB GROUP')
        valid_idx,test_idx =get_scan_index_split_bygroup(clinical_data_old,in_group=clinical_data['Scan Index'].unique())
        all_scanids=clinical_data_old['Scan Index'].astype(int).tolist()
        print(valid_idx,'VALID INDEX')
        clinical_data=clinical_data_old
    else:
        for criteria,val in criteria_dict.items():
            clinical_data=clinical_data[clinical_data[criteria]==val]
        print('Traditional split')
        valid_idx,test_idx =get_scan_index_patient_split(clinical_data,val_pct,test_pct,use_exclude=use_exclude)
        all_scanids = clinical_data['Scan Index'].astype(int).tolist()
    

    print('\n\n now')
    print(valid_idx)

    train_idx = (set(all_scanids)-set(test_idx))-set(valid_idx)
    train_idx=list(train_idx)
    # print(valid_idx,test_idx)

    # print()
    assert (len(train_idx)+len(valid_idx)+len(test_idx))==len(all_scanids)
    print(len(train_idx),len(valid_idx),len(test_idx),'lengths')
    print((len(set(train_idx))+len(set(valid_idx))+len(set(test_idx))),'unique lenths')
    
    file_count=0
    raw_data = {'train': [], 'valid': [], 'test': [],'all':[]} # save the train, valid dataset.

    # train_idx=list(train_idx)+list(test_idx)
    train_idx=list(train_idx)
    idxs_dict = {'train': train_idx, 'valid': valid_idx, 'test': test_idx,'all':all_scanids}
    # for i, data_item in enumerate(all_data):
    print(len(train_idx),len(valid_idx),len(test_idx),'lengths after add test to train')
    print(clinical_data.shape[0],'Cinical shape')

    valid_idtest=set([])
    test_idtest=set([])
    train_idtest=set([])


    all_idtest=set([])

    for i in range(clinical_data.shape[0]):
        row = clinical_data.iloc[i]
        scan_indx = row['Scan Index']
        scan = scan_data[scan_indx]
        if i in valid_idx:
            raw_data['valid'].append({'scan': scan, 'clinical': row})
        elif i in test_idx:
            raw_data['test'].append({'scan': scan, 'clinical': row})
            # raw_data['train'].append({'scan': scan, 'clinical': row})
        else:
            raw_data['train'].append({'scan': scan, 'clinical': row})

        raw_data['all'].append({'scan': scan, 'clinical': row})

        if scan_indx in all_idtest:
            print(scan_indx,"DOUBLE DIPPED")
        all_idtest.add(scan_indx)

        # print(scan.mean(),'scan mean',scan_indx,row['ID'])

        file_count += 1
        if file_count % 20 ==0:
            print('finished reading: %d' % file_count, end='\r')

    print(idxs_dict,'DICT')
    # sss
    print(len(idxs_dict['train']),len(idxs_dict['test']),len(idxs_dict['valid']),'length')
    return raw_data, idxs_dict


def make_fitted_pca(scan3d,n_components):
    """ takes in scans x roi x roi
        will flatten into (scans*roi) x roi so that we have all the features of every roi

    """
    scan3d=deepcopy(scan3d)
    pca= PCA(n_components=n_components)
    # pca= PCA(n_components='mle')
    PCA_X= scan3d
    # print(PCA_X.shape)
    xmax=PCA_X.max()
    for i in range(scan3d.shape[1]):
        PCA_X[:,i,i]=xmax

    all_rows = PCA_X.reshape(-1,PCA_X.shape[2])
    pca.fit(all_rows)
    out=pca.transform(all_rows)
    cum=0
    for i in range(len(pca.explained_variance_ratio_)):
        p=pca.explained_variance_ratio_[i]
        cum+=p
    print("PCT EXPLAINED: ", cum, 'by n_components: ',n_components)
    return pca,xmax,out.std()

def create_norm_functions(args):    ##identity options, set to 0, set to 1, set to min, set to max
    raw_data=np.load(args.raw_scan_file)
    band_adj = BAND_TO_INDEX[args.band]
    metric_adj = METRIC_TO_INDEX[args.metric]
    right_data = raw_data[:,band_adj,metric_adj]  ## will take more work to do multi-edges

    print(right_data.shape,"RIGHTOUT DATA")

    data_max = np.max(right_data)
    data_flat = []
    data_ut=[]
    for i in range(right_data.shape[1]):
        for j in range(i+1,right_data.shape[2]):
            data_ut.append(right_data[:,i,j])
    data_full=deepcopy(right_data)
    for s in range(data_full.shape[1]):
        data_full[:,s,s]=1
    data_flat=np.array(data_ut).flatten()


    data_flat_std=data_flat.std()
    data_flat_mean=data_flat.mean()
    data_full_std=data_full.std()
    data_full_mean=data_full.mean()


    if hasattr(args,'plv_pca_dim') and args.plv_pca_dim>0:
        fitted_pca,pca_max,pca_std = make_fitted_pca(right_data,args.plv_pca_dim)
    else:
        fitted_pca=None

    percentile_cache={'flat':{95:np.percentile(data_flat,95)}
                    ,'full':{95:np.percentile(data_full,95)}}
    # print(np.percentile(data_flat,75),80)
    # print(np.percentile(data_flat,80),80)
    # print(np.percentile(data_flat,85),85)
    # print(np.percentile(data_flat,90),90)
    # print(np.percentile(data_flat,95),95)
    # ssss
    # np.percentile(data_flat,90)
    # print(data_flat.shape,"180 x 90 x 90/2!!")
    print(np.percentile(data_flat,80),'80')
    print(np.percentile(data_flat,79),'79')
    print(np.percentile(data_flat,78),'78')

    # ss
    print(np.percentile(data_flat,75),'75')
    # # print(np.percentile(data_flat,.75))

    print(np.percentile(data_flat,85),'85')
    print(np.percentile(data_flat,90),'90')
    print(np.percentile(data_flat,95),'95')
    # pctile_cache
    # data_flat_show=np.clip(data_flat,.2,.65)
    # plt.hist(data_flat_show,bins=30)

    # here
    # plt.title(args.band+" Original distribution, mean:"+str(round(data_flat.mean(),3)))
    # plt.show()
    # sss


    ##### add self norm ie. normalizes based off single scan as before

    def add_noise_func(scan,rel_noise,mean=0):
        dataset_std=(np.std(data_flat))
        # print(dataset_std,'dataset std')
        # print(scan.std(),'single std')
        noise_mat = np.random.normal(mean,dataset_std*rel_noise,size=scan.shape)
        scan = scan+noise_mat
        return scan


    def mm_strech(scan,mm_min_pct=5,mm_max_pct=95,mm_clip_data=True,mm_clip_range=(0,1),identity_treatment='Raw',data_flat=data_flat,invert=False):  ## if you change data_flat to a single scan flat, you can retrofit to single norms
        scan=deepcopy(scan)
        if identity_treatment=='raw':
            pass
        elif identity_treatment=='set_to_max':

            for s in range(scan.shape[0]):
                scan[s,s]=np.max(data_flat)


        data_flat_use=1/data_flat if invert else data_flat
        scan_use = 1/scan if invert else scan


        adj_prob = (scan_use-np.percentile(data_flat_use,mm_min_pct))/((np.percentile(data_flat_use,mm_max_pct)-np.percentile(data_flat_use,mm_min_pct)))
        # full_prob = (data_flat_use-np.percentile(data_flat_use,mm_min_pct))/((np.percentile(data_flat_use,mm_max_pct)-np.percentile(data_flat_use,mm_min_pct)))


        # print(full_prob.mean(),full_prob.max(),'Strecht mean')
        # print(full_prob.min(),full_prob.max(),'min, max')
        if mm_clip_data:
            # full_prob=np.clip(full_prob,mm_clip_range[0],mm_clip_range[1])
            adj_prob=np.clip(adj_prob,mm_clip_range[0],mm_clip_range[1])

        data_ut=[]
        # print()
        for i in range(adj_prob.shape[0]):
            for j in range(i+1,adj_prob.shape[1]):
                # print(i,j,'PROBS')
                if i==j:
                    print(adj_prob[i,j],'ID')
                data_ut.append(adj_prob[i,j])

        data_ut=np.array(data_ut)
        # if 
        # print(adj_prob.shape,'ADJ PROB SHAPE?')
        # print(data_ut.mean(),'cleaned mean')
        # plt.hist(data_ut.flatten(),bins=50)
        # plt.show()
        # print(adj_prob.mean(),'ID mean')
        # plt.hist(adj_prob.flatten(),bins=50)
        # plt.show()
        # # akaka
        return adj_prob

    def ms_sigmoid(scan,r,t,identity_treatment='1'):
        # first find 
        # data_mean=
        scan=deepcopy(scan)
        ## hmmm
        def sig_func_sim(scan):
            prob_scan = 1 / (np.exp( -((scan - r) / t))+ 1.0)
            return prob_scan

        # if percentile_to_set<1:
        #     percentile_to_set*=100  ## should be full numbers!
        # if identity_treatment=='1':
        #     pass
        # elif identity_treatment=='set_to_pct':
        #     pass
        # elif identity_treatment=='set_to_95':

# 
        # plt.hist(scan.flatten(),bins=20)
        # plt.show()
        # print(adj_prob.mean(),'new ean should very')

        full_prob=sig_func_sim(data_flat)

        adj_prob=sig_func_sim(scan)

        if identity_treatment=='1':
            # aaa
            for s in range(adj_prob.shape[0]):
                adj_prob[s,s]=1

        # data_f = []
        data_ut=[]
        for i in range(adj_prob.shape[0]):
            for j in range(i+1,adj_prob.shape[1]):
                data_ut.append(adj_prob[i,j])
        data_f2=np.array(data_ut)

        # print(np.max(adj_prob),np.std(adj_prob),'pctinles should be same')
        # print(data_f2.mean(),'new ean should very')
        # print(np.max(full_prob),np.std(full_prob),'NO FULL')
        # print(full_prob.mean(),'FULL PROB')

        # print(np.max(data_f2),np.std(data_f2),'pctinles should be same')
        # print(data_f2.mean(),'new ean should very')

        # plt.hist(full_prob.flatten(),bins=30)
        # plt.show()
        # sksk
        return adj_prob

    def ms_norm(scan,mm_clip_data=False,mm_clip_range=(0,1),identity_treatment='Raw',percentile_to_set=95,data_flat=data_flat,include_id_std=False):

        data_flat=data_flat
        # data_mean=
        scan_og=scan
        scan=deepcopy(scan)
        if percentile_to_set<1:
            percentile_to_set*=100  ## should be full numbers!
        if identity_treatment=='raw':
            pass
        elif identity_treatment=='set_to_pct':
            
            for s in range(scan.shape[0]):
                # percentile calculations take fooooorever
                if percentile_to_set in percentile_cache['flat']:
                    pct_flat=percentile_cache['flat'][percentile_to_set]
                else:
                    pct_flat=np.percentile(data_flat,percentile_to_set) 
                    percentile_cache['flat'][percentile_to_set]=pct_flat

                if percentile_to_set in percentile_cache['full']:
                    pct_full=percentile_cache['full'][percentile_to_set]
                else:
                    pct_full=np.percentile(data_full,percentile_to_set) 
                    percentile_cache['full'][percentile_to_set]=pct_full
                # print(pct_flat,'pct_flat')
                # print(pct_full,'pct_full')
                scan[s,s]=pct_flat
                scan_og[s,s]=pct_full
                # scan[s,s]=1
                # scan_og[s,s]=1
        #     scan[s,s]=np.percentile(data_flat,percentile_to_set)
        # elif identity_treatment=='set_to_95':
        #     for s in range(scan.shape[0]):
        #         scan[s,s]=np.percentile(data_flat,95)

        # plt.hist(scan.flatten(),bins=20)
        # plt.show()
        if include_id_std:
            adj_prob=(scan_og-data_full_mean)/(data_full_std)
        else:
            adj_prob=(scan-data_flat_mean)/(data_flat_std)
        # adj_prob_old=(scan-data_flat_mean)/(data_flat_std)
        # print('new')
        # plt.hist(adj_prob.flatten(),bins=40)
        # plt.show()
        # print('other')
        # plt.hist(adj_prob_old.flatten(),bins=40)
        # plt.show()
        return adj_prob



    def identity(scan,identity_treatment='raw'):
        scan=deepcopy(scan)
        if identity_treatment=='1':
            # print('identity_treatment')
            for s in range(scan.shape[0]):
                scan[s,s]=1

        return scan

    def pca_transform(scan):
        if not fitted_pca:
            print('No PCA')
            return scan
        scan=deepcopy(scan)
        for s in range(scan.shape[0]):
            scan[s,s]=pca_max
        scan_red=fitted_pca.transform(scan)
        # print(scan_red.std())
        scan_red=scan_red/(pca_std)
        print(scan_red.mean())
        print(scan_red.std())
        return scan_red

    # data_warped=mm_strech(data_ut)

    # adj_prob=mm_strech(data_ut,mm_min_pct=1,mm_max_pct=99,mm_clip_data=True,identity_treatment='Raw') ## seems like set to max is already covered
    # plt.hist(adj_prob.flatten(),bins=20)
    # plt.title(args.band+" Final distribution, mean:"+str(adj_prob.mean()))
    # plt.show()
    return {'mm_strech':mm_strech,'ms_norm':ms_norm,'ms_sigmoid':ms_sigmoid,'identity':identity,'add_noise':add_noise_func,'pca_transform':pca_transform}



# def 
if __name__ == "__main__":
    preprocess(args)
