from sklearn.metrics import average_precision_score, accuracy_score, f1_score,roc_auc_score
from sklearn import metrics
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import torch.nn.functional as F

import numpy as np
import pandas as pd
# %matplotlib inline
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
plt.style.use('seaborn')
import seaborn as sns
import time
import sys
import os
import scipy.stats as st

# from statsmodels.stats.weightstats import DescrStats  ## gonna need that for weighted
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, f1_score,roc_auc_score
from sklearn.model_selection import cross_val_predict,GridSearchCV

from statsmodels.stats.weightstats import DescrStatsW
import statsmodels.api as sm
# from test import meg_roi_network_assigments
from copy import deepcopy
# %load_ext autoreload
# %autoreload 2
import argparse
from sklearn.model_selection import cross_val_predict,GridSearchCV,StratifiedKFold,LeaveOneOut,train_test_split,StratifiedShuffleSplit,RepeatedStratifiedKFold
from datetime import datetime
import random
import numpy as np
import os
import time
from utils import *
# from utils
from utils.dataloader_utils import load_data_graphalg,load_dataset, analyze_dataset_hyperbolicity,save_all_edge_dfs_handler, poincare_embeddings_all_handler
from params import *
import sys
from datetime import date
import pickle
from trials.dataset_config import config
import pickle
import optuna
import copy
from main import parse_default_args
from matplotlib import pyplot as plt
from data.MEG.get_data import preprocess,average_brain_edges
# from test import plot_meg

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import normalize as sk_normalize
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from manifolds.poincare import PoincareBall

from layers.layers import FermiDiracDecoder
from trials.dataset_config import config as default_config
import json
from main import parse_default_args
import networkx as nx
from datetime import datetime
import scipy.sparse as sp
from scipy import stats
import pandas as pd
from hyperbolic_learning_master.hyperbolic_kmeans.hkmeans import HyperbolicKMeans, plot_clusters,compute_mean
from hyperbolic_learning_master.utils.utils import poincare_distances,poincare_dist,hyp_lca_all
from diff_frech_mean.frechetmean import Poincare as Frechet_Poincare
from diff_frech_mean.frechet_agg import frechet_agg
from diff_frech_mean.frechetmean.frechet import frechet_mean

from utils.model_analysis_utils import save_embeddings
import os
from utils import *

subnet_labels_fn=["pDMN","aDMN","DAN","FPN","VN","VAN","SN","SMN",'N/A']
subnet_labels_fn=["pDMN","aDMN","DAN","FPN","VN","VAN","SN","SMN",'N/A']
subnet_pct_cols = [s+" Pct" for s in subnet_labels_fn[:-1]]
subnet_bin_cols = [s+" Bin" for s in subnet_labels_fn[:-1]]
# print(subnet_pct_cols)
# print(template_df[subnet_pct_cols])
# for i in range(len(template_df)):
#     print(template_df.iloc[i])

subnet_labels_sob=['R','L']
subnet_labels_all=['All']
subnet_labels_fnsob =[]
for s in subnet_labels_sob:
    for f in subnet_labels_fn:
        
    
        subnet_labels_fnsob.append(f+' '+s)
        
label_to_list = {'Functional Net':subnet_labels_fn,'SOB':subnet_labels_sob,'Func Net SOB':subnet_labels_fnsob,'All':subnet_labels_all}
      

BAND_TO_INDEX = {'theta':0,'alpha':1,'beta1':2,'beta2':3,'beta':4,'gamma':5}
METRIC_TO_INDEX = {'plv':0,'ciplv':1}


default_clinical=os.path.join(os.getcwd(),'data/MEG/MEG.clinical.csv')
default_MEG=os.path.join(os.getcwd(),'data/MEG/MEG.ROIs.npy')
default_temp=os.path.join(os.getcwd(),'data/MEG/AALtemplate.csv')
default_dataroot=os.path.join(os.getcwd(),'data/MEG')

# def fd_decod

def load_embedding(emb,template_df,node_col='RoiID',c=1.):
    # read in embedding coordinates
    ## right now, only works on 2d coords-- no reason can't be extended...
    # really shouldve made these z_0->z_n
    all_dim_cols = ['x','y','z','a','b','c','d','e']
    if type(emb)==str:
#         print('csv' in emb,'WELL IS IT OR ISNT IT?')
        print('Reading Embedding: {}'.format(emb))
        if 'csv' in emb:
            emb = pd.read_csv(emb)
            # print(emb,'EMBEDING')
            dim_cols = [ e for e in emb.columns if e!=node_col]

            print(len(dim_cols),'NATURAL DIM')
            emb.columns= ['node']+dim_cols
            # print("EMBEDDING CSV LIKE A GOOD CHRISTIAN")
        else: ### this is so wack
            emb = pd.read_table(emb, delimiter=' ')
            dims= int(emb.columns.values[1])
            # print(dims,'EMBEDDING SHAPE (load)')
            # print(dims,'DIMENISION')
            dim_cols = all_dim_cols[:dims]
            emb=emb.sort_index()
            emb = emb.reset_index()      
            emb.columns= ['node']+dim_cols
            emb = emb.sort_index()

    if 'r' not in emb.columns:
        if 'x' not in emb.columns or 'y' not in emb.columns:
            raise Exception('if no r, need both x and y')
        emb['theta'] = np.arctan2(emb['y'], emb['x'])
        emb['r'] = np.linalg.norm(emb[dim_cols],axis=1)
#         print(emb[['x','y','r']])
        
    if 'x' not in emb.columns:
        if 'r' not in emb.columns or 'theta' not in emb.columns:
            raise Exception('if no x, need both r and theta')
#         emb['r'] = emb.r / np.max(emb.r) - 1e-2

        # get cartesian coordinates
        x = []
        y = []
        for i in range(emb.shape[0]):
            x.append(emb.r[i]*np.cos(emb.theta[i]))
            y.append(emb.r[i]*np.sin(emb.theta[i]))
        emb['x'] = x
        emb['y'] = y
        
#     emb['hyp_r'] = emb.apply(lambda p: poincare_dist(np.array([p['x'],p['y']]),np.array([0,0])),axis=1)
    origin = np.zeros((len(dim_cols)))
    print(c,'CCCCCCCC')
    emb['hyp_r'] = emb.apply(lambda p: poincare_dist(np.array([p[dim_cols]]),origin,c),axis=1)

    # hyp_lcas,hyp_lca_rad = hyp_lca_all(np.array(emb[dim_cols]))

    # print(hyp_lcas,'')
    # print(hyp_lca_rad,'RAD')
    # emb['lca_origin'] = emb.apply(lambda p: hyp_lca(torch.from_numpy(np.array([p[dim_cols]]).astype(float)),torch.from_numpy(origin)),axis=1)
    # print(hyp_lcas,'hyp lca')
    
    # print(emb['lca_origin'],'OG')
    assign_cols = ['SOB','SOB Name','Functional Net Name','Functional Net','Func Net SOB','Func Net SOB Name']
    subnet_labels_fn=["pDMN","aDMN","DAN","FPN","VN","VAN","SN","SMN",'N/A']
    subnet_pct_cols = [s+" Pct" for s in subnet_labels_fn[:-1]]
    subnet_bin_cols = [s+" Bin" for s in subnet_labels_fn[:-1]]
    temp = template_df    
    emb[assign_cols] = emb.apply(
        lambda x: temp[assign_cols][temp[node_col]== x['node']].iloc[0],axis=1)
    emb[subnet_pct_cols] = emb.apply(
        lambda x: temp[subnet_pct_cols][temp[node_col]== x['node']].iloc[0],axis=1)
    emb[subnet_bin_cols] = emb.apply(
        lambda x: temp[subnet_bin_cols][temp[node_col]== x['node']].iloc[0],axis=1)
    
    emb["node_index"]=emb.apply(
        lambda x: temp["node_index"][temp[node_col]== x['node']].iloc[0],axis=1)
    
    emb['node_index']=emb['node_index'].astype(int)
    emb= emb.set_index('node_index',drop=False)
    emb=emb.sort_index()
    # print(emb[['node_index','node']],"NODULAS")
#     print(temp[['node_index','RoiID']],'ROIIZZZE')   
#     print(emb,'FINAL')
    
#     emb[assign_cols] = emb.apply( ## what we did before?
#         lambda x: temp[assign_cols][temp['Index']== int(x['node'])].iloc[0],axis=1)
#     emb[subnet_pct_cols] = emb.apply(
#         lambda x: temp[subnet_pct_cols][temp['Index']== int(x['node'])].iloc[0],axis=1)
#     print(emb,'embedding orders???')
    ### can't believe you were using iloc for this.. we need to assign keep node roi information 
    ### in the same dataframe in embedding space..
    
    return emb

def plot_embedding(emb, title=None, plot_edges=True,label='Functional Net'):
    n_clusters =len(emb[label].unique())
    if n_clusters <= 12:
        colors = ['b', 'r', 'g', 'y', 'm', 'c', 'k', 'silver', 'lime', 'skyblue', 'maroon', 'darkorange']
    elif 12 < n_clusters <= 20:
        colors = [i for i in plt.cm.get_cmap('tab20').colors]
    else:
        cmap = plt.cm.get_cmap(name='viridis')
        colors = cmap(np.linspace(0, 1, n_clusters))
    label_name = label+" Name"
    
    for i in range(n_clusters):
        plt.scatter(emb.loc[(emb[label] == i), 'y'], emb.loc[(emb[label] == i), 'z'],
                    color=colors[i], s=150, label=emb.loc[(emb[label] == i), label_name].values[0] );

    plt.xlim([-1.1,1.1])
    plt.ylim([-1.1,1.1])
    ax = plt.gca()
    circ = plt.Circle((0, 0), radius=1, edgecolor='black', facecolor='None', linewidth=3, alpha=0.5)
    ax.add_patch(circ)
#     if plot_edges:
#         edge_list_0 = get_edges(adj_conn_0)
#         for i in range(len(edge_list_0)):
#             x1 = emb.loc[(emb.iloc[:, 0] == edge_list_0[i][0]), ['x', 'y']].values[0]
#             x2 = emb.loc[(emb.id == edge_list_0[i][1]), ['x', 'y']].values[0]
#             _ = plt.plot([x1[0], x2[0]], [x1[1], x2[1]], '--', c='black', linewidth=1, alpha=0.25)
    if title != None:
        plt.title(title, size=16)



def fit_clusters(emb, num_updates=5,label='Functional Net',dims=2):
    # modify embedding labels
#     emb = emb[(emb.label.apply(lambda x: x in [2, 3, 4, 5]))].reset_index(drop=True)
#     emb['label'] = emb[label]
    ### can't really do this w/ weighted nodes

    n_clusters = len(emb[label].unique())
    # apply hkmeans to find centroids of 4 sub-networks
    dim_cols = ['x','y','z','a','b','c','d','e'][:dims]
    # dim_cols = ['x','y'] if dims==2 else ['x','y','z']
    emb_data = np.array(emb[dim_cols])
    hkmeans = HyperbolicKMeans(n_clusters=n_clusters,dims=dims)
    hkmeans.n_samples = emb_data.shape[0]
    hkmeans.init_centroids(radius=0.1)

    uni = list(emb[label].unique())
    uni.sort()
    assignments = np.zeros((hkmeans.n_samples, hkmeans.n_clusters))
    labels=emb[label].values
    hkmeans.init_assign(emb[label].values)
    print(emb_data.shape,"EMBEDDING SHAPE")
    for i in range(num_updates):  ## shouldn't this be 0?

        hkmeans.update_centroids(emb_data)
    return {'model': hkmeans, 'embedding': emb}



def create_legends(ax,main_labels,secondary_labels,secondary_type):
    network_to_full = {'DMN':'Default Mode Net','pDMN':'Posterior DMN',
                       'aDMN':'Anterior DMN','DAN':'Dorsal Att Net' , 'FPN':'Frontoparietal Net',
                       'VN':'Visual Net','VAN':'Ventral Att Net',
                       'SN':'Salience Net','SMN':'Sensorimotor Net','N/A':'No RSN' }
    
    main_labels=main_labels
    second_labels=secondary_labels
    colors =n_to_colors(len(main_labels))
    edge_colors = n_to_edgecolors(len(second_labels))
    print(edge_colors,"EDGE COLORS")
    shapes=n_to_marker(len(second_labels)) 
    
    print(second_labels,'SECOND')
    print(secondary_type,"SECONDARY TYPE")
    print(shapes,'SHAPES')
    print(second_labels[0])
    print(edge_colors[0])
#     fig2, ax2 = plt.subplots()
#     fig1, ax1 = plt.subplots()
    handles1=[plt.scatter([],[],color=colors[i],marker="o",
             label=network_to_full[main_labels[i]]) for i in range(len(main_labels))]
    if secondary_type=='edgecolor':
        print('second_home')    
        handles2=[plt.scatter([],[],color='green',marker="o",
                 edgecolors=edge_colors[i]) for i in range(len(second_labels))]
    elif secondary_type=='shape':
        print('SHAPE')
        handles2=[plt.scatter([],[],color='green',
                 marker=shapes[i]) for i in range(len(second_labels))] 
        print('onot business')

    elif secondary_type!= None:
        raise Exception("LEGENED NOT IMPLEMENTED"+str(secondary_type))
    
    
    artist1 = ax.legend(handles=handles1,labels=
                        [network_to_full[main_labels[i]] for i in range(len(main_labels))],
                    bbox_to_anchor=(0, 1), loc='upper right', ncol=1, edgecolor='black',prop={'size': 8})
    
    ax.add_artist(artist1)
    
    if secondary_type==None:
        print("NONE")
        return ax
    
    artist2 = ax.legend(handles=handles2,labels=
                        [second_labels[i] for i in range(len(second_labels))],
                        bbox_to_anchor=(0, .5), loc='upper right', ncol=1, edgecolor='black')    
    ax.add_artist(artist2)
#     ax.legend(bbox_to_anchor=(0, 1), loc='upper right', ncol=1, edgecolor='black')
    print('after hours')
#     print(ax.legend())
    return ax




def fit_and_plot_clusters(full_emb,title='',num_updates=5
                ,label='Functional Net',ignore_index=[8],
                          ignore_plot=False,secondary_label='SOB',save_file='',secondary_type='edgecolor'
                         ,use_legend=True,dims=2,ignore_alpha=0,show=False,norm_to_max=False):

    #### WILL NEED TO ADJUST IGNORE INDEX
#     secondary_type='edgecolor'
#     secondary_type='shape'
    label_list=label_to_list[label]
    label_list_second=label_to_list[secondary_label]

    # print(label_list,';ost')
    # print(label_trans,'transient')
    print(full_emb[label].unique())
    
#     print()
    label_name = [label+' Name']
    
    if label=='All' or secondary_label=='All':
        print('labelALL')
        full_emb['All']=0
    
    
    label_trans={} 
    for i in range(len(full_emb[label].unique())):
        if i in ignore_index:
            continue
        new_index= len(label_trans.keys())
        label_trans[i]=new_index

    full_emb_nets = full_emb[(full_emb[label].apply(lambda x: x not in ignore_index))].reset_index(drop=True)
    # full_emb_nets = full_emb[(full_emb[label].apply(lambda x: x))].reset_index(drop=True)
    # full_emb_nets = full_emb
    full_emb_nets[label] = full_emb_nets.apply(lambda x: label_trans[x[label]],axis=True) ### will need to adjust this when ignore index isnt the top
#     full_emb_nets[label] = full_emb_nets.apply(lambda x: label_trans[x[label]],axis=True)
    subnet_labels ={label_trans[i]:label_list[i] for i in range(len(label_list)) if i not in ignore_index}
    # subnet_labels ={label_trans[i]:label_list[i] for i in range(len(label_list))}
#     label_name_list =[label_names[i]:label_list[i] for i in range(len(label_list)) if i not in ignore_index]
    
     
#     print(full_emb_nets['label'])
    print(subnet_labels,'SUBNET LABELS')
    print(label,'LABEL')
    clustering = fit_clusters(full_emb_nets,label=label,dims=dims)
    
    if ignore_alpha==0:
        labels=[label_list[i] for i in range(len(label_list)) if i not in ignore_index]
    else:
        labels=[label_list[i] for i in range(len(label_list))]
    subnet_pct_cols =[l+" Pct" for l in labels]
    emb = clustering['embedding']
#     print(emb,'EMBEDDING!')
    emb['label']=emb[label]
    print(subnet_pct_cols,'subnet pct')
    emb['secondary_label']=emb[secondary_label]
    hkmeans = clustering['model'] 
    n_clusters = len(emb[label].unique())
    n_shapes = len(emb[secondary_label].unique())
    colors =n_to_colors(n_clusters)
    edge_colors = n_to_edgecolors(n_shapes)
    shapes=n_to_marker(n_shapes)
    emb['Shape'] = full_emb_nets.apply(lambda x: shapes[x[secondary_label]],axis=True)
    emb['color'] = full_emb_nets.apply(lambda x: colors[x['label']],axis=True)
    emb['edgecolors'] = full_emb_nets.apply(lambda x: edge_colors[x[secondary_label]],axis=True)
    print(emb.shape,'EMB SHAPE')

    print(subnet_labels,"SUBMEEEEEEEEEEEt")
    if not ignore_plot:
        # plot healthy control
#         plt.subplot(121)
        ax = plt.gca()
        if secondary_type=='shape' and secondary_label!='All':
        
            for i in range(n_clusters):

                alpha=1
                if i in ignore_index:
                    print('ignored!!')

                    if ignore_alpha>0:
                        alpha=ignore_alpha
                    else:
                        continue

                for s in range(n_shapes):
                    marker=full_emb_nets.loc[(emb.label == i), 'Shape'].values

                    plt.scatter(emb.loc[((emb.label == i)& (emb.secondary_label == s)), 'x'],
                            emb.loc[((emb.label == i)& (emb.secondary_label == s)), 'y'],color=colors[i], 
                            marker=shapes[s],alpha=alpha);
        
        
        elif (secondary_type=='edgecolor') and (secondary_label!='All'):
            print("EEEEEEEEEEEEEDGE COLORS")
            scatter = ax.scatter(emb['x'],emb['y'],color=emb['color'],edgecolors=emb['edgecolors'])
        else:
            scatter = ax.scatter(emb['x'],emb['y'],color=emb['color'])
        print(len(labels),'SHOULD BE SHORT')
        if secondary_label=="All":
            print("NONE")
            secondary_type=None
#         print(secondary_label,'SECONDARY LABEL')
#         print()
        if use_legend:
            ax = create_legends(ax,labels,secondary_labels=label_list_second,secondary_type=secondary_type)
#         ax = create_legends(ax,labels_names,secondary_labels=label_list_second,secondary_type=secondary_type)
#         plt.legend(bbox_to_anchor=(0, 1), loc='upper right', ncol=1, edgecolor='black');
        fig = plt.gcf()
        circ = plt.Circle((0, 0), radius=1, edgecolor='black', facecolor='None', linewidth=3, alpha=0.5)
        ax.add_patch(circ)
       

        ax.set_aspect('equal')
        if not norm_to_max:
            ax.set_xlim(xmin=-1.1,xmax=1.1)
            ax.set_xbound(lower=-1.1, upper=1.1)
            ax.set_ylim(ymin=-1.1,ymax=1.1)
            ax.set_ybound(lower=-1.1, upper=1.1)
            ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
   
        plt.title(title,size=16)
        
#         plt.xlim([-1.1,1.1])
#         plt.ylim([-1.1,1.1])

        plt.tick_params(
            axis='y',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            left=False,      # ticks along the bottom edge are off
            right=False,         # ticks along the top edge are off
            labelleft=False) # labels along the bottom edge are off
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off
#         fig.set_size_inches(8, 8,forward=True)
#         plt.show()
#         if title != None:
#             plt.suptitle('Hyperbolic K-Means - ' + title, size=16);
        if save_file!='':
            plt.savefig(save_file,bbox_inches='tight',dpi=300)
            # plt.savefig(save_file,bbox_inches='tight',dpi=300)
        if show:
            plt.show();     
    return {'model': hkmeans, 'embedding': emb,'labels':labels}



def create_distance_mat(embeddings,c):
    ## basically the same as poincare_distances, but I want to make sure order is right
    distance_mat = np.zeros((len(embeddings),len(embeddings)))
    for i in range(len(embeddings)):
        for j in range(len(embeddings)):
            distance_mat[i,j]= poincare_dist(embeddings[i], embeddings[j],c=c)
    return distance_mat    



def create_origin_array(embeddings,c):
    origin_distances = np.array([poincare_dist(x, [0,0],c=c) for x in A])
    return origin_distances

def create_inclusion_mat(A_inclusion,B_inclusion,zero_identity=True):
    ## think about normalizing these so we don't get such small weightings?
    inclusion_mat = np.zeros((len(A_inclusion),len(B_inclusion)))
    for i in range(len(A_inclusion)):
        for j in range(len(B_inclusion)):
            if (i==j) and zero_identity:
                inclusion_mat[i,j]=0
            else:
                inclusion_mat[i,j]= A_inclusion[i]*B_inclusion[j]
    return inclusion_mat

def distance_between_weighted(D_mat, inclusion_mat, metric='average',use_binary=True,r=2.,t=1.):
    # methods for intercluster distances
    distances = D_mat.flatten()
    weights = inclusion_mat.flatten()
    
    d_nonzero= distances[weights>0]
    w_nonzero= weights[weights>0]
    
    
#     print(distances)
#     print(w_nonzero)

#     for i in range(distances.shape[0]):
#         if weights[i]>.01:
#             print(i,distances[i],weights[i])
    #     print(d_nonzero,w_nonzero)
#     print(stats.mean,'WITHIN AVERAGE')
    if len(w_nonzero)==0:
        raise Exception("No overal")
    # if stats.mean<.01:
        # print(stats.mean,'???')
    if use_binary:
        if metric == 'average': ## why not add a var?
            probs=1/(np.exp((d_nonzero - r) / t)+ 1.0)
            return np.mean(d_nonzero),np.mean(probs)
        raise Exception('Not implemented')
    stats = DescrStatsW(d_nonzero, weights=w_nonzero, ddof=1)

    if metric == 'average': ## why not add a var?

        return stats.mean
#         return np.mean(distances)
    elif metric == 'max':
        return stats.max
#         return np.max(distances)
    elif metric == 'min':
        return np.min(distances)
    elif metric == 'variance':
        return stats.std_mean
#         return stats.var
    else:
        print('Invalid metric specified')
        return
        
def distance_within_weighted(D_mat, inclusion_mat, metric='variance',use_binary=True,r=2.,t=1.):
    # methods to compute cohesion within cluster
    ### should be exact same as between, but the inclusion mat is a self matrix
    ### and we will want different stats
    distances = D_mat.flatten()
    weights = inclusion_mat.flatten()
    d_nonzero= distances[weights>0]
    w_nonzero= weights[weights>0]
    # print(weights,'BEST BE ALL 1s!!!')
    
#     print(stats,'stat')
    
#     print(distances,wieghts)
#     print(stats.mean,'WITHIN AVERAGE')

#     orin
    if use_binary:
        if metric == 'variance':
            return np.std(d_nonzero)
#         return stats.var
        elif metric == 'diameter':
            return np.max(d_nonzero)
        elif metric == 'pairwise':
            probs=1/(np.exp((d_nonzero - r) / t)+ 1.0)
            return np.mean(d_nonzero),np.mean(probs)
            # return np.mean(d_nonzero)  
            # return np.sum(d_nonzero)  /len(D_mat)

    stats = DescrStatsW(d_nonzero, weights=w_nonzero, ddof=1)
    if metric == 'variance':
        return stats.std_mean
#         return stats.var
    elif metric == 'diameter':
        return stats.max
    elif metric == 'pairwise':
        return stats.mean
#         return np.sum(pairwise_distances) / len(A)
    else:
        print('Invalid metric specified')
        return

def radius_metrics_weighted(rad_array,inclusion_array, metric='average',use_binary=True,r=2.,t=1.):
    ### should have dist from origin passed in
#     origin_distances = np.array([poincare_dist(x, [0,0]) for x in A])
    rads= rad_array[inclusion_array>0]
    # print(rads,'rads')
    # print(rads.mean(),'RATION')
    metric='prob'
    metric='average'
    if use_binary:
        if metric == 'variance':
            return np.std(rads)
#         return stats.var
        elif metric == 'average':
            # eoiwhf;woeifhsSSS
            probs=1/(np.exp((rads - r) / t)+ 1.0)
            return np.mean(rads),np.mean(probs)

        elif metric=='prob':
            # print(rads)
            prob= 1/(np.exp((rads - r) / t)+ 1.0)
            # print(prob,"PROBS")
            # sddds
            return np.mean(prob)
    stats= DescrStatsW(rad_array, weights=inclusion_array, ddof=1)
    if metric == 'average':
        return stats.mean
    elif metric == 'variance':
        return stats.std_mean
#         return stats.var
    else:
        print('Invalid metric specified')
        return

def cluster_polar_metrics_weighted(A,inclusion_array):
    sub=A[inclusion_array>0]
    # print(sub.columns,'all colm=umns should be nums') 
    # print(sub.mean(axis=0))
    avg_df= pd.DataFrame(data=[sub.mean(axis=0)],columns=sub.columns)
    # print(avg_df)
    # wwwww
    return avg_df
    # return sub.mean(axis=0)

def cluster_features_weighted(emb,centroids, use_binary=True,wc_metric='pairwise',
                              bc_metric='average',rc_metric='average',dims=2,node_polar=None,c=1.,r=2.,t=1.):
                              ##,subnet_bin_cols=subnet_bin_cols,subnet_pct_cols=subnet_pct_cols) ## this used to belong here...
    ### need to pass in weighted columns..
    cluster_pct_cols = subnet_bin_cols if use_binary else subnet_pct_cols
    # dim_cols = ['x','y'] if dims==2 else ['x','y','z']
    dims_cols = ['x','y','z','a','b','c','d','e'][:dims]
    emb_data = np.array(emb[dims_cols])
    within_cluster = []
    radius_cluster=[]
    radius_prob_cluster=[]
    within_prob_cluster=[]
    hk_cluster=[]
#     n_clusters= model.n_clusters
    n_clusters=len(centroids)
    between_cluster = np.zeros((n_clusters,n_clusters))
    between_prob_cluster = np.zeros((n_clusters,n_clusters))
#     print(emb,'embeddings')
    d_mat = create_distance_mat(emb_data,c=c)
    origin = np.zeros((dims))
#     d_origin_man  = np.array([poincare_dist(x, origin) for x in emb_data])
    d_origin  = np.array(emb['hyp_r'])

    
#     assert dims==3
    
#     print(cluster_pct_cols)
#     print(emb.columns,'EMB COLUMS')
#     print(n_clusters,"NUM CLUSTERS")
    
#     j_inclusion_array=np.array(emb[cluster_pct_cols[j]])

    for i in range(n_clusters):
#         print(i,'NEW I')
        i_inclusion_array=np.array(emb[cluster_pct_cols[i]])
#         print(i_inclusion_array,'BETTER BE ALL 1s and zeroes@')
#         print(i_inclusion_array.sum(),('sum inclusions'))
#         print(cluster_pct_cols[i],i_inclusion_array,'SELF INCLUSION')
        self_inclusion_mat= create_inclusion_mat(i_inclusion_array,i_inclusion_array,zero_identity=True)
#         print(cluster_pct_cols[i],self_inclusion_mat,"SELF MAT")
        wc_d,wc_prob=distance_within_weighted(d_mat,self_inclusion_mat, metric=wc_metric,use_binary=use_binary,r=2.,t=1.)
        rc_d,rc_prob=radius_metrics_weighted(d_origin,i_inclusion_array, metric=rc_metric,use_binary=use_binary,r=2.,t=1.)

        within_cluster.append(wc_d)
        radius_cluster.append(rc_d) ## rethink\
        within_prob_cluster.append(wc_prob)
        radius_prob_cluster.append(rc_prob) ## rethink\

        # within_prob_cluster.append(distance_within_weighted(d_mat,self_inclusion_mat, metric=wc_metric,use_binary=use_binary,metric=prob,c=c,r=r,t=t))
        # radius_prob_cluster.append(radius_metrics_weighted(d_origin,i_inclusion_array, metric=rc_metric,use_binary=use_binary,metric=prob,c=c,r=r,t=t)) ## rethink
        # rad_prob_cluster.append(radius_metrics_weighted(d_origin,i_inclusion_array, metric=rc_metric,use_binary=use_binary)) ## rethink

        hk_cluster.append(cluster_polar_metrics_weighted(node_polar,i_inclusion_array)) ## rethink
        # hk_cluster.append(0)
        # hk_cluster.append()
        #### Perfect place to put avg radius
        for j in range(i+1, n_clusters):
            j_inclusion_array=np.array(emb[cluster_pct_cols[j]])
#             print(cluster_pct_cols[i],i_inclusion_array,j_inclusion_array,"BOTH INCLUSIONS")
#             j_inclusion_array = emb[cluster_pct_cols[j]]
#             print(cross_inclusion)
            cross_inclusion_mat=create_inclusion_mat(i_inclusion_array,j_inclusion_array,zero_identity=True)
#             print(cross_inclusion_mat,"CROSSWAYS")
            btw_dist,btw_dist_prob = distance_between_weighted(d_mat,cross_inclusion_mat, metric=bc_metric,use_binary=use_binary,r=2.,t=1.)
            # btw_dist_prob = distance_between_weighted(d_mat,cross_inclusion_mat, metric=bc_metric,use_binary=use_binary)
            between_cluster[i,j]=btw_dist
            between_cluster[j,i]=btw_dist

            between_prob_cluster[i,j]=btw_dist_prob
            between_prob_cluster[j,i]=btw_dist_prob
#     eee
    
            
    return {'within':np.array(within_cluster), 'between': np.array(between_cluster),'radius': np.array(radius_cluster),
    'rad_prob': np.array(radius_prob_cluster), 'between_prob': np.array(between_prob_cluster),'within_prob': np.array(within_prob_cluster),'hk_dist':(hk_cluster)}


def distance_between(A, B, metric='average'):
    # methods for intercluster distances
    distances = []
    for a in A:
        distances += [poincare_dist(a, b) for b in B]
    if metric == 'average':
        return np.mean(distances)
    elif metric == 'max':
        return np.max(distances)
    elif metric == 'min':
        return np.min(distances)
    else:
        print('Invalid metric specified')
        return
        
def distance_within(A, centroid, metric='pairwise'):
    # methods to compute cohesion within cluster
    centroid_distances = np.array([poincare_dist(x, centroid) for x in A])
    pairwise_distances = poincare_distances(A)
    if metric == 'variance':
        return np.mean(centroid_distances**2)
    elif metric == 'diameter':
        return np.max(pairwise_distances)
    elif metric == 'pairwise':
        return np.mean(pairwise_distances)
#         return np.sum(pairwise_distances) / len(A)
        
    else:
        print('Invalid metric specified')
        return

def radius_metrics(A, metric='average'):
    
    origin_distances = np.array([poincare_dist(x, np.zeros(x.shape[0])) for x in A])
    if metric == 'average':
        return np.mean(origin_distances)
    elif metric == 'variance':
        return np.var(origin_distances)
    else:
        print('Invalid metric specified')
        return

def hkmeans_metrics(A,hkmeans_full,use_origin=True):
    if hkmeans_full is None:
        return np.zeros(A.shape[0])

    node_dist = hkmeans_full.transform(A,use_origin=use_origin)
    # print(node_dist.shape,'node shape...')
    clust_dist= np.mean(node_dist,axis=0)
    # print(clust_dist.shape,'final shape')
    # print(clust_dist,'final shape')
    # dddddd
    return clust_dist

def cluster_polar_metrics(A):
    # print(A.columns,'all colm=umns should be nums') 
    # print(A.mean(axis=0))
    avg_df= pd.DataFrame(data=A.mean(axis=0),columns=A.columns)
    # print(avg_df)
    # wwwww
    return avg_df
    # return A.mean(axis=0)


def cluster_features(emb,centroids, wc_metric='pairwise', bc_metric='average',rc_metric='average',dims=2,hkmeans_full=None,node_polar=None):
    # dim_cols = ['x','y'] if dims==2 else ['x','y','z']
    dims_cols = ['x','y','z','a','b','c','d','e'][:dims]
    emb_data = np.array(emb[dims_cols])
    within_cluster = []
    radius_cluster=[]
    
#     n_clusters= model.n_clusters
    n_clusters=len(centroids)
    between_cluster = np.zeros((n_clusters,n_clusters))
    hk_cluster=[]



    for i in range(n_clusters):
        within_cluster.append(distance_within(emb_data[emb.label == i], centroid=centroids[i], metric=wc_metric))
        radius_cluster.append(radius_metrics(emb_data[emb.label == i], metric=rc_metric))
        hk_cluster.append(hkmeans_metrics(emb_data[emb.label == i], hkmeans_full=hkmeans_full))

        print(radius_cluster[-1],'RADIUS CLUSTER')
        print(hk_cluster[-1][-1],'HYPERBOLIC BACK')

        #### Perfect place to put avg radius
        for j in range(i+1, n_clusters):
            btw_dist = distance_between(emb_data[emb.label == i], emb_data[emb.label == j], metric=bc_metric)
            between_cluster[i,j]=btw_dist
            between_cluster[j,i]=btw_dist

    # print(hk_cluster,'HK CLUSTERS')
            

    return {'within':np.array(within_cluster),
            'between': np.array(between_cluster),'radius': np.array(radius_cluster),'hk_dist':np.array(hk_cluster)}

def plot_distribution(adj_mat,mean=0,vari=1,adj_threshold=.32):
#     adj_flat=adj_mat.flatten()
    adj_flat=[]
    for j in range(adj_mat.shape[0]):

        for k in range(j+1, adj_mat.shape[0]):
            adj_flat.append(adj_mat[j, k])
    adj_flat=np.array(adj_flat)
    plt.hist(adj_flat,bins=20)
    plt.show()
#     adjusted = flat-
    adjusted = (adj_flat-adj_flat.mean())/adj_flat.std()
    new_adjusted = (adj_flat-np.min(adj_flat))/(np.max(adj_flat)-np.min(adj_flat))
    new_adjusted_clipped = (adj_flat-np.percentile(adj_flat,5))/(np.percentile(adj_flat,95)-np.percentile(adj_flat,5))
    new_adjusted_clipped=np.clip(new_adjusted_clipped,0,1)
    plt.hist(new_adjusted,bins=20)
    plt.show()
    
#     print(unweighted,'UNWEIGHTED')
    plt.hist(new_adjusted_clipped)
    plt.show()  
    unweighted=np.where(adj_flat>adj_threshold,1,0)
    print(unweighted,'UNWEIGHTED')
    plt.hist(unweighted)
    plt.show()
    
    print(unweighted.mean(),'unweighted mean')
    print(new_adjusted.mean(),'adjusted mean')
    print(new_adjusted_clipped.mean(),'clipped mean')
    print(adj_flat.mean(),'ful mean')
    
    
#     data – np.min(data)) / (np.max(data) – np.min(data)
    
def adj_emb_comparison(adj_mat,embedding,use_weight=True,adj_threshold=.32,dims=2):
    print(embedding,'EMBEDDING')
#     print(embedding[i])
#     poincare_d = poincare_distances(poincare_dist(np.array([p['x'],p['y']]))

    poincare_d=poincare_distances(np.array(embedding[['x', 'y']])) if dims == 2 else poincare_distances(np.array(embedding[['x', 'y','z']]))
    adj_flat = []
    pairwise_flat = []
    if not use_weight:
        print(np.where(adj_mat>adj_threshold,1,0))
        print(adj_mat)
        adj_mat=np.where(adj_mat>adj_threshold,1,0)
    
    
    for j in range(adj_mat.shape[0]):
#         print(j,'JK')
#         print(embedding.iloc[j],'J')
            
        for k in range(j+1, adj_mat.shape[0]):
            adj_flat.append(adj_mat[j, k])
            pairwise_flat.append(poincare_d[j, k])
    pairwise_flat=np.array(pairwise_flat)
    adj_flat = 1/np.array(adj_flat)
    a, b = np.polyfit(adj_flat, pairwise_flat, 1)
    
    adj_flat=np.array(adj_flat)
    
    print(a)
    print(adj_flat)
    a=float(a)

    #add points to plo

    #add line of best fit to plot
         
    plt.scatter(x=adj_flat,y=pairwise_flat)
    plt.plot(adj_flat, a*adj_flat+b,c='k') 
    plt.ylim(top=12)
    plt.ylim(bottom=0)
    plt.show()
#     return metrx_flat,label_flat
    

def get_shared_nodes(emb_root,graph_ids,template_df,suffix):
    nodes_seen=set([])
    nodes_all = set([])
    means=set([])
    std=set([])
    for g_id in graph_ids.astype(int):   ##### easy... create_file list?? then we can just index in w/ graph ids
        print(g_id,'JEZY')
        
        emb_file =os.path.join(emb_root,str(int(g_id))+"_embeddings"+suffix)
        emb = load_embedding(emb_file,template_df)
        print(g_id,'JEZY')
        print('x gonna', emb['x'],'x gonna')
        nodes = emb['node_index'].values
        nodes_seen=nodes_seen.union(nodes)
        if emb['x'].mean() in means:
            print(emb['x'].mean(),'bad mean')
            print(means)
#             raise Exception()
        if emb['x'].std() in std:
            print('also bad std')
#             raise Exception()
        means.add(emb['x'].mean())
        std.add(emb['x'].std())
        
#         sss
        if len(nodes_all)==0:
            nodes_all=set(nodes)
        nodes_all=nodes_all.intersection(nodes)
#     nodes_all= list(nodes_all)
    excluded_nodes = nodes_seen-nodes_all
    return nodes_all,excluded_nodes


def pairwise_matrx_flatten_and_name(metrx_mat,label_list):

        metrx_flat = []
        label_flat = []
        # print(label_list,'label list')
        for j in range(metrx_mat.shape[0]):
            for k in range(j+1, metrx_mat.shape[0]):
                metrx_flat.append(metrx_mat[j, k])
                try:
                    label_flat.append(str(label_list[j])+'_'+str(label_list[k]))
                except:
#                     print(metrx_mat.shape,'SHOULD BE NO MORE THAN 89 right')
                    label_flat.append(str(j)+'_'+str(k))
#                     print(j,k,'jk')
#                     print()
#                     aaaa
        return metrx_flat,label_flat

# def group_cartesian_to_polar_hy(c_df):
def cartesian_to_polar_hyp(c_df,c):
    dims = c_df.shape[1]
    dims_cols = ['x','y','z','a','b','c','d','e'][:dims]
    ang_cols = ['theta','phi','alpha','beta','chi','delta','eps','phi'][:dims-1]
    sin_cols = ['sin_'+a for a in ang_cols]
    cos_cols = ['cos_'+a for a in ang_cols]


    angles = []
    r = np.linalg.norm(c_df,axis=1)
    running_r=r
    for i in range(dims-1):
        # print(c_df[:,i:].shape,'whats are norm')
        running_r=np.linalg.norm(c_df[:,i:],axis=1)

        # print(running_r,'RUNNING RRRRR')
        # print(running_r.shape,'radius shape')
        x_i=c_df[:,i]
        a_i = np.arccos(x_i/running_r)
        if i ==dims-2:
            a_i = np.where(x_i>=0,a_i,2*np.pi-a_i)
            # a_i = 2*np.pi()-a_i
        angles.append(a_i)
    angles.reverse() ## so theta is first
    # print(angles)
    angles=np.array(angles).T
    # print(angles.shape)


    # print(c_df[:,0]**2 + c_df[:,1]**2 +  c_df[:,2]**2,'before')
    # print(r,"RRRRR")
    # print(r.shape,'r shape')
    theta = np.arctan2( c_df[:,1], c_df[:,0])
    phi = np.arccos(c_df[:,2]/r) if c_df.shape[1]>2 else 0
    polar_df = pd.DataFrame(columns=['r','phi','theta'])
    polar_df['r']=r
    # print(r,'R')
    # print(polar_df['r'],'POLAR R')
    polar_df['phi_old']=phi
    polar_df['theta_old']=theta

    origin = np.zeros((dims))


    polar_df[ang_cols]=angles
    polar_df[sin_cols]=np.sin(polar_df[ang_cols])
    polar_df[cos_cols]=np.cos(polar_df[ang_cols])
    polar_df['hyp_r'] =np.array([poincare_dist(p,origin,c) for p in c_df])
    # print(polar_df['hyp_r'],'lets go')


    # polar_df['cos_theta']= np.cos(polar_df['theta'])
    # polar_df['sin_theta']= np.sin(polar_df['theta'])
    # polar_df['sin_phi']= np.sin(polar_df['phi'])
    # polar_df['cos_phi']= np.cos(polar_df['phi'])
    # polar_df['x']= c_df[:,0]
    # polar_df['y']= c_df[:,1]
    # polar_df['z']= c_df[:,2] if c_df.shape[1]>2  else 0
    # print(dims,'DIMS')
    # print(c_df.shape,'SHAPELY')
    polar_df[dims_cols]=c_df[:,:dims]
    return polar_df

def stack_embeddings(emb_root,graph_ids,template_df,weighted=False,use_binary=True,bin_rank_threshold=10,
                       net_threshold=.2,suffix='.tsv',dims=3):
    node_coords=[]
    template_df= template_df.apply(
        lambda row: meg_roi_network_assigments(
            row,combine_dmn=False,network_thresh=net_threshold,rank_thresh_inclusive=bin_rank_threshold),axis=1) 
    dim_cols = ['x','y','z','a','b','c'][:dims]
    for g_id in graph_ids.astype(int):   ##### easy... create_file list?? then we can just index in w/ graph ids
        print(g_id,'JEZY')
        emb_file =os.path.join(emb_root,str(int(g_id))+"_embeddings"+suffix)
        emb = load_embedding(emb_file,template_df)
        emb_coords=emb[dim_cols]

        node_coords.append(emb_coords)
    return node_coords

    
def embedding_analysis(emb_root,clinical_df,cluster_label,template_df,
                       weighted=False,
                       wc_metric='pairwise', bc_metric='average',rc_metric='average',
                       preloaded_emb=False,emb_df=None,use_binary=True,bin_rank_threshold=10,
                       net_threshold=.2,suffix='.tsv',dims=3,hkmeans_full=None,balanced_atl=True,c=1,r=2,t=1):
#     shared_nodes,excluded_nodes=get_shared_nodes(emb_root,graph_ids,template_df,suffix=suffix)
        
    """
    clinical_df was subbed in for graph_ids to avoid any mistakes with labels
    clinical_df should ONLY contain the graphs that you would like to be analize
        ie. if your model has a critera CogTr=1, then only those should be passed in in the clinical_df
    """

    if use_binary:
        thresh_str = 'Rank'+str(bin_rank_threshold)
    else:
        thresh_str= 'PCTt'+str(net_threshold)
    file_str='embedding_stats_'+str(cluster_label)+'_'+thresh_str+'.npy'
    csv_str='embedding_stats_'+str(cluster_label)+'_'+thresh_str+'.csv'

    out_dir=os.path.join(emb_root,file_str)
    out_dir_csv=os.path.join(emb_root,csv_str)
    print(out_dir_csv,'out dir')
    # eeee
    if os.path.exists(out_dir) and os.path.exists(out_dir_csv):
        stat_dict = pickle.load( open(out_dir , "rb" ) )
        full_stat_df = pd.read_csv(out_dir_csv)
        return stat_dict,full_stat_df
        print(stat_dict)

    print(c,'C')
    print(r,'R')
    print(t,'T')
        # eff
    label_names= []
    wc_features = []
    rc_features = []
    bc_features = []
    rad_prob_features = []
    wc_prob_features = []
    bc_prob_features = []
    node_features= []
    node_rad_prob_features= []
    nodepw_prob_features= []
    node_polar=[]
    c_coord_features=[]
    node_coord_features=[]
    nodepw_features=[]
    hk_features=[]

    dims_cols = ['x','y','z','a','b','c','d','e','f'][:dims]
    stat_dict={'wc':{},'rc':{},'bc':{},'node_rad':{},
    'pw':{},'c_coord':{},'node_coord':{},'hkc':{},'rad_prob':{},'wc_prob':{},'bc_prob':{},'pw_prob':{},'node_rad_prob':{}} 
    ### we really should include all stats right? ie. lets
    
    if weighted and use_binary:
        print("CANNOT USE WEIGHT AND BINARY- GOING W/ BINARY")
        weighted=False
        
    if (weighted or use_binary) and cluster_label!='Functional Net':
        print(cluster_label,'cluster lble')
        print('cant do weighted on SOB we dont know what to weight')
        weighted=False
        use_binary=False
    
    first=True
    if not use_binary:
        print('making bin rank thresh -1 bc not using binary')
        bin_rank_threshold=-1

    template_df= template_df.apply(
        lambda row: meg_roi_network_assigments(
            row,combine_dmn=False,network_thresh=net_threshold,rank_thresh_inclusive=bin_rank_threshold,use_binary=use_binary
            ,balanced=balanced_atl),axis=1)    

    networks=["pDMN","aDMN","DAN","FPN","VN","VAN","SN","SMN"]
    bin_cols=[n+' Bin' for n in networks]
    # print(pct_or_thresh_name,'thresh it')
    print(template_df[bin_cols].sum(),'eyyy')
    roi_to_name=get_roi_to_name(template_df,node_col='RoiID')
    graph_ids=clinical_df['Scan Index'].values
    graph_id_ordered=[]
    all_files=os.listdir(emb_root)
    # for g_id in graph_ids.astype(int):   ##### easy... create_file list?? then we can just index in w/ graph ids
    for g_id in graph_ids.astype(int):   ##### easy... create_file list?? then we can just index in w/ graph ids
        # if g_id<90:
        #     continue

        emb_name=str(int(g_id))+"_embeddings"+suffix
        emb_file =os.path.join(emb_root,emb_name)
        if emb_name not in all_files:
            print('Skipping {} because not in embeddings list. make sure this is not an error'.format(g_id))
            continue
        graph_id_ordered.append(g_id)

        emb = load_embedding(emb_file,template_df,c=c)
        # dims_cols_to_use = [e for e in emb.columns if e not in dims_cols]
        clustering= fit_and_plot_clusters(emb,title='',ignore_index=[8,17],
                                          label=cluster_label,ignore_plot=True,dims=dims) ## can easily grab clusters from this

        cluster_emb = clustering['embedding']
        emb_coords = np.array(emb[dims_cols])
        emb_use = cluster_emb
        emb_coords = np.array(emb_use[dims_cols])
        hkmeans = clustering['model']
        c_coord_features.append(cartesian_to_polar_hyp((hkmeans.centroids),c=c))

        node_polars=cartesian_to_polar_hyp(emb_coords,c)
        # node_rads=cartesian_to_polar_hyp((emb_coords))
        node_rads = np.array(node_polars['hyp_r'])
        node_features.append(node_rads)

        node_rad_probs = 1/(np.exp((node_rads - r) / t)+ 1.0)
        node_rad_prob_features.append(node_rad_probs)
        if weighted:
            cluster_feats = cluster_features_weighted(cluster_emb,hkmeans.centroids,
                                            wc_metric=wc_metric, bc_metric=bc_metric,rc_metric=rc_metric
                                                     ,dims=dims,use_binary=False)
        elif use_binary:
             cluster_feats=cluster_features_weighted(cluster_emb,hkmeans.centroids,
                                            use_binary=True,wc_metric=wc_metric, bc_metric=bc_metric,rc_metric=rc_metric
                                                     ,dims=dims,node_polar=node_polars)           
        else:
            print("FEATES")
            print('still using binary weight, even though pcts used to make inclusion')
            # cluster_feats = cluster_features(cluster_emb,hkmeans.centroids,
                                            # wc_metric=wc_metric, bc_metric=bc_metric,rc_metric=rc_metric
                                            # ,dims=dims,hkmeans_full=hkmeans_full)
            cluster_feats=cluster_features_weighted(cluster_emb,hkmeans.centroids,
                                            use_binary=True,wc_metric=wc_metric, bc_metric=bc_metric,rc_metric=rc_metric
                                                     ,dims=dims,node_polar=node_polars,c=c,r=r,t=t)   
        # print(cluster_feats['hk_dist'],'forr BW')
        wc_features.append(cluster_feats['within'])
        rc_features.append(cluster_feats['radius'])
        hk_features.append(cluster_feats['hk_dist'])

        wc_prob_features.append(cluster_feats['within_prob'])
        rad_prob_features.append(cluster_feats['rad_prob'])
        # hk_prob_features.append(cluster_feats['hk_dist_prob'])

        if hkmeans_full is None:
            hk_labels=['None']
        else:
            hk_labels=[str(c) for c in np.arange(hkmeans_full.centroids.shape[0])]
            print(cluster_feats['hk_dist'].shape,'hk_dist shape')

            if cluster_feats['hk_dist'].shape[1]>len(hk_labels):
                hk_labels.append('origin') #### we must use one of the origin features
        
        bc_prelim = cluster_feats['between']
        bc_prob_prelim = cluster_feats['between_prob']

        bc_flat,bc_labels = pairwise_matrx_flatten_and_name(bc_prelim,clustering['labels'])
        bc_features.append(bc_flat)    

        bc_prob_flat,bc_prob_labels = pairwise_matrx_flatten_and_name(bc_prob_prelim,clustering['labels'])
        bc_prob_features.append(bc_prob_flat)       

        D = poincare_distances(np.array(emb_use[dims_cols]),c=c) 
        D_probs=fd_decoder(D,r,t)

        node_pairwise,pw_labels =pairwise_matrx_flatten_and_name(D,emb_use['node'].apply(lambda x:roi_to_name[x]))             
        # print(emb_use['hyp_r'],'should changed right?')                                  
        node_pairwise_prob,pw_labels =pairwise_matrx_flatten_and_name(D_probs,emb_use['node'].apply(lambda x:roi_to_name[x]))   
        # node_features.append(np.array(emb_use['hyp_r']))
        # print(node_features[-1],"HEY")
        nodepw_features.append(np.array(node_pairwise))
        nodepw_prob_features.append(np.array(node_pairwise_prob))
        
        if first:
#             'c_coords':{},'node_coords':{}
            stat_dict['wc']['labels']=clustering['labels']
            print(clustering['labels'],'ABESL')
            stat_dict['wc_prob']['labels']=clustering['labels']
            stat_dict['rc']['labels']=clustering['labels']
            stat_dict['rad_prob']['labels']=clustering['labels']
            stat_dict['c_coord']['labels']=clustering['labels']
#             
            stat_dict['bc']['labels']=bc_labels
            stat_dict['bc_prob']['labels']=bc_labels
            stat_dict['node_rad']['labels']=emb_use['node'].apply(lambda x:roi_to_name[x]) ### add node_name to load embeddings? but we don't want to overload if we
            stat_dict['node_rad_prob']['labels']=emb_use['node'].apply(lambda x:roi_to_name[x]) ### add node_name to load embeddings? but we don't want to overload if we
                                ### go into higher dims
            stat_dict['node_coord']['labels']=stat_dict['node_rad']['labels']
            stat_dict['pw']['labels']=pw_labels
            stat_dict['pw_prob']['labels']=pw_labels
            stat_dict['hkc']['labels']= clustering['labels']

            ### create the empty data frame
            rad_cols=[ c+'_Rad' for c in clustering['labels']]
            coh_cols=[ c+'_Coh' for c in clustering['labels']]
            btw_cols=[ c+'_Btw' for c in bc_labels]
            radprob_cols=[ c+'_Radprob' for c in clustering['labels']]
            cohprob_cols=[ c+'_Cohprob' for c in clustering['labels']]
            btwprob_cols=[ c+'_Btwprob' for c in bc_labels]
            ## could probably do this at the end, but this will avoid any confusion?

    stat_dict['wc']['stat_df']= np.array(wc_features)
    stat_dict['rad_prob']['stat_df']= np.array(rad_prob_features)
    stat_dict['rc']['stat_df']= np.array(rc_features)
    stat_dict['bc']['stat_df']= np.array(bc_features)

    stat_dict['bc_prob']['stat_df']= np.array(bc_prob_features)
    stat_dict['wc_prob']['stat_df']= np.array(wc_prob_features)

    stat_dict['node_rad']['stat_df']= np.array(node_features)
    stat_dict['pw']['stat_df']= np.array(nodepw_features)

    stat_dict['node_rad_prob']['stat_df']= np.array(node_rad_prob_features)
    stat_dict['pw_prob']['stat_df']= np.array(nodepw_prob_features)
    graph_id_ordered=np.array(graph_id_ordered)[:,None]

    stat_dict['hkc']['stat_df']= hk_features
    stat_dict['c_coord']['stat_df']= c_coord_features
    stat_dict['node_coord']['stat_df']= node_coord_features
    print(graph_ids[:,None].shape,stat_dict['rc']['stat_df'].shape)
    full_stat_array=np.concatenate([
        graph_id_ordered,stat_dict['rc']['stat_df'],stat_dict['wc']['stat_df'],stat_dict['bc']['stat_df']
        ,stat_dict['rad_prob']['stat_df'],stat_dict['wc_prob']['stat_df'],stat_dict['bc_prob']['stat_df']
        ],axis=1)
    # full_stat_array=np.concatenate([graph_ids,stat_dict['rc']['stat_df'],stat_dict['wc']['stat_df'],stat_dict['bc']['stat_df']],axis=0)
    full_stat_df=pd.DataFrame(columns=['Scan Index']+rad_cols+coh_cols+btw_cols+radprob_cols+cohprob_cols+btwprob_cols,data=full_stat_array)
    full_stat_df['Scan Index']=full_stat_df['Scan Index'].astype(int)


    print(full_stat_df)
    print(clinical_df)
    # right join just to keep cols in easy order
    full_stat_df=clinical_df.join(full_stat_df,on=['Scan Index'],rsuffix='_',how='right')
    pickle.dump( stat_dict, open( out_dir, "wb" ) )
    full_stat_df.to_csv(out_dir_csv)

    # pickle.dump( full_stat_df, open( out_dir_csv, "wb" ) )
    return stat_dict,full_stat_df

def get_cluster_feats(label_col,graph_ids,emb_root):
    wc_features = []
    bc_features = []
    rc_features=[]
    label=label_options[label_col]
    label_names=[]
    for g_id in graph_ids.astype(int):
        emb_file =os.path.join(emb_root,str(int(g_id))+"_embeddings.tsv")
        emb = load_embedding(emb_file)
        clustering= fit_and_plot_clusters(emb,title='',ignore_index=[8,17],label=label,ignore_plot=True)
        emb = clustering['embedding']
        hkmeans = clustering['model']

        emb = clustering['embedding']
        hkmeans = clustering['model']
        wc1 = cluster_features(emb, hkmeans.centroids, wc_metric='pairwise')['within']
        wc2 = cluster_features(emb, hkmeans.centroids, wc_metric='variance')['within']
        wc3 = cluster_features(emb, hkmeans.centroids, wc_metric='diameter')['within']
        wc_features.append(np.hstack((wc1, wc2, wc3)))

    #     print(wc1.shape,wc2.shape,wc3.shape)
        bc1 = cluster_features(emb, hkmeans.centroids, bc_metric='average')['between']
        bc2 = cluster_features(emb, hkmeans.centroids, bc_metric='max')['between']
        bc3 = cluster_features(emb, hkmeans.centroids, bc_metric='min')['between']
        bc_features.append(np.hstack((bc1.flatten(), bc2.flatten(), bc3.flatten())))
    #     print(wc1.shape,wc2.shape,wc3.shape)

    #     print(np.array(wc_features).shape,'washcloth')
    #     print(np.array(bc_features).shape,'brian celler')
        rc1 = cluster_features(emb, hkmeans.centroids, rc_metric='average')['radius']
        rc2 = cluster_features(emb, hkmeans.centroids, rc_metric='variance')['radius']
        rc_features.append(np.hstack((rc1, rc2)))
        if len(label_names)==0:
            label_names=clustering['labels']
            
    return np.array(wc_features),np.array(bc_features),np.array(rc_features),label_names
    

def fd_decoder(dist,r,t):
    return 1/(np.exp((dist - r) / t)+ 1.0) 
def to_embedding_dict(emb,dims=3,no_suffix=False):
    all_dim_cols = ['x','y','z','a','b','c']
    dim_cols=all_dim_cols[:dims]
    emb_dict = {}
    for i in range(emb.shape[0]):
        row = emb.iloc[i]
        # if no_suffix:
        name_to_use=int(row['node'].split('_')[0]) if no_suffix else row['node']

        emb_dict[name_to_use]=np.array(row[dim_cols])
    return emb_dict

# def get_embedding_scores(emb_root,relation_dir,graph_ids,template_df,suffix='.tsv',dims=2):
# def get_embedding_scores(emb_root,relation_dir,graph_ids,template_df,suffix='.tsv',dims=2):



def get_roi_to_name(template_df,node_col):
	roi_to_name={}
	for i in range(template_df.shape[0]):
	    r=template_df.iloc[i]
	    roi_to_name[r[node_col]]=r['Nicknames'][2:-2]
	# print(roi_to_name)
	return roi_to_name
def n_to_colors(n):
    if n <= 12:
        colors = ['b', 'r', 'g', 'y', 'm', 'c', 'k', 'silver', 'lime', 'skyblue', 'maroon', 'darkorange']
    elif 12 < n <= 20:
        colors = [i for i in plt.cm.get_cmap('tab20').colors]
    else:
        cmap = plt.cm.get_cmap(name='viridis')
        colors = cmap(np.linspace(0, 1, n_clusters))
    return colors

def n_to_marker(n):
    colors = ['o','^','X','D']
    return colors

def n_to_edgecolors(n):
    colors = ['k','darkorange','b']
    return colors





   

# def 

        
    ### 
    

def combine_patient_feats(X,y,clinical_df,
                          add_cog=True,add_diag=False,conditional=None,average_scans=False,shuffle=False):
    ## feat_df should be # scans by # feats
    ## assume every patient has 2 scans, one pre one post with same diagnosis and CogTr code
#     assert X.shape[0]==180 ## for this dataset
#     pats = clinical_df['ID'].unique()
    use_first=False
    use_second=False
    use_diff= False
    use_avg=False
    use_stack=True
    use_concat=False
    add_cog=False
    add_diag=False
    
    try:
        assert conditional==None
        pats = clinical_df['ID'].unique()
    except:
        print('conditational')
        print(conditional)
        pats = clinical_df[conditional]['ID'].unique()
        
#     print(pats,'PATS')
    pats=np.sort(pats)
    pat_to_indx={}
    
    for i in range(len(pats)):
        pat_to_indx[pats[i]]=i
        
    outX_dim = X.shape[1]*2 if not add_cog else (X.shape[1]*2)+1
    
#     outputX = np.array((len(pats),outX_dim))
#     outputY = np.array((len(pats),Y.shape[1]*2)) ## may be oneD
    outputX = []
    outputY=[]
    for p in pats:
        pat_rows = clinical_df[clinical_df['ID']==p]
        assert len(pat_rows)==2
        pre_row = pat_rows[pat_rows['Pre']==1].iloc[0]
        post_row = pat_rows[pat_rows['Pre']==0].iloc[0]
#         print(post_row)
#         print(post_row.shape)
#         assert len(post_row)==1
#         assert len(pre_row)==1
        assert pre_row['CogTr']==post_row['CogTr']
        assert pre_row['diagnosis']==post_row['diagnosis']
        pre_scan=pre_row['Scan Index']
        post_scan=post_row['Scan Index']
#         print(pre_scan)
#         X[pre_scan]
        if use_avg:
            cat =np.vstack((X[pre_scan],X[post_scan]))
#             print(cat.shape,'CAT SHAPE')
            patX = np.mean(cat,axis=0)
#             print(patX.shape,'out shape'
        elif use_first:
            patX=np.array(X[pre_scan])
        elif use_second:
            patX=np.array(X[post_scan])
        elif use_diff:
            patX=np.array(X[post_scan])-np.array(X[pre_scan])
        elif use_stack:
#             print(X[post_scan].shape,'SHAPE OG')
#             patX=np.concatenate((X[pre_scan],X[post_scan]),axis=1)
#             print(patX.shape,'AXIS1')
            patX = np.vstack((X[pre_scan],X[post_scan]))
            patX =patX.T
#             print(patX.shape,'AXIS0')
        elif use_concat:
            patX=np.concatenate((X[pre_scan],X[post_scan]))
        else:
            raise Exception('ALL METHODS FALSE')

    
#         print(patX.shape,'X SHAPE')
        patCog = pre_row['CogTr']
        patDiag = pre_row['diagnosis']
        if add_cog:
            patX=np.concatenate((patX,np.array([patCog])))
        if add_diag:
            patX=np.concatenate((patX,np.array([patDiag])))
#         print(patX.shape,'add one?')
#         ass
#         assert y[pre_scan]==pre_row['diagnosis']
        outputX.append(patX)
        outputY.append(y[pre_scan])
    random_order= np.arange(len(outputX))
    np.random.shuffle(random_order)

    

    
    outputX=np.array(outputX)
    outputY=np.array(outputY)
    outputX_shuff=outputX[random_order]
    outputY_shuff=outputY[random_order]
#     print(outputY_shuff[5])
#     print(outputY[random_order[5]])
#     print(outputX_shuff[5])
#     print(outputX[random_order[5]])
    if shuffle:
        print('SHUFFLE')
        return outputX_shuff,outputY_shuff
    print('NO SHUFFLE')
    return outputX,outputY,pat_to_indx
        

def multiple_embedding_analysis(emb_roots,clinical_df,label,net_threshold,
                               template_df,save_dir='',model_dir='',
                               bin_rank_threshold=10,use_binary=False,
                                suffix='.csv',balanced_atl=True
                               , average=True):
    """
    clinical_df was added in place of scan_ids-- we are going to chage all of it around
    to avoid any potential mistakes with labels... you are such a dickhead sometimes.. best to avoid any potential for mistakes..

    takes all the same args as multiple embedding_analysis BUT list of emb_roots
    first will run embedding_analysis on all embs
    """
    ## takes average of all statistics
# wc METRIC
# rc METRIC
# bc METRIC
# node_rad METRIC
# pw METRICFmetric

    if use_binary:
        thresh_str = 'Rank'+str(bin_rank_threshold)
    else:
        thresh_str= 'PCTt'+str(net_threshold)
    clinical_df['Scan Index']=clinical_df['Scan Index'].astype(int)
    useable_mets=set(['wc','rc','bc','pw','node_rad','wc_prob','rc_prob','bc_prob','pw_prob','node_rad_prob']) #### these are transductivable, we can average
    first=True
    
    all_stat_dicts=[]
    all_stat_dfs=[]
    running_c=0

    for root in emb_roots:
        if ('.pt' in root) or ('.csv' in root) or ('relations' in root):
            continue
        root_files=set(os.listdir(root))
        # check to make sure embeddings are complete 
        if model_dir:
            print('passed manual model_dir')
        elif ('final_model.pth' not in root_files) and ('model.pth' not in root_files) and ('finish_train.txt' not in root_files):
            print('training not complete or stopped midway, skipping analysis for {}'.format(root))
            continue
        else:
            model_dir=root
        if 'scan_info.csv' not in os.listdir(root):
            print('embeddings not yet saved for {}'.format(root))
            print('Saving now.')
            save_embeddings(root,overwrite=False)

        emb_root=os.path.join(root,'embeddings')
        # print(emb_root,'embedding root')

        model_path = os.path.join(model_dir,"{}.pt".format('model'))  ### args should be saved with the model
        # printmodel_path,'model path'
        model = th.load(model_path,map_location=torch.device('cpu'))

        try:
            scan_path =  os.path.join(root,'scan_info.csv')
            scan_info = pd.read_csv(scan_path)
        except:
            print('testing changes to scan info with diff names')
            scan_path =  os.path.join(root,'scan_info_full.csv')
            scan_info = pd.read_csv(scan_path)
        r=model.dc.r.item()
        t=model.dc.t.item()
        c=model.c.item()
        dims=model.args.output_dim

        for l in range(model.args.num_layers):
            print(model.encoder.curvatures[l],'layer of c',l)
            print(c,'listed c')

        running_c+=model.encoder.curvatures[model.args.num_layers-1]
        
            # print(model.encoder.layers[l].c,'layer of c',l)
            # print(model.encoder.layers[l].c,'layer of c',l)
        print(r,t,c)    
        print('r','t','c')
        print(model.args.band,'BANDS')
        print(model.args.use_identity,'ID please?')
        print(model.args.use_plv,'PLV')
        print(model.args.use_norm,'graph norm?')
        # we should probably filter clinical_df by model.args??
        print(model.args.idxs_dict,'ARGUMENT')
        all_scan_ids=set([int(s) for s in model.args.idxs_dict['all']])
        print(all_scan_ids,'All scans?')

        clinical_df_subset=clinical_df[clinical_df['Scan Index'].isin(all_scan_ids)]
        # why not just psass in the whole clinical_df?
        stat_dict_new,full_stat_df_new=embedding_analysis(emb_root,clinical_df_subset,label,net_threshold=net_threshold,
                               template_df=template_df,
                               bin_rank_threshold=bin_rank_threshold,
                           use_binary=use_binary,suffix=suffix,dims=dims,balanced_atl=balanced_atl,
                                        t=t,c=c,r=r)
        
        all_stat_dicts.append(stat_dict_new)
        all_stat_dfs.append(full_stat_df_new)
        # break
    if not all_stat_dicts:
        print('No successful embeddings')
        return None,None,None,None
    print(running_c/len(emb_roots),'AVERAGE C')
    print('out stat dict')
    if not average:
        return all_stat_dicts,None,all_stat_dfs,None
    
    ### now we average all the vals, hope it works!
    full_dict={}
    first=True
#     data_dict={}  ### this will hold a running count all the stat_dfs, for every key that can then be avged
                    ### and fill in to the full_df
    last=False
    for i in range(len(all_stat_dicts)):
        stat_dict=all_stat_dicts[i]
        if first==True:
#             print('FIRST')
            
            full_dict=deepcopy(stat_dict)
#             print(full_dict['wc']['stat_df'])
            first=False
            continue

        # print('full dict')
        if i == len(all_stat_dicts)-1:
            last=True
        
#         print(full_dict[''])
        for metric,metric_dict in stat_dict.items():
            met_stat_df=metric_dict['stat_df']
#             print(metric,'METRIC')
            if metric not in useable_mets:
#                 print(metric,'SKIP')
                continue
            full_dict[metric]['stat_df']
            full_dict[metric]['stat_df']+=met_stat_df
            if last:
                full_dict[metric]['stat_df']
                full_dict[metric]['stat_df']/=len(all_stat_dicts)

    print(all_stat_dfs[0].shape,'og shape')
    for df in all_stat_dfs:
        print(df.shape,'DF SHAPE')
    stat_df_cols=all_stat_dfs[0].columns
    for i in range(0,len(all_stat_dfs)):
        # col_1=set(all_stat_dfs[i].columns)
        bad_cols=[c for c in all_stat_dfs[i].columns if 'Unnamed' in c]
        if bad_cols:
            # print(bad_cols,'bad cols')
            all_stat_dfs[i]=all_stat_dfs[i].drop(columns=bad_cols)
    # DataFrame.drop(columns=[None]
    # turning all dfs into a stacked numpy array so we can take the mean
    try:
        df_concat = np.stack([df for df in all_stat_dfs],axis=2)
    except:
        for i in range(1,len(all_stat_dfs)):
            col_1=set(all_stat_dfs[i].columns)
            col_0=set(all_stat_dfs[i-1].columns)
            print(col_0 ^ col_1 , 'disjoint of two sets')
        raise Exception('MESSED UP stack')
    # creating empy array to turn back into pandas
    avg_stat_array=[]
    #used to keep track of our columns to make a df
    used_columns=[]
    for i in range(df_concat.shape[1]):
        if stat_df_cols[i]=='Scan Index':
            # avg_stat_np[:,i]=df_concat[:,i,0]
            avg_stat_array.append(df_concat[:,i,0])
        elif stat_df_cols[i] in clinical_df.columns:
            # then same as clinical,should all be the same
            # going to skip and then join on the scan index
            # avg_stat_np[:,i]=df_concat[:,i,0]
            continue
        else:
            ## takes mean in last dimension
            avg_stat_array.append(df_concat[:,i].mean(axis=-1))
            # avg_stat_np[:,i]=df_concat[:,i].mean(axis=-1)
        used_columns.append(stat_df_cols[i])
    avg_stat_np=np.array(avg_stat_array).T
    print(avg_stat_np.shape,'conccat')
    # avg_stat_np=np.array(used_columns).T
    # avg_stat_df=pd.DataFrame(columns=used_columns,data=avg_stat_np.T)
    avg_stat_df=pd.DataFrame(columns=used_columns,data=avg_stat_np)
    full_stat_df=clinical_df.join(avg_stat_df,on=['Scan Index'],rsuffix='_',how='right')
    print(avg_stat_df.shape,'SHAPE')
    print(full_stat_df,'avg stat df')
    
    csv_str='average_'+str(len(all_stat_dfs))+'_embedding_stats_'+str(label)+'_'+thresh_str+'.csv'
    out_dir_csv=os.path.join(save_dir,csv_str)
    if save_dir:
        full_stat_df.to_csv(out_dir_csv)
    return all_stat_dicts,full_dict,all_stat_dfs,full_stat_df




def load_model(model_dir):
    model_name ='model'
    model_path = os.path.join(model_dir,"{}.pt".format(model_name))  ### args should be saved with the model
    config_path = os.path.join(model_dir,'config.json')  ### args should be saved with the model
    # print(config_path,'CONFI')
    out_embedding_path = model_dir
    model = th.load(model_path,map_location=torch.device('cpu'))
    # print(model.args)
    try:
        model.args
    except:
        print(config_path,'CONFIGURATION')
        print(os.path.exists(config_path),'DO WE EXIST')
        with open(config_path, 'r') as j:
             model_config = json.loads(j.read())

        print('not a real model')
        print(model_config,'CRAZY MODEL eh>')
        set_params = default_config['set_params']

        args = parse_default_args(set_params) 
        Model = LPModel
        for k,v in model_config.items():
            print(k,v)
            if hasattr(args,k):
                setattr(args,k,v)
        print(args,'ARGUING')
        args.feat_dim = args.num_feature
        # args.r
        model = Model(args).to(args.device)


    # asdkug
    return model
#     print(full_dict)
       
# X_weight_use,y=labels_to_use
# conditionals = (clinical_df['Pre']==0)
# # conditionals = (clinical_df['diagnosis']==)
# # conditionals = (clinical_df['Pre']==0)
# x,y,pat_to_indx0=combine_patient_feats(X_weight,graph_labels,clinical_df,conditional=conditionals)
    
    


    
# print(x.shape)
