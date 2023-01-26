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
import pingouin as pg
from numpy_ml.neural_nets.losses import CrossEntropy
# from statsmodels.stats.weightstats import DescrStats  ## gonna need that for weighted
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, f1_score,roc_auc_score
from sklearn.model_selection import cross_val_predict,GridSearchCV
from statsmodels.stats.multitest import fdrcorrection
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

from hyperbolic_learning_master.utils.embed import train_embeddings, load_embeddings,load_embeddings_npy, evaluate_model,mean_average_precision
from hyperbolic_learning_master.utils.utils import poincare_distances,poincare_dist,hyp_lca_all

from diff_frech_mean.frechetmean import Poincare as Frechet_Poincare
from diff_frech_mean.frechet_agg import frechet_agg
from diff_frech_mean.frechetmean.frechet import frechet_mean

import os

from utils import *
from hyperbolic_learning_master.utils import *
from hyperbolic_learning_master.hyperbolic_kmeans.hkmeans import HyperbolicKMeans, plot_clusters,compute_mean

# from sknetwork.data import karate_club
from scipy.cluster.hierarchy import dendrogram, fcluster, leaves_list, set_link_color_palette,linkage
# from sknetwork.hierarchy import tree_sampling_divergence, Paris,dasgupta_score,dasgupta_cost  ####### WILL NEED THIS!!
# from sknetwork.data import house
import matplotlib as mpl
from matplotlib.pyplot import cm
from scipy.cluster import hierarchy
# from HypHC_master.utils.lca import hyp_lca
# from HypHC_master.utils.visualization import plot_geodesic
# import HypHC_master.utils.visualization as viz
from heapq import heapify,heappush,heappop

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
    all_dim_cols = ['x','y','z','a','b','c','d','e']
    if type(emb)==str:
#         print('csv' in emb,'WELL IS IT OR ISNT IT?')
        if 'csv' in emb:
            emb = pd.read_csv(emb)
            # print(emb,'EMBEDING')
            dim_cols = [ e for e in emb.columns if e!=node_col]

            print(len(dim_cols),'NATURAL DIM')
            emb.columns= ['node']+dim_cols
            print("EMBEDDING CSV LIKE A GOOD CHRISTIAN")
        else: ### this is so wack
            emb = pd.read_table(emb, delimiter=' ')
            dims= int(emb.columns.values[1])
            # print(emb,'emb')
            # print(emb.shape,'EMBEDDING SHAPE (load)')
            print(dims,'EMBEDDING SHAPE (load)')
            # print(dims,'DIMENISION')
            dim_cols = all_dim_cols[:dims]
#             dim_cols = [ e for e in emb.columns if e!=node_col]
#             dim_cols = ['node']+dim_cols
            # print(emb,"EMBEDDING!!")
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
                       net_threshold=.2,suffix='.tsv',dims=2,hkmeans_full=None,balanced_atl=True,c=1,r=2,t=1):
#     shared_nodes,excluded_nodes=get_shared_nodes(emb_root,graph_ids,template_df,suffix=suffix)
        
    """
    clinical_df was subbed in for graph_ids to avoid any mistakes with labels
    """
#     print(len(shared_nodes),'SAHRED NODES')
#     print(excluded_nodes,'didnt make the cut')
    
#     eeee

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
    # if False:
        stat_dict = pickle.load( open(out_dir , "rb" ) )
        full_stat_df = pd.read_csv(out_dir_csv)
        # print(stat_dict,'STAT DICT')
        # print(full_stat_df,'FULL STATS')
        return stat_dict,full_stat_df
        print(stat_dict)

        # print('\n\n\n')

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
    # dim_cols = ['x','y'] if dims==2 else ['x','y','z']
    # stat_dict={'wc':{},'rc':{},'bc':{},'node_rad':{},'pw':{},'c_coord':{},'node_coord':{}} 
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
    print(net_threshold)
    
    if not use_binary:
        print('making bin rank thresh -1 bc not using binary')
        bin_rank_threshold=-1

    # print
    template_df= template_df.apply(
        lambda row: meg_roi_network_assigments(
            row,combine_dmn=False,network_thresh=net_threshold,rank_thresh_inclusive=bin_rank_threshold,use_binary=use_binary
            ,balanced=balanced_atl),axis=1)    

    # print(template_df[template_df['SMN Bin']>0],'TEMPLATE BF')
    # ee

    networks=["pDMN","aDMN","DAN","FPN","VN","VAN","SN","SMN"]
    bin_cols=[n+' Bin' for n in networks]
    # print(pct_or_thresh_name,'thresh it')
    print(template_df[bin_cols].sum(),'eyyy')
    # suugg


    # print()
    roi_to_name=get_roi_to_name(template_df,node_col='RoiID')
    graph_ids=clinical_df['Scan Index'].values
    graph_id_ordered=[]
    # for g_id in graph_ids.astype(int):   ##### easy... create_file list?? then we can just index in w/ graph ids
    for g_id in graph_ids.astype(int):   ##### easy... create_file list?? then we can just index in w/ graph ids
        print(g_id,'JEZY')
        # if g_id>2:
            # break

        graph_id_ordered.append(g_id)
        emb_file =os.path.join(emb_root,str(int(g_id))+"_embeddings"+suffix)
        emb = load_embedding(emb_file,template_df,c=c)
#         emb=emb[emb['node_index'].isin(shared_nodes)]
#         print(label,'LABEL')
#         print(cluster_label,'Cluster LABEL')
        # print(emb,'EMY')
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
        # node_rad_prob_features.append(np.array(node_polars['hyp_r_prob']))
        # print(np.array(node_polars['hyp_r']))
#         node_coords.append(cartesian_to_polar_hyp((hkmeans.centroids)))
#         print(hkmeans.centroids,'SNETROID')
#         print(polar_centroids,'new man')hc)
#         rgffr
#         print(cluster_emb,'CLUSTERING EMBEDDING!!')
        if weighted:
#             raise Exception
#             www
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

        # print(roi_to_name,'ROI TO NAME')
        
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
            # full_stat_df_row=pd.DataFrame(columns=['Scan Index']+rad_cols+coh_cols+btw_cols,data=[[g_id]+stat_dict['rc']['stat_df']+stat_dict['wc']['stat_df']+stat_dict['bc']['stat_df']])
            # full_stat_df=full_stat_df
        #     first=False
        # else:
        #     full_stat_df_row=pd.DataFrame(columns=['Scan Index']+rad_cols+coh_cols+btw_cols,data=[[g_id]+stat_dict['rc']['stat_df']+stat_dict['wc']['stat_df']+stat_dict['bc']['stat_df']])
        #     full_stat_df=pd.concatenate((full_stat_df,full_stat_df_row))
        # print(full_stat_df,'full_stat_df')

    ### here we can add to full_stat_df
    # ss
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
    # sdds
    # right join just to keep cols in easy order
    full_stat_df=clinical_df.join(full_stat_df,on=['Scan Index'],rsuffix='_',how='right')
    # full_stat_df=pd.DataFrame(columns=['Scan Index']+rad_cols,data=full_stat_array)
    # np.save()
    # print(full_stat_df)

    

    
    pickle.dump( stat_dict, open( out_dir, "wb" ) )
    full_stat_df.to_csv(out_dir_csv)
    # pickle.dump( full_stat_df, open( out_dir_csv, "wb" ) )
    return stat_dict,full_stat_df

def standard_anova(data,columns,clinical_df):
    # planning on mixing with mixed_anova
    df = pd.DataFrame(data)
    stat_columns=df.columns
    df[columns]=np.array(clinical_df[columns])

    first=True
    for c in stat_columns:
        df[c]=pd.to_numeric(df[c])
        res = pg.anova(dv=c, between=columns,data=df)
        res['metric']=c
        if first:
            combined_results=res
            first=False
        else:
            combined_results=pd.concat([combined_results,res],axis=0)
    
    # alternative could do label only before the error correction...
    # but for now we're still sig
    combined_results=combined_results.reset_index()
    fdr=fdrcorrection(pvals=combined_results['p-unc'].values,alpha=.05)    
    combined_results[['Significant','q_value']]= np.array([fdr[0],fdr[1]]).T
    combined_results['Significant']=fdr[0]
    combined_results['q_value']=fdr[1]
    combined_results['p_value']=combined_results['p-unc']
    
    # label_only=combined_results[combined_results['Source']=='label']
    # label_only=label_only.reset_index()
    # fdr=fdrcorrection(pvals=label_only['p-unc'].values,alpha=.05) 
    # label_only[['Significant_alt','q_value_alt']]= np.array([fdr[0],fdr[1]]).T
    return combined_results

def mixed_anova(data,split_labels,clinical_df):
    """
    should be to pass in arg to make purely repeated measure
    """
    df = pd.DataFrame(data)
    stat_columns=df.columns
    df['label']=split_labels
    df['time_labels']=np.array(clinical_df['Pre'])
    df['ID']=np.array(clinical_df['ID'])

    first=True
    for c in stat_columns:
        df[c]=pd.to_numeric(df[c])
        res = pg.mixed_anova(dv=c, between='label',within='time_labels',subject='ID', data=df)
        res['metric']=c
        if first:
            combined_results=res
            first=False
        else:
            combined_results=pd.concat([combined_results,res],axis=0)
    
    # alternative could do label only before the error correction...
    # but for now we're still sig
    combined_results=combined_results.reset_index()
    print(combined_results,'COMBINED RESULTS')

    fdr=fdrcorrection(pvals=combined_results['p-unc'].values,alpha=.05)    
    combined_results[['Significant','q_value']]= np.array([fdr[0],fdr[1]]).T
    combined_results['Significant']=fdr[0]
    combined_results['q_value']=fdr[1]
    combined_results['p_value']=combined_results['p-unc']
    
    label_only=combined_results[combined_results['Source']=='label']
    label_only=label_only.reset_index()
    time_only=combined_results[combined_results['Source']=='time_labels']
    time_only=time_only.reset_index()
    # fdr=fdrcorrection(pvals=label_only['p-unc'].values,alpha=.05) 
    # label_only[['Significant_alt','q_value_alt']]= np.array([fdr[0],fdr[1]]).T
    return combined_results,label_only,time_only


# def welch_ttest(entity,g1, g2): 
def welch_from_columns(g1,g2,equal_var=False):
        t, p = stats.ttest_ind(g1, g2, equal_var = equal_var) 
        
        return [p,t]
    
def welch_from_columns_onepop(g1,popmean=0):
    t, p = stats.ttest_1samp(g1, popmean = popmean) 
    return [p,t]

def conf_interval(pop,conf=.95):
    mean = pop.mean()
    z_score = conf_to_z(conf)
    SE = pop.std()/(pop.shape[0]**(1/2))
    offset = z_score*SE
#     print((pop.shape[0]**(1/2)),'popshape')
#     print(offset)
    conf_int = (mean-offset,mean+offset)
    return conf_int
    
def conf_to_z(conf):
    st.norm.ppf(1-(1-0.95)/2)
    return st.norm.ppf(1-(1-conf)/2)

def welch_ttest(df,split_labels,equal_var=False): 
    ## Welch-Satterthwaite Degrees of Freedom ##
    ### lost our node information... this is dangerous?
    # print(df,'DF')
    split_options = np.array(list(set(list(split_labels)))).astype(int)
    split_labels=np.array(split_labels).astype(int)
    split_options.sort()
    # print(split_options,'split options')
    # print(split_labels)
    # eeffe
    split_0 = df[split_labels==split_options[0]]
    if len(split_options)==1:
    
        data = np.array([[r]+welch_from_columns_onepop(split_0[:,r],popmean=0) for r in range(df.shape[1])])
        dataframe = pd.DataFrame(data=data,columns=['welch_ttest','p_value','t_value'])
    else:
        split_1 = df[split_labels==split_options[1]]
        data = np.array([[r]+welch_from_columns(split_0[:,r],split_1[:,r],equal_var=equal_var) for r in range(df.shape[1])])
    dataframe = pd.DataFrame(data=data,columns=['welch_ttest','p_value','t_value'])
    #### this is where we drop rsns
#     dataframe['conf1']=
    
    # print(dataframe,'dedd')
    fdr_alpha=.05
    fdr= fdrcorrection(dataframe['p_value'].values,alpha=fdr_alpha)
    print("ALPHA AT {}".format(fdr_alpha))
    
#     print(np.array(fdr))
    dataframe[['Significant','q_value']]= np.array([fdr[0],fdr[1]]).T
    return dataframe

def anova_test(X,y,side_X=None,first_stat=None):
    print(X.shape)

    y=y[:,None]
    y_int = np.concatenate((y,y**2),axis=1)
    
    print(y_int.shape)
    
    y_use=y
    use_first=False

    if side_X is not None:
        side_X_col=['age_stand','age_sq']

        # side_X_col=[]
        side_X_val=np.array(side_X[side_X_col])
        ## adding extra dimension if only a vector
        side_X_val=side_X_val[:,None] if len(side_X_val.shape)<2 else side_X_val


        # print(side_X_val.shape)
        y_use=np.concatenate((y_use,side_X_val),axis=1)
    # print(y_use.shape,'final y shape')
    # y_use=y_int
    p_val_cols=['p_value_'+str(i) for i in range(y_use.shape[1])]
    param_cols=['coef_'+str(i) for i in range(y_use.shape[1])]
    if use_first and (first_stat is not None):
        p_val_cols.append('p_value_'+str(len(p_val_cols)))
        param_cols.append('coef_'+str(len(p_val_cols)))

    y_use=sm.add_constant(y_use)

    cols=['ID']+p_val_cols+param_cols
    anova_df=pd.DataFrame(data=[],columns=cols)
    for c in range(X.shape[1]):
        # print(c)
        col=X[:,c]
        colsq=col**2

        # col = sm.add_constant(col)
        # results = sm.OLS(y, col).fit()
        # col_int = np.concatenate((col,colsq),axis=1)
        ## here we concatenate the corresponding first stat (useful if we are analyzing the changes over time ie. 
        ## does the change over time depend on the initial conditions)
        if use_first and (first_stat is not None):
            print(first_stat[:,c].shape)
            y_use_final=np.concatenate((y_use,first_stat[:,c:c+1]),axis=1)
        else:
            y_use_final=y_use

       

        
        results = sm.OLS(col, y_use_final).fit()
        print(col,'colssss')
        print(y_use_final,'Y USE')
        print(results.summary())

        print(cols,'COLUMS')
        # print([c]+list(esults.pvalues[1:])+list(results.params[1:]),'data lists')
        print([c])
        print(results,'RESULTS')
        print(results.pvalues,'PVALES')

        print(list(results.pvalues[1:]))
        print(list(results.params[1:]))

        row=pd.DataFrame(data=[[c]+list(results.pvalues[1:])+list(results.params[1:])],columns=cols)
        # print(row,'ROW')
        anova_df=anova_df.append(row)
        # print(results.summary())# get all features from clustering analysis
        # fig = sm.graphics.plot_fit(results,'x1')
        # fig.tight_layout(pad=1.0)
        # plt.show()

    fdr_alpha=.05
    for j in range(len(p_val_cols)):
        p_col=p_val_cols[j]
        fdr= fdrcorrection(anova_df[p_col].values,alpha=fdr_alpha)
        anova_df[['Significant_'+str(j),'q_value_'+str(j)]]= np.array([fdr[0],fdr[1]]).T
    print(anova_df)


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
    

# print(conf_to_z(.95))
def remove_rsn(stat_df,labels ,ignore=''):
    include=[]
    for l in labels:

        rsn_split=l.split('_')
        if len(rsn_split)<2:
            include.append(True)

        elif rsn_split[0]==ignore:
            include.append(False)
        elif (len(rsn_split)==2) and (rsn_split[1]==ignore):
            # print('REMOVE')
            include.append(False)
        else:
            # print('TRUTH')
            include.append(True)
    include=np.array(include)
    # print(stat_df.shape,'STAT SHAPE')
    # print(include.shape,'include SHAPE')
    include_df=stat_df[:,np.array(include)]
    include_labels=np.array(labels)[np.array(include)]

    # print('include_labe')
    return include_df,include_labels

def metric_analysis(graph_by_metric_df,entity_names,graph_labels,metric_cols=[],y_axis='',analyze_time=False,
                    plot_title='',graph_label_names=['Healthy Control','SCI'],sort_vals=False,max_plot=100,column_name=[]
                   ,print_pval=True,print_qval=True,df_save_path='',ignore_rsn='',group_analysis=True,clinical_df=None,first_stat=None,plot_save_path=''):
    """
    graph_by_metric_df- should be df w/ flattened metric for every graph ie. 90x 8000 for 90 graphs w/ 8000 metrics
    entity_labels-the name we're gonna use for plotting each of these. if one metric per node, entity is node 1-90, if cross node should be 90*89/2
    #graph_labels. graphx1, whatever we want to group by
    column_name- whatever we want to label the y_axis ie. the name of the metric. Doesn't have to correspond to anything
    


    """
    # sig_df = welch_ttest(eval_df,graph_labels).sort_values(by='p_value')

    # graph_by_metric_df
    # print(graph_by_metric_df,'GRAPH LABELS PRE')

    # graph_by_metric_df,entity_names=remove_rsn(graph_by_metric_df,entity_names,ignore_rsn)
    # print(graph_by_metric_df,'GRAPH LABELS POST')

    # raise Exception('GAME PLAN---- functionalize plotting, so that you can split eval_df into 2,')
    if metric_cols:
        pass
    elif graph_label_names== ['Healthy Control', 'SCD']:
        metric_col='diagnosis'
    elif graph_label_names== ['CogTr', 'Control']:
        metric_col='CogTr'
    elif graph_label_names== ['Post', 'Pre']:
        metric_col='Pre'
    else:
        skip_val=True


    if metric_col:

        val_col=np.array(clinical_df[metric_col])
        val_col=val_col-val_col.min()
        print(val_col,'val col')
        print(graph_labels,'graph met')
        assert np.sum(np.abs((val_col-graph_labels)))==0
        print('ALL EQUAL!')



    eval_df=graph_by_metric_df

    # savepath_df=os.path.join(save_dir,column_name[0]+'_df.csv')

    # print(eval_df,'EVALUATION DF')
    # print(eval_df.shape)
    # sss
    # print(entity_names,'ENTITY NAMES')
#     print(eval_df,'evals')
    # sig_df = welch_ttest(eval_df,graph_labels).sort_values(by='p_value')  
    # sig_df = welch_ttest(eval_df,graph_labels).sort_values(by='p_value')  

    print(graph_labels,'LABELS')
    # if False:
    if clinical_df is not None and len(clinical_df['ID'].unique())*2<=len(clinical_df['ID']):
        repeated=True
        print('measures are repeated, doing mixed anaylsis instead of ttest')
        sig_df_combined,sig_df_label,sig_df_time = mixed_anova(eval_df,graph_labels,clinical_df)
        sig_df_list=[sig_df_label,sig_df_time]
        graph_label_name_list=[graph_label_names,['Post','Pre']]
        sig_df=sig_df_list[1]
        graph_label_names=graph_label_name_list[1]
        graph_labels=clinical_df['Pre'].values

        print(graph_labels,'GRAPH LABELS')
        # if analyze_time:
        #     sig_df=sig_df_list[1]
        #     graph_label_names=graph_label_name_list[1]
        # else:
        #     sig_df=sig_df_list[0]
        #     graph_label_names=graph_label_name_list[0]
    else:
        repeated=False
        sig_df = welch_ttest(eval_df,graph_labels)
        sig_df_combined=sig_df
        sig_df_list=[sig_df]
        graph_label_name_list=[graph_label_names]


    print(sig_df_combined,'SIG NF COMBINED')
    # ekeke

    print(sig_df,'SIG DF')
    print(sig_df.shape,'SIG DF')
    # sksks
    
    n_plot=min(sig_df.shape[0],max_plot)
    keepers = list(sig_df[:n_plot].index.astype(int))
# 
    # print(sig_df[:max_plot],'shoudl be top')
    # print(keepers,'keeps!')
    use_full_names=True
    full_entity_names=[]
    str_qs=[]
    str_ps=[]
    
    

    for i in sig_df.index.astype(int):
        ### full sorted list of new names.. 
        
        new_name=entity_names[i]
        
        p =sig_df[sig_df.index==i]['p_value'].values[0]
        q =sig_df[sig_df.index==i]['q_value'].values[0]
    
        if p<.0001:
           str_p =np.format_float_scientific(p, precision = 0, exp_digits=1)
        elif p<.001:
            str_p=str(p)[1:6]
            
        else:
            str_p=str(p)[1:5]
#         str_q='q-value: '+str(q)[1:5]
#         str_p='p-value: '+str_p
       
        str_p='p: '+str_p
        
        if q<.0001:
           str_q =np.format_float_scientific(q, precision = 0, exp_digits=1)
        elif q<.001:
            str_q=str(q)[1:6]
            
        else:
            str_q=str(q)[1:5]
         
        str_q='q: '+str_q
            
        if q<.05:
            str_q+='*'
        if p<.05:
            str_p+='*'
#         str_q='q: '+str(q)[1:5] ,
        mean_str=get_group_mu_str(eval_df[:,i],graph_labels)
        add_mean_str=False
        new_name = entity_names[i]+'\n'+str_p
        if True:
            new_name+='\n'+str_q
        if add_mean_str:
            new_name+='\n'+mean_str
#         if q!=p:
#             new_name+='\n'+str_q
        full_entity_names.append(new_name)
        str_qs.append(str_q)
        str_ps.append(str_p)

    show_data=np.array([entity_names,str_ps,str_qs]).T
    show_df=pd.DataFrame(columns=[['StatName','p_value','q_value']],data=show_data)
    if df_save_path:
        sig_df_combined.to_csv(df_save_path)

    if sort_vals:
#         eval_df=eval_df[eval_df['entity'].isin(keepers)]
        eval_df=eval_df[:,keepers]
        
        if use_full_names:  ## these will have already been sorted... you're all twisted
            entity_names=full_entity_names[:n_plot]
        else:
            entity_names=[entity_names[i] for i in keepers]
    
    eval_df = pd.DataFrame(np.ravel(eval_df), columns=y_axis)  ### ADD BIG ASS STARS
    num_labels = eval_df.shape[0]/graph_labels.shape[0]
    assert num_labels - int(num_labels) == 0 ### make sure this makes a whole number
    eval_df['label'] = np.repeat(graph_labels,num_labels)
    eval_df['label'] = eval_df.label.apply(lambda x: graph_label_names[int(x)])
    print('done with label')
    eval_df['network'] = np.tile(entity_names, graph_labels.shape[0]) ### where the names, pvals etc. gets set
    print(eval_df.shape)
    print('eval df shape')
    print('tile up')
#     print(eval_df['network'])
#     print(eval_df['network'].shape,'what is this mystery?')
    sns.set(style="whitegrid")
# ax = sns.boxplot(x=tips["total_bill"], whis=[5, 95])
    cis= np.array([[-0.35, 0.50] for i in range(len(keepers))])
    ax = sns.boxplot(x="network", y=y_axis[0], hue="label", data=eval_df, palette="pastel",whis=1.,notch=True,bootstrap=1000,showfliers = False)
#     ax = sns.swarmplot(x="network", y=column_name[0], hue="label", dta=eval_df)
    print('bout to show')
    plt.title(plot_title, size=16)
    if plot_save_path:
        print("SAVING")
        print(plot_save_path)
        plt.savefig(plot_save_path)

        # plt.save()
    plt.show();
    # lkjfg
# def plot_analysis()

def get_group_mu_str(eval_df,graph_labels):
    label_options=list(set(graph_labels))
    label_options.sort()
    # print(label_options)
    # print(eval_df,'EVAL DF')
    metr_str=''
    for o in label_options:

        sub_df=eval_df[graph_labels==o]
        # print(sub_df.mean(),'mean')
        # metr_str+='\n'+o+' '+str(sub_df.mean())
        metr_str+='\n'+str(o)+' '+ np.format_float_positional(np.median(sub_df),precision=3)
        metr_str+='\n'+ np.format_float_positional(sub_df.std(),precision=3)
        metr_str+='\n'+ str(sub_df.shape[0])

    return metr_str
def plot_boxplot(eval_df,entity_names,graph_labels,plot_title, column_name='',graph_label_names=[''],max_to_plot=-1):
    eval_df = pd.DataFrame(np.ravel(eval_df), columns=column_name)
    num_labels = eval_df.shape[0]/graph_labels.shape[0]
    assert num_labels - int(num_labels) == 0 ### make sure this makes a whole number
    eval_df['label'] = np.repeat(graph_labels,num_labels)
    
    # print(graph_labels,'graph labels')
    # print(num_labels,'num labels')
    # print(graph_label_names,'label names')
    # print(eval_df.label.unique(),'UNIQUE LABLES')

    eval_df['label'] = eval_df.label.apply(lambda x: graph_label_names[int(x)])
    eval_df['network'] = np.tile(entity_names, graph_labels.shape[0])
#     print(eval_df['network'])
#     print(eval_df['network'].shape,'what is this mystery?')
    sns.set(style="whitegrid")
# ax = sns.boxplot(x=tips["total_bill"], whis=[5, 95])
    # if max_to_plot>
    # cis= np.array([[-0.35, 0.50] for i in range(len(keepers))])
    sns.stripplot(x="network", y=column_name[0], hue="label", data=eval_df, palette="pastel")
    # ax = sns.boxplot(x="network", y=column_name[0], hue="label", data=eval_df, palette="pastel",whis=1.,notch=True,bootstrap=1000)
#     ax = sns.swarmplot(x="network", y=column_name[0], hue="label", dta=eval_df)
    plt.title(plot_title, size=16)
    plt.show();

def stack_angular(stat_df):
    angles=['theta','phi','alpha','beta']  ### need to expand to more angles
    angle_feats=['sin_theta']+['cos_'+a for a in angles]
    angle_feats=[af for af in angle_feats if af in stat_df[0][0].columns]
    thetasin_df=[]
    thetacos_df=[]
    phi_df=[]
    radius_df=[]
    all_angle_df=[]
    angle_feat_dict={af:[] for af in angle_feats}


    for s in range(len(stat_df)): ### TO AUTOMATE FOR MULTIPLE DIMS IF YOU WANT!!
        clust_list_all_ang=[]
        for af in angle_feats:
            clust_list_af=[]
            for c in stat_df[s]:
                clust_list_af.append(c[af].values)
                clust_list_all_ang.append(c[af].values)
            angle_feat_dict[af].append(clust_list_af)
        all_angle_df.append(clust_list_all_ang)
    for af,vals in angle_feat_dict.items():
        # print(np.array(vals).shape,'OG ANGLE SHAPE')
        angle_feat_dict[af]=np.array(vals)[:,:,0]


    for s in range(len(stat_df)):
            clus_list=[]
            for c in stat_df[s]:
        #         print(c['sin_theta'])
                clus_list.append(c['hyp_r'].values)

            radius_df.append(clus_list)
    # for s in range(len(stat_df)):
    #     clus_list=[]
    #     for c in stat_df[s]:
    # #         print(c['sin_theta'])
    #         clus_list.append(c['sin_theta'].values)
    #     thetasin_df.append(clus_list)
        
    # thetacos_df=[]
    # for s in range(len(stat_df)):
    #     clus_list=[]
    #     for c in stat_df[s]:
    # #         print(c['sin_theta'])
    #         clus_list.append(c['cos_theta'].values)
    #     thetacos_df.append(clus_list)

    # # for s in range(len(stat_df)):
    # #         clus_list=[]
    # #         for c in stat_df[s]:
    # #     #         print(c['sin_theta'])
    # #             clus_list.append(c['cos_phi'].values)
    # #         phi_df.append(clus_list)

    # for s in range(len(stat_df)):
    #         clus_list=[]
    #         for c in stat_df[s]:
    #     #         print(c['sin_theta'])
    #             clus_list.append(c['hyp_r'].values)

    #         radius_df.append(clus_list)

    # thetasin_df=np.array(thetasin_df)[:,:,0]
    # thetacos_df=np.array(thetacos_df)[:,:,0]
    # phi_df=np.array(phi_df)[:,:,0]
    radius_df=np.array(radius_df)[:,:,0]
    all_angle_df=np.array(all_angle_df)[:,:,0]
    # print(radius_df,'RADIUS DUH')
    return radius_df,angle_feat_dict,all_angle_df
    # return radius_df,thetasin_df,thetacos_df,""
    # return radius_df,thetasin_df,thetacos_df,phi_df

def digest_stat_dict(stat_dict,useable_ids):
    wc = stat_dict['wc']['stat_df'][useable_ids]
    wc_labels = stat_dict['wc']['labels']
    bc = stat_dict['bc']['stat_df'][useable_ids]
    bc_labels = stat_dict['bc']['labels']
    rc = stat_dict['rc']['stat_df'][useable_ids]
    rc_labels = stat_dict['rc']['labels']
    node = stat_dict['node_rad']['stat_df'][useable_ids]
    node_labels = stat_dict['node_rad']['labels']
    pw = stat_dict['pw']['stat_df'][useable_ids]
    pw_labels = stat_dict['pw']['labels']

    hkc=stat_dict['hkc']['stat_df']
    # radius_df,thetasin_df,thetacos_df,_=stack_angular(hkc)
    radius_df,angle_feat_dict,all_angle_df=stack_angular(hkc)
    radius_df=radius_df[useable_ids]
    all_angle_df=all_angle_df[useable_ids]
    for af,vals in angle_feat_dict.items():
        angle_feat_dict[af]=np.array(vals)[useable_ids]
    # thetasin_df=thetasin_df[useable_ids]
    # thetacos_df=thetacos_df[useable_ids]
    return wc,bc,rc,node,pw,radius_df,angle_feat_dict,all_angle_df,wc_labels,bc_labels,rc_labels,node_labels,pw_labels

def digest_stat_dict_prob(stat_dict,useable_ids):
    wc_prob = stat_dict['wc_prob']['stat_df'][useable_ids]
    wc_labels = stat_dict['wc']['labels']
    bc_prob = stat_dict['bc_prob']['stat_df'][useable_ids]
    bc_labels = stat_dict['bc']['labels']
    rc_prob = stat_dict['rad_prob']['stat_df'][useable_ids]
    rc_labels = stat_dict['rc']['labels']
    node_prob = stat_dict['node_rad_prob']['stat_df'][useable_ids]
    node_labels = stat_dict['node_rad_prob']['labels']
    pw_prob = stat_dict['pw_prob']['stat_df'][useable_ids]
    pw_labels = stat_dict['pw_prob']['labels']
    return wc_prob,bc_prob,rc_prob,node_prob,pw_prob,wc_labels,bc_labels,rc_labels,node_labels,pw_labels

def drop_rsn_metrics(stat_dict,ignore_abrevs=set(['SMN'])):
    hjhhke

def slice_metrics(stat_dict,metric_col,clinical_df,conditionals,use_difference=False,
    use_second_stat=False,stat_dict2=None,save_dir='',ignore_rsn=''):

    # print()
    # ignore_rsn=''
    group_analysis=True
    if use_second_stat:
        assert metric_col=='band'
    if metric_col in ('diagnosis','diagnosis_inv'):
        group_labels = ['Healthy Control', 'SCD']
    if metric_col == 'CogTr':
        # group_labels=['Control','CogTr']
        group_labels=['CogTr','Control']
    if metric_col == 'Pre':
        group_labels=['Post','Pre']
    if metric_col in ('age','age_dyn','diagnosis_age','diagnosis_train'):
        group_labels=['Age']
        group_analysis=False
    if metric_col=='diagnosis_train_cat':
        group_labels= ['HC Train', 'HC No Train','SCD Train', 'SCD No Train']
        # group_labels= ['HC Train','SCD Train', 'HC No Train', 'SCD No Train']
        # group_labels= ['HC Train','SCD Train']
    if metric_col=='band':
        assert use_second_stat
        group_labels=['Alpha','Gamma']
    if metric_col=='constant':
        group_labels=['All']

    # side_X=clinical_df['side']

    # if use_second_stat:
        # group_labels
    


    # patient_ids=clinical_df[conditionals]['ID'].values
    # conditionals=True

    # print(stat_dict)

    if use_difference:
        clinical_df=clinical_df.sort_values('Scan Index')
        # useable_1=clinical_df[conditionals&(clinical_df.Pre==1)].sort_values('ID')
        # useable_2=clinical_df[conditionals&(clinical_df.Pre==0)].sort_values('ID')

        useable_1=clinical_df[conditionals&(clinical_df.Pre==1)]
        useable_2=clinical_df[conditionals&(clinical_df.Pre==0)]

        scan_labels = clinical_df[conditionals&(clinical_df.Pre==0)][metric_col].values  #### better be the same!
        clinical_use=clinical_df[conditionals&(clinical_df.Pre==0)]
        pats1=useable_1['ID'].values.tolist()
        pats2=useable_2['ID'].values.tolist()
        print(pats1)
        print(pats2)
        assert pats1==pats2
        useable_ids1=useable_1['Scan Index'].values
        useable_ids2=useable_2['Scan Index'].values
        print('still need to change since we switched stack ang')
        wc1,bc1,rc1,node1,pw1,radius_df1,angle_feat_dict1,all_angle_df1,wc_labels,bc_labels,rc_labels,node_labels,pw_labels=digest_stat_dict(stat_dict,useable_ids1)
        thetasin_df1=angle_feat_dict1['sin_theta']
        thetacos_df1=angle_feat_dict1['cos_theta']
        wc2,bc2,rc2,node2,pw2,radius_df2,angle_feat_dict2,all_angle_df2,wc_labels2,bc_labels2,rc_labels2,node_labels2,pw_labels2=digest_stat_dict(stat_dict,useable_ids2)

        thetasin_df2=angle_feat_dict2['sin_theta']
        thetacos_df2=angle_feat_dict2['cos_theta']
        wc =wc2-wc1
        bc = bc2-bc1
        rc = rc2-rc1

        node = node2-node1
        pw = pw2-pw1
        radius_df=radius_df2-radius_df1
        thetasin_df=thetasin_df2-thetasin_df1
        thetacos_df=thetacos_df2-thetacos_df1

        wc_prob1,bc_prob1,rc_prob1,node_prob1,pw_prob1,wc_labels,bc_labels,rc_labels,node_labels,pw_labels= digest_stat_dict_prob(stat_dict,useable_ids1)
        wc_prob2,bc_prob2,rc_prob2,node_prob2,pw_prob2,wc_labels,bc_labels,rc_labels,node_labels,pw_labels= digest_stat_dict_prob(stat_dict,useable_ids2)

        wc_prob =wc_prob2-wc_prob1
        bc_prob = bc_prob2-bc_prob1
        rc_prob = rc_prob2-rc_prob1

        node_prob = node_prob2-node_prob1
        pw_prob = pw_prob2-pw_prob1

        # hkc=hkc1-hkc2


    elif use_second_stat:
        useable_both=clinical_df[conditionals]['Scan Index'].values
        clinical_use=clinical_df[conditionals]

        scan_labels = np.array([0 for i in range(len(useable_both))]+[1 for i in range(len(useable_both))])  #### better be the same!

        # pats1=useable_both['ID'].values.tolist()

        # assert pats1==pats2
        wc1,bc1,rc1,node1,pw1,radius_df1,angle_feat_dict1,all_angle_df1,wc_labels,bc_labels,rc_labels,node_labels,pw_labels=digest_stat_dict(stat_dict,useable_both)
        thetasin_df1=angle_feat_dict1['sin_theta']
        thetacos_df1=angle_feat_dict1['cos_theta']

        wc2,bc2,rc2,node2,pw2,radius_df2,angle_feat_dict2,all_angle_df2,wc_labels2,bc_labels2,rc_labels2,node_labels2,pw_labels2=digest_stat_dict(stat_dict2,useable_both)
        thetasin_df2=angle_feat_dict2['sin_theta']
        thetacos_df2=angle_feat_dict2['cos_theta']

        # wc,bc,rc,node,pw,radius_df,thetasin_df,thetacos_df,wc_labels,bc_labels,rc_labels,node_labels,pw_labels=digest_stat_dict_prob(stat_dict,useable_ids)
        wc_prob1,bc_prob1,rc_prob1,node_prob1,pw_prob1,wc_labels,bc_labels,rc_labels,node_labels,pw_labels= digest_stat_dict_prob(stat_dict,useable_both)
        wc_prob2,bc_prob2,rc_prob2,node_prob2,pw_prob2,wc_labels,bc_labels,rc_labels,node_labels,pw_labels= digest_stat_dict_prob(stat_dict2,useable_both)
        # print('hello>>>')

        wc =np.vstack([wc1,wc2])
        bc = np.vstack([bc1,bc2])
        rc =np.vstack([rc1,rc2])
        node = np.vstack([node1,node2])
        pw = np.vstack([pw1,pw2])

        wc_prob =np.vstack([wc_prob1,wc_prob2])
        bc_prob = np.vstack([bc_prob1,bc_prob2])
        rc_prob =np.vstack([rc_prob1,rc_prob2])
        node_prob = np.vstack([node_prob1,node_prob2])
        pw_prob = np.vstack([pw_prob1,pw_prob2])
        print(wc.shape,'should be twice?')
        # hkc=hkc1-hkc2
        radius_df=radius_df1-radius_df2
        thetasin_df=thetasin_df1-thetasin_df2
        thetacos_df=thetacos_df1-thetacos_df2
    else:    
        useable_ids = clinical_df[conditionals]['Scan Index'].values
        clinical_use=clinical_df[conditionals]
        # useable_ids=[0,1,2,3,4]
        # print(useable_ids)
        scan_labels = clinical_df[conditionals][metric_col].values
        wc,bc,rc,node,pw,radius_df,angle_feat_dict,all_angle_df,wc_labels,bc_labels,rc_labels,node_labels,pw_labels=digest_stat_dict(stat_dict,useable_ids)
        wc1=None
        bc1=None
        rc1=None
        wc_prob1=None
        bc_prob1=None
        rc_prob1=None
        thetasin_df=angle_feat_dict['sin_theta']
        thetacos_df=angle_feat_dict['cos_theta']
        # wc,bc,rc,node,pw,radius_df,thetasin_df,thetacos_df,wc_labels,bc_labels,rc_labels,node_labels,pw_labels=digest_stat_dict(stat_dict,useable_ids)
        wc_prob,bc_prob,rc_prob,node_prob,pw_prob,wc_labels,bc_labels,rc_labels,node_labels,pw_labels= digest_stat_dict_prob(stat_dict,useable_ids)
        # print('hello>>>')
        # print(wc_prob)
    # phi_df=phi_df[usable_ids]

    # print(phi_df.shape)
    # sss

    print(wc.shape)
    print(wc_labels)

    # reoih

    print(scan_labels,'SCAN 1')
    if min(scan_labels)>0:
        scan_labels=scan_labels-1
    print(scan_labels,'SCAN 2!')
    # if scan_labels.max()-scan_labels.min()>30:
        ### if non-catagorical, normalize!
        # print(scan_labels)
        # scan_labels=StandardScaler().fit_transform(scan_labels.reshape(-1, 1)).T[0]
        # scan_labels=sk_normalize(scan_labels.reshape(1, -1) ,axis=1)
        # print(scan_labels)

    
#     print(len(node))
#     print(node_labels,'node labels')
# #     labels = clinical_df[usable_ids][metric_col]

#     print(wc.shape)
#     print(wc_labels)
#     print(scan_labels)
    
#     # print(node,'NDODE NODE NODE')
#     print(scan_labels.shape,'SCAN SHAPE')

    # sss
    # kjg
    # print(wc,'WC')
    # aaa

    print('coh')
    metric_analysis(wc,wc_labels,scan_labels,column_name=['Cluster Cohesion'],
                    plot_title='',graph_label_names=group_labels,sort_vals=True,max_plot=8,save_dir=save_dir,ignore_rsn=ignore_rsn,group_analysis=group_analysis,clinical_df=clinical_use,first_stat=wc1)
    print('rad')
    metric_analysis(rc,rc_labels, scan_labels,column_name=['Cluster Radius from Origin'],
                    plot_title='',graph_label_names=group_labels,sort_vals=True,max_plot=8,save_dir=save_dir,ignore_rsn=ignore_rsn,group_analysis=group_analysis,clinical_df=clinical_use,first_stat=rc1)

    
    # ddd
    metric_analysis(node,node_labels,scan_labels,column_name=['Node Radius from Origin'],
                    plot_title='',graph_label_names=group_labels,sort_vals=True,max_plot=8,save_dir=save_dir,ignore_rsn=ignore_rsn,group_analysis=group_analysis,clinical_df=clinical_use)
    print('btw clust')
    metric_analysis(bc,bc_labels,scan_labels,column_name=['Dist Btw Clusters'],
                    plot_title='',graph_label_names=group_labels,sort_vals=True,max_plot=6,save_dir=save_dir,ignore_rsn=ignore_rsn,group_analysis=group_analysis,clinical_df=clinical_use,first_stat=bc1)
    metric_analysis(pw,pw_labels,scan_labels,column_name=['Dist Btw Node'],
                    plot_title='',graph_label_names=group_labels,sort_vals=True,max_plot=8,save_dir=save_dir,ignore_rsn=ignore_rsn,group_analysis=group_analysis,clinical_df=clinical_use,first_stat=bc1)

    if 'wc_prob' in stat_dict:
        print('coh prob')

        metric_analysis(wc_prob,wc_labels,scan_labels,column_name=['Cluster Cohesion Prob'],
                        plot_title='',graph_label_names=group_labels,sort_vals=True,max_plot=8,save_dir=save_dir,ignore_rsn=ignore_rsn,group_analysis=group_analysis,clinical_df=clinical_use,first_stat=wc_prob1)
        print('rad prob')
        metric_analysis(rc_prob,rc_labels, scan_labels,column_name=['Cluster Radius from Origin Prob'],
                        plot_title='',graph_label_names=group_labels,sort_vals=True,max_plot=8,save_dir=save_dir,ignore_rsn=ignore_rsn,group_analysis=group_analysis,clinical_df=clinical_use,first_stat=rc_prob1)

        metric_analysis(node_prob,node_labels,scan_labels,column_name=['Node Radius from Origin Prob'],
                        plot_title='',graph_label_names=group_labels,sort_vals=True,max_plot=8,save_dir=save_dir,ignore_rsn=ignore_rsn,group_analysis=group_analysis,clinical_df=clinical_use,)

        metric_analysis(bc_prob,bc_labels,scan_labels,column_name=['Dist Btw Clusters Prob'],
                        plot_title='',graph_label_names=group_labels,sort_vals=True,max_plot=6,save_dir=save_dir,ignore_rsn=ignore_rsn,group_analysis=group_analysis,clinical_df=clinical_use,first_stat=bc_prob1)
        metric_analysis(pw_prob,pw_labels,scan_labels,column_name=['Dist Btw Node Prob'],
                        plot_title='',graph_label_names=group_labels,sort_vals=True,max_plot=8,save_dir=save_dir,ignore_rsn=ignore_rsn,group_analysis=group_analysis)


    # metric_analysis(phi_df,rc_labels,scan_labels,column_name=['Phi Cos Cluster'],
                    # plot_title='',graph_label_names=['Healthy Control','SCI'],sort_vals=True,max_plot=8)
    # metric_analysis(thetasin_df,rc_labels,scan_labels,column_name=['Theta Sin Cluster'],
                    # plot_title='',graph_label_names=['Healthy Control','SCI'],sort_vals=True,max_plot=8,group_analysis=group_analysis)
    # metric_analysis(thetacos_df,rc_labels,scan_labels,column_name=['Theta Cos Cluster'],
                    # plot_title='',graph_label_names=['Healthy Control','SCI'],sort_vals=True,max_plot=8,group_analysis=group_analysis)



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

def edges_from_thresh(scan,thresh):
    adj_mat = np.where(scan>thresh,1,0)
    # print(adj_mat.shape,'SHAPE UP')  
    x, y = sp.triu(adj_mat,k=1).nonzero()
    pos_edges = np.array(list(zip(x, y))).astype(int)
    x, y = sp.triu(sp.csr_matrix(1. - adj_mat),k=1).nonzero()
    neg_edges = np.array(list(zip(x, y))).astype(int)
    return pos_edges,neg_edges

def get_embedding_scores_thresholds(root,thresholds,use_natural=True,val_only=False,args={}):


    ### check shapes and go from there...
    scan_path =  os.path.join(root,'scan_info.csv')
    
    
    if args!={}:
        # args=args
        model=None
    else:
        model_path = os.path.join(root,"{}.pt".format('model'))  ### args should be saved with the model

        model=load_model(root)
        # model = th.load(model_path,map_location=torch.device('cpu'))
        args=model.args
    # print(args,'ARGS')
    if os.getcwd() not in args.raw_clinical_file:
        args.raw_clinical_file=default_clinical

    if os.getcwd() not in args.raw_scan_file:
        args.raw_scan_file=default_MEG

    scan_df = pd.read_csv(scan_path)
    if val_only:
        scan_df=scan_df[scan_df['train']==0]

    print(scan_df.shape,'HOW MUCH VAL?')


    output_str='_'.join(root.split('\\')[-3:])

    last_str='val_scores_' if val_only else 'scores_'
    outdir=os.path.join(root,output_str+last_str)

    out_dir=os.path.join(root,last_str+output_str+'.csv')
    rel_dir=os.path.join(root,'relations')




    template = os.path.join(r"C:\Users\coleb\OneDrive\Desktop\Fall 2021\Neuro\hgcn\data\MEG","AAltemplate_balanced.csv")
    template_df = pd.read_csv(template)
    template_df= template_df.apply(lambda row: meg_roi_network_assigments(row,combine_dmn=False),axis=1)

    t_trained=args.adj_threshold
    band_adj = BAND_TO_INDEX['alpha']
    metric_adj = METRIC_TO_INDEX[args.metric]
    raw_scans = np.load(args.raw_scan_file)
    scans = raw_scans[:,band_adj,metric_adj] ### this is not changed by any hyperparamters

    dim_cols=['x','y','z','a','b','c','d','e']
    print(dim_cols,'HOW MANY DIMMIES??')
    if not model:
        c=1
    else:
        c = model.c
    fd_r=args.r
    fd_t=args.t

    # optimize = 'min' if args.use_weighted_loss else 'BCE'
    if not model:
        dev_losses=[-1]
    else:
        stats=model.metrics_tracker
    # stat_to_plot
        dev_losses=[t['loss'] for t in stats['dev']]
    # best_train = min(train_losses) if optimize=='min' else max(train_losses)
    best_dev = min(dev_losses)
    # val_
    # print(mode)

    all_embeddings={}

    # print(output_str,'EY YO')
    # print(out_dir)
    nontrain_indx=scan_df['train']==0

    if use_natural and not (t_trained in thresholds):
        print('adding natural thresh to thresholds')
        thresholds=[t_trained]+thresholds

    # print('SCANNABLE')

    for i in range(len(scan_df)):
        print(i,'i')

        ### here we're gonna calculate all embedding related things for reuse over thresholds
        row=scan_df.iloc[i]
        # print(row,'ROW')
        g_id=row['graph_id']
        emb_file = row['save_location']
        emb_df = load_embedding(emb_file,template_df,c=c)
        dim_cols=[e for e in emb_df.columns if e in dim_cols]
        print(dim_cols,'DIM COLS')
        emb=np.array(emb_df[dim_cols])
        print(emb.shape,'shap!')
        dist_pw=poincare_distances(emb,c=c,full=True)
        scores_pw=fd_decoder(dist_pw,t=fd_t,r=fd_r)


        # print(dist_pw,'DISTANC PWIZZLE')
        emb_dict = to_embedding_dict(emb_df,len(dim_cols),no_suffix=True)
        all_embeddings[g_id]={}
        all_embeddings[g_id]['emb']=emb
        all_embeddings[g_id]['dist_pw']=dist_pw
        all_embeddings[g_id]['scores_pw']=scores_pw
        all_embeddings[g_id]['emb_dict']=emb_dict
        all_embeddings[g_id]['train']=row['train']
    # print()


        # if len(all_embeddings.keys())>4:
            # break

    mAPs = []
    mean_ranks=[]
    mean_degrees=[]
    mean_rocs=[]
    mean_rocsdist=[]
    mean_BCEs=[]
    mAPs_blind = []
    mean_ranks_blind=[]
    mean_degrees_blind=[]
    mean_rocs_blind=[]
    mean_rocsdist_blind=[]
    mean_BCEs_blind=[]

    trained_losses=[]
    thresholds_all=[]


    roots=[]

    t_traineds=[]
    # columns=['threshold','mAPs','Mean Ranks','Mean Degrees','ROC','MAE','T Trained','Root']
    columns=['threshold',
    'mAPs','Mean Ranks','Mean Degrees','ROC','ROCDist','BCE',
    'mAPs_blind','Mean Ranks_blind','Mean Degrees_blind','ROC_blind','ROCDist_blind','BCE_blind',
    'Dev Loss','T Trained','Root']
    if not os.path.exists(rel_dir):
        os.makedirs(rel_dir)

    for t in thresholds:
        
        mAPs_t = []
        mean_ranks_t=[]
        mean_degrees_t=[]    
        mean_roc_t=[]
        mean_rocdist_t=[]
        mean_BCE_t=[]

        mAPs_t_blind = []
        mean_ranks_t_blind=[]
        mean_degrees_t_blind=[]    
        mean_roc_t_blind=[]
        mean_rocdist_t_blind=[]
        mean_BCE_t_blind=[]

        rel_dir_t=os.path.join(rel_dir,str(t))
        if not os.path.exists(rel_dir_t):
            os.makedirs(rel_dir_t)        
        # for i in range(len(scan_df)):
        for g_id,i_dict in all_embeddings.items():
            # print(i)
            print(g_id,'gidydy')
            rel_dir_ti=os.path.join(rel_dir_t,str(g_id)+"_embeddings.npy")
            scan=scans[g_id]
            ## calculate edges
            # if os.path.exists(rel_dir_ti):  ### gonna need to do this for pos and neg edges
            #     pos_edges=np.load(rel_dir_ti)
            # else:
            #     pos_edges,neg_edges= edges_from_thresh(scan,t)
            #     np.save(rel_dir_ti, pos_edges)
            pos_edges,neg_edges= edges_from_thresh(scan,t)
            # np.save(rel_dir_ti, pos_edges)




            emb=i_dict['emb']
            emb_dict=i_dict['emb_dict']
            dist_pw=i_dict['dist_pw']
            scores_pw=i_dict['scores_pw']
            is_train=i_dict['train']
            mean_rank,mAP,mean_degree=embed.mean_average_precision_from_pw(pos_edges,dist_pw)

            # if 

            if len(scores_pw)<90:
                print('Not all included so repeating last')
                try:
                    BCE=last_bce
                    roc=last_bce
                    roc_dist=last_roc_dist
                except:
                    print('yikes guess this was first')

                    print('giving .9 for grins')
                    BCE=.90
                    roc=.90
                    roc_dist=.90
            else:
                pos_scores=np.array([scores_pw[e[0],e[1]] for e in pos_edges])
                neg_scores=np.array([scores_pw[e[0],e[1]] for e in neg_edges])

                pos_dist=np.array([dist_pw[e[0],e[1]] for e in pos_edges])
                neg_dist=np.array([dist_pw[e[0],e[1]] for e in neg_edges])

                preds = list(pos_scores) + list(neg_scores)
                labels = [1] * pos_scores.shape[0] + [0] * neg_scores.shape[0]

                print(torch.tensor(preds),'preds')
                print(torch.tensor(labels),'labels')
                # BCE=F.binary_cross_entropy(torch.from_numpy(np.array(preds)).to(torch.float32), torch.tensor(labels)).item()
                BCE=F.binary_cross_entropy(torch.tensor(preds).double(), torch.tensor(labels).double()).item()
                preds = np.array(list(pos_scores) + list(neg_scores))
                preds_dist = np.array(list(pos_dist) + list(neg_dist))
                roc=roc_auc_score(labels,preds)
                roc_dist=roc_auc_score(labels,-preds_dist)
                last_bce=BCE
                last_roc=roc
                last_roc_dist=roc_dist
# 
            # print(pos_edges)
            # print(emb_dict,'EMB DICT')
            # print("PAUSE \n\n\n NOW FOR THE OLDY")
            # old_MAP=mAP
            # old_rank=mean_rank
            # mean_rank,mAP,mean_degree=embed.mean_average_precision(pos_edges,emb_dict,hyperbolic=True,c=c)
            # print(mean_rank,mAP,'RANK AND MAP')
            # if old_MAP!=mAP:
                # raise Exception("WHAT THE FREAK")
            mean_ranks_t.append(mean_rank)
            mean_degrees_t.append(mean_degree)
            mAPs_t.append(mAP)
            mean_roc_t.append(roc)
            mean_rocdist_t.append(roc_dist)
            mean_BCE_t.append(BCE)
            if not is_train:
                mean_ranks_t_blind.append(mean_rank)
                mean_degrees_t_blind.append(mean_degree)
                mAPs_t_blind.append(mAP)
                mean_roc_t_blind.append(roc)
                mean_rocdist_t_blind.append(roc_dist)
                mean_BCE_t_blind.append(BCE)


            # if len(mAPs)>20:
                # break
        nontrain_indx=nontrain_indx[:len(mAPs)]  ### for debugging
            
            # break



        maps_t_mean = np.mean(mAPs_t)
        mean_ranks_t_mean = np.mean(mean_ranks_t)
        mean_degrees_t_mean=np.mean(mean_degrees_t)
        mean_roc_t_mean = np.mean(mean_roc_t)
        mean_rocdist_t_mean = np.mean(mean_rocdist_t)
        mean_BCE_t_mean=np.mean(mean_BCE_t)


        maps_t_mean_blind = np.mean(mAPs_t_blind)
        mean_ranks_t_mean_blind = np.mean(mean_ranks_t_blind)
        mean_degrees_t_mean_blind=np.mean(mean_degrees_t_blind)
        mean_roc_t_mean_blind = np.mean(mean_roc_t_blind)
        mean_rocdist_t_mean_blind = np.mean(mean_rocdist_t_blind)
        mean_BCE_t_mean_blind=np.mean(mean_BCE_t_blind)

        # maps_t_mean = np.mean(mAPs_t)
        # mean_ranks_t_mean = np.mean(mean_ranks_t)
        # mean_degrees_t_mean=np.mean(mean_degrees_t)
        # mean_roc_t_mean = np.mean(mean_roc_t)
        # mean_BCE_t_mean=np.mean(mean_BCE_t)

        thresholds_all.append(t)
        mAPs.append(maps_t_mean)
        mean_ranks.append(mean_ranks_t_mean)
        mean_degrees.append(mean_degrees_t_mean)
        mean_rocs.append(mean_roc_t_mean_blind)
        mean_rocsdist.append(mean_rocdist_t_mean)
        mean_BCEs.append(mean_BCE_t_mean)

        mAPs_blind.append(maps_t_mean_blind)
        mean_ranks_blind.append(mean_ranks_t_mean_blind)
        mean_degrees_blind.append(mean_degrees_t_mean_blind)
        mean_rocs_blind.append(mean_roc_t_mean_blind)
        mean_rocsdist_blind.append(mean_rocdist_t_mean_blind)
        mean_BCEs_blind.append(mean_BCE_t_mean_blind)


        trained_losses.append(best_dev)
        t_traineds.append(t_trained)
        roots.append(root)
        # break

    score_df = pd.DataFrame(
        data=np.array([thresholds_all,
            mAPs,mean_ranks,mean_degrees,mean_rocs,mean_rocsdist,mean_BCEs,
            mAPs_blind,mean_ranks_blind,mean_degrees_blind,mean_rocs_blind,mean_rocsdist_blind,mean_BCEs_blind,
            trained_losses,t_traineds,roots]).T,columns=columns)

    score_df.to_csv(out_dir)
    return score_df


    # r=model.dc.r.item()
    # t=model.dc.t.item()
    # c=model.c.item()

def get_embedding_scores(scan_df_root,relation_dir,graph_ids,template_df,dims=2,hyperbolic=True):
    ## calculates scores and facts from embeddings, IE mean avg percision
    ##should be no reason for template df, just load like normal
    emb_root = scan_df_root
    scan_df = pd.read_csv(os.path.join(scan_df_root,'scan_info.csv'))
    scan_df= scan_df.sort_values(by='train')
    # print  
    mAPs = []
    mean_ranks=[]
    mean_degrees=[]
    rel_files =[]
    emb_files =[]
    g_ids=[]
    train_indx=scan_df['train']>0
    nontrain_indx=scan_df['train']==0
    print(scan_df['train'])
    print(scan_df[nontrain_indx])
    # nontrain_indx=nontrain_indx.values
    print(nontrain_indx,'NON TRAIN')


    # for g_id in graph_ids.astype(int):   ##### easy... create_file list?? then we can just index in w/ graph ids

    all_embeddings={}
    for i in range(len(scan_df)):
        row=scan_df.iloc[i]
        g_id=row['graph_id']
        emb_file = row['save_location']
        # if row['train']>0:
        #     print('SKIPS')
        #     continue
        # emb_file = os.path.join(emb_root,str(int(g_id))+"_embeddings"+suffix)
        rel_file = os.path.join(relation_dir,str(int(g_id))+"_relations.csv")
        emb_files.append(emb_file)
        rel_files.append(rel_file)
        g_ids.append(g_id)

        rels = np.array(pd.read_csv(rel_file))
        emb = load_embedding(emb_file,template_df)
        emb_dict = to_embedding_dict(emb,dims)
        all_embeddings[i]={}
        all_embeddings[i]['emb']=emb
        all_embeddings[i]['emb_dict']=emb_dict


        mean_rank,mAP,mean_degree=embed.mean_average_precision(rels,emb_dict,hyperbolic=hyperbolic)
        mean_ranks.append(mean_rank)
        mean_degrees.append(mean_degree)
        mAPs.append(mAP)

        print(mAP,'AVERAGE PRECISION')
        print(g_id,'Jeeze')

        # if len(mAPs)>20:
            # break
    nontrain_indx=nontrain_indx[:len(mAPs)]  ### for debugging
    overall_prec_blind = np.mean(np.array(mAPs)[nontrain_indx])
    mean_rank_blind = np.mean(np.array(mean_ranks)[nontrain_indx])
    mean_degree_blind=np.mean(np.array(mean_degrees)[nontrain_indx])
    overall_prec = np.mean(mAPs)
    mean_rank = np.mean(mean_ranks)
    mean_degree=np.mean(mean_degrees)

    mAPs.append(overall_prec_blind)
    mean_ranks.append(mean_rank_blind)
    mean_degrees.append(mean_degree_blind)
    emb_files.append('BlindScans')
    rel_files.append('N/A')

    mAPs.append(overall_prec)
    mean_ranks.append(mean_rank)
    mean_degrees.append(mean_degree)
    emb_files.append('AllScans')
    rel_files.append('N/A')

    score_df = pd.DataFrame(data=np.array([mAPs,mean_ranks,mean_degrees,rel_files,emb_files]).T,columns=['mAPs','Mean Ranks','Mean Degrees','rel_files','emb_files'])

    score_path = os.path.join(emb_root,'embedding_scores.csv')
    score_df.to_csv(score_path)
    




    ### RETURN A WHOLE DATAFRAME!!!
    return score_df

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




from sklearn.decomposition import PCA

from scipy.cluster.hierarchy import inconsistent
#k=2
#fcluster(Z, k, criterion='maxclust')
int_to_marker=['o','^','s','p','H','d','*','8','1','2','3','4']
pad=100
for p in range(pad):
    int_to_marker+=['.']
def ints_to_marker(int_list,max_int=10):
    markers = ["v",'^','<','>','1','2','3','4','8']
    markers = ['o','^','s','p','P','h','8','*','1','2','3','4']
    # pad=100
    # for p in pad:
    #     markers+=['.']
    # print()
    marker_list= [markers[min(i,max_int)] for i in int_list]

    return marker_list
    
class Dendrogram(): ###goddammit this should be one super dendrogram class, oneway takes a linkage matrix the other takes other shit fuck!
    def __init__(self,data,linkage_matrix,seperate_adj=False,adj_mat=None,merge_method='weighted',data_reduction={'name':'pca','dim':2}):
        if seperate_adj:
            self.adj_mat=adj_mat
        else:
            self.adj_mat=data
        self.data=data

        self.linkage_matrix=linkage_matrix
        self.N= data.shape[0] ## double check this
        self.nodes=[]
        self.lines=[]
        self.tree= nx.DiGraph()
        self.frechet_man = Frechet_Poincare()
#         print(data.shape,'DATA SAHPE')
        for i in range(data.shape[0]):
            node = self.create_node(i)

            self.nodes.append(node)
        for l in linkage_matrix:
            self.add_line(l,merge_method)
            
        self.full_data=np.array([n.coords for n in self.nodes])


        # print(self.full_data)
        self.node_df=self.create_df()
        self.plot_coords=self.get_plot_coords_df(self.full_data,data_reduction)
        # for n in self.nodes:
            # print(n.node_id,'node_id',n.coords)
            # print(self.full_data[n.node_id],'coord_df og')
            # print(self.plot_coords[n.node_id],'coord_df')
        # print(self.plot_coords.shape,'DATAREDICTIOM')
        self.edges=self.get_edges()

        self.lca_dict=self.make_lca_dict()
        self.desc_dict=self.make_desc_dict()
        self.trip_subtree_dict=self.make_triple_subtree_dict()
        # print()
#         print(self.data,'DATA')
#         print(self.full_data,'FULL DATA')
#         print(self.full_data.shape,'should be same dim shape but w/ nodes+lines')

    def make_triple_subtree_dict(self):
        leafs = [l.node_id for l in self.nodes if l.is_leaf]
        lca_dict=self.lca_dict
        desc_dict=self.desc_dict
        subtree_dict={}
        for i in range(len(leafs)):
            Li = leafs[i]
            for j in range(i+1,len(leafs)):
                Lj=leafs[j]
                lca_ij=lca_dict[(Li,Lj)]
                for k in range(j+1,len(leafs)):
                    Lk=leafs[k]
                    lca_ik=lca_dict[(Li,Lk)]
                    lca_jk=lca_dict[(Lj,Lk)]
        #             print()
                    if min(lca_ik,lca_jk,lca_ij)<150:
                        pass

                    C_jk_i= lca_jk in desc_dict[lca_ik]
                    C_jk_i2= lca_jk in desc_dict[lca_ij]

                    C_ik_j= lca_ik in desc_dict[lca_ij]
                    C_ik_j2= lca_ik in desc_dict[lca_jk]

                    C_ij_k= lca_ij in desc_dict[lca_ik]
                    C_ij_k2= lca_ij in desc_dict[lca_jk]
                    assert C_jk_i==C_jk_i2
                    assert C_ik_j==C_ik_j2
                    assert C_ij_k==C_ij_k2
                    subtree_dict[(j,k,i)]=0
                    subtree_dict[(k,j,i)]=0
                    subtree_dict[(i,k,j)]=0
                    subtree_dict[(k,i,j)]=0
                    subtree_dict[(i,j,k)]=0
                    subtree_dict[(j,i,k)]=0              
                    if C_jk_i:
                        subtree_dict[(j,k,i)]=1
                        subtree_dict[(k,j,i)]=1
                    if C_ik_j:
                        subtree_dict[(i,k,j)]=1
                        subtree_dict[(k,i,j)]=1
                    if C_ij_k:
                        subtree_dict[(i,j,k)]=1
                        subtree_dict[(j,i,k)]=1   
        return subtree_dict        
    def make_desc_dict(self):
        desc_dict={}
        for n in self.nodes:
            desc_dict[n.node_id]=set([d for d in n.child_nodes])
        return desc_dict
    def make_lca_dict(self):
        lca_dict={}
        T = self.tree
        for node_tup,lca in nx.tree_all_pairs_lowest_common_ancestor(T):
            if node_tup[0]>90 or node_tup[1]>90:
                continue
            if node_tup[0]==node_tup[1]:
                continue
            tup1=node_tup
            tup2=(node_tup[1],node_tup[0])                                     
            lca_dict[tup1]=lca
            lca_dict[tup2]=lca
        return lca_dict       
    def score(self,score_type='tree_samp'):
        if score_type=='tree_samp':
            score = tree_sampling_divergence(self.adj_mat, self.linkage_matrix)
        elif score_type=='dasguspta_score':
            score = dasgupta_score(self.adj_mat, self.linkage_matrix)
            # score = score/
        elif score_type=='dasguspta_cost':
            score = dasgupta_cost(self.adj_mat, self.linkage_matrix)
            score=score/(self.adj_mat.mean()*100)
        else:
            raise Exception('Invalid choice: {}, \
                            options are {}'.format((score_type,
                                                    ('tree_samp','dasguspta_score','dasguspta_score'))))
        return score
    
    def create_node(self,node_id,add_to_tree=True):
        coords = self.data[node_id]
#         print(node_id,'NODE')
#         print(coords,'NODAL COORd')
        child_nodes=[]
        # child_nodes=[node_id] ## oh come on this is fucked
        node=Node(node_id,direct_children=[],
                  child_nodes=child_nodes,child_leafs=[coords],layer=0,coords=coords) ### may be best 
                                                ##to leave self out of child_nodes
        # print(node,'node?')
        if add_to_tree:
            self.tree.add_node(node.node_id)

        return node
    
    def add_line(self,line,merge_method='weighted',add_to_tree=True):
#         print(line,'LINE')
        assert (int(line[0])*10)==((line[0])*10)
        node1=self.nodes[int(line[0])]
        node2=self.nodes[int(line[1])]

        # print(node1,node2,'NODES')

        distance=line[2]
        num_leaf_nodes=line[3]
        self.lines.append(line)
        new_node=self.merge_nodes(node1,node2,coord_merge_method=merge_method)
        # if add_to_tree:
        #     # self.tree.add_edge(node1.node_id,node2.node_id)
        #     self.tree.add_edge(new_node.node_id,node2.node_id)
        #     self.tree.add_edge(new_node.node_id,node1.node_id)
        assert num_leaf_nodes==new_node.weight
#         print(new_node.node_id)
#         print(len(self.nodes))
        assert len(self.nodes)==(new_node.node_id)
        assert new_node.weight<=self.N
        self.nodes.append(new_node)
    
    def merge_nodes(self,node1,node2,coord_merge_method='weighted',lca_coords=None,node_id=-1,add_to_tree=True):
        ### weighted takes average of coords of the node and weights by # of nodes that make that node up
        ## I believe valid for euclidean
        ## lca_coords gives new coord if merge_method=hyp_lca
        new_layer= max(node1.layer,node2.layer)+1
#         new_id = len(self.lines)+self.N-1
        new_id = node_id if node_id>0 else len(self.lines)+self.N-1
        child_nodes = node1.child_nodes+node2.child_nodes+[node1.node_id,node2.node_id]
        # print(node1.node_id,node2.node_id,'who are the parents?')
        # print(node1.child_nodes,node2.child_nodes,'parents first')
        # print(child_nodes,'okay children')
        child_leafs = node1.child_leafs+node2.child_leafs
        if coord_merge_method=='weighted':
            # print('weighted')
            to_merge=np.array([node1.coords*node1.weight,node2.coords*node2.weight]).T
            coords = np.sum(to_merge,axis=1)/(node1.weight+node2.weight)      
            to_merge_full=np.array([l for l in child_leafs]).T
            coords_full = np.mean(to_merge_full,axis=1)

        elif coord_merge_method=='hyp_lca':
            # print('hyp_lca ............')
            # lca_coords==None
#                 print(lca_coords==None,'NO LCA COORDS')
            # print('')
            # print(node1.coords,node2.coords,'NODE COORDS')
            lca_coords,_=hyp_lca(torch.from_numpy(node1.coords)
                                          ,torch.from_numpy(node2.coords),return_numpy=True)

            # print(lca_coords,'NEED THESE TO MATCH')
            # try:
#                 lca_coords==None
# #                 print(lca_coords==None,'NO LCA COORDS')
#                 # print('')
#                 lca_coords,_=hyp_lca(torch.from_numpy(node1.coords)
#                                               ,torch.from_numpy(node2.coords),return_numpy=True)
#                 # print(lca_coords,'get through?')
#             except:

#                 # pass
# #                 print('truth value of blahb blah blah')
#                 print('lca coords already exist')
#                 pass

            coords=lca_coords

        elif coord_merge_method=='hyp_lca_centroid':
            # print(child_leafs)
#             leaf_nodes=node1.child_lefas
            leafs= torch.Tensor(child_leafs)
            coords=frechet_mean(leafs, self.frechet_man,return_numpy=True)
        else:
            print('huh coords')
            to_merge=np.array([l.coords for l in child_leafs]).T
            coords = np.mean(to_merge,axis=1)    

        direct_children=[node1,node2]       
        new_node=Node(new_id,direct_children,child_nodes,child_leafs,new_layer,coords)
        if add_to_tree:
            self.tree.add_node(new_node.node_id)
            self.tree.add_edge(new_node.node_id,node1.node_id)
            self.tree.add_edge(new_node.node_id,node2.node_id)
        # print(new_node.coords,'ARE WE DONE YUET?',new_node.node_id)
        # print([d.coords for d in new_node.direct_children],'direct children')
        node1.parent_node=new_node
        node2.parent_node=new_node
        node1.has_parent=True
        node2.has_parent=True
        return new_node
#     def get_clusters(self,metric='maxclust',t=5,depth=4):
#     def get_clusters(self,metric='maxclust',t=5,depth=4):
    def get_clusters(self,metric='distance',t=1.5,depth=4):
        if metric!='inconsistency' and depth!=4:
            print('DEPTH UNNECESSARY IF NOT USING INCONSITENCY')
        clusters=fcluster(self.linkage_matrix,t,criterion=metric)
        
        # print(len(clusters),'how many clusters')
        #### would be sick to do internal nodes as well
        return clusters
#         if metric==k:clus
#             if max_d>0:
#                 print('IGNORING MAX D')
#         elif max_d>0:
            
#         elif:
#             raise Exception('must have either k or d')
    def create_df(self):
        columns=['node_id','is_leaf','layer']
        data= [[n.node_id,n.is_leaf,n.layer] for n in self.nodes]
        data_df = pd.DataFrame(columns=columns,data=data)
        # print(data_df)
        return data_df
        
        
        
    def get_plot_coords_df(self,coord_df,data_reduction={'name':'pca','dim':2}):
#         if self.has_reduction:
#             pca
        
        if data_reduction['name']=='pca':
            pca = PCA(n_components=data_reduction['dim'])
            data_plot = pca.fit_transform(coord_df)
        else:
            data_plot=coord_df
        # print(data_plot,"DATA PLOT")


        return data_plot
    
    def get_edges(self):
        data_to_plot = self.plot_coords
        edges=[]
        # for n in self.nodes:
            # if not n.has_parent:
                # continue
            # edges.append([data_to_plot[n.node_id],data_to_plot[n.parent_node.node_id]])

        for n in self.nodes:
            if len(n.direct_children)<2:
                continue
            n1=n.direct_children[0]
            n2=n.direct_children[1]
            edges.append([data_to_plot[n1.node_id],data_to_plot[n2.node_id]])


        edges=np.array(edges)
        return edges
#         print(edges.shape)

    def plot_cluster_progression(self,hyp_edges=True):
        # fig, ax = plt.subplots()
        data_to_plot = self.plot_coords
        leafs=data_to_plot[self.node_df['is_leaf']==True]
        print(len(leafs),'how many leafs?')
        # plt.scatter(leafs[:,0],leafs[:,1],c='gray',alpha=.1)

        for n_id in range(len(self.nodes)):

            n=self.nodes[n_id]


            if n.is_leaf:
                continue

            print(n.node_id,n.coords,self.plot_coords[n.node_id],'NEW NODE')

            fig, ax = plt.subplots()
            ax.scatter(leafs[:,0],leafs[:,1],c='gray',alpha=.5)
            # internal_so_far = data_to_plot[[ch.node_index for ch in self.]]
            # print(n.child_leafs,'CHILD LEAFS')
            # print(n.child_nodes,'node children')
            # print([ch.node_index for ch in n.child_leafs])
            # print([ch.node_index for ch in n.child_leafs if ch not in n.direct_children])
            child_node_ids=[ch for ch in n.child_nodes if ((self.nodes[ch] not in n.direct_children) and (self.nodes[ch].is_leaf))]
            child_coords = data_to_plot[child_node_ids]
            direct_child_coords= data_to_plot[[ch.node_id for ch in n.direct_children ]] ## highest priority
            internal_child_node_ids=[ch for ch in n.child_nodes if ((self.nodes[ch] not in n.direct_children) and (not self.nodes[ch].is_leaf))]
            internal_child_coords = data_to_plot[internal_child_node_ids] ## all these may overlap, so just go exclusive to non exclusive or do ifs and elses
            #### should be split by color for the two direct_child_coords
            e = direct_child_coords
            viz.plot_geodesic(e[0],e[1],ax=ax)

            for ic_id in n.child_nodes:
                ic=self.nodes[ic_id]
                if len(ic.direct_children)<2:
                    continue
                # if ic in n.direct_children
                # if self.nodes[]
                dc=ic.direct_children
                viz.plot_geodesic(dc[0],dc[1],ax=ax,ls='--',alpha=.5,linewidth=.8)

            ax.scatter(direct_child_coords[:,0],direct_child_coords[:,1],)
            if len(internal_child_coords>0):
                ax.scatter(internal_child_coords[:,0],internal_child_coords[:,1],alpha=.4,marker='.',color='blue')
            if len(child_coords)>0:
                ax.scatter(child_coords[:,0],child_coords[:,1],alpha=.4,color='blue')

            # print(n.coords,'n parent')
            # print(direct_child_coords,'direct_children')
            # circ = plt.Circle((0, 0), radius=1, edgecolor='black', facecolor='None', linewidth=3, alpha=0.5)
            # ax.add_patch(circ)
           
            
            ax.set_aspect('equal')
            ax.scatter(n.coords[0],n.coords[1],marker='^',color='green')
            plt.show()

            # if n_id>94:
                # break




    def plot_nodes(self,color_metric='cluster',
                   cluster_metric='maxclust',cluster_t=5,max_layer=2,hyp_edges=False):
        pad_color='orange'
        pad=50
        colors=['b', 'r', 'g', 'y', 'm', 'c', 'k', 'silver', 'lime', 'skyblue', 'maroon', 'darkorange']+[pad_color]*pad

        # hyp_edges=False
#         colors = cm.rainbow(np.linspace(0, 1, 14))
        ## take p argument, that
        data_to_plot = self.plot_coords

        # print(Dat)
        leafs=data_to_plot[self.node_df['is_leaf']==True]
#         internal_df = 
        internals = data_to_plot[self.node_df['is_leaf']==False]
        internal_layers = self.node_df[self.node_df['is_leaf']==False]['layer']
#         print(data_to_plot,'PLOTTING')
#         int_layer_markers=ints_to_marker(internal_layers,max_plot=6)
        if data_to_plot.shape[1]>2:
            print("TOO MANY DIMS, USING FIRST 2")
        if color_metric=='cluster':
#             p
            clusters=self.get_clusters(cluster_metric,cluster_t)
            # print(np.max(clusters),'NUM CLUSTERS')
            colors=[colors[c] for c in clusters]
            
        else:
            raise Exception('COLORS')
        ax = plt.gca()
        plt.scatter(leafs[:,0],leafs[:,1],c=colors,alpha=.5)

        print(internal_layers.unique(),'how manx??')
        
        for l in internal_layers.unique():
            if l>max_layer:
                continue
            to_plot = data_to_plot[self.node_df['layer']==l]
            mark=int_to_marker[l]
            plt.scatter(to_plot[:,0],to_plot[:,1],marker=mark,c='blue')
            
        if hyp_edges:
            for e in self.edges:
                # print(e[:,0],'edge one?')
                # viz.plot_geodesic(e[:,0],e[:,1],ax=ax)
                viz.plot_geodesic(e[0],e[1],ax=ax)
                
        else:
            for e in self.edges:
                plt.plot(e[:,0],e[:,1],c='blue',linestyle='dashed',linewidth=1)
        fig = plt.gcf()
        circ = plt.Circle((0, 0), radius=1, edgecolor='black', facecolor='None', linewidth=3, alpha=0.5)
        ax.add_patch(circ)
       
        
        ax.set_aspect('equal')
        ax.set_xlim(xmin=-1.1,xmax=1.1)
        ax.set_xbound(lower=-1.1, upper=1.1)
        ax.set_ylim(ymin=-1.1,ymax=1.1)
        ax.set_ybound(lower=-1.1, upper=1.1)
   
        plt.show()

    # def
        
class Node():
    def __init__(self,node_id,direct_children,child_nodes,child_leafs,layer,coords):
        self.dims=coords.shape[0]
        origin = np.zeros((self.dims))
        self.node_id=node_id
        self.direct_children=direct_children
        self.child_nodes=child_nodes
        self.child_leafs=child_leafs
        self.layer=layer  ### maybe add layers for p situations....
        self.coords=coords
        self.hyp_radius= poincare_dist(coords,origin)
        self.weight=len(child_leafs)
        self.is_leaf=True if self.layer==0 else False
        self.has_parent=False
        self.parent_node=None
#         print(self.direct_children,'direct')
        # print(node_id,[c.node_id for c in self.direct_children])
        
        
    ### node needs id, merge function, coords, children
        

class HypClustering():
    def __init__(self, embeddings,metric='deepestLCA',method='deepestLCA'):
        """
        method- what will we be taking LCA of/ what will represent merged nodes
            -cumLCA: merged nodes will be represented by their LCA
            -centroid: merged nodes will be represented by the hyperbolic centroid of all of the leaf nodes
            -averageIndy: merged nodes will be represented by all it's nodes individually. 
                        The distance between clusters will be the average LCA height of all nodes
                        This is equivalent to agglomerativeClustering where similarity is LCA
            -singleIndy: will implement alg from Chami, order nodes by deepest LCA, 
                        cluster the trees they belong to
        
        metric- 'deepestLCA' or Hyperbolic distance.. but right now just using deepestLCA
        """
#         print(embedding,'EMBEDDING')
        self.hyp_lca,self.hyp_lca_rad=hyp_lca_all(embeddings)
#         self.node_rads = 
        self.embeddings=embeddings
        self.coords=self.embeddings ## maybe change so embeddings has more info?
        self.metric=metric
        self.merge_method=method
        self.merge_heap=[]
        self.N=embeddings.shape[0]
        self.tree= nx.DiGraph()
        self.nodes=[self.create_node(n) for n in range(self.N)]
        # print('radiii')
        # for n in self.nodes:
            # print(n.node_id)
            # print(np.linalg.norm(n.coords),'eucl rad')
            # print(n.hyp_radius,'HYP RADIUS')
        self.forget_nodes=set([]) ### for some algs, we want to forget nodes, ie forget them if they're popped
        self.lines=[]
        self.frechet_man = Frechet_Poincare()
        self.origin=np.zeros((self.embeddings.shape[1]))
        for i in range(self.N):    
            for j in range(i+1,self.N):
                lca_rad= -self.hyp_lca_rad[i,j] ## give us deepest lca
                heappush(self.merge_heap,(lca_rad,i,j))
        self.lca_dict,self.lca_rad_dict=self.create_lca_dicts()
        self.clustered=False
        self.link_mat=None
        self.dendro_obj=None
#         self.merge_method=met
        
    def get_dendro(self,ground_adj):
        if not self.clustered:
            raise Exception('CANT MAKE DENDRO IF HAVENT CLUSTERED')
        dendi=Dendrogram(self.coords,self.link_mat,
                         seperate_adj=True,adj_mat=ground_adj,merge_method=self.merge_method,
                        data_reduction={'name':'hyperbolic'})
        self.dendro_obj=dendi
        return dendi
    def plot_dendrogram(self,p=5,truncate_mode='level'):
        dendrogram(self.link_mat, truncate_mode=truncate_mode, p=p)
        plt.show()
    def plot_nodes(self,cluster_t=5,max_layer=2,hyp_edges=True):
        if not self.dendro_obj:
            raise Exception('Bro you should fix all this')
        print(max_layer,'maximum layer')
        self.dendro_obj.plot_nodes(cluster_t=cluster_t,max_layer=max_layer,hyp_edges=hyp_edges)
        plt.show()
    def plot_cluster_progression(self):
        self.dendro_obj.plot_cluster_progression(hyp_edges=True)
    def run_cluster(self):
        self.lines=[]
        min_dist=False
        first=True
        while len(self.merge_heap)>0:
#             print(len(self.merge_heap),'MERGE LEN')
#             print(len(self.forget_nodes),'Already forgotten')
            deepest_lca,n1,n2 = heappop(self.merge_heap)
            if n1 in self.forget_nodes:
#                 print(n1,'FORGET ABOUT IT')
                continue
            if n2 in self.forget_nodes:
#                 print(n2,'FORGET ABOUT IT')
                continue
            lca_coords=self.lca_dict[n1][n2]        
            # print('one time')
            merged=self.merge_nodes(self.nodes[n1],self.nodes[n2],
                                    coord_merge_method=self.merge_method,
                                    lca_coords=lca_coords,node_id=len(self.nodes))
            distance=deepest_lca
            # print(n1,n2,'lca_rad',deepest_lca,'lca_coords',lca_coords,)
            
            
            min_rad=min(self.nodes[n1].hyp_radius,self.nodes[n2].hyp_radius)
            if abs(distance)>(min_rad+.0001) and self.merge_method=='hyp_lca':
                print('what the hell')
                print(abs(distance),'lcarad',self.nodes[n1].hyp_radius,self.nodes[n2].hyp_radius,'rad 1 and 2')
                raise Exception('what the hells going on here')
            new_line=[n1,n2,distance,merged.weight]
            self.lines.append(new_line)
            self.nodes.append(merged)
            self.forget_nodes.add(n1)
            self.forget_nodes.add(n2)
            cluster_as_lca=self.merge_method=='hyp_lca_centroid'
            # print('before updat')
            self.update_lcas(merged,cluster_as_lca)## adds new lcas to lca_dicts, and those to merge_heaps

        # print('out?')
        self.clustered=True
        self.link_mat=np.array(self.lines)
        # print(self.link_mat[:,2])
        # print(self.link_mat[:,2].min())
        self.link_mat[:,2]=self.link_mat[:,2]-self.link_mat[:,2].min()
        return self.link_mat
    def update_lcas(self,node,cluster_as_lca=False):
        node_coords=node.coords
        self.lca_dict[node.node_id]={}
        self.lca_rad_dict[node.node_id]={}
        lcas_to_add=[] ## must do this so we don't modify looping list
#         print(len(self.merge_heap),'HEAP LENGTH BEFORE')
#         for n3 in self.merge_heap:
        for n3 in self.nodes:
            if n3 in self.forget_nodes:
#                 print('skipping node n3') ## not completely necissary but speeds things up
                continue 
            n3_coords= n3.coords
            if n3.node_id==node.node_id:
#                 print('same guy?')
                continue
            if cluster_as_lca:
#                 new_lca,new_lca_rad = self.two_node_frechet_mean(node,n3)
                new_lca,new_lca_rad = self.two_node_frechet_mean_grad(node,n3)
#                 print(new_lca_rad,'OG RAD')
                new_lca_rad=-new_lca_rad
#                 print(new_lca_rad,'neg RAD')
                
            else:
                new_lca,new_lca_rad = hyp_lca(torch.from_numpy(node.coords)
                                              ,torch.from_numpy(n3_coords),return_numpy=True)
                new_lca_rad=-new_lca_rad[0]
#             new_lca=new_lca[0]
            self.lca_dict[node.node_id][n3.node_id]=new_lca
            self.lca_dict[n3.node_id][node.node_id]=new_lca
            self.lca_rad_dict[node.node_id][n3.node_id]=new_lca_rad
            self.lca_rad_dict[n3.node_id][node.node_id]=new_lca_rad
            
            heappush(self.merge_heap,(new_lca_rad,node.node_id,n3.node_id))
#             print(len(self.merge_heap),'ADDED ONE W/',node.node_id)
#         print(len(self.merge_heap),'HEAP LENGTH AFTER')

    def merge_nodes(self,node1,node2,coord_merge_method='weighted',lca_coords=None,node_id=-1,add_to_tree=True):
        ### weighted takes average of coords of the node and weights by # of nodes that make that node up
        ## I believe valid for euclidean
        ## lca_coords gives new coord if merge_method=hyp_lca
        
        new_layer= max(node1.layer,node2.layer)+1
#         new_id = len(self.lines)+self.N-1
        new_id = node_id if node_id>0 else len(self.lines)+self.N-1
        child_nodes = node1.child_nodes+node2.child_nodes
        child_leafs = node1.child_leafs+node2.child_leafs
        # print(coord_merge_method,'MERGE METHOD')
        if coord_merge_method=='weighted':
            to_merge=np.array([node1.coords*node1.weight,node2.coords*node2.weight]).T
            coords = np.sum(to_merge,axis=1)/(node1.weight+node2.weight)      
            to_merge_full=np.array([l for l in child_leafs]).T
            coords_full = np.mean(to_merge_full,axis=1)
        elif coord_merge_method=='hyp_lca':
            try:
                lca_coords==None
#                 print(lca_coords==None,'NO LCA COORDS')
                lca_coords=hyp_lca(node1.coords,node2.coords)
            except:
                pass
#                 print('truth value of blahb blah blah')
#                 print('lca coords already exist').
            coords=lca_coords
        elif coord_merge_method=='hyp_lca_centroid':
            ### do we need an initial theta?
            ### could we actually use the hyp_lca as initial theta?
#             theta_k = poincare_pt_to_hyperboloid(self.centroids[i])
#             fmean_k = compute_mean(theta_k, H_k, alpha=0.1)
#             new_centroids[i] = hyperboloid_pt_to_poincare(fmean_k)
            ### should this be the centroid of all nodes? or just the leaf nodes?
                      ### probably just the leaf nodes.
#             print(child_leafs)
#             leaf_nodes=node1.child_lefas
            coords,_ = self.two_node_frechet_mean_grad(node1,node2)
#             leafs= torch.Tensor(child_leafs)

#             coords_calc=frechet_mean(leafs, self.frechet_man,return_numpy=True)
#             print(coords,'grad')
#             print(coords_calc,'calculated')
        else:
            to_merge=np.array([l.coords for l in child_leafs]).T
            coords = np.mean(to_merge,axis=1)          
        direct_children=[node1,node2]       
        new_node=Node(new_id,direct_children,child_nodes,child_leafs,new_layer,coords)
        if add_to_tree:
            self.tree.add_node(new_node.node_id)
            self.tree.add_edge(new_node.node_id,node1.node_id)
            self.tree.add_edge(new_node.node_id,node2.node_id)
        node1.parent_node=new_node
        node2.parent_node=new_node
        node1.has_parent=True
        node2.has_parent=True


        
#         print(node1.coords,'node1')
#         print(node2.coords,'node2')
#         print(new_node.coords,'new_node')
#         print([d.coords for d in new_node.direct_children],'COORDS')
        return new_node
    def create_node(self,node_id,add_to_tree=True):
        coords = self.coords[node_id]
#         print(node_id,'NODE')
#         print(coords,'NODAL COORd')
        node=Node(node_id,direct_children=[],
                  child_nodes=[node_id],child_leafs=[coords],layer=0,coords=coords) ### may be best 
                                                ##to leave self out of child_nodes
        if add_to_tree:
            self.tree.add_node(node.node_id)

        return node
    
    def create_lca_dicts(self):
        lca_dict={}
        lca_rad_dict={}
        for k in range(self.N):
            lca_dict[k]={}
            lca_rad_dict[k]={}
            for t in range(self.N):
                if k==t: ## could be an infinity situation to guarantee value nmw
                    continue
                b = max(k,t)
                a= min(k,t)
#                 print(a,b,'AB')
#                 print(self.hyp_lca.shape,'FULL SHAPE')
                
                lca_dict[k][t]=self.hyp_lca[a,b]
                lca_rad_dict[k][t]=-self.hyp_lca_rad[a,b]
        return lca_dict,lca_rad_dict
    
    def get_score(self,score_type='dasgupta_score'):
        return self.dendro_obj.score(score_type)
    
    def two_node_frechet_mean_grad(self,node1,node2):
        theta,_=hyp_lca(torch.from_numpy(node1.coords)
                                              ,torch.from_numpy(node2.coords),return_numpy=True)
        child_leafs=node1.child_leafs+node2.child_leafs
#         leafs= torch.Tensor(child_leafs)
        leafs=np.array(child_leafs)
        coords=compute_mean(theta=theta,X=leafs)
        rad = poincare_dist(coords,self.origin)
        return coords,rad
    
    def two_node_frechet_mean(self,node1,node2):
#         print(self.origin.shape,'origin shape ')
        child_leafs=node1.child_leafs+node2.child_leafs
        leafs= torch.Tensor(child_leafs)
        coords=frechet_mean(leafs, self.frechet_man,return_numpy=True)
        rad = poincare_dist(coords,self.origin)
        if rad<0:
            raise Exception('RADIUS SHOULDNt BE <0:')
        return coords,rad
   

# def 
def dasgupta_wang(W,T_dict):
    ## W is the similarity matrix (ie. plv mat or whatever)
    ## T_dict takes any triple of leaf nodes (i,j,k), and tells you if i,j are deepest of ij,jk,ik should be subtracted
                ### T_dict should have permutations (i,j,k)=(j,i,k)
    second_term_up=0
    second_term_low=0
    second_term_true=0
    first_term=0
    n_first=0
    n_second=0
#     n_triples
    for i in range(len(W)):
        # for j in range(i+1,len(adj)):
        for j in range(i+1,len(W)):
            if i==j:
                continue
            w_ij=W[i,j]
            first_term+=w_ij
            n_first+=1
            # for k in range(len(adj)):
            for k in range(j+1,len(W)):
                if k in (i,j):
                    continue

                tij_k=T_dict[(i,j,k)]
                tik_j=T_dict[(i,k,j)]
                tjk_i=T_dict[(k,j,i)]
                assert(tij_k+tik_j+tjk_i)==1

                w_ik=W[i,k]
                w_jk=W[j,k]

                w_nojk = w_ij+w_ik
                w_noik = w_ij+w_jk
                w_noij = w_ik+w_jk

                # true= -1*(tjk_i*w_jk+tij_k*w_ij+tik_j*w_ik)
                true= max(tjk_i*w_nojk,
                    tik_j*w_noik,tij_k*w_noij) #### Wang's Dasgupta-- cancel out the weight of the deepest pair
                                ### if that pair is the biggest weight, true=second_term_low=min(2 options)
#                 print(t1,t2,t3,'t1,t2,t3')
                second_term_low+=min(w_nojk,w_noik,w_noij)
                second_term_up+=max(w_nojk,w_noik,w_noij)
                second_term_true+=true
                n_second+=1
    n_triples=n_second
    # n_second=1
    # n_first=1
    lb = (second_term_low)/n_second+(2*first_term)/n_first
    ub = (second_term_up)/n_second+(2*first_term)/n_first
    cost = (second_term_true)/n_second+(2*first_term)/n_first

    lb = (second_term_low)+(2*first_term)
    ub = (second_term_up)+(2*first_term)
    cost= (second_term_true)+(2*first_term)
    # print(n_triples,'how many triples?')
    ratio=ub/lb
    lb/=n_first
    ub/=n_first
    cost/=n_first
#     lb/=n_triples
#     ub/=n_triples
    # print(2*first_term,'FIRST TERM')
    return cost,ub,lb, ratio

def upper_lower_bounds(adj):
    second_term_up=0
    second_term_low=0
    first_term=0
    n_first=0
    n_second=0
#     n_triples
    for i in range(len(adj)):
        # for j in range(i+1,len(adj)):
        for j in range(i+1,len(adj)):
            if i==j:
                continue
            w_ij=adj[i,j]
            first_term+=w_ij
            n_first+=1
            # for k in range(len(adj)):
            for k in range(j+1,len(adj)):
                if k in (i,j):
                    continue
                w_ik=adj[i,k]
                w_jk=adj[j,k]
                t1 = w_ij+w_ik
                t2=w_ij+w_jk
                t3 = w_ik+w_jk
#                 print(t1,t2,t3,'t1,t2,t3')
                second_term_low+=min(t1,t2,t3)
                second_term_up+=max(t1,t2,t3)
                n_second+=1
    n_triples=n_second
    n_second=1
    n_first=1
    lb = (second_term_low)/n_second+(2*first_term)/n_first
    ub = (second_term_up)/n_second+(2*first_term)/n_first
    lb = (second_term_low)+(2*first_term)
    ub = (second_term_up)+(2*first_term)
    ratio=ub/lb
#     lb/=n_triples
#     ub/=n_triples
#     lb/=n_triples
#     ub/=n_triples
    return ub,lb, ratio
                



def plot_clusters(node_coords,hkmeans,p_show=1):
    colors=['b', 'r', 'g', 'y', 'm', 'c', 'k', 'silver', 'lime', 'skyblue', 'maroon', 'darkorange']
    if len(hkmeans.centroids)>len(colors):
        colors = [i for i in plt.cm.get_cmap('tab20').colors]
        
    cluster_labels = np.array([colors[np.argmax(l)] for l in hkmeans.predict(node_coords)])
    to_show= np.random.choice(node_coords.shape[0], size=int(node_coords.shape[0]*p_show), replace=False)
    centroids=hkmeans.centroids
    centroid_ids = np.array([colors[int(l)] for l in np.arange(centroids.shape[0])] )
    
    
#     plt.figure(figsize=(width,height))
    plt.xlim([-1.0,1.0])
    plt.ylim([-1.0,1.0])
    ax = plt.gca()
    circ = plt.Circle((0, 0), radius=1, edgecolor='black', facecolor='None', linewidth=3, alpha=0.5)
    ax.add_patch(circ)
    fig = plt.gcf()
    fig.set_size_inches(8, 8,forward=True)
    ax.scatter(node_coords[:,0][to_show],node_coords[:,1][to_show],color = cluster_labels[to_show])
    
    plt.scatter(centroids[:, 0], centroids[:, 1], s=750, color = centroid_ids, linewidth=2, marker='*');
    plt.show()
#     ax.scatter(node_coords[:,1])
def hyp_cluster_full(node_coords,k=20,add_origin_centroid=False,plot_cents=True):
    """
    node_coords should be nscans x nodes x dims
    """
    n_graphs,n_nodes,n_dims=node_coords.shape
    flat_coords =node_coords.reshape(-1,node_coords.shape[2])
    
    print(node_coords.shape,'OG SHAP')
    print(flat_coords.shape,'FLATTENED SHAPE')
    hkmeans = HyperbolicKMeans(n_clusters=k,dims=n_dims)
    hkmeans.fit(flat_coords, max_epochs=10)  
#     print(n)
    
    k_out = k+1 if add_origin_centroid else k
    node_centroid_dists=np.zeros((n_graphs,n_nodes,k_out))
    graph_centroid_dists=np.zeros((n_graphs,k_out))
    if plot_cents:
        plot_clusters(flat_coords,hkmeans,p_show=.2)
    
    for g in range(n_graphs):
        g_nodes = node_coords[g]
#         print(g_nodes.shape,'node shape')
        node_dist = hkmeans.transform(g_nodes,use_origin=add_origin_centroid) ##node x k  ### could be a good spot to insert origin?
#         print(node_dist.shape,'node dist shape')
        graph_dist = np.mean(node_dist,axis=0) ## k,
#         print(graph_dist.shape,'graph shape')
        node_centroid_dists[g]=node_dist
        graph_centroid_dists[g]= graph_dist 
        
        
        
    ### add some plotting
    
    return graph_centroid_dists,node_centroid_dists,hkmeans
        
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
                               template_df,
                               bin_rank_threshold=10,use_binary=False,
                                suffix='.csv',dims=3,balanced_atl=True
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
# pw METRIC
    useable_mets=set(['wc','rc','bc','pw','node_rad','wc_prob','rc_prob','bc_prob','pw_prob','node_rad_prob']) #### these are transductivable, we can average
    first=True
    
    all_stat_dicts=[]
    all_stat_dfs=[]
#     print(len(emb_roots),'count!')
    # print(emb_roots)
    # sss
    running_c=0
    for root in emb_roots:
        if ('.pt' in root) or ('.csv' in root) or ('relations' in root):
            continue
        print(root,'ERRRR')
#         root = er.split(' ')
        emb_root=os.path.join(root,'embeddings')
        # print(emb_root,'embedding root')

        model_path = os.path.join(root,"{}.pt".format('model'))  ### args should be saved with the model
        # printmodel_path,'model path'
        model = th.load(model_path,map_location=torch.device('cpu'))
        scan_path =  os.path.join(root,'scan_info.csv')
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
        # return 'a','s'
        # continue
        print(r,t,c)    
        print('r','t','c')
        print(model.args.band,'BANDS')
        print(model.args.use_identity,'ID please?')
        print(model.args.use_plv,'PLV')
        print(model.args.use_norm,'graph norm?')
        # why not just psass in the whole clinical_df?
        stat_dict_new,full_stat_df_new=embedding_analysis(emb_root,clinical_df,label,net_threshold=net_threshold,
                               template_df=template_df,
                               bin_rank_threshold=bin_rank_threshold,
                           use_binary=use_binary,suffix=suffix,dims=dims,balanced_atl=balanced_atl,
                                        t=t,c=c,r=r)
        
        all_stat_dicts.append(stat_dict_new)
        all_stat_dfs.append(full_stat_df_new)
        # break
    print(running_c/len(emb_roots),'AVERAGE C')
    print('out stat dict')
    if not average:
        return all_stat_dicts,_,all_stat_dfs
    
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
            # print(metric_dict,'metric test')
            # print(metric,'METRIC')
            # if metric=='node_prob':
                # print(metric_dict['stat_df'],'METRIC DICT')
#                 continue
            met_stat_df=metric_dict['stat_df']
#             print(metric,'METRIC')
            if metric not in useable_mets:
#                 print(metric,'SKIP')
                continue
            full_dict[metric]['stat_df']
            full_dict[metric]['stat_df']+=met_stat_df
#             print(i,'I')
#             print(met_stat_df)
            if last:
#                 print(len(all_stat_dicts),'LENGHT')
#                 print(full_dict[metric]['stat_df'])
                full_dict[metric]['stat_df']
                full_dict[metric]['stat_df']/=len(all_stat_dicts)
#                 print("AFTEEEERRRR")
#                 print(full_dict[metric]['stat_df'])
                
#                 ekjge
    print(all_stat_dfs[0].shape,'og shape')
    for df in all_stat_dfs:
        print(df.shape,'DF SHAPE')
    stat_df_cols=all_stat_dfs[0].columns
    # turning all dfs into a stacked numpy array so we can take the mean
    df_concat = np.stack([df for df in all_stat_dfs],axis=2)
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
    

    return all_stat_dicts,full_dict,all_stat_dfs,full_stat_df


def study_get_scores(root,thresholds,val_only=False,force_args={}):
    roots = [os.path.join(root,r) for r in os.listdir(root) if ('.pt' not in r) and ('.csv' not in r)]
    print(roots,'ROOTS')
    output_str='_'.join(root.split('\\')[-3:])
#     print(roots,'ROOTs')
    last_str='val_full_score.csv' if val_only else 'full_score.csv'
    outdir=os.path.join(root,output_str+last_str)
    score_df = multiple_get_scores(roots,outdir,thresholds,val_only,force_args=force_args)
    return score_df

def multiple_get_scores(roots,outdir,thresholds,val_only=False,force_args={}):
    first=True
    for i in range(len(roots)):
        r=roots[i]

        score_r=get_embedding_scores_thresholds(root=r,thresholds=thresholds,val_only=val_only,args=force_args)
        # try:
            # score_r=get_embedding_scores_thresholds(root=r,thresholds=thresholds)
        # except:
            # continue
        score_r['ID']=i
        if first:
            score_df=score_r
            first=False
        else:
            score_df=pd.concat([score_df,score_r])
            
    score_df.to_csv(outdir)
    return score_df

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
    
def run_combos_list(model_type,feature_list,names=[],times=2,
    use_agg=False,agg_fit=None,agg_pred=None,
    hyp_search=True,passed_params=None,y_comb=[],
    use_nested=True):
# nested_params={'use_nested':True,'inner_loo':True,'inner_splits':5}):
    ##linear only matters for svc
    def execute(X,y):
        if use_agg:
            assert agg_fit is not None
            assert agg_pred is not None
            if agg_fit=='split_sample' and agg_pred=='same':
                ## BRO WHAT???? THIS IS CHEATING ISN"T IT????
                temp_model=AggregateClassifier(model_type,fit_method=agg_fit,predict_method=agg_pred)
                X,y=temp_model.reform_inputs(X,y) ### only use temp model for this reform
                
                
            print(y.shape,'y?')
            print(X.shape,'X')
            print(passed_params,'PASS')
            # X_norm = MinMaxScaler().fit_transform(X)
            # res=make_classifier_agg(X,y,base_model=model_type,
            #             fit_method=agg_fit,predict_method=agg_pred,
            #                         hyp_search=hyp_search,passed_params=passed_params)
            if use_nested:
                res=make_classifier_nested(X,y,base_model=model_type,
                        fit_method=agg_fit,predict_method=agg_pred,
                                    hyp_search=hyp_search,passed_params=passed_params)   
            else:
                res=make_classifier_agg(X,y,base_model=model_type,
                        fit_method=agg_fit,predict_method=agg_pred,
                                    hyp_search=hyp_search,passed_params=passed_params)
            print(res,'res')
            return res
        else:
            return make_classifier(X,y=y,model=model_type)

#     print()
    f1_list=np.zeros((len(feature_list),times))
    roc_list=np.zeros((len(feature_list),times))
    for t in range(times):
        
        for i in range(len(feature_list)):
            feats=feature_list[i]
            res=execute(feats,y=y_comb)
            f1_list[i,t]=res[0]
            roc_list[i,t]=res[1]
#             f1_list[i,t]=i
#             roc_list[i,t]=i
    
    f1_means=f1_list.mean(axis=1)
    roc_means=roc_list.mean(axis=1)
    for j in range(len(f1_means)):
        print(f1_means[j],roc_means[j],names[j])

    
def run_combos(model_type,times=2,use_agg=False,agg_fit=None,agg_pred=None,hyp_search=True,passed_params=None):
    ##linear only matters for svc
    def execute(X,y):
        if use_agg:
            assert agg_fit is not None
            assert agg_pred is not None
            if agg_fit=='split_sample' and agg_pred=='same':
                ## BRO WHAT???? THIS IS CHEATING ISN"T IT????
                temp_model=AggregateClassifier(model_type,fit_method=agg_fit,predict_method=agg_pred)
                X,y=temp_model.reform_inputs(X,y) ### only use temp model for this reform
                
                
            print(y.shape,'y?')
            print(passed_params,'PASS')
            res=make_classifier_agg(X,y,base_model=model_type,
                        fit_method=agg_fit,predict_method=agg_pred,
                                    hyp_search=hyp_search,passed_params=passed_params)   
            print(res,'res')
            return res
        else:
            return make_classifier(X,y=y,model=model_type)
    weighted_f1s=[]
    X_wtncs=[]
    X_wtccs=[]
    X_wtnccs=[]
    X_nccs=[]
    X_wradwithcluss = []
    X_wradbetcluss = []
    X_wradbetwitcluss=[]
    
    weighted_roc=[]
    X_wtncs_roc=[]
    X_wtccs_roc=[]
    X_wtnccs_roc=[]
    X_nccs_roc=[]
    X_wradwithcluss_roc = []
    X_wradbetcluss_roc = []
    X_wradbetwitcluss_roc=[]
    print(y_comb)
    print(y_comb.mean(),y_comb.std(),'HOW MANY OF EACH?')

#     print()
    for t in range(times):
        res=execute(X_weight_comb,y=y_comb, )
        weighted_f1s.append(res[0])
        weighted_roc.append(res[1])
        
        res=execute(X_rawstack,y=y_comb, )  ### stack raw
        X_wtncs.append(res[0])
        X_wtncs_roc.append(res[1])
        
        res=execute(X_allemb,y=y_comb, )  ### other w/   ### node+cluster only
        X_wtnccs.append(res[0])
        X_wtnccs_roc.append(res[1])
        
        res=execute(X_wradbetwitclus,y=y_comb, )  ## otgre w/ all clust
        X_wtccs.append(res[0])
        X_wtccs_roc.append(res[1])  ## ours,other w/ all clust
        
#         X_wradbetwitclus=np.hstack((X_weight_g_comb,x_cc_comb,X_wc_use_comb,X_bc_use_comb))
# X_wradwitclus=np.hstack((X_weight_comb,X_weight_g_comb,x_cc_comb,X_wc_use_comb,X_bc_use_comb))
        res=execute(X_ncwit,y=y_comb, ) ## clust only
        X_wradbetcluss.append(res[0])
        X_wradbetcluss_roc.append(res[1])
        
        res=execute(X_wradwitclus,y=y_comb, ) ## alpha w/ cluster
        X_wradbetwitcluss.append(res[0])
        X_wradbetwitcluss_roc.append(res[1])
#         break

#         X_nodekm_comb,y_comb = combine_patient_feats(X_node_km_dist,graph_labels,clinical_df,conditional=conditionals)
# X_fnkm_comb
#         X_wtccs.append( execute(X_wtcc,y=y_comb, ))
        
#         X_nccs.append(execute(X_ncc,y=y_comb,
#                                           ))
# #         X_wradwithcluss.append(execute(X_wradwitclus,y=y_comb,
# #                                           ))
# #         X_wradbetcluss.append(make_classifier(X_wradbetclus,y=y_comb,
# #                                           model=model_type))
#         X_wradbetcluss.append(execute(X_wfnkm,y=y_comb,))
#         X_wradbetwitcluss.append(execute(X_wbetclus,y=y_comb))
        
#     print(X_wfnkm.shape,'FN SHAPEL')
#     print(X_wnodekm.shape,'FN SHAPEL should be bulk of before')
#     print(X_weight_comb.shape,'weighted')
    print(np.array(weighted_f1s).mean(),np.array(weighted_roc).mean(),'weighted')
    print(np.array(X_wtncs).mean(),np.array(X_wtncs_roc).mean(),'raw stack')
    print(np.array(X_wtnccs).mean(),np.array(X_wtnccs_roc).mean(),'w+node+cluster__rn node avg cent')
    print(np.array(X_wtccs).mean(),np.array(X_wtccs_roc).mean(),'all cluster stuff')
    print(np.array(X_nccs).mean(),np.array(X_nccs_roc).mean(),'node+cluster')
    print(np.array(X_wradwithcluss).mean(),np.array(X_wradwithcluss_roc).mean(),'Weight and all clus')
    print(np.array(X_wradbetcluss).mean(),np.array(X_wradbetcluss_roc).mean(),'node + cluster info')
    print(np.array(X_wradbetwitcluss).mean(),np.array(X_wradbetwitcluss_roc).mean(),'weight+Cluster rad')


    

base_param_grid_svc = [
#   {'C': [.2,.5,1, 10, 1000], 'kernel': ['linear']},
  {'C': [.01,.1,.5,1, 10, 1000], 'gamma': [1,.5,.1,0.01,0.001], 'kernel': ['rbf'],'class_weight':['balanced']},
#     {'C': [.1,.3, 1000], 'gamma': [0.01,0.001, 0.0001], 'kernel': ['rbf'],'class_weight':[None,'balanced']},
 ]
base_param_grid_svc = [
  {'C': [.1,.2,.5,1, 10,100, 1000], 'kernel': ['linear'],'class_weight':['balanced']},
  {'C': [.1,.2,.5,1, 10,100, 1000], 'degree': [2,3,4], 'kernel': ['poly'],'class_weight':['balanced']},
  {'C': [.1,.2,.5,1, 10,100, 1000], 'gamma': [1,.5,.1,0.01,0.001], 'kernel': ['rbf'],'class_weight':['balanced']},
#     {'C': [.1,.3, 1000], 'gamma': [0.01,0.001, 0.0001], 'kernel': ['rbf'],'class_weight':[None,'balanced']},
 ]

base_param_grid_linear=[
  {'C': [.2,.5,1,2,3,5, 10,100,1000],'penalty':['l1'], 'solver': ['liblinear']},
    {'C': [.2,.5,1,2,3,5, 10,100,1000],'penalty':['l2'], 'solver': ['lbfgs']},
 ]
base_param_grid_linear=[
  {'C': [.5,1,5, 10,100,1000],'penalty':['l1'], 'solver': ['liblinear'],'class_weight':['balanced']},
    {'C': [.5,1,5, 10,100,1000],'penalty':['l2'], 'solver': ['lbfgs'],'class_weight':['balanced']},
 ]
# base_param_grid_linear=[
  # {'C': [1000],'penalty':['l1'], 'solver': ['liblinear']},
    # {'C': [5],'penalty':['l2'], 'solver': ['lbfgs']},
 # ]
# base_param_grid_linear=[
  # {'C': [1,2,3,5, 10,100,1000],'penalty':['l1'], 'solver': ['liblinear']} ]
# 
base_param_grid_rfc=[
  {'sampling_method':['uniform'],'objective':['binary:logistic'],'eval_metric':['logloss']},
 ]

# n_estimators=100, *, criterion='gini', max_depth=None, min_samples_split=2
# base_param_grid_linear=[
#   {'C': [.2,.5,1,2,3,5, 10,100,1000],'penalty':['l1'], 'solver': ['liblinear']},
#     {'C': [.2,.5,1,2,3,5, 10,100,1000],'penalty':['l2'], 'solver': ['lbfgs']},
#  ]


def agg_params_grid(base_model,fit_method='concatenate',predict_method='same'):   
    agg_params={'base_model':base_model,'fit_method':fit_method,'predict_method':predict_method}
    full_param_list=[]
    base_param_list=base_param_grid_svc if base_model==SVC else agg_param_grid_linear
    for p in base_param_grid_svc:
        merged_dict={**p,**agg_params}
        full_param_list.append(merged_dict)
    return full_param_list
        # # params={'base_model':SVC,'fit_method':'split_sample','base_model_params':svc_params}
# params={'base_model':SVC,'fit_method':'split_sample'}

# full_params = {**params,**svc_params}

# def run_grid_search(X,y,model,parameters,scoring="roc_auc",cv=5,p_test=.2):
# def run_grid_search(X,y,model,parameters,scoring="roc_auc",cv=5,p_test=.2):
def run_grid_search(X,y,model,parameters,scoring="f1_macro",cv=5,p_test=.2):
#     splitter = StratifiedKFold(n_splits=cv,shuffle=True)
#     splitter = StratifiedKFold(n_splits=cv)
#     cv=10
    # splitter = StratifiedShuffleSplit(n_splits=cv, test_size=p_test)
    # splitter= RepeatedStratifiedKFold(n_splits=8, n_repeats=2)
    # splitter= RepeatedStratifiedKFold(n_splits=5, n_repeats=2)  ### like to use big splits for this so that     
    #                                                     ## the removal of one item in nested doesn't change hyperparameters drastic st. we find the marginally worst set
    #                                                     ## of params for that particular set
    splitter = LeaveOneOut()
    # clf = GridSearchCV(model, parameters,scoring=scoring,cv=splitter)
    # print(splitter,'splitter!!')
    # print(model,'MODEL')

    clf = GridSearchCV(model, parameters,scoring=scoring,cv=splitter)
    clf.fit(X, y)
    clf_df =pd.DataFrame(clf.cv_results_)
    clf_df=clf_df.sort_values(by=['rank_test_score'])

    # print(clf_df,'CLF DF')
    # print(cv,'what is going on?')
    # print(clf.cv,'GET CV')
    # pd.set_option('display.max_colwidth', None)
#     with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    # print(clf_df.iloc[:3][['params','mean_test_score','rank_test_score']])
    
    return clf
    
def make_classifier_nested(X,y,base_model,fit_method='concatenate',
                        predict_method='same',hyp_search=True, passed_params=None):
    
        X_clone= deepcopy(X)
        model = AggregateClassifier(base_model=base_model
                                    ,fit_method=fit_method,predict_method=predict_method)

                                    
        params = base_param_grid_svc if base_model==SVC else base_param_grid_linear
        print(X.shape,'SHAPPE  UP')
        if base_model==SVC:
            params=base_param_grid_svc
        elif base_model==LogisticRegression:
            params=base_param_grid_linear
        else:
            params=base_param_grid_rfc

        # gs = run_grid_search(deepcopy(X),y,model,params,cv=8)
        # gs_df = pd.DataFrame(gs.cv_results_)
        # best = gs_df[gs_df['rank_test_score']==1]

        # best_params = best['params']
        # f1=gs.best_score_
        # print(f1,gs.n_splits_,'num_split?')
        # best_model = gs.best_estimator_  
        # best_params_full = gs.best_params_  

        # print(best_params,'BEST PARAMS??')
        # print()
#             best_
        # best_model = AggregateClassifier(base_model=base_model
        #                             ,fit_method=fit_method,predict_method=predict_method
        #                                 ,base_model_params=best_params)
#         use_params=best_params
        


        # outer_splitter = StratifiedKFold(n_splits=10,shuffle=True)
        # inner_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=np.random.randint(0,40))
        # outer_splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=np.random.randint(0,40))
        outer_splitter = LeaveOneOut()
# 
        prediction=np.zeros_like(y)
        probs=np.zeros_like(y)
        y_vals=[]
        probs=[]
        prediction=[]
        current_start=0
        ## to test this, first take the hyperparam search out of the loop and do it like nonest
        for train_index, test_index in outer_splitter.split(X, y):
            print(len(probs),'NUM PROBS')
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            num_test=y_test.shape[0]
            current_end=current_start+num_test
            # assert prediction[current_start:current_end].sum()==0 ## double check we arent overwriting anything

            gs = run_grid_search(deepcopy(X_train),y_train,model,params)

            gs_df = pd.DataFrame(gs.cv_results_)
            best = gs_df[gs_df['rank_test_score']==1]
            best_params = best['params']
            f1=gs.best_score_
            print(f1,gs.n_splits_,'num_split?')
            best_model = gs.best_estimator_  
            best_params = gs.best_params_  
            # best_params=best_params_full

            print(best_params,'BEST PARAMS??')
            # print(f1,gs.n_splits_,'num_split?')
            print()
#             best_
            best_model_sub = AggregateClassifier(base_model=base_model
                                        ,fit_method=fit_method,predict_method=predict_method
                                            ,base_model_params=best_params)

            best_model_sub.fit(X_train,y_train)
            # best_model.fit(X_train,y_train)
            test_pred=best_model_sub.predict(X_test)
            test_prob=best_model_sub.predict_proba(X_test)
            # print(test_prob.shape)
            if base_model==SVC:
                print('SVC')
                roc_prob=test_prob
            else:
                print('LIN DOG')
                roc_prob=test_prob[:,1]

            # prediction[current_start:current_end]=test_pred
            probs.extend(roc_prob)
            prediction.extend(test_pred)
            y_vals.extend(y_test)
            current_start=current_end
            if len(probs)>0:
                try:
                    cur_roc = roc_auc_score(y_vals, probs)
                    print(cur_roc,'Current ROC')
                except:
                    pass
        probs=np.array(probs)
        prediction=np.array(prediction)
        y_vals=np.array(y_vals)

        print(prediction)
        print(prediction.shape)
        print(y.shape,'Y I AOUGHTA')

        print(probs)
        print(probs.shape)




        
        # if base_model==SVC:
        #    roc_prob=proba
        # else:
        #     roc_prob=proba[:,1]

        f1 = f1_score(y_vals, prediction, average='macro')
        print(classification_report(y_vals, prediction))
        print(accuracy_score(y_vals, prediction),'accuracy')
#         pred_roc = best_model.
        roc = roc_auc_score(y_vals, probs)
        # f1=.7
        print(roc,'roc')
        return f1,roc


def make_classifier_agg(X,y,base_model,fit_method='concatenate',
                        predict_method='same',hyp_search=True, passed_params=None):
    
        X_clone= deepcopy(X)
        model = AggregateClassifier(base_model=base_model
                                    ,fit_method=fit_method,predict_method=predict_method)
                                    
        params = base_param_grid_svc if base_model==SVC else base_param_grid_linear
        if base_model==SVC:
            params=base_param_grid_svc
        elif base_model==LogisticRegression:
            params=base_param_grid_linear
        else:
            params=base_param_grid_rfc
#         params= agg_params_grid(base_model,fit_method='fit_method','predict_method'=predict_method) 
        if hyp_search:
            gs = run_grid_search(deepcopy(X),y,model,params,cv=8)

            gs_df = pd.DataFrame(gs.cv_results_)
            best = gs_df[gs_df['rank_test_score']==1]

            best_params = best['params']
            f1=gs.best_score_
            print(f1,gs.n_splits_,'num_split?')
            best_model = gs.best_estimator_  
            best_params = gs.best_params_  

            print(best_params,'BEST PARAMS??')
            print()
#             best_
            best_model = AggregateClassifier(base_model=base_model
                                        ,fit_method=fit_method,predict_method=predict_method
                                            ,base_model_params=best_params)
            use_params=best_params

        else:
#             passed_params['probability']=False
            best_model=AggregateClassifier(base_model=base_model
                                        ,fit_method=fit_method,predict_method=predict_method
                                            ,base_model_params=passed_params)
            use_params=passed_params
            
#         use_params_proba=use_params
#         if base_model=='SVC':
#             use_params_proba['probability']=True
#         best_model_proba=AggregateClassifier(base_model=base_model
#                                     ,fit_method=fit_method,predict_method=predict_method
#                                         ,base_model_params=use_params_proba)

        
#         splitter = StratifiedKFold(n_splits=5,shuffle=True)
        splitter = LeaveOneOut()
#         plitter = StratifiedShuffleSplit(n_splits=20, test_size=.2)

#         print(best_model,"BEST MODEL ")
#         print(use_params,'BEST PARAMS')
#         print(base_model,'base model>')
#         print(best_model.base_model,'BASE_MODEL')
        print('NEW!')
        print(y,'EXAMPLE Y')
        print(y.shape,'EXAMPLE Y')
        print(X.shape,'new')
        print(y.shape,'clone')
        
#         best

        print(X.shape,' X shape???sss')
        print(splitter)
#         y=y-y.min()
        print(y,'YYY')
        prediction = cross_val_predict(best_model,deepcopy(X) , y, cv=splitter,method='predict')   
        
#         best_mo
        
        proba =cross_val_predict(best_model,deepcopy(X) , y, cv=splitter,method='predict_proba') 
    
#         print(proba.shape,'what?')
        
        if base_model==SVC:
           roc_prob=proba
        else:
            roc_prob=proba[:,1]
#         print(prediction,'PREDs')
#         print(proba,'PROBA')
#         roc_prob=proba
#         print(prediction.shape,'huh') 
#         print(y.shape,'y')
#         print(pre)
#         f1 = f1_score(y, pred, average='macro')
        f1 = f1_score(y, prediction, average='macro')
        print(classification_report(y, prediction))
        print(accuracy_score(y, prediction),'accuracy')
#         pred_roc = best_model.
        roc = roc_auc_score(y, roc_prob)
        print(roc,'roc')
        return f1,roc
def make_classifier(X,y,model=SVC,class_weight='balanced',hyp_search=True):
    if model in (RFC, XGBClassifier):
        hyp_search=False
#     if hyp_search:
    if hyp_search:
#         if model
        params = base_param_grid_svc if model==SVC else base_param_grid_linear
        gs = run_grid_search(X,y,model(),params,cv=5)
        gs_df = pd.DataFrame(gs.cv_results_)
        best = gs_df[gs_df['rank_test_score']==1]
        best_params = best['params']
        f1=gs.best_score_
        print(f1,gs.n_splits_,'num_split?')
        best_model = gs.best_estimator_  
        
#         print(best_model.coefs_,'COEFFIENCIETS')
#         print(np.max((best_model.coefs_),'MAX COEFF'))
        pred = cross_val_predict(model(best_params), X, y, cv=5)
        pred = cross_val_predict(best_model, X, y, cv=5)
        print(classification_report(y, pred))
    else:
        if model==XGBClassifier:
            pred = cross_val_predict(model(), X, y, cv=5)
        else:
            pred = cross_val_predict(model(class_weight=class_weight,bootstrap=True,
                                      n_estimators=400), X, y, cv=5)
        
#         f1 = f1_score(y, pred, average='weighted')
        f1 = f1_score(y, pred, average='macro')
        print(classification_report(y, pred))
        print(accuracy_score(y, pred),'accuracy')
        roc = roc_auc_score(y, pred)
        print(roc,'roc')
    return f1,roc