import argparse
from datetime import datetime
import random
import numpy as np
import os
import time
from utils import *
from utils.dataloader_utils import load_data_graphalg,load_dataset, analyze_dataset_hyperbolicity,relations_to_degree_analysis
from params import *
import sys
from datetime import date
import pickle
import train_inductive
import train
from trials.dataset_config import config
import pickle
import optuna
import copy
from main import parse_default_args
from matplotlib import pyplot as plt
from data.MEG.get_data import preprocess
from data.MEG.create_relations_dir import create_relations
if __name__ == '__main__':
    # tunable_param_fn = config['tunable_param_fn']

    # set_params = config['set_params']
    # params = config
    # print()

    # args = parse_default_args(set_params)
# 
    # print(args,'args?')
    # train_file,dev_file,test_file = preprocess(args)
    # results = analyze_dataset_hyperbolicity(args,'graph',80,node_samples=100)

    # relations_dir=r"C:\Users\Cole S Baker\Desktop\Thesis\Thesis\hgcn\data\MEG\35\plv\alpha\relations")
    # output_dir=r"C:\Users\Cole S Baker\Desktop\Thesis\Thesis\hgcn\data\MEG\35\plv\alpha")
    # C:\Users\Cole S Baker\Desktop\Thesis\Thesis\hgcn\data\MEG\32\plv\alpha
    # relations_to_degree_analysis(relations_dir,output_dir=output_dir)

    # (?# model_dir= r"C:\Users\Cole S Baker\Desktop\Thesis\Thesis\hgcn\logs\meg\lp\2022_5_11\55" ## allpha id_sim loss= .381 w/ 32noisey. seems to not be helping anything. training was unstable, slow)
    # use_band='gamma'
    use_band='alpha'
    use_percentiles=True
    plot_percentiles=True
    model_dir=r"C:\Users\Cole S Baker\Desktop\Thesis\Thesis\hgcn\logs\meg\lp\2022_5_11\60"
    model_name ='model'
    model_path = os.path.join(model_dir,"{}.pt".format(model_name))  ### args should be saved with the model
    out_embedding_path = model_dir
    model = th.load(model_path)
    raw_scans = np.load(model.args.raw_scan_file)
    setattr(model.args,'refresh_data',1)
    setattr(model.args,'train_noise_num',0)
    setattr(model.args,'use-plv',0)
    setattr(model.args,'use-identity',1)
    setattr(model.args,'use-identity-sim',0)
    setattr(model.args,'band',use_band)
    BAND_TO_INDEX = {'theta':0,'alpha':1,'beta1':2,'beta2':3,'beta':4,'gamma':5}

    args=model.args
    # print(set_params,'set_params')

    
    if use_percentiles:
        percentiles=[70,72,74,75,77,78,79,80,81,82,84,86]
        # percentiles=[75,77,79]
        scan_data = raw_scans[:,BAND_TO_INDEX[use_band],0]
        train_scans = scan_data
                

        X_weight = np.array([a[np.triu_indices(a.shape[0],k=1)] for a in train_scans])
        threshold=[np.percentile(X_weight,p) for p in percentiles]
        # dffff
    else:
        raise Exception('insisting on using percentile')
    # threshold = [.33,.35]
    # threshold = [.17,.18,.19,.2,.21,.22,.23,.24,.25,.26,.28,.29,.3,.31,.32,.33,.35,.36,.37,.38,.39,.4]

    # threshold = [.93,.94]
    # n_samp = [10000,50000]
    n_samp=[2000]
    # samps={}
    results = []
    # hyps=[1.0, 1.0285714285714285, 1.0714285714285714, 1.1785714285714286, 1.292857142857143, 1.457142857142857, 1.5071428571428571, 1.5357142857142858]
    props = []
    stat_results={}
    groups = ['all','scd','hc']
    stats= ['hyp_mean','hyp_std','edge_prop','largest_comp']
    title_stats = ['Mean Hyperbolicity','STD Hyperbolicity','% nodes pairs w/ edge','Avg Largest Componented']
    handle_stats = ['Mean Hyper','STD Hyper','%edges','Lgst Comp/100']
    for s in stats:
        stat_results[s]={}
        for g in groups:
            stat_results[s][g]=[]
    for t in threshold:
        t=t
        setattr(args,'adj_threshold',t)
        train_file,valid_file,test_file,all_file,idxs_dict,indx_file = preprocess(args)
        # d
        # d
        setattr(args,'train_file',train_file)
        setattr(args,'dev_file',valid_file)
        setattr(args,'test_file',test_file)
        setattr(args,'all_file',all_file)
        
        for n in n_samp:
            print(n,'MPDE SAMPS')
            results = analyze_dataset_hyperbolicity(args,'graph',20,node_samples=n)
            for g in groups: 
                for s in stats:
                    stat_results[s][g].append(results[g][s])

    x_label=percentiles if plot_percentiles else False
    colors = ['red','blue','green','yellow']
    
    
    handles = []
    # fake= [1,2]
    num_stats = len(stats)
    num_cols=2
    num_rows = num_stats//num_cols
    fig, axes = plt.subplots(num_rows, num_cols)  ##TOP TO BOTTOM?
    # fig.legend( lines, labels, loc = (0.5, 0), ncol=5 )
    first =True
    last=False
    for a in range(num_stats):
        s  =stats[a]
        title =title_stats[a]
        col=0
        col=a%num_cols
        row= a //num_cols
        ax = axes[row,col] 
        ax.title.set_text(title)
        if row!=(num_rows-1):
            ax.xaxis.set_visible(False)
        for i in range(len(groups)):
            g=groups[i]
            line, = ax.plot(x_label,stat_results[s][g])
            # line, = ax.plot(fake,[1+i,2+i])
            handles.append(line)
        if first:
            ax.legend(handles,groups,loc='lower right')
            first=False

    plt.show()
    fig, axes = plt.subplots(num_rows, num_cols)  ##TOP TO BOTTOM?
    first =True
    last=False
    handles = []
    for a in range(num_stats):
        s  =stats[a]
        title =title_stats[a]
        col=0
        col=a%num_cols
        row= a //num_cols
        ax = axes[row,col] 
        ax.title.set_text(title)
        if row!=(num_rows-1):
            ax.xaxis.set_visible(False)
        for i in range(len(groups)):

            g=groups[i]
            if g!='all':
                continue
            line, = ax.plot(x_label,stat_results[s][g])
            # line, = ax.plot(fake,[1+i,2+i])
            handles.append(line)
        if first:
            ax.legend(handles,['all'],loc='lower right')
            first=False
    fig, ax = plt.subplots(1)  ##TOP TO BOTTOM?
    first =True
    last=False
    handles = []
    for a in range(num_stats):
        s  =stats[a]
        title ="All stats"
        ax.title.set_text(title)
        for i in range(len(groups)):

            g=groups[i]
            if g!='all':
                continue
            stat = stat_results[s][g] if s!='largest_comp' else [val/100 for val in stat_results[s][g]]
            line, = ax.plot(x_label,stat)
        handles.append(line)

    ax.legend(handles,handle_stats,loc='best')
    plt.show()
    # plt.plot(threshold,props)
    # plt.show()
    # plt.plot(threshold,hyps)
    
    # plt.plot(threshold,props)
    # plt.show()
    # print(samps)

