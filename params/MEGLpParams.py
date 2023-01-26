#!/usr/bin/env/python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os


band_thresholds= { 
           'alpha':.329,
           'gamma':.2491,
           'beta': .2505,
           'theta': .323}

band='alpha'
override_threshold=-1
use_override_thresh=False
if use_override_thresh and override_threshold>0:
    adj_threshold=override_threshold
else:
    adj_threshold=band_thresholds[band]
override_c=None
# set_c
use_override_c=False
band_c= { 
           'alpha':.54,
           'gamma':.66,
           'beta': .74,
            'theta':None
}

if (use_override_c) and ((override_c is None) or (override_c>0) ):
    c=override_c
else:
    c=band_c[band]
if c:
    c=float(c)

use_norm=0
overide_norm=1
# if (use_norm+overide_norm)<1 and (c==None):
#     print('Cannot skip use norm without set c unless overridden')
#     print('Use Norm set to 1')
#     use_norm=1
use_plv=1
use_identity=0

# model='HNN'
model='HGCN'
use_virtual=1
# hyperbolic=False
hyperbolic=True
manifold='PoincareBall'
if hyperbolic:
    optimizer='RiemannianAdam'
else:
    model='GCN'
    manifold='Euclidean'
    optimizer='Adam'
    
if model=='HNN':
    use_plv=1
    use_identity=0
    use_virtual=0


def add_params(parser):
    ##Cole added args
    parser.add_argument('--adj_threshold', type=float, default=adj_threshold)
    parser.add_argument('--use_weight', type=bool, default=False)
    parser.add_argument('--is_inductive', type=int, default=1)
    parser.add_argument('--is_regression', type=int, default=0)
    parser.add_argument("--n_classes", type=int, default=2)  
    parser.add_argument('--prop_idx', type=int, default=0)
    parser.add_argument('--use_virtual', type=int, default=use_virtual)

    ## Cole big changes to existing
    parser.add_argument('--output_act', type=str, default=None, choices=['relu','leaky_relu', 'elu', 'selu',None,'None'])
    parser.add_argument('--output_agg', type=bool, default=True)
    # parser.add_argument('--ignore_act', type=int, default=True)
    parser.add_argument('--output_dim', type=int, default=3)

    parser.add_argument('--use_weighted_loss', type=int, default=1)

    parser.add_argument('--band', type=str, default=band)
    parser.add_argument('--metric', type=str, default='plv')

    parser.add_argument('--use-ciplv', type=int, default=0)
    parser.add_argument('--use-plv', type=int, default=use_plv)
    parser.add_argument('--use-identity', type=int, default=use_identity)
    parser.add_argument('--use-beta', type=int, default=0)
    parser.add_argument('--use-degree', type=int, default=0)
    parser.add_argument('--use-region', type=int, default=0)
    parser.add_argument('--use-coords', type=int, default=0)
    parser.add_argument('--use-volume', type=int, default=0)
    parser.add_argument('--use_norm', type=int, default=use_norm)

    parser.add_argument('--use_batch', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=8)
    ## Cole added data args
    parser.add_argument('--refresh_data', type=int, default=1)
    parser.add_argument('--raw_clinical_file', type=str, default=os.path.join(os.getcwd(),'data/MEG/MEG.clinical.csv'))
    parser.add_argument('--raw_scan_file', type=str, default=os.path.join(os.getcwd(),'data/MEG/MEG.ROIs.npy'))
    # a=
    parser.add_argument('--raw_atlas_file', type=str, default=os.path.join(os.getcwd(),'data/MEG/AALtemplate.csv'))
    parser.add_argument('--data_root', type=str, default=os.path.join(os.getcwd(),'data/MEG'))
    # (?# 'data_root':r"C:\Users\Cole S Baker\Desktop\Thesis\Thesis\hgcn\data\MEG")
    parser.add_argument('--train_file', type=str, default=os.path.join(os.getcwd(),'data/MEG/meg_train_MEG_{}.json'.format(adj_threshold)))  #### WILL BE CHANGED IF REFRESH_DATA (see dataloader_utils)
    parser.add_argument('--dev_file', type=str, default=os.path.join(os.getcwd(),'data/MEG/meg_valid_MEG_{}.json'.format(adj_threshold)))
    parser.add_argument('--test_file', type=str, default=os.path.join(os.getcwd(),'data/MEG/meg_test_MEG_{}.json'.format(adj_threshold)))
    parser.add_argument('--all_file', type=str, default=os.path.join(os.getcwd(),'data/MEG/meg_all_MEG_{}.json'.format(adj_threshold)))

    # parser.add_argument('--all_file', type=str, default=os.path.join(os.getcwd(),'data/MEG/meg_all_MEG_{}.json'.format(adj_threshold)))


    parser.add_argument('--num_feature', type=int, default=91) ## if higher than num features, will increase dimension
    parser.add_argument('--edge_type', type=int, default=-1) ## do
    parser.add_argument('--normalization', type=int, default=0)  ## ignore for now and let our guy take care of it 

    ##training config
    parser.add_argument('--lr', type=float, default=0.021)
    # parser.add_argument('--lr', type=float, default=0.02)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--cuda', type=float, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--optimizer', type=str, 
                        default=optimizer, choices=['Adam', 'RiemannianAdam']) 
    parser.add_argument('--momentum', type=float, default=0.999)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--sweep-c', type=float, default=0)
    parser.add_argument('--lr-reduce-freq', type=float, default=20)
    parser.add_argument('--gamma', type=float, default=.9)
    parser.add_argument('--print-epoch', type=bool, default=True)
    parser.add_argument('--grad-clip', type=float, default=100)
    parser.add_argument('--min-epochs', type=int, default=2)

    parser.add_argument('--stretch_pct', type=int, default=98)


    ##model_config
    parser.add_argument('--model', type=str, default=model, choices=['Shallow', 'MLP', 'HNN', 'GCN', 'GAT', 'HGCN'])
    parser.add_argument('--dim', type=int, default=6)  
    parser.add_argument('--manifold', type=str, default=manifold, choices=['Euclidean', 'Hyperboloid', 'PoincareBall'])
    parser.add_argument('--c', default=c)
    parser.add_argument('--r', default=2)
    parser.add_argument('--t', default=1)
    parser.add_argument('--fermi_freq', default=-1)
    parser.add_argument('--fermi_use', default=0)
    parser.add_argument('--pretrained-embeddings', default=None)
    parser.add_argument('--pos-weight', type=int, default=0)
    parser.add_argument('--num-layers', type=int, default=2)
    # parser.add_argument('--num-layers', type=int, default=1)
    parser.add_argument('--bias', type=int, default=1)
    parser.add_argument('--act', type=str, default='selu', choices=['relu','leaky_relu', 'elu', 'selu',None,'None'])
    # parser.add_argument('--hyp_act', type=str, default=True)
    parser.add_argument('--hyp_act', type=str, default=False)

    parser.add_argument('--n-heads', type=int, default=2)
    parser.add_argument('--alpha', type=float, default=.2)
    parser.add_argument('--double-precision', type=int, default=1)
    parser.add_argument('--use-att', type=int, default=0)
    parser.add_argument('--local-agg', type=int, default=0)
    parser.add_argument('--use_frechet_agg', type=int, default=1)
    parser.add_argument('--train_only', type=int, default=1)  #### now the one with val stuff.
    parser.add_argument('--use_val', type=int, default=1) 
    parser.add_argument('--val_sub', type=int, default=0) 

    parser.add_argument('--train_noise_level', type=int, default=.01) 
    parser.add_argument('--train_noise_prob', type=int, default=.5) 
    parser.add_argument('--train_noise_num', type=int, default=0) 

    # parser.add_argument('--refresh_data', type=int, default=1) 

    ##data_config

    parser.add_argument('--use-feats', type=int, default=1)
    parser.add_argument('--normalize-feats', type=int, default=1)
    parser.add_argument('--normalize-adj', type=int, default=1)
    parser.add_argument('--val-prop', type=int, default=.2)
    parser.add_argument('--test-prop', type=int, default=.01)


def change_threshold(args,t):
    setattr(args,'adj_threshold',t)
    setattr(args,'train_file','data/MEG/meg_train_MEG_{}.json'.format(t))
    setattr(args,'dev_file','data/MEG/meg_valid_MEG_{}.json'.format(t))
    setattr(args,'test_file','data/MEG/meg_test_MEG_{}.json'.format(t))
    setattr(args,'all_file','data/MEG/meg_all_MEG_{}.json'.format(t))
    setattr(args,'indx_file','data/MEG/meg_all_MEG_{}.json')
    indx_file='C:\\Users\\coleb\\OneDrive\\Desktop\\Fall 2021\\Neuro\\hgcn\\data/MEG\\meg_MEG_0'+str(t)+'_latestindx.json'
    return args
    # parser.add_argument('--adj_threshold', type=float, default=adj_threshold)
    # parser.add_argument('--train_file', type=str, default=os.path.join(os.getcwd(),'data/MEG/meg_train_MEG_{}.json'.format(adj_threshold)))  #### WILL BE CHANGED IF REFRESH_DATA (see dataloader_utils)
    # parser.add_argument('--dev_file', type=str, default=os.path.join(os.getcwd(),'data/MEG/meg_valid_MEG_{}.json'.format(adj_threshold)))
    # parser.add_argument('--test_file', type=str, default=os.path.join(os.getcwd(),'data/MEG/meg_test_MEG_{}.json'.format(adj_threshold)))
    # parser.add_argument('--all_file', type=str, default=os.path.join(os.getcwd(),'data/MEG/meg_all_MEG_{}.json'.format(adj_threshold)))