#!/usr/bin/env/python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from utils import str2bool
import sys

def add_params(parser):
    parser.add_argument('--lr', type=float, default=0.01)
    # parser.add_argument('--lr_hyperbolic', type=float, default=0.01)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--model', type=str, default='HGCN', choices=['Shallow', 'MLP', 'HNN', 'GCN', 'GAT', 'HGCN'])
    parser.add_argument('--manifold', type=str, default='PoincareBall', choices=['Euclidean', 'Hyperboloid', 'PoincareBall'])
    parser.add_argument('--optimizer', type=str, 
                        default='RiemannianAdam', choices=['Adam', 'RiemannianAdam']) 

    # parser.add_argument('--lr_scheduler', type=str, 
    #                     default='none', choices=['exponential', 'cosine', 'cycle', 'none'])      
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--lr-reduce-freq', type=float, default=None)
    parser.add_argument('--is_inductive', type=int, default=1)
    # parser.add_argument("--num_class", type=int, default=3)  
    parser.add_argument("--n_classes", type=int, default=3)  
    # parser.add_argument("--num_centroid", type=int, default=100)
    # parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=100)
    # parser.add_argument('--num-layers', type=int, default=3) 
    parser.add_argument('--grad_clip', type=float, default=1)
    parser.add_argument('--dropout', type=float, default=0)
    
    # parser.add_argument('--leaky_relu', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--add_neg_edge', type=str2bool, default='False')
    # parser.add_argument('--proj_init', type=str, 
    #                     default='xavier', 
    #                     choices=['xavier', 'orthogonal', 'kaiming', 'none'])
    # parser.add_argument('--embed_manifold', type=str, default='euclidean', choices=['euclidean', 'hyperbolic'])  
    parser.add_argument('--embed_size', type=int, default=5)    
    parser.add_argument('--dim', type=int, default=2)    
    parser.add_argument('--num_feature', type=int, default=500) 
    # parser.add_argument('--eucl_vars', type=list, default=[]) 
    # parser.add_argument('--hyp_vars', type=list, default=[])
    # parser.add_argument('--tie_weight', type=str2bool, default="False") 
    parser.add_argument('--apply_edge_type', type=str2bool, default="False") 
    parser.add_argument('--is_regression', type=str2bool, default=False) 
    parser.add_argument('--train_file', type=str, default='data/synthetic/synthetic_train.pkl')
    parser.add_argument('--dev_file', type=str, default='data/synthetic/synthetic_dev.pkl')
    parser.add_argument('--test_file', type=str, default='data/synthetic/synthetic_test.pkl')
    # parser.add_argument('--dist_method', type=str, default='all_gather', choices=['all_gather', 'reduce']) 
    parser.add_argument('--prop_idx', type=int, default=0) 
    parser.add_argument('--remove_embed', type=str2bool, default=False)

    parser.add_argument('--c', default=None)
    parser.add_argument('--r', default=2)
    parser.add_argument('--t', default=1.)

    parser.add_argument('--pretrained-embeddings', type=float, default=None)
    parser.add_argument('--pos-weight', type=int, default=0)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--bias', type=int, default=1)
    parser.add_argument('--act', type=str, default='relu', choices=['relu','leaky_relu', 'elu', 'selu'])
    parser.add_argument('--n-heads', type=int, default=2)
    parser.add_argument('--alpha', type=float, default=.2)
    parser.add_argument('--double-precision', type=int, default=0)
    parser.add_argument('--use-att', type=int, default=1)
    parser.add_argument('--local-agg', type=int, default=1)

    parser.add_argument('--use-feats', type=int, default=0)
    parser.add_argument('--normalize-feats', type=int, default=0)
    parser.add_argument('--normalize-adj', type=int, default=0)

        # 'use-feats': (0, 'whether to use node features or not'),
        # 'normalize-feats': (0, 'whether to normalize input node features'),
        # 'normalize-adj': (0, 'whether to row-normalize the adjacency matrix'),


        # 'pretrained-embeddings': (None, 'path to pretrained embeddings (.npy file) for Shallow node classification'),
        # 'pos-weight': (0, 'whether to upweight positive class in node classification tasks'),
        # 'num-layers': (3, 'number of hidden layers in encoder'),
        # 'bias': (1, 'whether to use bias (1) or not (0)'),
        # 'act': ('relu', 'which activation function to use (or None for no activation)'),
        # 'n-heads': (2, 'number of attention heads for graph attention networks, must be a divisor dim'),
        # 'alpha': (0.2, 'alpha for leakyrelu in graph attention networks'),
        # 'double-precision': ('0', 'whether to use double precision'),
        # 'use-att': (0, 'whether to use hyperbolic attention or not'),
        # 'local-agg': (1, 'whether to local tangent space aggregation or not')