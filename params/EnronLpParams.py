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
    ##Cole added args
    parser.add_argument('--is_inductive', type=int, default=0)
    parser.add_argument('--is_regression', type=int, default=0)
    parser.add_argument("--n_classes", type=int, default=0)  
    ##training config
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--cuda', type=float, default=0)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--optimizer', type=str, 
                        default='Adam', choices=['Adam', 'RiemannianAdam']) 
    parser.add_argument('--momentum', type=float, default=0.999)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--sweep-c', type=float, default=0)
    parser.add_argument('--lr-reduce-freq', type=float, default=None)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--print-epoch', type=bool, default=True)
    parser.add_argument('--grad-clip', type=float, default=None)
    parser.add_argument('--min-epochs', type=int, default=100)


    ##model_config
    parser.add_argument('--model', type=str, default='HGCN', choices=['Shallow', 'MLP', 'HNN', 'GCN', 'GAT', 'HGCN'])
    parser.add_argument('--dim', type=int, default=16)  
    parser.add_argument('--manifold', type=str, default='PoincareBall', choices=['Euclidean', 'Hyperboloid', 'PoincareBall'])  
    parser.add_argument('--c', default=1.0)
    parser.add_argument('--r', default=2.)
    parser.add_argument('--t', default=1.)
    parser.add_argument('--pretrained-embeddings', default=None)
    parser.add_argument('--pos-weight', type=int, default=0)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--bias', type=int, default=1)
    parser.add_argument('--act', type=str, default='relu', choices=['relu','leaky_relu', 'elu', 'selu'])
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--alpha', type=float, default=.2)
    parser.add_argument('--double-precision', type=int, default=0)
    parser.add_argument('--use-att', type=int, default=0)
    parser.add_argument('--local-agg', type=int, default=0)

    ##data_config

    parser.add_argument('--use-feats', type=int, default=1)
    parser.add_argument('--normalize-feats', type=int, default=1)
    parser.add_argument('--normalize-adj', type=int, default=1)
    parser.add_argument('--val-prop', type=int, default=.05)
    parser.add_argument('--test-prop', type=int, default=.1)
