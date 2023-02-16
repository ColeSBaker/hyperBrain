#!/usr/bin/env/python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from datetime import datetime
import random
import numpy as np
# from task import *
import os
import time
from utils import *
from params import *
import sys
# from manifold import *
# from gnn import RiemannianGNN
import pickle
import train_inductive

def set_up_fold(args):
    # if args.task in ['dd', 'enzymes', 'proteins', 'reddit', 'collab']:
    if args.dataset in ['dd', 'enzymes', 'proteins', 'reddit', 'collab']: # cole change
        args.train_file = (args.train_file % args.fold)
        args.dev_file = (args.dev_file % args.fold)
        args.test_file = (args.test_file % args.fold)

def add_embed_size(args):
    # add 1 for Lorentz as the degree of freedom is d - 1 with d dimensions
    if args.select_manifold == 'lorentz':
        args.embed_size += 1

def overwrite_with_manual(args,manual_args):
    if manual_args:
        # print(manual_args,'should have a few')
        for k,v in manual_args.items():

            if hasattr(args,k):
                # print('matching new to old: {} ({}->{})'.format(k,getattr(args,k),v))
                # print(k,v)
                setattr(args,k,v)
                # print(getattr(args,k))
            else:
                # print('no match: {}={}'.format(k,v))
                # print(k,v)
                setattr(args,k,v)
                # print(getattr(args,k))
    return args
def parse_default_args(manual_args=None):
    """
couldn't find a cleaner way to add in args without commandline for hyperparam search -Cole
manual_args will overwrite default if they exist
"""
    parser = argparse.ArgumentParser(description='RiemannianGNN')
    # raise Exception('First')
    parser.add_argument('--name', type=str, default='{:%Y_%m_%d_%H_%M_%S_%f}'.format(datetime.now()))
    parser.add_argument('--task', type=str, default='',choices=['lp', 'nc',''])


    parser.add_argument('--dataset', type=str, choices=['cora','pubmed','disease_nc','disease_lp','airport',
                                                     'synthetic','proteins','meg','enron','meg_ln'  #should add more to include transductive
                                                     ])
    parser.add_argument('--use_pretrained', type=int, default=0)

    
   
    # parser.add_argument('--seed', type=int, default=int(time.time()))
    # parser.add_argument('--save_dir', type=str, default=os.path.join(os.getcwd(),'results'))
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--save_id', default='')
    parser.add_argument('--save', default=1)
    parser.add_argument('--save_model', default=False)

    # parser.add_argument('--batch_size', type=int, default=-1)
    # for distributed training
    parser.add_argument('--log-freq', type=int,default=5)
    parser.add_argument('--eval-freq', type=int,default=2)

    ## can ignore these two
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--distributed_method", default='None', choices=['None',None,'multi_gpu', 'slurm'])
    parser.add_argument("--max_per_epoch", type=int, default=-1)

    # parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--split-seed", type=int, default=1234)
    # parser.add_argument("--seed", type=int, default=1234)
    
    args, _ = parser.parse_known_args()


    overwrite_with_manual(args,manual_args)
    print(args.task,'should match manual')
    assert args.task

    if args.dataset=='meg' and args.task in ('lp','ds'):
        MEGLpParams.add_params(parser)
    elif args.dataset=='meg_ln' and args.task in ('lp','ds'):
        MEGLinkNodeLpParams.add_params(parser)
    elif args.dataset=='enron' and args.task in ('lp','ds'):
        EnronLpParams.add_params(parser)
    else:
        raise Exception ('need to pick proper kind of dataset and task: {} {}'.format(args.dataset,args.task))



    args, _ = parser.parse_known_args()
    args = overwrite_with_manual(args,manual_args)  ## we must do twice bc we only edited args, not the parser

    # args.use_batch = True if args.batch_size>0 else False

    print(args.task,'3')
    args.train_only=1 if args.task=='ds' else 0
    set_up_fold(args)
    # add_embed_size(args)
    args.device  = 'cuda' if th.cuda.is_available() else 'cpu'


    return args

if __name__ == '__main__':

    args = parse_default_args()
    print(args.save_dir,'SAVE DIR?')
    # assert False
    loss=train_inductive.train(args)

    

