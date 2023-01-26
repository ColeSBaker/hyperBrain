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
        print(manual_args,'should have a few')
        for k,v in manual_args.items():

            if hasattr(args,k):
                print('match new to old')
                print(k,v)
                setattr(args,k,v)
                print(getattr(args,k))
            else:
                print('no match')
                # print(k,v)
                setattr(args,k,v)
                print(getattr(args,k))
    return args
def parse_default_args(manual_args=None):
    """
couldn't find a cleaner way to add in args without commandline for hyperparam search -Cole
manual_args will overwrite default if they exist
"""
    parser = argparse.ArgumentParser(description='RiemannianGNN')
    parser.add_argument('--name', type=str, default='{:%Y_%m_%d_%H_%M_%S_%f}'.format(datetime.now()))
    parser.add_argument('--task', type=str, choices=['lp', 'nc'])


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
    # model-specific params
    if args.dataset=='synthetic' and args.task=='lp':
        SyntheticHyperbolicParams.add_params(parser)
    elif args.dataset=='cora' and args.task=='lp':
        CoraLpParams.add_params(parser)
    elif args.dataset=='pubmed' and args.task=='lp':
        PubMedLpParams.add_params(parser)
    elif args.dataset=='airport' and args.task=='lp':
        AirportLpParams.add_params(parser)
    elif args.dataset=='disease_lp' and args.task=='lp':
        DiseaseLpParams.add_params(parser)
    elif args.dataset=='disease_nc' and args.task=='lp':
        DiseaseLpParams.add_params(parser)
    elif args.dataset=='proteins' and args.task=='lp':
        ProteinsLpParams.add_params(parser)
    elif args.dataset=='meg' and args.task=='lp':
        MEGLpParams.add_params(parser)
    elif args.dataset=='meg_ln' and args.task=='lp':
        MEGLinkNodeLpParams.add_params(parser)
    elif args.dataset=='enron' and args.task=='lp':
        EnronLpParams.add_params(parser)

    # if args.dataset=='synthetic' and args.task=='lp':
    #     SyntheticHyperbolicParams.add_params(parser)
    elif args.dataset=='cora' and args.task=='nc':
        CoraNcParams.add_params(parser)
    elif args.dataset=='pubmed' and args.task=='nc':
        PubMedNcParams.add_params(parser)
    elif args.dataset=='airport' and args.task=='nc':
        AirportNcParams.add_params(parser)

    elif args.dataset=='meg' and args.task=='nc':
        MEGNcParams.add_params(parser)

    elif (args.dataset=='disease_nc' or args.dataset=='disease_lp') and args.task=='nc':
        if hasattr(args,'use_pretrained') and args.use_pretrained:
            print('PRETRAINED DISEASE')
            DiseaseNcPretrainedParams.add_params(parser)
        else:
            print('OOOOOOOOOOOOOOO NC NO PRETRAINING')
            DiseaseNcParams.add_params(parser)
    elif args.dataset=='proteins' and args.task=='nc':
        ProteinsNcParams.add_params(parser)

    else:
        raise Exception("Dataset: {} and task: {} not implemented together".format(args.dataset,args.task))
    args = parser.parse_args()
    args = overwrite_with_manual(args,manual_args)  ## we must do twice bc we only edited args, not the parser

    # args.use_batch = True if args.batch_size>0 else False

    print(args.task,'3')
    set_up_fold(args)
    # add_embed_size(args)
    args.device  = 'cuda' if th.cuda.is_available() else 'cpu'


    return args

if __name__ == '__main__':

    args = parse_default_args()
    loss=train_inductive.train(args)
    print(loss,'final loss')

