import argparse
from datetime import datetime
import random
import numpy as np
import os
import time
from utils import *
from params import *
import sys
from datetime import date
import pickle
import train_inductive
from trials.hyperparam_config import config
import pickle
import optuna
import copy
from utils.train_utils import get_dir_name, format_metrics
from main import parse_default_args

def save_study(args,study):
    ## should adjust to deal name based on stable params?
    study_dir='' if not hasattr(args,'study_dir') else args.study_dir
    if study_dir=='':
        study_dir = os.path.join(os.getcwd(),'study')
        if not os.path.exists(study_dir):
            os.mkdir(study_dir)

        study_dir = os.path.join(study_dir,args.dataset)
        if not os.path.exists(study_dir):
            os.mkdir(study_dir)   
        study_dir = os.path.join(study_dir,args.task)
        if not os.path.exists(study_dir):
            os.mkdir(study_dir)
        today_string = date.today().strftime("%b_%d")
        study_dir = os.path.join(study_dir, today_string)  
        if not os.path.exists(study_dir):
            os.mkdir(study_dir)

        save_id = str(len(os.listdir(study_dir))//2)

        path = os.path.join(study_dir, save_id)


        study_path = os.path.join(study_dir,'{}_study.pt'.format(path))
        args_path = os.path.join(study_dir,'{}_args.pt'.format(path)) ## will have to clean all this up
    else:
        study_path = os.path.join(study_dir,'study.pt')
        args_path = os.path.join(study_dir,'args.pt') ## will have to clean all this up
    with open(study_path, 'wb') as f:
        pickle.dump(study, f)
    with open(args_path, 'wb') as f:
        pickle.dump(args, f)

def load_study(study_path,args_path):
    with open(study_path, 'rb') as f:
        study = pickle.load(f)
    with open(args_path, 'rb') as f:
        args = pickle.load(f)
    return study, args


def create_study_output_dir(args):
    dt = datetime.now()
    date = f"{dt.year}_{dt.month}_{dt.day}"
    models_dir = os.path.join(os.path.join(os.getcwd(),'study'), args.dataset,args.task, date)
    # save_dir = get_dir_name(models_dir)

    meg_dir= os.path.join(os.path.join(os.getcwd(),'study'), args.dataset)
    #? train only... how to specify? how to specify subset?
    #"link_preds"
    #"stats"
    hgcn_str=args.model
    if args.use_virtual and args.use_weight:
        feat_str='full'
    elif args.use_virtual:
        feat_str='novirt'
    elif args.use_weight:
        feat_str='nowt'
    else:
        feat_str='nowtvirt'

    if args.use_plv:
        inp_str='plv'
    elif args.use_identity:
        inp_str='id'
    else:
        raise Exception('what are we doing here?')

    if not args.c:
        c_str='findc'
    else:
        c_str=str(args.c)+'c'

    subset_str=''
    for criteria,v in subset_str:
        subset_str+='{}{}'.format(criteria,v)

    model_str="{}_{}_{}_{}".format(hgcn_str,feat_str,c_str,inp_str)
    specific_dir=os.path.join(subset_str,"L{}".format(args.output_dim),model_str)
    full_dir=os.path.join(meg_dir,specific_dir)


    print(full_dir,'FULL DIR')
    if not os.path.exists(full_dir):
        os.makedirs(full_dir)
    return specific_dir

def create_objective_fun(args,tunable_param_fn):


    def objective_fn(trial):
        args_copy = copy.deepcopy(args) ### 
        print('start TRIALLLLLL \n\n\n')
        print('Args')
        # print(len(args_copy.eucl_vars))
        # print(len(args_copy.hyp_vars))
        tunable_params = tunable_param_fn(trial)
        args_copy.save_dir = get_dir_name(args.study_dir)

        ### write down all possible tunable items.

        for k,v in tunable_params.items():
            print(k,v)
            if hasattr(args_copy,k):
                print('match')
                print(k,v)
                setattr(args_copy,k,v)

                if k =='batch_size' and v>0:
                    setattr(args_copy,'use_batch',True)
        # print(trial,'trial')
        args_copy.trial=trial

        if args.is_inductive: ### eventually consolidate? in a run_gnn...
            # best_dev = train_inductive.train(args_copy)
            try:
                best_dev = train_inductive.train(args_copy)
                # print(best_dev,'best dev')
            except Exception as e:
                print(e)
                best_dev=2
        else:
            best_dev = train.train(args_copy)
        

        
        # best_dev,best_test = gnn_task.run_gnn() ## what does this return

        return best_dev
    return objective_fn

if __name__ == '__main__':
    # print(config)
    # s
    
    #    ### read in hyperparam configs
    # config['set_params']['batch_size']=2
    # config['set_params']['max_per_epoch']=10
    ## set like this because we must have the trial set before our hyperparam configuration can be set
    tunable_param_fn = config['tunable_param_fn']

    set_params = config['set_params']
    # print()

    args = parse_default_args(set_params) ### still need to call this with args for dataset, possibly others
    setattr(args,'study_dir',create_study_output_dir(args))

    # # print(args,'arguments')
    # dd

    args.pruner = set_params['pruner']
    # direction = "minimize" if args.is_regression else "maximize"  ## figure this ot
    # direction = "minimize" if args.is_regression else "maximize"
    direction= "minimize"
    objective = create_objective_fun(args,tunable_param_fn)
    print(args.task)

    print(args.fermi_freq,'ARG FREQ')
    study = optuna.create_study(direction=direction, sampler=set_params['sampler'],pruner=set_params['pruner'])  ### adjust based on args
    study.optimize(objective, n_trials=5)

    save_study(args,study)
    #    #### everything below relies on hyperparams ####

    #    gnn_task.run_gnn()


	#    ### save study and hyperparam configs
	# for key, value in best_trial.params.items():
	#     print("{}: {}".format(key, value))  
