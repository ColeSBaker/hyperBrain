import argparse
from datetime import datetime
import random
import shutil
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
from utils.model_analysis_utils import save_embeddings_all,save_embeddings
from utils.embedding_utils import multiple_embedding_analysis
from main import parse_default_args

DATA_ROOT=r'C:\Users\coleb\OneDrive\Desktop\Fall 2021\Neuro\hyperBrain\data\MEG'


def ingest_all_embeddings_in_subtree(root):
    clinical_file=os.path.join(DATA_ROOT,'MEG.clinical.csv')
    clinical_df = pd.read_csv(clinical_file)
    template_file = os.path.join(DATA_ROOT,'AALtemplate_balanced.csv')
    template_df = pd.read_csv(template_file)
    label_options = {'FN':'Functional Net','SOB':'SOB','FN_SOB':'Func Net SOB','All':'All'}
    label=label_options['FN']
    for (subroot,dirs,files) in os.walk(root, topdown=True):
        print('NEXT LEVEL??')
    #     print(root,dirs,files)
        if not dirs :
            continue
        embedding_roots=[]
    #     full_dirs=p
    #         os.path.join(s,r) for r in os.listdir(s) if '.csv' not in r  and ('.json' not in r) ]
        for d in dirs:
            full_dir=os.path.join(subroot,d)
            subfiles=os.listdir(full_dir)
            if ('finish_train.txt' in subfiles) or ('model.pth' in subfiles) or ('final_model.pth' in subfiles):
                embedding_roots.append(full_dir)
        if embedding_roots:
            print(subroot,embedding_roots)
            
        all_stat_dicts,full_dict,all_stat_dfs,full_stat_df = multiple_embedding_analysis(embedding_roots,clinical_df,label,net_threshold=.29,
                                       template_df=template_df,save_dir=subroot,suffix='.csv',balanced_atl=True,average=True)

def save_study(args,study):
    ## should adjust to deal name based on stable params?s
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

    # this will be an issue if called from other placees
    # use absolute path to change.
    meg_dir= os.path.join(os.path.join(os.getcwd(),'study'), args.dataset)
    #? train only... how to specify? how to specify subset?
    #"link_preds"
    #"stats"
    hgcn_str=args.model
    
    if args.use_virtual and args.use_weighted_loss:
        feat_str='full'
    elif args.use_weighted_loss:
        feat_str='novirt'
    elif args.use_virtual:
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

    dp_str='dp' if args.double_precision else 'nodp'

    if not args.criteria_dict:
        subset_str='all_pats'
    else:
        subset_str='pats'
        for criteria,v in args.criteria_dict.items():
            subset_str+='_{}{}'.format(criteria,v)

    val_pct_st=args.val_prop if not args.train_only else 0
    test_pct_st=args.test_prop if not args.train_only else 0
    val_str='vpct{}_tpct{}'.format(val_pct_st,test_pct_st) if not args.val_sub else 'val_excl_group'

    hpt_dir='e{}_p{}_lr{}_{}_strchinp{}_strchloss{}_b{}'.format(args.epochs,args.patience,args.lr,val_str,args.stretch_pct,args.stretch_loss,args.batch_size)


    model_str="{}_{}_{}_{}_{}".format(hgcn_str,feat_str,c_str,inp_str,dp_str)
    specific_dir=os.path.join(args.task,subset_str,"L{}".format(args.output_dim),model_str,hpt_dir)
    full_dir=os.path.join(meg_dir,specific_dir)
    print(model_str,'model')
    return full_dir

def create_objective_fun(args,tunable_param_fn):


    def objective_fn(trial):
        args_copy = copy.deepcopy(args) ### s
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

def complete_study_one_arg(args,n,erase_empty=True):
    """
    this will run the a study of trains n times
    for one static set of args
    """
    print()
    print()
    create_study_output_dir(args)
    setattr(args,'study_dir',create_study_output_dir(args))
    print(args.study_dir,'study dir')
    # assert os.path.exists(args.study_dir)

    path=args.study_dir
    # path=r"C:\Users\coleb\OneDrive\Des\Fall 2021\Neuro\hyperBrain\study\meg\L3\HGCN_novirt_0.54c_id"
    if os.path.exists(path):
        print(os.path.exists(path),'exists for sure')
        print(os.listdir(path))

        subdirs=[ name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
        competed=[]
        # this will have MAJOR problems for parrallelism
        for sd in subdirs:
            subdir_path=os.path.join(path,sd)
            subdir_files=os.listdir(subdir_path)

            # if 'final_model.pth' in subdir_files:
            if ('finish_train.txt' in subdir_files) or ('model.pth' in subdir_files):
                competed.append(sd)

                save_embeddings(subdir_path,overwrite=False)
            # else:
            #     shutil.rmtree( subdir_path )
        num_completed=len(competed)
    else:
        num_completed=0
    # returns
    print('we are in here??')
    n_trials=n-num_completed
    if n_trials<=0:
        print('negative trials')
        pass
    else:
        for n in range(n_trials):
            # print(n,'')
            args_copy = copy.deepcopy(args)
            print(args.study_dir,'study directory!')
            args_copy.save_dir = get_dir_name(args.study_dir)
            train_inductive.train(args_copy)

    save_embeddings_all(args.study_dir)
    ingest_all_embeddings_in_subtree(args.study_dir)

    return args.study_dir

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
