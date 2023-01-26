### configs for next hyperparam trial
import optuna
from datetime import datetime

import numpy as np


def train_fn(trial):
    # t=trial.suggest_uniform("t", .5,3
    # t = trial.suggest_int("t",1, 8,step=1)
    # r=trial.suggest_uniform("r", 0, 10)
    # r=t*2
    # t = trial.suggest_uniform("t", 1.0, 4.0),
    # rt_prop = trial.suggest_uniform("rt_prop", .5, 2)
    # print(t,'t!')
    # prunt()
    params = {

    'use_batch':0
    # 'num_centroid':trial.suggest_int("num_centroid", 20, 30,step=10),
    # 'num_centroid': 100,
    # 'embed_size': 5, 
    

    # 'embed_manifold':'hyperbolic'
    # 'c':trial.suggest_categorical('c', [None,1]),
    # 'output_dim':trial.suggest_int("output_dim", 2, 4),
    # 'dim':trial.suggest_int("dim", 2, 8,step=2),
    # 'output_dim':2
    # 'dim':trial.suggest_int("dim", 2, 16),
    
    # 't':t,
    # 'r':r,
    # 'rt_prop':rt_prop
    # 'use-att':trial.suggest_categorical("use-att", [1]),
    # 'r':r,
    # 't':t

    }
    return params

REGRESSIONS = set(['qm8','qm9','zinc'])
dataset = 'meg'
task = 'lp'
set_embedding = True
minimize = True if dataset in REGRESSIONS else False
# pruner =optuna.pruners.SuccessiveHalvingPruner(reduction_factor=2,min_resource=3) ### adjust NopPruner
pruner =optuna.pruners.NopPruner()
sampler= optuna.samplers.TPESampler()
# sampler =optuna.samplers.RandomSampler(seed=np.random.randint(1200))

# assert manifold in {'poincare', 'euclidean','lorentz'}
config = {    
     'set_params':{
     'task':task,  ###
     'dataset':dataset,
     # 'select_manifold':manifold,
     # 'max_per_epoch':50,
     # 'max_epochs':1,
     'epochs':10,
     # 'graph_norm': True,

     'patience':10,
     'pruner': pruner,
     'sampler':sampler,
     },
    'tunable_param_fn':train_fn
}

