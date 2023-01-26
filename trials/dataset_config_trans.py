### configs for next hyperparam trial
import optuna
from datetime import datetime


dataset = 'cora'
task = 'lp'
config = {    
     'set_params':{
     'task':task,  ###
     'dataset':dataset,
     }
}

