### configs for next hyperparam trial
import optuna
from datetime import datetime



# def train_fn(trial):
#     t=trial.suggest_uniform("t", .6, 1.5)
#     r=trial.suggest_uniform("r", t, t*2.4)
#     params = {
#     # 'lr_hyperbolic': trial.suggest_loguniform('lr_hyperbolic', .01, .1),
#     'lr': trial.suggest_loguniform('lr', .001, .05),
#     # trial.suggest_categorical('lr_scheduler', ['e01xponential', 'cosine', 'cycle', 'none'])
#     # 'lr_scheduler':trial.suggest_categorical('lr_scheduler', ['exponential', 'cosine', 'cycle', 'none']),  ### to use exponential, need to supply gamma lr
#     # 'lr_scheduler':trial.suggest_categorical('lr_scheduler', ['cosine', 'cycle', 'none']),
#     # 'lr_scheduler':'cosine',
#     # 'dropout': trial.suggest_uniform("dropout", 0, .1), 
#     # 'gnn_layer': trial.suggest_int("gnn_layer", 4, 5),
#     'num_layers': trial.suggest_int("layers", 3, 5,step=1),
#     # 'batch_size': trial.suggest_int("batch_size", 2, 6,step=2),
#     # 'batch_size': 16,
#     # 'num_centroid':trial.suggest_int("num_centroid", 20, 30,step=10),
#     # 'num_centroid': 100,
#     # 'embed_size': 5, 
    

#     # 'embed_manifold':'hyperbolic'
#     'c':trial.suggest_categorical('c', [None,1]),
#     # 'r':trial.suggest_uniform("r", .6, 3),
#     # 't':trial.suggest_uniform("t", .6, 1.5)
#     'r':r,
#     't':t

#     }
#     return params

REGRESSIONS = set(['qm8','qm9','zinc'])
dataset = 'meg'
task = 'lp'
set_embedding = True
minimize = True if dataset in REGRESSIONS else False
# pruner =optuna.pruners.SuccessiveHalvingPruner(reduction_factor=2,min_resource=3) ### adjust NopPruner
# pruner =optuna.pruners.NopPruner()
# sampler =optuna.samplers.TPESampler()

# assert manifold in {'poincare', 'euclidean','lorentz'}
config = {    
     'set_params':{
     # 'start_time':datetime.now(),
     # 'adj_threshold':.32,
     'task':task,  ###
     'dataset':dataset,
     'data_root':r"C:\Users\Cole S Baker\Desktop\Thesis\Thesis\hgcn\data\MEG",
     # 'graph_samples':90
     }
}

