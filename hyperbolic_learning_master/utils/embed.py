# simple wrapper for gensim's poincare model
# source: https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/poincare.py

# import libraries
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split, ShuffleSplit, KFold
from gensim.models.poincare import PoincareModel, PoincareRelations
import logging
logging.basicConfig(level=logging.INFO)
import time
import os
import sys
import math

# import modules within repository
# my_path = 'C:\\Users\\dreww\\Desktop\\hyperbolic-learning\\utils' # path to utils folder
# sys.path.append(my_path)
# my_path = 'C:\\Users\\dreww\\Desktop\\hyperbolic-learning\\hyperbolic_svm'
# sys.path.append(my_path)

from hyperbolic_learning_master.utils.utils import *
from hyperbolic_learning_master.utils.datasets import * 
from hyperbolic_learning_master.utils.platt import *

def train_embeddings(input_path, # path to input edge relations
                     delimiter, # input file delim
                     output_path, # path to output embedding vectors 
                     size=2, # embed dimension
                     alpha=0.1, # learning rate
                     burn_in=30, # burn in train rounds
                     burn_in_alpha=0.01, # burn in learning rate
                     workers=1, # number of training threads used
                     negative=10, # negative sample size
                     epochs=200, # training rounds
                     print_every=500, # print train info
                     batch_size=10): # num samples in batch
    
    # load file with edge relations between entities

# os.path.join(os.getcwd(),'data/MEG/MEG.clinical.csv')
    temp_root=os.path.join(os.getcwd(),'temp')
    if not os.path.exists(temp_root):
        os.makedirs(temp_root)
    # temp_root=r"C:\Users\Cole S Baker\Desktop\Thesis\Thesis\hgcn\data\MEG\temp")
    read_path=os.path.join(temp_root,'temp.csv')
    relations=pd.read_csv(input_path) #### TO DOUBLE CHECK MAKE SURE NO HEADERS
    relations.to_csv(read_path,index=False,header=False)    #### this is wack, maybe rerun w/ links?

    relations = PoincareRelations(file_path=read_path, delimiter=delimiter)


    print(relations,'relations!')
    # for r in relations:
        # print(r,'RELATION')


     ### use this to make sure a self loop exists!--- actually... back on that... we can still sort without?
    #### if a node doesn't exist, is that the worst thing?, regardless, at this point you're boned


    
    # train model   ## why size = 2?
    model = PoincareModel(train_data=relations, size=size, alpha=alpha, burn_in=burn_in,
                          burn_in_alpha=burn_in_alpha, workers=workers, negative=negative)
    model.train(epochs=epochs, print_every=print_every,batch_size=batch_size)
    
    # save output vectors
    model.kv.save_word2vec_format(output_path)
    
    return

def load_embeddings(file_path, delim=' ',use_index=True):

    ### create new functinon for loading embeddings and combining with assignments
    # load pre-trained embedding coordinates

    print(file_path,'FO:E PATH')
    emb = pd.read_table(file_path, delimiter=' ')
    print(emb,'EMB')
    emb = emb.reset_index()
    # print()
    for e in emb['index']:
        print(e)
    # print(emb,'EMBEDDINGS')
    # print
    # emb = emb.reset_index()
    # print(emb,'EMBEDDINGS')
    # for e in emb['index']:
        # print(e)

    ogshape=emb.shape
    emb.columns = ['node', 'x', 'y']


    if emb.dtypes['node'] != np.number and ogshape[1]<3:

        try:
            emb = emb.loc[(emb.node.apply(lambda x: x not in ['u', 'v'])), :]
            emb['node'] = emb.node.astype('int')
            emb = emb.sort_values(by='node').reset_index(drop=True)
        except ValueError as e:
            pass

    emb['hyp_r'] = emb.apply(lambda p: poincare_dist(np.array([p['x'],p['y']]),np.array([0,0])),axis=1)
    return emb

def load_embeddings_npy(file_path, delim=' '):
    # load pre-trained embedding coordinates
    emb = np.load(file_path)  ## if things look fishy- double check order
    print(emb.shape,"EMB SHAPE")
    emb=emb[:,:2]
    node_array = np.array([i for i in range(emb.shape[0])])[:,None]
    emb = np.concatenate((node_array, emb), axis=1)
    emb_df = pd.DataFrame(columns=['node','x','y'],data=emb)
    # emb_df['node']=[i for i in range(emb.shape[0])]
    # emb = emb.reset_index()
    # emb.columns = ['node', 'x', 'y']
    # if emb.dtypes['node'] != np.number:
    #     try:
    #         emb = emb.loc[(emb.node.apply(lambda x: x not in ['u', 'v'])), :]
    #         emb['node'] = emb.node.astype('int')
    #         emb = emb.sort_values(by='node').reset_index(drop=True)
    #     except ValueError as e:
    #         passhowe
    return emb_df

# K-fold cross validation
def evaluate_model(model, X, y, max_epochs=10, cv=5, report=True, classifier='hkmeans', scorer='f1', alpha=None):
    # print classification report with other metrics
    if report:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        if classifier == 'hsvm':
            model.fit(poincare_pts_to_hyperboloid(X_train, metric='minkowski'), y_train)
            y_pred = model.predict(poincare_pts_to_hyperboloid(X_test, metric='minkowski'))
        elif classifier == 'hgmm':
            model.fit(poincare_pts_to_hyperboloid(X_train, metric='minkowski'), y_train, max_epochs=max_epochs, alpha=alpha)
            y_pred = model.predict(poincare_pts_to_hyperboloid(X_test, metric='minkowski'))
        else:
            model.fit(X_train, y_train, max_epochs=max_epochs)
            y_pred = np.argmax(model.predict(X_test), axis=1)
        print(classification_report(y_test, y_pred))
    
    # cross validation with macro f1-score metric
    kf = KFold(n_splits=cv)
    cv_scores = []
    for train, test in kf.split(X):
        if classifier == 'hsvm':
            model.fit(poincare_pts_to_hyperboloid(X[train], metric='minkowski'), y[train])
            y_pred = model.predict(poincare_pts_to_hyperboloid(X[test], metric='minkowski'))
        elif classifier == 'hgmm':
            model.fit(poincare_pts_to_hyperboloid(X[train], metric='minkowski'), y[train], max_epochs=max_epochs, alpha=alpha)
            y_pred = model.predict(poincare_pts_to_hyperboloid(X[test], metric='minkowski'))
        else:
            model.fit(X[train], y[train], max_epochs=max_epochs)
            y_pred = np.argmax(model.predict(X[test]), axis=1)
        if scorer == 'precision':
            cv_scores.append(precision_score(y[test], y_pred, average='macro'))
        else:
            cv_scores.append(f1_score(y[test], y_pred, average='macro'))
    return cv_scores

def positive_ranks_from_pw(item, relations, distance_pw):
    #### will only work w/ indices 
    theta = distance_pw[item]
    distances= distance_pw[item]
    # print(distances,'DISTANCES')

    positives = [x[1] for x in relations if x[0] == item] + [x[0] for x in relations if x[1] == item]
    ranks=np.argsort(distances)
    # print(ranks,'RANKS')
    pos_ranks = [j for j in range(len(ranks)) if ranks[j] in positives]
    # print(pos_ranks,'POS RANKS')
    return pos_ranks
    # if hyperbolic:
        # distances=
def positive_ranks(item, relations, embedding_dict,hyperbolic=True,c=-1,precomputed=False):
    theta = embedding_dict[item]
    if c==-1:
        print("AUTOMATIC CCCCCCCCC")
        print("BEWARE")
        c=1
    if hyperbolic:
        distances = [poincare_dist(theta, x,c=c) for x in np.array(list(embedding_dict.values()))]
    else: 
        distances  = [np.linalg.norm((theta-x)) for x in np.array(list(embedding_dict.values()))]
    positives = [x[1] for x in relations if x[0] == item] + [x[0] for x in relations if x[1] == item]  #### should be both ways???
    keys = list(embedding_dict.keys())
    ranks = [keys[i] for i in np.argsort(distances)]
    # print(ranks,'RANKS')
    pos_ranks = [j for j in range(len(ranks)) if ranks[j] in positives]

    # print(pos_ranks,'positive ranks')
    return pos_ranks


def avg_precision(item, relations, embedding_dict,hyperbolic=True,c=-1):
    if c==-1:
        print("AUTOMATIC CCCCCCCCC")
        print("BEWARE")
        c=1
    ranks = positive_ranks(item, relations, embedding_dict,hyperbolic=hyperbolic,c=c)
    # print(item,ranks,'RANLY')
    avg_precision = ((np.arange(1, len(ranks) + 1) / np.sort(ranks)).mean())
    return avg_precision,ranks

def avg_precision_from_pw(item, relations,distance_pw):
    ranks = positive_ranks_from_pw(item, relations, distance_pw)
    # print(item,ranks,'RANKS PW')
    avg_precision = ((np.arange(1, len(ranks) + 1) / np.sort(ranks)).mean())
    return avg_precision,ranks

def mean_average_precision_from_pw(relations,distance_pw):
    avg_precisions = []
    ranks = []
    degrees=[]
    # for item in list(embedding_dict.keys()):
    for item in range(len(distance_pw)):
        # if item>5:
            # continue
        avg_prec,pos_ranks=avg_precision_from_pw(item, relations,distance_pw)
        if (not np.isnan(avg_prec)) and (not avg_prec==math.inf):
            avg_precisions.append(avg_prec)
            ranks.extend(pos_ranks)
            degrees.append(len(pos_ranks))

            if avg_prec>10000:
                print(avg_prec)
                raise Exception()
        else:
            pass
    mAP = np.mean(avg_precisions)
    mean_rank = np.mean(ranks)
    mean_degree= np.mean(degrees)
    return [mean_rank, mAP,mean_degree]   

def mean_average_precision(relations, embedding_dict,hyperbolic=True,c=-1):
    if c==-1:
        print("AUTOMATIC CCCCCCCCC")
        print("BEWARE")
        c=1

    avg_precisions = []
    ranks = []
    degrees=[]
    for item in list(embedding_dict.keys()):
        # if item>5:
            # continue
        avg_prec,pos_ranks=avg_precision(item, relations, embedding_dict,hyperbolic=hyperbolic,c=c)
        # print(avg_prec,'AVG PREC')
        if (not np.isnan(avg_prec)) and (not avg_prec==math.inf):
            avg_precisions.append(avg_prec)
            ranks.extend(pos_ranks)
            degrees.append(len(pos_ranks))

            if avg_prec>10000:
                print(avg_prec)
                raise Exception()
        else:
            pass

    mAP = np.mean(avg_precisions)
    mean_rank = np.mean(ranks)
    mean_degree= np.mean(degrees)
    return [mean_rank, mAP,mean_degree]


if __name__=='__main__':
    relations_path = r"C:\Users\Cole S Baker\Desktop\Thesis\Thesis\hgcn\hyperbolic-learning-master\data\disease\disease_lp.edges.csv"
    delimiter=','
    output_path = r'C:\Users\Cole S Baker\Desktop\Thesis\Thesis\hgcn\hyperbolic-learning-master\data\disease\disease_pc_embeddings.tsv'
    train_embeddings(relations_path,delimiter,output_path)