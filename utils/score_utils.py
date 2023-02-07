import numpy as np
import pandas as pd
import pd

import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp

from embedding_utils import load_embedding, load_model,to_embedding_dict
from hyperbolic_learning_master.utils import *

def edges_from_thresh(scan,thresh):
    adj_mat = np.where(scan>thresh,1,0)
    # print(adj_mat.shape,'SHAPE UP')  
    x, y = sp.triu(adj_mat,k=1).nonzero()
    pos_edges = np.array(list(zip(x, y))).astype(int)
    x, y = sp.triu(sp.csr_matrix(1. - adj_mat),k=1).nonzero()
    neg_edges = np.array(list(zip(x, y))).astype(int)
    return pos_edges,neg_edges

def get_embedding_scores_thresholds(root,thresholds,use_natural=True,val_only=False,args={}):


    ### check shapes and go from there...
    scan_path =  os.path.join(root,'scan_info.csv')
    
    
    if args!={}:
        # args=args
        model=None
    else:
        model_path = os.path.join(root,"{}.pt".format('model'))  ### args should be saved with the model

        model=load_model(root)
        # model = th.load(model_path,map_location=torch.device('cpu'))
        args=model.args
    # print(args,'ARGS')
    if os.getcwd() not in args.raw_clinical_file:
        args.raw_clinical_file=default_clinical

    if os.getcwd() not in args.raw_scan_file:
        args.raw_scan_file=default_MEG

    scan_df = pd.read_csv(scan_path)
    if val_only:
        scan_df=scan_df[scan_df['train']==0]

    print(scan_df.shape,'HOW MUCH VAL?')


    output_str='_'.join(root.split('\\')[-3:])

    last_str='val_scores_' if val_only else 'scores_'
    outdir=os.path.join(root,output_str+last_str)

    out_dir=os.path.join(root,last_str+output_str+'.csv')
    rel_dir=os.path.join(root,'relations')




    template = os.path.join(r"C:\Users\coleb\OneDrive\Desktop\Fall 2021\Neuro\hgcn\data\MEG","AAltemplate_balanced.csv")
    template_df = pd.read_csv(template)
    template_df= template_df.apply(lambda row: meg_roi_network_assigments(row,combine_dmn=False),axis=1)

    t_trained=args.adj_threshold
    band_adj = BAND_TO_INDEX['alpha']
    metric_adj = METRIC_TO_INDEX[args.metric]
    raw_scans = np.load(args.raw_scan_file)
    scans = raw_scans[:,band_adj,metric_adj] ### this is not changed by any hyperparamters

    dim_cols=['x','y','z','a','b','c','d','e']
    print(dim_cols,'HOW MANY DIMMIES??')
    if not model:
        c=1
    else:
        c = model.c
    fd_r=args.r
    fd_t=args.t

    # optimize = 'min' if args.use_weighted_loss else 'BCE'
    if not model:
        dev_losses=[-1]
    else:
        stats=model.metrics_tracker
    # stat_to_plot
        dev_losses=[t['loss'] for t in stats['dev']]
    # best_train = min(train_losses) if optimize=='min' else max(train_losses)
    best_dev = min(dev_losses)
    # val_
    # print(mode)

    all_embeddings={}

    # print(output_str,'EY YO')
    # print(out_dir)
    nontrain_indx=scan_df['train']==0

    if use_natural and not (t_trained in thresholds):
        print('adding natural thresh to thresholds')
        thresholds=[t_trained]+thresholds

    # print('SCANNABLE')

    for i in range(len(scan_df)):
        print(i,'i')

        ### here we're gonna calculate all embedding related things for reuse over thresholds
        row=scan_df.iloc[i]
        # print(row,'ROW')
        g_id=row['graph_id']
        emb_file = row['save_location']
        emb_df = load_embedding(emb_file,template_df,c=c)
        dim_cols=[e for e in emb_df.columns if e in dim_cols]
        print(dim_cols,'DIM COLS')
        emb=np.array(emb_df[dim_cols])
        print(emb.shape,'shap!')
        dist_pw=poincare_distances(emb,c=c,full=True)
        scores_pw=fd_decoder(dist_pw,t=fd_t,r=fd_r)


        # print(dist_pw,'DISTANC PWIZZLE')
        emb_dict = to_embedding_dict(emb_df,len(dim_cols),no_suffix=True)
        all_embeddings[g_id]={}
        all_embeddings[g_id]['emb']=emb
        all_embeddings[g_id]['dist_pw']=dist_pw
        all_embeddings[g_id]['scores_pw']=scores_pw
        all_embeddings[g_id]['emb_dict']=emb_dict
        all_embeddings[g_id]['train']=row['train']


    mAPs = []
    mean_ranks=[]
    mean_degrees=[]
    mean_rocs=[]
    mean_rocsdist=[]
    mean_BCEs=[]
    mAPs_blind = []
    mean_ranks_blind=[]
    mean_degrees_blind=[]
    mean_rocs_blind=[]
    mean_rocsdist_blind=[]
    mean_BCEs_blind=[]

    trained_losses=[]
    thresholds_all=[]


    roots=[]

    t_traineds=[]
    # columns=['threshold','mAPs','Mean Ranks','Mean Degrees','ROC','MAE','T Trained','Root']
    columns=['threshold',
    'mAPs','Mean Ranks','Mean Degrees','ROC','ROCDist','BCE',
    'mAPs_blind','Mean Ranks_blind','Mean Degrees_blind','ROC_blind','ROCDist_blind','BCE_blind',
    'Dev Loss','T Trained','Root']
    if not os.path.exists(rel_dir):
        os.makedirs(rel_dir)

    for t in thresholds:
        
        mAPs_t = []
        mean_ranks_t=[]
        mean_degrees_t=[]    
        mean_roc_t=[]
        mean_rocdist_t=[]
        mean_BCE_t=[]

        mAPs_t_blind = []
        mean_ranks_t_blind=[]
        mean_degrees_t_blind=[]    
        mean_roc_t_blind=[]
        mean_rocdist_t_blind=[]
        mean_BCE_t_blind=[]

        rel_dir_t=os.path.join(rel_dir,str(t))
        if not os.path.exists(rel_dir_t):
            os.makedirs(rel_dir_t)        
        # for i in range(len(scan_df)):
        for g_id,i_dict in all_embeddings.items():
            # print(i)
            print(g_id,'gidydy')
            rel_dir_ti=os.path.join(rel_dir_t,str(g_id)+"_embeddings.npy")
            scan=scans[g_id]

            pos_edges,neg_edges= edges_from_thresh(scan,t)
            # np.save(rel_dir_ti, pos_edges)

            emb=i_dict['emb']
            emb_dict=i_dict['emb_dict']
            dist_pw=i_dict['dist_pw']
            scores_pw=i_dict['scores_pw']
            is_train=i_dict['train']
            mean_rank,mAP,mean_degree=embed.mean_average_precision_from_pw(pos_edges,dist_pw)

            # if 

            if len(scores_pw)<90:
                print('Not all included so repeating last')
                try:
                    BCE=last_bce
                    roc=last_bce
                    roc_dist=last_roc_dist
                except:
                    raise Exception('fix this issue.')
                    print('yikes guess this was first')

                    print('giving .9 for grins')
                    BCE=.90
                    roc=.90
                    roc_dist=.90
            else:
                pos_scores=np.array([scores_pw[e[0],e[1]] for e in pos_edges])
                neg_scores=np.array([scores_pw[e[0],e[1]] for e in neg_edges])

                pos_dist=np.array([dist_pw[e[0],e[1]] for e in pos_edges])
                neg_dist=np.array([dist_pw[e[0],e[1]] for e in neg_edges])

                preds = list(pos_scores) + list(neg_scores)
                labels = [1] * pos_scores.shape[0] + [0] * neg_scores.shape[0]


                #do we really need torch in this case?
                print(torch.tensor(preds),'preds')
                print(torch.tensor(labels),'labels')
                # BCE=F.binary_cross_entropy(torch.from_numpy(np.array(preds)).to(torch.float32), torch.tensor(labels)).item()
                BCE=F.binary_cross_entropy(torch.tensor(preds).double(), torch.tensor(labels).double()).item()
                preds = np.array(list(pos_scores) + list(neg_scores))
                preds_dist = np.array(list(pos_dist) + list(neg_dist))
                roc=roc_auc_score(labels,preds)
                roc_dist=roc_auc_score(labels,-preds_dist)
                last_bce=BCE
                last_roc=roc
                last_roc_dist=roc_dist

            mean_ranks_t.append(mean_rank)
            mean_degrees_t.append(mean_degree)
            mAPs_t.append(mAP)
            mean_roc_t.append(roc)
            mean_rocdist_t.append(roc_dist)
            mean_BCE_t.append(BCE)
            if not is_train:
                mean_ranks_t_blind.append(mean_rank)
                mean_degrees_t_blind.append(mean_degree)
                mAPs_t_blind.append(mAP)
                mean_roc_t_blind.append(roc)
                mean_rocdist_t_blind.append(roc_dist)
                mean_BCE_t_blind.append(BCE)


            # if len(mAPs)>20:
                # break
        nontrain_indx=nontrain_indx[:len(mAPs)]  ### for debugging
            
            # break



        maps_t_mean = np.mean(mAPs_t)
        mean_ranks_t_mean = np.mean(mean_ranks_t)
        mean_degrees_t_mean=np.mean(mean_degrees_t)
        mean_roc_t_mean = np.mean(mean_roc_t)
        mean_rocdist_t_mean = np.mean(mean_rocdist_t)
        mean_BCE_t_mean=np.mean(mean_BCE_t)


        maps_t_mean_blind = np.mean(mAPs_t_blind)
        mean_ranks_t_mean_blind = np.mean(mean_ranks_t_blind)
        mean_degrees_t_mean_blind=np.mean(mean_degrees_t_blind)
        mean_roc_t_mean_blind = np.mean(mean_roc_t_blind)
        mean_rocdist_t_mean_blind = np.mean(mean_rocdist_t_blind)
        mean_BCE_t_mean_blind=np.mean(mean_BCE_t_blind)

        # maps_t_mean = np.mean(mAPs_t)
        # mean_ranks_t_mean = np.mean(mean_ranks_t)
        # mean_degrees_t_mean=np.mean(mean_degrees_t)
        # mean_roc_t_mean = np.mean(mean_roc_t)
        # mean_BCE_t_mean=np.mean(mean_BCE_t)

        thresholds_all.append(t)
        mAPs.append(maps_t_mean)
        mean_ranks.append(mean_ranks_t_mean)
        mean_degrees.append(mean_degrees_t_mean)
        mean_rocs.append(mean_roc_t_mean_blind)
        mean_rocsdist.append(mean_rocdist_t_mean)
        mean_BCEs.append(mean_BCE_t_mean)

        mAPs_blind.append(maps_t_mean_blind)
        mean_ranks_blind.append(mean_ranks_t_mean_blind)
        mean_degrees_blind.append(mean_degrees_t_mean_blind)
        mean_rocs_blind.append(mean_roc_t_mean_blind)
        mean_rocsdist_blind.append(mean_rocdist_t_mean_blind)
        mean_BCEs_blind.append(mean_BCE_t_mean_blind)


        trained_losses.append(best_dev)
        t_traineds.append(t_trained)
        roots.append(root)
        # break

    score_df = pd.DataFrame(
        data=np.array([thresholds_all,
            mAPs,mean_ranks,mean_degrees,mean_rocs,mean_rocsdist,mean_BCEs,
            mAPs_blind,mean_ranks_blind,mean_degrees_blind,mean_rocs_blind,mean_rocsdist_blind,mean_BCEs_blind,
            trained_losses,t_traineds,roots]).T,columns=columns)

    score_df.to_csv(out_dir)
    return score_df


    # r=model.dc.r.item()
    # t=model.dc.t.item()
    # c=model.c.item()

def get_embedding_scores(scan_df_root,relation_dir,graph_ids,template_df,dims=2,hyperbolic=True):
    ## calculates scores and facts from embeddings, IE mean avg percision
    ##should be no reason for template df, just load like normal
    emb_root = scan_df_root
    scan_df = pd.read_csv(os.path.join(scan_df_root,'scan_info.csv'))
    scan_df= scan_df.sort_values(by='train')
    # print  
    mAPs = []
    mean_ranks=[]
    mean_degrees=[]
    rel_files =[]
    emb_files =[]
    g_ids=[]
    train_indx=scan_df['train']>0
    nontrain_indx=scan_df['train']==0
    print(scan_df['train'])
    print(scan_df[nontrain_indx])
    # nontrain_indx=nontrain_indx.values
    print(nontrain_indx,'NON TRAIN')


    # for g_id in graph_ids.astype(int):   ##### easy... create_file list?? then we can just index in w/ graph ids

    all_embeddings={}
    for i in range(len(scan_df)):
        row=scan_df.iloc[i]
        g_id=row['graph_id']
        emb_file = row['save_location']
        # if row['train']>0:
        #     print('SKIPS')
        #     continue
        # emb_file = os.path.join(emb_root,str(int(g_id))+"_embeddings"+suffix)
        rel_file = os.path.join(relation_dir,str(int(g_id))+"_relations.csv")
        emb_files.append(emb_file)
        rel_files.append(rel_file)
        g_ids.append(g_id)

        rels = np.array(pd.read_csv(rel_file))
        emb = load_embedding(emb_file,template_df)
        emb_dict = to_embedding_dict(emb,dims)
        all_embeddings[i]={}
        all_embeddings[i]['emb']=emb
        all_embeddings[i]['emb_dict']=emb_dict


        mean_rank,mAP,mean_degree=embed.mean_average_precision(rels,emb_dict,hyperbolic=hyperbolic)
        mean_ranks.append(mean_rank)
        mean_degrees.append(mean_degree)
        mAPs.append(mAP)

        print(mAP,'AVERAGE PRECISION')
        print(g_id,'Jeeze')

        # if len(mAPs)>20:
            # break
    nontrain_indx=nontrain_indx[:len(mAPs)]  ### for debugging
    overall_prec_blind = np.mean(np.array(mAPs)[nontrain_indx])
    mean_rank_blind = np.mean(np.array(mean_ranks)[nontrain_indx])
    mean_degree_blind=np.mean(np.array(mean_degrees)[nontrain_indx])
    overall_prec = np.mean(mAPs)
    mean_rank = np.mean(mean_ranks)
    mean_degree=np.mean(mean_degrees)

    mAPs.append(overall_prec_blind)
    mean_ranks.append(mean_rank_blind)
    mean_degrees.append(mean_degree_blind)
    emb_files.append('BlindScans')
    rel_files.append('N/A')

    mAPs.append(overall_prec)
    mean_ranks.append(mean_rank)
    mean_degrees.append(mean_degree)
    emb_files.append('AllScans')
    rel_files.append('N/A')

    score_df = pd.DataFrame(data=np.array([mAPs,mean_ranks,mean_degrees,rel_files,emb_files]).T,columns=['mAPs','Mean Ranks','Mean Degrees','rel_files','emb_files'])

    score_path = os.path.join(emb_root,'embedding_scores.csv')
    score_df.to_csv(score_path)
    




    ### RETURN A WHOLE DATAFRAME!!!
    return score_df

def study_get_scores(root,thresholds,val_only=False,force_args={}):
    roots = [os.path.join(root,r) for r in os.listdir(root) if ('.pt' not in r) and ('.csv' not in r)]
    print(roots,'ROOTS')
    output_str='_'.join(root.split('\\')[-3:])
#     print(roots,'ROOTs')
    last_str='val_full_score.csv' if val_only else 'full_score.csv'
    outdir=os.path.join(root,output_str+last_str)
    score_df = multiple_get_scores(roots,outdir,thresholds,val_only,force_args=force_args)
    return score_df

def multiple_get_scores(roots,outdir,thresholds,val_only=False,force_args={}):
    first=True
    for i in range(len(roots)):
        r=roots[i]

        score_r=get_embedding_scores_thresholds(root=r,thresholds=thresholds,val_only=val_only,args=force_args)
        # try:
            # score_r=get_embedding_scores_thresholds(root=r,thresholds=thresholds)
        # except:
            # continue
        score_r['ID']=i
        if first:
            score_df=score_r
            first=False
        else:
            score_df=pd.concat([score_df,score_r])
            
    score_df.to_csv(outdir)
    return score_df