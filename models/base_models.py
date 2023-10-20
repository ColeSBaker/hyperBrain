
"""Base model class."""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from layers.layers import FermiDiracDecoder
import layers.hyp_layers as hyp_layers
import manifolds
import models.encoders as encoders
from models.decoders import model2decoder

from hyperbolic_learning_master.utils.embed import train_embeddings, load_embeddings,load_embeddings_npy, evaluate_model,mean_average_precision

from utils.eval_utils import acc_f1,binary_acc,get_calibration_metrics,draw_reliability_graph
from utils.EarlyStoppingCriterion import EarlyStoppingCriterion
import random
import math
from datetime import datetime
import scipy
from matplotlib import pyplot as plt
class BaseModel(nn.Module):
    """
    Base model for graph embedding tasks.
    """

    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.manifold_name = args.manifold
        self.args=args
        if args.c is not None:
            self.c = torch.tensor([args.c])
            if not args.cuda == -1:
                self.c = self.c.to(args.device)
        else:
            self.c = nn.Parameter(torch.Tensor([1.]))

        self.manifold = getattr(manifolds, self.manifold_name)()
        if self.manifold.name == 'Hyperboloid':
            args.feat_dim = args.feat_dim + 1
        # self.nnodes = args.n_nodes ##we won't know ahead of time
        print(getattr(encoders, args.model),'aTTRs')
        print(args.num_feature,'NUMBER FEATs')
        self.encoder = getattr(encoders, args.model)(self.c, args)
        # print('big adjustment')
        self.c=self.encoder.layers[-1].agg.c
        # self.encoder.layers[-1].agg.c=self.c
        
        # self.encoder.layers[-1].agg.c=self.c

        # print(self.c,'SLEF C')
        # print(self.encoder.layers[-1].agg.c,'SLEF C')
        # self.EarlyStoppingCriterion(args.patience,'max')

        # print(hasattr(args,'pruner'))
        # print(hasattr(args,'trial'))
        if hasattr(self.args,'pruner'):
            print('pruner')
            self.pruner=self.args.pruner
        else:
            print('no pruner')
            self.pruner=None
        if hasattr(args,'trial'):
            print('trial')
            self.trial=self.args.trial
        else:
            print('no trial')
            self.trial=None

    def encode(self, x, adj):
        if self.manifold.name == 'Hyperboloid':
            o = torch.zeros_like(x)
            x = torch.cat([o[:, 0:1], x], dim=1)

        # print(x.dtype,'XXX')
        # print(x.shape)
        # print(adj,'SHAPED UP')
        # print(adj)
        h = self.encoder.encode(x, adj)
        return h

    def compute_metrics(self, embeddings, data, split):
        raise NotImplementedError

    def init_metric_dict(self):
        raise NotImplementedError

    # def 

    def has_improved(self, m1, m2):
        raise NotImplementedError

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

class LPModel(BaseModel):
    """
    Base model for link prediction task.
    """

    def __init__(self, args):
        print(LPModel)

        # print(self,'SEKF')
        super(LPModel, self)
        print(type(self))
        super(LPModel, self).__init__(args)
        self.dc = FermiDiracDecoder(r=args.r, t=args.t) 
        #### ADD TEMPERATURE DECODER RIGHT HERE!!!!
        self.args = args
        if (not hasattr(args,'is_inductive')) or  (not args.is_inductive):
            self.nb_false_edges = args.nb_false_edges
            self.nb_edges = args.nb_edges
            self.is_inductive=False
        else:
            self.is_inductive=True


        if (hasattr(args,'train_only')):
            self.train_only=args.train_only  ### this could be clearner
        else:
            self.train_only=False


        self.task='lp'
        # if args.n_classes > 2:
        #     self.f1_average = 'micro'
        # else:
        self.metrics_tracker = {}
        self.f1_average = 'binary' ## just yes or no if there's an edge
        self.reset_epoch_stats(-1,'start')
        self.previous_edges_false=None
        


    def test(self): ## unclear
         r = 2


    def decode(self, h, idx):
        ###h = embeddings
        ###idx= edge list
        normalize_euc=False
        if normalize_euc and self.manifold_name == 'Euclidean':
            h = self.manifold.normalize(h)
        if self.manifold_name == 'Euclidean':
            h = self.manifold.normalize(h) 

        emb_in = h[idx[:, 0], :]
        emb_out = h[idx[:, 1], :]
        sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)
        # print(sqdist,'sqdist')
        # print(sqdist.max(),'max squizzle?')
        probs = self.dc.forward(sqdist)
        # print(probs,'probs')
        return probs

    def true_probs(self,adj_probs,idx):
        # print(adj_probs.shape,'adj prob!')
        true_probs = adj_probs[idx[:,0],idx[:,1]]
        return true_probs

    def sample_distance_loss(self,embeddings,edges,false_dict,num_sample=10):

        ### gotta be smarter with this... could basically loop through edges false and match up embeddings
        true_samples = []
        false_samps = []
        # false_dict= args.edges_false_dict[split]
        loss=0
        # print(false_dict)
        # print('starting sample')

        dims=self.args.output_dim
        true_in = embeddings[edges[:, 0], :]
        true_out = embeddings[edges[:, 1], :]
        # print(true_in.shape,'true 3?')
        # print(true_in[:6].shape)

        fisrt=datetime.now()
        true_dists = torch.exp(-self.manifold.sqdist(true_in, true_out, self.c))
        # print(true_dists.shape,'tTRUE DONE')

        # true_dists1 = torch.exp(-self.manifold.sqdist(h1, h2, self.c))
        # false_samps = torch.zeros((true_dists.shape[0],num_sample))
        # false_samps = torch.zeros((true_dists.shape[0],2,num_sample,2))
        false_samps1 = torch.zeros((true_dists.shape[0]*num_sample,dims))
        false_samps2 = torch.zeros((true_dists.shape[0]*num_sample,dims))

        repeat1 = torch.zeros((true_dists.shape[0]*num_sample,dims))
        repeat2 = torch.zeros((true_dists.shape[0]*num_sample,dims))  ##### is 2 just the dims?

        before_loop=datetime.now()

        # print(before_loop-fisrt,"inti")
        for i in range(len(edges)):
            e =edges[i]
            n1=e[0].item()
            n2=e[1].item()
            h1 = embeddings[n1]
            h2=embeddings[n2]
            # print(false_dict,'FALSLY')
            max1 = min(num_sample,len(false_dict[n1]))

            max2 = min(num_sample,len(false_dict[n2]))

            ### this random function takes forever
            ### would be easier if we plit into num samples
            if max1>0:
                # print(embeddings.shape)
                split = embeddings.shape[0]//max1

                false_samps1[i:i+max1] = torch.vstack([embeddings[f1] for f1 in random.sample(false_dict[n1], k=max1)])
                repeat1[i:i+max1] = h1.expand(max1,-1)


            if max2>0:
                false_samps2[i:i+max2] = torch.vstack([embeddings[f2] for f2 in random.sample(false_dict[n2], k=max2)])
                repeat2[i:i+max2]=h1.expand(max2,-1)

        after_loop=datetime.now()
        # print(after_loop-before_loop,'looping time')
        eps=.001
        flat_dists1 = torch.exp(-(self.manifold.sqdist(false_samps1,repeat1, self.c)+eps))
        flat_dists2 = torch.exp(-(self.manifold.sqdist(false_samps2,repeat2, self.c)+eps)) 

        # sq_dist = -self.manifold.sqdist(false_samps1,repeat1, self.c)/2.
        # sq_dist1 = -self.manifold.sqdist(false_samps1,repeat1, self.c)/2.
        # print(sq_dist.min(),'min',sq_dist.max(),'max')
        # print(sq_dist1.min(),'min',sq_dist1.max(),'max')

        # print(flat_dists1.min(),'min',flat_dists1.max(),'max')
        # print(flat_dists2.min(),'min',flat_dists2.max(),'max')
        # print(torch.isnan(flat_dists1).any(),"NAN1")
        # print(torch.isnan(flat_dists2).any(),"NAN@")

        sumdist1 = (flat_dists1.reshape((true_dists.shape[0],num_sample)).sum(axis=1))
        sumdist2 = (flat_dists2.reshape((true_dists.shape[0],num_sample)).sum(axis=1))

        prelog1 = true_dists/(sumdist1)
        prelog2 =  true_dists/(sumdist2)
# 
        return (torch.mean(prelog1.log())+torch.mean(prelog2.log()))
        # return (torch.sum(prelog1.log())+torch.sum(prelog2.log()))
        # return (prelog1.log())+torch.mean(prelog2.log())
 
    def distance_loss(self,h,edges,edges_false,num_sample=2):

        ### gotta be smarter with this... could basically loop through edges false and match up embeddings
        true_samples = []
        true_in = h[edges[:, 0], :]
        true_out = h[edges[:, 1], :]

        false_in =  h[edges_false[:, 0], :]
        false_out =  h[edges_false[:, 1], :]

        true_dist=self.manifold.sqdist(true_in, true_out, self.c).mean()  ## so can be arb samples
        false_dist=self.manifold.sqdist(false_in, false_out, self.c).mean()
        loss = true_dist-false_dist
        # print(true_dist,'true distance')
        # print(false_dist,'false distance')

        return loss

    def get_embedding_score(self,h,edges):
        emb_dict={}
        for i in range(h.shape[0]):
            emb=h[i]
            emb_dict[i]=emb
        hyperbolic=True if self.args.manifold!='Euclidean' else False
        mean_rank,mAP,mean_degree=mean_average_precision(edges,emb_dict,hyperbolic=hyperbolic)
        print(mean_rank,mAP,'scores')
        return mean_rank,mAP,mean_degree

    def get_embedding_correlation(self,embeddings,adj_mat_true):
        emb_dict={}
        calc_dists=[]
        calc_probs=[]
        calc_dists_sq=[]
        true_dists=[]
        true_probs=[]
        true_dists_inv=[]
        dist_mat_true=1-np.array(adj_mat_true).squeeze()
        dist_mat_true_inv=1/np.array(adj_mat_true).squeeze()
        adj_mat_true=np.array(adj_mat_true).squeeze()
        print(embeddings.shape,'EMBEDDING SHAPE')
        print(dist_mat_true.shape,'dist_mat_true SHAPE')

        if dist_mat_true.shape[0]!=embeddings.shape[0]:
            assert  dist_mat_true.shape[0]==embeddings.shape[0]-1, 'if not equal shape, must be one off due to virtual node'
        for i in range(dist_mat_true.shape[0]):
            embi=torch.tensor(embeddings[i])
            
            # print(embi.shape,'FIRST EMBEDDING')

            for j in range(i+1,dist_mat_true.shape[0]):
                embj=torch.tensor(embeddings[j])
                # print(embj.shape,'SECOND EMBEDDING')
                calc_dist_sq=self.manifold.sqdist(embi, embj, self.c).detach().numpy()[0]
                calc_dist=(calc_dist_sq)**(1/2)
                # print(calc_dist)
                true_dist=dist_mat_true[i,j]
                true_prob=adj_mat_true[i,j]
                true_dist_inv=dist_mat_true_inv[i,j]
                # print(true_dist,'true dist')
                true_probs.append(true_prob)
                true_dists.append(true_dist)
                true_dists_inv.append(true_dist_inv)
                calc_dists.append(calc_dist)
                calc_dists_sq.append(calc_dist_sq)
                calc_probs.append(self.dc.forward(calc_dist))
        # print(true_dists,'TRUE DIST')
        # print(calc_dists,'TRUE DIST')
        # spearman=scipy.stats.spearmanr(true_dists,true_dists)
        # spearman=scipy.stats.spearmanr(true_dists,calc_dists)[0]
        # print(spearman,'spear')
        # pearson=scipy.stats.pearsonr(true_dists,calc_dists)[0]

        # print(o)
        # print(pearson,'PEAR')

        # plt.scatter(true_dists,calc_dists)
        # plt.text(y=1,x=1,s='Spearman: '+str(spearman))
        # plt.text(y=1.5,x=1.5,s='Pearson: '+str(pearson))
        # plt.show()

        # print(true_dists_inv)
        # print(calc_dists)
        # calc_probs = self.dc.forward(np.array(calc_dists_sq))

        # print(len(true_dists_inv),len(calc_dists),'first shapes')
        # print(len(true_dists),len(calc_probs),'first shapes')
        # print(calc_probs)


        spearman=scipy.stats.spearmanr(true_dists_inv,calc_dists)[0]
        # spearman2=scipy.stats.spearmanr(true_probs,calc_probs)[0]
        # print(spearman,'spear')
        # pearson=scipy.stats.pearsonr(true_dists_inv,calc_dists)[0]
        # pearson2=scipy.stats.pearsonr(true_probs,calc_probs)[0]
        pearson=0
        # print(spearman,pearson,'correlations')

        # plt.scatter(true_dists_inv,calc_dists)
        # plt.text(y=1,x=1,s='Spearman: '+str(spearman))
        # plt.text(y=1.5,x=1.5,s='Pearson: '+str(pearson2))
        # plt.text(y=1.5,x=1.5,s='Inv dist to dist ')
        # plt.show()

        # plt.scatter(true_probs,calc_probs)
        
        # plt.text(y=1,x=1,s='Spearman: '+str(spearman2))
        # plt.text(y=1.5,x=1.5,s='Pearson: '+str(pearson2))
        # # plt.text(y=1.5,x=1.5,s='Inv dist to dist ')
        # plt.show()

        # plt.hist(true_probs,bins=40)
        # plt.hist(calc_probs,bins=40)
        # plt.show()

        # plt.text(y=1.5,x=1.5,s='Pearson: '+str(pearson))
        # plt.show()



        return spearman,pearson

    def train_calibration_model(self):
        calibration_model = self.calibration_model if self.calibration_model is not None else ''


    def calibration_metrics(self,cal_type='ECE'):
        pass




    def loss_handler(self,edges,edges_false,pos_probs,neg_probs,pos_scores,neg_scores,num_graphs):

        if hasattr(self.args,'use_weighted_loss') and self.args.use_weighted_loss:

            if hasattr(self.args,'use_weighted_bce') and self.args.use_weighted_bce:
                l_func=F.binary_cross_entropy

            else:
                l_func=F.mse_loss


            # print(hasattr(self.args,'unify_pos_neg_loss'),'do we have unifying')
            if hasattr(self.args,'unify_pos_neg_loss') and self.args.unify_pos_neg_loss:

                # loss = l_func(pos_scores,pos_probs)
                # neg_loss = l_func(neg_scores,neg_probs)  
                # # print(loss,'POS LOSS OG')
                # # print(neg_loss,'NEG LOSS OG')
                # # print(pos_scores.size,'pos_scores shape')
                # # print(neg_scores.size,'neg_scores shape')
                # ratio=neg_scores.shape[0]/pos_scores.shape[0]
                # print(ratio,'ratio?')
                # loss+=neg_loss*ratio
                # print(pos_scores.shape,'pos_scores shape')
                # print(neg_scores.shape,'neg_scores shape')
                scores=torch.cat([pos_scores,neg_scores], dim=-1)

                probs=torch.cat([pos_probs,neg_probs], dim=-1)
                loss = l_func(scores,probs)*2
                # print(loss,'FINAL LOSS OG')
                # print(scores.shape,'score shape')
            else:
                if torch.any(torch.isnan(pos_scores)):
                    print(pos_scores,'pos scores??')
                    print('NANS IN POS SCORES')
                    pos_scores=torch.where(torch.isnan(pos_scores),.01,pos_scores)
                    print(pos_scores,'new scores')
                    print(torch.any(torch.isnan(pos_scores)),'any left??')
                    # assert False
                if torch.any(torch.isnan(neg_scores)):
                    print('NANS IN NEG SCORES')
                    neg_scores=torch.where(torch.isnan(neg_scores),.99,neg_scores)     
                    print(neg_scores,'new scores')
                    print(torch.any(torch.isnan(neg_scores)),'any left??')
                    # assert False               
                loss = l_func(pos_scores,pos_probs)
                neg_loss = l_func(neg_scores,neg_probs)  
                loss+=neg_loss
                # l_func=F.mse_loss

        else:
            # print(pos_scores,'POSITIVE SCORES????')
            # if/
            loss = F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores)) ### is this average or sum??
        # #     # print(loss,'pos loss')
            neg_loss=F.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores))
            loss+=neg_loss

        
        if pos_scores.is_cuda:
            pos_scores = pos_scores.cpu()
            neg_scores = neg_scores.cpu()

        labels = [1] * pos_scores.shape[0] + [0] * neg_scores.shape[0]

        preds =np.array(list(pos_scores.data.cpu().numpy()) + list(neg_scores.data.cpu().numpy()))
        # labels= torch.where(torch.isnan(preds),.01,labels)
        # preds= torch.where(torch.isnan(preds),.99,preds)
        # preds=np.array(preds)
        roc = roc_auc_score(labels, preds)
        ap = average_precision_score(labels, preds)
        acc = binary_acc(labels,preds)
        # try:
        #     roc = roc_auc_score(labels, preds)
        #     ap = average_precision_score(labels, preds)
        #     acc = binary_acc(labels,preds)
        # except Exception as e:
        #     print(e)
        #     print('error in roc')
        #     print(torch.isnan(preds),labels,'labels')
        #     print(torch.isnan(preds),preds,'preds')
        #     # labels=torch.where(torch.isnan(preds),.01,labels)
        #     # preds=torch.where(torch.isnan(preds),.99,preds)
        #     # roc = roc_auc_score(labels, preds)
        #     # ap = average_precision_score(labels, preds)
        #     # acc = binary_acc(labels,preds)
        #     roc=torch.Tensor(0)
        #     ap=torch.Tensor(0)
        #     acc=torch.Tensor(0)
        # ECE,MCE = get_calibration_metrics(labels,preds,one_side=False)
        ECE,MCE=0,0
        metrics = {'loss': loss, 'roc': roc, 'ap': ap,'acc':acc,'ECE':ECE,
        'num_edges_true':len(edges),'num_edges_false':len(edges_false),'num_edges':len(edges)+len(edges_false),'num_graphs':num_graphs}
        # self.epoch_stats['num_total'] += label.size(0)
        return metrics

    def get_edges(self,embeddings,data,split):
        #### helper for repeated code
        if not self.is_inductive:
            edges_false = data[f'{split}_edges_false'] ### goddammit why are you so stupid- this is where the split happens shitbird
            edges = data[f'{split}_edges']
            # false_dict = data[f'{split}_false_dict']
        else: ## because train / val splits naturally are unbalanced-- maybe try one without balancing?
            edges_false = data['edges_false']
            edges = data['edges'] 
            # false_dict = data[f'false_dict']
        # sample_rate=10
        # print NOT SAMPLING NO REASON!! 
        # we can do all negative edge
        sample_rate=-1
        # if split == 'train' or (self.args.train_only): 
        if split == ('train') and (sample_rate>0): 
            raise Exception('We should not be sampling right now.')
        #     ### here we should sample ?
            try:
                self.train_only
            except:
                print('NO TRAIN ONLY')
                self.train_only=False
            if self.train_only and not self.is_inductive:  ## train only comes in at different location (GraphDataset) for inductive train noly
                # edges=[]
                # edges_false=[]
                splits=['val','test']
                for s in splits:
                    edges_false = torch.concat([edges_false,data[f'{s}_edges_false']]) ### goddammit why are you so stupid- this is where the split happens shitbird
                    # edges+=data[f'{s}_edges']
                    edges = torch.concat([edges,data[f'{s}_edges']]) 

            edges_false = edges_false[np.random.randint(0, len(edges_false),len(edges)*sample_rate)]
            # try:
                # edges_false = edges_false[np.random.randint(0, len(edges_false),len(edges)*sample_rate)]
            # except:
                # print(data['adj_mat'].to_dense(),'No negative edges??')
            self.previous_edges_false=edges_false
        return edges, edges_false

    def compute_metrics_multiple(self,embeddings_list,data_list,split):

        edges_full=[]
        edges_false_full=[]
        pos_probs_full=[]
        neg_probs_full=[]
        pos_scores_full=[]
        neg_scores_full=[]

        for i in range(len(embeddings_list)):
            embeddings=embeddings_list[i]
            data=data_list[i]
            edges,edges_false = self.get_edges(embeddings,data,split)
            # print(edges.shape,'num edges')
            # print(edges_false.shape,'num edges_false')

            pos_scores = self.decode(embeddings, edges)
            neg_scores = self.decode(embeddings, edges_false)

            if len(pos_scores.shape)>1:
                assert pos_scores.shape[1]==1
                pos_scores=pos_scores[:,0]
                neg_scores=neg_scores[:,0]
            adj_prob = data['adj_prob']
            neg_probs = self.true_probs(adj_prob,edges_false)
            pos_probs = self.true_probs(adj_prob,edges)


            edges_full.append(edges)
            edges_false_full.append(edges_false)
            pos_probs_full.append(pos_probs)
            neg_probs_full.append(neg_probs)
            pos_scores_full.append(pos_scores)
            neg_scores_full.append(neg_scores)

        edges_comb=torch.cat(edges_full)
        edges_false_comb=torch.cat(edges_false_full)
        pos_probs_comb=torch.cat(pos_probs_full)
        neg_probs_comb=torch.cat(neg_probs_full)
        pos_scores_comb=torch.cat(pos_scores_full)
        neg_scores_comb=torch.cat(neg_scores_full)

        # print(edges_comb.shape)
        # print(pos_probs_comb.shape)
        # print(neg_scores_comb.shape)

        # print(edges_comb.shape)
        # print(pos_probs_comb.shape)
        # print(neg_scores_comb.shape)


        ### edges,edges_false only used for their length, so don't matter
        metrics = self.loss_handler(edges_comb,edges_false_comb,pos_probs_comb,neg_probs_comb,pos_scores_comb,neg_scores_comb,num_graphs=len(embeddings_list))
        # print(metrics,'')
        return metrics

  
    def compute_metrics(self, embeddings, data, split,verbose=False): ## need new one for inductive edges ie. brand new graphs
        edges,edges_false = self.get_edges(embeddings,data,split)

        pos_scores = self.decode(embeddings, edges)
        neg_scores = self.decode(embeddings, edges_false)

        # print(pos_scores.shape,'POS SCORES')
        # print(neg_scores.shape,'NEG SCORES')

        # print(edges.shape,'POS EDGES')
        # print(edges_false.shape,'NEG EDGES')
        # print(embeddings.shape,'embeddings')
        # print(edges.shape,'edges')
        # print(pos_scores.shape,'pos scores')

        if len(pos_scores.shape)>1:
            assert pos_scores.shape[1]==1
            pos_scores=pos_scores[:,0]
            neg_scores=neg_scores[:,0]
        adj_prob = data['adj_prob']
        neg_probs = self.true_probs(adj_prob,edges_false)
        pos_probs = self.true_probs(adj_prob,edges)
        metrics = self.loss_handler(edges,edges_false,pos_probs,neg_probs,pos_scores,neg_scores,num_graphs=1)
        return metrics

        # print(adj_prob.shape,'probably')
        # print(edges_false.shape,'EDGES FALSE')

        # if hasattr(self.args,'use_weighted_loss') and self.args.use_weighted_loss:
        #     adj_prob = data['adj_prob']
        #     neg_probs = self.true_probs(adj_prob,edges_false)
        #     pos_probs = self.true_probs(adj_prob,edges)

        #     # print(pos_scores.shape,'POS SCORES SHAPE')
        #     loss = F.mse_loss(pos_scores,pos_probs)
        #     neg_loss= F.mse_loss(neg_scores,neg_probs)

        # else:
        #     # print(pos_scores,'POSITIVE SCORES????')
        #     loss = F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores)) ### is this average or sum??
        # # #     # print(loss,'pos loss')
        #     neg_loss=F.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores))
        # loss+=neg_loss
        # if pos_scores.is_cuda:
        #     pos_scores = pos_scores.cpu()
        #     neg_scores = neg_scores.cpu()

        # labels = [1] * pos_scores.shape[0] + [0] * neg_scores.shape[0]
        # preds = np.array(list(pos_scores.data.cpu().numpy()) + list(neg_scores.data.cpu().numpy()))
        # roc = roc_auc_score(labels, preds)
        # ap = average_precision_score(labels, preds)
        # acc = binary_acc(labels,preds)
        # # ECE,MCE = get_calibration_metrics(labels,preds,one_side=False)
        # ECE,MCE=0,0
        # metrics = {'loss': loss, 'roc': roc, 'ap': ap,'acc':acc,'ECE':ECE,'num_edges_true':len(edges),'num_edges_false':len(edges_false),'num_edges':len(edges)+len(edges_false)}
        # self.epoch_stats['num_total'] += label.size(0)
        # metrics = self.loss_handler(embeddings,edges,edges_false,pos_probs,neg_probs,pos_scores,neg_scores)
        # return metrics

    def init_metric_dict(self):
        return {'roc': -1, 'ap': -1,'loss':100000}

    def has_improved(self, m1, m2):
        # print("HAS IMPROVED")
        # print(m1['loss'],'NEW',m2['loss'],'old')
        return m1['loss']>m2['loss']
        # return 0.5 * (m1['roc'] + m1['ap']) < 0.5 * (m2['roc'] + m2['ap'])

    def reset_epoch_stats(self, epoch, prefix):
        """
        prefix: train/dev/test
        """
        ## why no keep track of all stats (roc, acc, prec) for all ?
        # print(epoch,prefix,'resetting')

        if (prefix!='start') and ('start'!= self.epoch_stats['prefix']):
            # print(self.epoch_stats)
            # print(prefix , self.epoch_stats['prefix'],'both of em?')
            if self.epoch_stats['prefix'] not in self.metrics_tracker:
                self.metrics_tracker[self.epoch_stats['prefix'] ]=[]
            avg_stats,_ = self.report_epoch_stats()
            avg_stats['r']=self.dc.r.item()
            avg_stats['t']=self.dc.t.item()
            self.metrics_tracker[self.epoch_stats['prefix']].append(avg_stats) ### adds new stats before resetting

            # print(self.metrics_tracker,'METRICS TRACK NOW')

            
        self.epoch_stats = {
            'prefix': prefix,
            'epoch': epoch,
            'loss': 0,
            'roc': 0,
            'ap': 0,
            'acc': 0,
            'ECE':0,
            'num_correct': 0,## add to acc_f1 funct
            'num_true':0,
            'num_false':0,
            'num_graphs':0,
            'num_total': 0,
            'num_updates':0

        }
        return

    def update_epoch_stats(self, metrics, split):
        with th.no_grad():
            ## if loss is mean but num total scales with batch size, will lead to problem

            # metrics = {'loss': loss, 'acc': acc, 'f1': f1}
            old_total  = self.epoch_stats['num_total']
            new_total = old_total  + metrics['num_edges']
            # self.epoch_stats['ap'] = ((metrics['num_edges'] * metrics['ap'].item())+ old_total*self.epoch_stats['ap'])/ new_total
            # self.epoch_stats['roc'] += ((metrics['num_edges'] * metrics['roc'].item())+ old_total*self.epoch_stats['roc'])/ new_total

            self.epoch_stats['ap'] +=  metrics['ap'].item() 
            self.epoch_stats['ECE'] +=  metrics['ECE']

            self.epoch_stats['acc'] +=  metrics['acc']
            self.epoch_stats['roc'] +=  metrics['roc'].item() ### check this- is it average roc? then we need to weight per graph-- absolutely need to weight this- because we are probably skewed towards smaller graphs that have more edges

            self.epoch_stats['loss'] += metrics['loss'].item()
            
            # self.epoch_stats['num_correct'] = self.epoch_stats['num_total']*self.epoch_stats['acc']
            # self.epoch_stats['num_correct'] += (metrics['num_edges_true']+metrics['num_edges_false'])*metrics['acc']
 
            self.epoch_stats['num_total'] = new_total

            # print(self.epoch_stats['num_graphs'],'graphs')
            # print(self.epoch_stats['acc'],'accuracy')



            # self.epoch_stats['ap'] += metrics['ap'].item()
            # self.epoch_stats['roc'] += metrics['roc'].item()

            # self.epoch_stats['num_total'] += metrics['num_edges_true']+metrics['num_edges_false']
            self.epoch_stats['num_true'] += metrics['num_edges_true']
            self.epoch_stats['num_false'] += metrics['num_edges_false']
            self.epoch_stats['num_graphs'] += metrics['num_graphs']
            self.epoch_stats['num_updates'] += 1
            # print(self.epoch_stats['num_graphs'],metrics['num_graphs'],'num_graphs')
            # self.epoch_stats['num_correct'] = self.epoch_stats['num_total']*self.epoch_stats['acc']
            # self.epoch_stats['num_correct'] = 0
        return self.epoch_stats

    def report_epoch_stats(self):
        statistics = [self.epoch_stats['num_correct'], self.epoch_stats['num_total'], self.epoch_stats['loss']] ## could put back with train in (or not distributed)
        # precision = float(self.epoch_stats['ap'])/self.epoch_stats['num_total']
        # roc = float(self.epoch_stats['roc'])/self.epoch_stats['num_total']
        # loss = float(self.epoch_stats['loss'])/self.epoch_stats['num_total']
        # acc =  float(self.epoch_stats['acc'])/self.epoch_stats['num_total']

        print(self.epoch_stats['num_updates'])

        precision = float(self.epoch_stats['ap'])/self.epoch_stats['num_updates']
        roc = float(self.epoch_stats['roc'])/self.epoch_stats['num_updates']
        loss = float(self.epoch_stats['loss'])/self.epoch_stats['num_updates']
        acc =  float(self.epoch_stats['acc'])/float(self.epoch_stats['num_updates'])
        ECE =  float(self.epoch_stats['ECE'])/float(self.epoch_stats['num_updates'])

        # precision = float(self.epoch_stats['ap'])/self.epoch_stats['num_graphs']
        # roc = float(self.epoch_stats['roc'])/self.epoch_stats['num_graphs']
        # loss = float(self.epoch_stats['loss'])/self.epoch_stats['num_graphs']
        # acc =  float(self.epoch_stats['acc'])/float(self.epoch_stats['num_graphs'])
        # ECE =  float(self.epoch_stats['ECE'])/float(self.epoch_stats['num_graphs'])

        avg_stats = {
            'prefix': self.epoch_stats['prefix'],
            'epoch':  self.epoch_stats['epoch'],
            'loss': loss,
            'roc': roc,
            'ap': precision,
            'acc': acc,
            'ECE':ECE
            # 'num_correct': 0,## add to acc_f1 funct
            # 'num_true':0,
            # 'num_false':0,
            # 'num_graphs':0,
            # 'num_total': 0,
        }

        



        # roc = float(self.epoch_stats['roc'])/self.epoch_stats['num_total']
        # loss = float(self.epoch_stats['loss'])/self.epoch_stats['num_total']
        stat_string="%s phase of epoch %d: precision %.6f,roc %.6f, loss %.6f, acc %d, ECE %d #edges %d, #graphs %d" % (
                self.epoch_stats['prefix'],
                self.epoch_stats['epoch'],
                precision, 
                roc,
                loss,
                acc, 
                ECE,
                self.epoch_stats['num_total'],
                self.epoch_stats['num_graphs'])
        # if self.epoch_stats['prefix'] != 'test': ### I am so fucking confused right now, why is acc printing out as 0?

        #     print(stat_string)
            
        if self.epoch_stats['prefix'] == 'dev':
            print(loss,acc,'loss acc')
        if (self.pruner) and (self.epoch_stats['prefix'] == 'dev'): ## all added by Cole
            print('reported')
            metric = loss
            self.trial.report(metric, self.epoch_stats['epoch'])
            if self.trial.should_prune():
                print('PRUNED')
                raise optuna.exceptions.TrialPruned()
        return avg_stats,stat_string ## still shows higher numbers but thats okay
