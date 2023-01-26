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
from utils.eval_utils import acc_f1,binary_acc
from utils.EarlyStoppingCriterion import EarlyStoppingCriterion


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
        self.encoder = getattr(encoders, args.model)(self.c, args)
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
        h = self.encoder.encode(x, adj)
        return h

    def compute_metrics(self, embeddings, data, split):
        raise NotImplementedError

    def init_metric_dict(self):
        raise NotImplementedError

    def has_improved(self, m1, m2):
        raise NotImplementedError


class NCModel(BaseModel):
    """
    Base model for node classification task.
    """

    def __init__(self, args):
        super(NCModel, self).__init__(args)
        self.decoder = model2decoder[args.model](self.c, args)
        if args.n_classes > 2:
            self.f1_average = 'micro'
        else:
            self.f1_average = 'binary'
        if args.pos_weight:
            self.weights = torch.Tensor([1., 1. / data['labels'][idx_train].mean()])
        else:
            self.weights = torch.Tensor([1.] * args.n_classes)
        if not args.cuda == -1:
            self.weights = self.weights.to(args.device)



    def decode(self, h, adj, idx):
        output = self.decoder.decode(h, adj) ### simply change decoder to allow
        return F.log_softmax(output[idx], dim=1)

    def compute_metrics(self, embeddings, data, split):
        idx = data[f'idx_{split}']
        output = self.decode(embeddings, data['adj_train_norm'], idx)
        loss = F.nll_loss(output, data['labels'][idx], self.weights)
        acc, f1 = acc_f1(output, data['labels'][idx], average=self.f1_average)
        metrics = {'loss': loss, 'acc': acc, 'f1': f1}
        return metrics

    def init_metric_dict(self):
        return {'acc': -1, 'f1': -1}

    def has_improved(self, m1, m2):
        return m1["f1"] < m2["f1"]




    def reset_epoch_stats(self, epoch, prefix):
        """
        prefix: train/dev/test
        """
        self.epoch_stats = {
            'prefix': prefix,
            'epoch': epoch,
            'loss': 0,
            'acc': 0,
            'f1': 0,
            'num_correct': 0,## add to acc_f1 funct
            'num_total': 0,
        }
        return

    def update_epoch_stats(self, embeddings, data, split):
        with th.no_grad():
            ## if loss is mean but num total scales with batch size, will lead to problem

            idx = data[f'idx_{split}']
            output = self.decode(embeddings, data['adj_train_norm'], idx)
            loss = F.nll_loss(output, data['labels'][idx], self.weights)
            acc, f1 = acc_f1(output, data['labels'][idx], average=self.f1_average)
            metrics = {'loss': loss, 'acc': acc, 'f1': f1}
            self.epoch_stats['loss'] += loss.item()
            self.epoch_stats['acc'] += acc.item()
            self.epoch_stats['f1'] += f1.item()
            self.epoch_stats['num_total'] += label.size(0)
            self.epoch_stats['num_correct'] = self.epoch_stats['num_total']*self.epoch_stats['acc']
        return self.epoch_stats

    def report_epoch_stats(self):
        statistics = [self.epoch_stats['num_correct'], self.epoch_stats['num_total'], self.epoch_stats['loss']] ## could put back with train in (or not distributed)
        accuracy = float(self.epoch_stats['acc'])/self.epoch_stats['num_total']
        f1 = float(self.epoch_stats['f1'])/self.epoch_stats['num_total']
        loss = float(self.epoch_stats['loss'])/self.epoch_stats['num_total']

        if self.epoch_stats['prefix'] != 'test':
            self.logger.info(
                "rank %d, %s phase of epoch %d: accuracy %.6f,f1 %.6f, loss %.6f, num_correct %d, total %d" % (
                self.args.distributed_rank,
                self.epoch_stats['prefix'],
                self.epoch_stats['epoch'],
                accuracy, 
                f1,
                loss,
                self.epoch_stats['num_correct'], 
                self.epoch_stats['num_total']))

        if (self.pruner) and (self.epoch_stats['prefix'] == 'dev'): ## all added by Cole
            print('reported')
            metric = loss
            self.trial.report(metric, self.epoch_stats['epoch'])
            if self.trial.should_prune():
                print('PRUNED')
                raise optuna.exceptions.TrialPruned()

        # if (self.pruner) and (self.epoch_stats['prefix'] == 'dev'): ## all added by Cole
        #     print('reported')
        #     # metric = loss if self.criterion=='min' else accuracy
        #     metric = loss
        #     # self.pruner.report(metric, self.epoch_stats['epoch'])
        #     # if self.pruner.should_prune():
        #     #   print('PRUNED')
        #     #   raise optuna.exceptions.TrialPruned()
        #     self.trial.report(metric, self.epoch_stats['epoch'])
        #     if self.trial.should_prune():
        #         print('PRUNED')
        #         raise optuna.exceptions.TrialPruned()

        return self.epoch_stats

class LPModel(BaseModel):
    """
    Base model for link prediction task.
    """

    def __init__(self, args):
        super(LPModel, self).__init__(args)
        self.dc = FermiDiracDecoder(r=args.r, t=args.t)
        self.args = args
        if (not hasattr(args,'is_inductive')) or  (not args.is_inductive):
            self.nb_false_edges = args.nb_false_edges
            self.nb_edges = args.nb_edges
            self.is_inductive=False
        else:
            self.is_inductive=True
        # if args.n_classes > 2:
        #     self.f1_average = 'micro'
        # else:
        self.f1_average = 'binary' ## just yes or no if there's an edge
        self.reset_epoch_stats(-1,'start')

    def decode(self, h, idx):
        if self.manifold_name == 'Euclidean':
            h = self.manifold.normalize(h)

        # print(h.type(),h.shape,'h.shape')
        ### what the hell is going on w/ the indexing
        ### we can use the squared distance as minimizer
        emb_in = h[idx[:, 0], :]
        emb_out = h[idx[:, 1], :]
        sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)
        probs = self.dc.forward(sqdist)
        return probs

    def compute_metrics(self, embeddings, data, split): ## need new one for inductive edges ie. brand new graphs
        if not self.is_inductive:
            edges_false = data[f'{split}_edges_false'] ### goddammit why are you so stupid- this is where the split happens shitbird
            edges = data[f'{split}_edges']
        else: ## because train / val splits naturally are unbalanced-- maybe try one without balancing?
            edges_false = data['edges_false']
            edges = data['edges'] 

        # if split == 'train' or self.is_inductive: 
        #     ### here we should sample ?
        #     edges_false = edges_false[np.random.randint(0, len(edges_false),len(edges))]
        if split == 'train': 
        #     ### here we should sample ?
            edges_false = edges_false[np.random.randint(0, len(edges_false),len(edges))]
            ### alternative-- don't decode, just get hyperbolic distance diff. and use that as loss.
            ### or compare difference in probabilites? but then you still rely on the same problem.
        pos_scores = self.decode(embeddings, edges)
        neg_scores = self.decode(embeddings, edges_false)
        # print(pos_scores.shape,edges.type(),'pos scores!')
        loss = F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores)) ### is this average or sum??
        loss += F.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores))
        if pos_scores.is_cuda:
            pos_scores = pos_scores.cpu()
            neg_scores = neg_scores.cpu()
        labels = [1] * pos_scores.shape[0] + [0] * neg_scores.shape[0]
        preds = list(pos_scores.data.cpu().numpy()) + list(neg_scores.data.cpu().numpy())
        # print(len(labels),'labels')
        # print(len(preds),'preds')
        # print(preds[0].shape,'preds')
        # print(labels[0].shape,'labels')
        roc = roc_auc_score(labels, preds)
        # print(labels,preds)
        # print(len(labels),len(preds),'preds')
        ap = average_precision_score(labels, preds)
        acc = binary_acc(labels,preds)
        metrics = {'loss': loss, 'roc': roc, 'ap': ap,'acc':acc,'num_edges_true':len(edges),'num_edges_false':len(edges_false),'num_edges':len(edges)+len(edges_false)}
        # self.epoch_stats['num_total'] += label.size(0)
        return metrics

    def init_metric_dict(self):
        return {'roc': -1, 'ap': -1}

    def has_improved(self, m1, m2):
        return 0.5 * (m1['roc'] + m1['ap']) < 0.5 * (m2['roc'] + m2['ap'])

    def reset_epoch_stats(self, epoch, prefix):
        """
        prefix: train/dev/test
        """
        ## why no keep track of all stats (roc, acc, prec) for all ?
        self.epoch_stats = {
            'prefix': prefix,
            'epoch': epoch,
            'loss': 0,
            'roc': 0,
            'ap': 0,
            'acc': 0,
            'num_correct': 0,## add to acc_f1 funct
            'num_true':0,
            'num_false':0,
            'num_graphs':0,
            'num_total': 0,
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
            self.epoch_stats['num_graphs'] += 1
            # self.epoch_stats['num_correct'] = self.epoch_stats['num_total']*self.epoch_stats['acc']
            # self.epoch_stats['num_correct'] = 0
        return self.epoch_stats

    def report_epoch_stats(self):
        statistics = [self.epoch_stats['num_correct'], self.epoch_stats['num_total'], self.epoch_stats['loss']] ## could put back with train in (or not distributed)
        # precision = float(self.epoch_stats['ap'])/self.epoch_stats['num_total']
        # roc = float(self.epoch_stats['roc'])/self.epoch_stats['num_total']
        # loss = float(self.epoch_stats['loss'])/self.epoch_stats['num_total']
        # acc =  float(self.epoch_stats['acc'])/self.epoch_stats['num_total']

        precision = float(self.epoch_stats['ap'])/self.epoch_stats['num_graphs']
        roc = float(self.epoch_stats['roc'])/self.epoch_stats['num_graphs']
        loss = float(self.epoch_stats['loss'])/self.epoch_stats['num_graphs']
        acc =  float(self.epoch_stats['acc'])/float(self.epoch_stats['num_graphs'])

        avg_stats = {
            'prefix': self.epoch_stats['prefix'],
            'epoch':  self.epoch_stats['epoch'],
            'loss': loss,
            'roc': roc,
            'ap': precision,
            'acc': acc,
            # 'num_correct': 0,## add to acc_f1 funct
            # 'num_true':0,
            # 'num_false':0,
            # 'num_graphs':0,
            # 'num_total': 0,
        }

        



        # roc = float(self.epoch_stats['roc'])/self.epoch_stats['num_total']
        # loss = float(self.epoch_stats['loss'])/self.epoch_stats['num_total']
        stat_string="%s phase of epoch %d: precision %.6f,roc %.6f, loss %.6f, acc %d, edges total %d, graphs total %d" % (
                self.epoch_stats['prefix'],
                self.epoch_stats['epoch'],
                precision, 
                roc,
                loss,
                acc, 
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

    # def format_epoch_stats()