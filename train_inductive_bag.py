from __future__ import division
from __future__ import print_function

import datetime
import json
import logging
import os
import pickle
import time

import numpy as np
import optimizers
import torch
import torch as th
from config import parser
from models.base_models import LPModel
from utils.data_utils import load_data
from utils.train_utils import get_dir_name, format_metrics,checkpoint_params,has_model_changed
from utils.dataloader_utils import load_data_graphalg,load_dataset
import copy
from diff_frech_mean.frechet_agg import frechet_B
from torchtest import assert_vars_change

def train(args):
    from models.base_models import LPModel
    # args.seed=1234
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)

    args.device= 'cuda' if th.cuda.is_available() else 'cpu'
    if int(args.double_precision):
        torch.set_default_dtype(torch.float64)
    # if int(args.cuda) >= 0:
    #     torch.cuda.manual_seed(args.seed)
    # args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
    args.patience = args.epochs if not args.patience else  int(args.patience)
    logging.getLogger().setLevel(logging.INFO)
    if args.save:
        if not args.save_dir:
            dt = datetime.datetime.now()
            date = f"{dt.year}_{dt.month}_{dt.day}"
            try:
                models_dir = os.path.join(os.environ['LOG_DIR'], args.task, date)
            except:
                
                models_dir = os.path.join(os.path.join(os.getcwd(),'logs'), args.dataset,args.task, date)
            save_dir = get_dir_name(models_dir)
        else:
            save_dir = args.save_dir
        logging.basicConfig(level=logging.INFO,
                            handlers=[
                                logging.FileHandler(os.path.join(save_dir, 'log.txt')),
                                logging.StreamHandler()
                            ])

    logging.info(f'Using: {args.device}')
    # logging.info("Using seed {}.".format(args.seed))
    torch.autograd.set_detect_anomaly(True)

    # Load data
    datapath = os.path.join(os.getcwd(),'data')
    args.feat_dim = args.num_feature
    # args.feat_dim = args.num_feature
    model = LPModel(args).to(args.device)
    # model = LPModel(args).to(args.device)
    # if 
    # getting a very annoying answer when initializing the model @ model = LPModel(args).to(args.device)
    # in super(LPModel, self).__init__(args) that self is not a subtype of LPModel, which it is
    # this only happens if calling the LP Model *after* load_dataset
    # if I call model = LPModel(args).to(args.device) before, dataloading, nothing goes wrong
    # if I call it before, then after nothing goes wrong. what could this be?
    # okay I think I get it
    # I think that if you run from a notebook without reloading the page, LPModel changes...
    # sksks
    # train_loader, dev_loader, test_loader = load_data(args, os.path.join(datapath, args.dataset))
    print(args.dataset,'ARG DATASET')
    if 'synthetic' == args.dataset: ### check what youre doing with datapath 
        train_loader, dev_loader, test_loader =load_dataset(args,'synthetic')
    if 'meg_ln' == args.dataset: ### check what youre doing with datapath 
        print("SPECIAL LOAD")
        train_loader, dev_loader, test_loader =load_dataset(args,'meg_ln')
    else:
        # fff
        train_loader, dev_loader, test_loader =load_dataset(args,'graph')
    # print(data,'data')
    # args.n_nodes, args.feat_dim = data['features'].shape
    
    # train_loader, dev_loader, test_loader =load_data_graphalg(args)
    if args.task == 'nc':
        raise Exception('only doing LP training right now')
        Model = NCModel
        # args.n_classes = int(data['labels'].max() + 1)
        args.n_classes = args.n_classes
        logging.info(f'Num classes: {args.n_classes}')
    else:
        Model = LPModel
        # args.nb_false_edges = len(data['train_edges_false'])
        # args.nb_edges = len(data['train_edges'])
        # if args.task == 'lp':
        #     Model = LPModel
        # else:
        #     Model = RECModel
        #     # No validation for reconstruction task
        #     args.eval_freq = args.epochs + 1

    if not args.lr_reduce_freq:  ### CAREFUL- ACCIDENTALY PUT LR_scheduler step in inner loop, w/ epochs=5 so we shot down
        args.lr_reduce_freq = args.epochs

    print(args.fermi_freq,'ARG FREQ')

    # ljbjj
    print(args,'ARGS???')
    print(Model,'MODEL??')
    # p
    model = LPModel(args).to(args.device)
    # Model and optimizer
    
    logging.info(str(model))

    optimizer_full = getattr(optimizers, args.optimizer)(params=filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                                    weight_decay=args.weight_decay)

    # model.freeze()
    model.dc.unfreeze()
    print(model.dc)
    print(model.dc.t,'t')

    model.dc.freeze()

    optimizer=optimizer_full
    # optimizer = getattr(optimizers, args.optimizer)(params=model.parameters(), lr=args.lr,
                                                    # weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(args.lr_reduce_freq),
        gamma=float(args.gamma)
    )
    tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    logging.info(f"Total number of parameters: {tot_params}")
    # if args.cuda is not None and int(args.cuda) >= 0 :
    #     os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
    #     model = model.to(args.device)
    #     for x, val in data.items():
    #         if torch.is_tensor(data[x]):
    #             data[x] = data[x].to(args.device)


    # Train model
    t_total = time.time()
    counter = 0
    best_val_metrics = model.init_metric_dict()
    best_test_metrics = None
    best_emb = None
    args_to_save = copy.deepcopy(args)
    if hasattr(args_to_save,'pruner'):
        setattr(args_to_save,'pruner','Cannot pickel pruner')
    if hasattr(args_to_save,'sampler'):
        setattr(args_to_save,'sampler','Cannot pickel sampler')
    if hasattr(args_to_save,'trial'):
        setattr(args_to_save,'trial','Cannot pickel trial')
    if hasattr(args_to_save,'change_threshold'):
        setattr(args_to_save,'change_threshold','Cannot pickel function')
    args.frech_B_dict={}
    args.frech_B_list=[]
    fermi_freq= args.fermi_freq

    for epoch in range(args.epochs): ## eventually check  if this randomizes so we can go with shorter epochs..

        show_fermi_post=True
        if show_fermi_post:
            print(model.dc.r.item(),model.dc.t.item(),'FERMI epoch:',epoch)
        model.unfreeze()
        model.dc.freeze()
        print(model.c,'MODEL C')
        
        model.reset_epoch_stats(epoch, 'train')
        setattr(args,'currently_training',True)
        print('reset')
        t = time.time()
        print(len(train_loader),'train length')
        initial_params_epoch=checkpoint_params(model)
        model.train()
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            for x, val in data.items():
                if torch.is_tensor(data[x]):
                    data[x] = data[x].to(args.device)
                    # data[x]
            data['adj_mat']  =  data['adj_mat'].to_dense()
            batch_size=data['features'].shape[0]
            embeddings_list=[]
            data_list=[]
            # print(data,'DATA')
            # print(data['features'].shape,'DATA SHAPE!!')
            # ssksk
            # print(batch_size,'BATCH SIZE')
            loss=0

            for i in range(batch_size):
                embeddings,data_i=run_data_single(data, i,model,no_grad=False)
                # embeddings,data_i=run_data_single(data, i,model,no_grad=freeze)
                metrics = model.compute_metrics(embeddings, data_i, 'train')
                _,data_i=run_data_single(data, i,model,skip_embedding=True)
                loss+=metrics['loss']
                embeddings_list.append(embeddings)
                data_list.append(data_i)
                # loss.backward()
                model.update_epoch_stats(metrics, 'train')   
                # model.update_epoch_stats(metrics, 'train')   
            # print(len(data_list),'LEN DATA LIS')
            # optimizer.zero_grad()
            # embeddings_list,_= run_data_batch(data,model)
            # embeddings=embeddings_list[0]  ### just for saving?? get rid of this shit

            # run_data_batch, return stacked embedding (ie. embedding_list and processed data ie. data_list)


            # train_metrics=model.compute_metrics_multiple(embeddings_list,data_list,'train')
            # model.update_epoch_stats(train_metrics, 'train')     
            # train_metrics['loss'].backward()

            loss=loss/batch_size
            loss.backward()
            print(loss,'loss by avg')

            if model.c<.1:
                print('Low model:')
                print(model.c)
            if model.c<.02:
                print('CRITICALLY LOW MODEL!!')
                print(model.c)
                model.c=torch.clip(model.c,0.02)

            # loss.backward(retain_graph=True)
            if args.grad_clip is not None:
                max_norm = float(args.grad_clip)
                all_params = list(model.parameters())
                for param in all_params:
                    torch.nn.utils.clip_grad_norm_(param, max_norm)
            optimizer.step()
            optimizer.zero_grad()
            # lr_scheduler.step() Naughty naughty
            # if self.hyperbolic and len(self.args.hyp_vars) != 0:
            #     hyperbolic_optimizer.step()
            # if self.args.is_regression and self.args.metric == "mae":
            #     loss = th.sqrt(loss)
            # ##check all this w/ batch situation/
            # model.update_epoch_stats(train_metrics, 'train')         
            if model.epoch_stats['num_graphs'] % 23 <batch_size:
            # if True:
                avg_stats,stat_string = model.report_epoch_stats()
                print(stat_string)
            all_change,any_change=has_model_changed(model,initial_params_epoch)
            assert all_change==any_change==True, "All params of model should change"
            # break
            # batch_loss.backward()


        epoch_stats,stat_string = model.report_epoch_stats()
        if True:
            logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1),
                                   'lr: {}'.format(lr_scheduler.get_last_lr()[0]),
                                   stat_string,
                                   'time: {:.4f}s'.format(time.time() - t)
                                   ]))
        print('training epoch done in ', 'time: {:.4f}s'.format(time.time() - t))
        if (epoch + 1) % args.eval_freq == 0:
            if args.eval_train:
                val_metrics,val_embeddings,val_string=train_metrics,'',stat_string
                test_metrics,test_embeddings,test_string=train_metrics,'',stat_string
            else:
                val_metrics,val_embeddings,val_string = evaluate(epoch, dev_loader, 'dev', model,freeze=True)
                optimizer.zero_grad()
                test_metrics,test_embeddings,test_string = evaluate(epoch, test_loader, 'test', model,freeze=False)
                optimizer.zero_grad()

            optimizer.zero_grad()
            # print(val_metrics,val_embeddings ,'val')
            # print(test_metrics,test_embeddings,'test')
            # if (epoch + 1) % args.log_freq == 0:
            logging.info(" ".join(['Dev','Epoch: {:04d}'.format(epoch + 1), val_string]))
            # logging.info(" ".join(['Test','Epoch: {:04d}'.format(epoch + 1), test_string]))
            if model.has_improved(best_val_metrics, val_metrics):
                best_test_metrics = test_metrics
                best_test_string = test_string
                best_emb = embeddings.cpu()
                if args.save:
                    # np.save(os.path.join(save_dir, 'embeddings.npy'), best_emb.cpu().detach().numpy())
                    np.save(os.path.join(save_dir, 'embeddings.npy'), best_emb.detach().cpu().numpy())
                    if hasattr(model.encoder, 'att_adj'):
                        filename = os.path.join(save_dir, args.dataset + '_att_adj.p')
                        pickle.dump(model.encoder.att_adj.cpu().to_dense(), open(filename, 'wb'))
                        print('Dumped attention adj: ' + filename)



                    json.dump(vars(args_to_save), open(os.path.join(save_dir, 'config.json'), 'w'))
                    torch.save(model.state_dict(), os.path.join(save_dir, 'model_state.pth'))
                    torch.save(model, os.path.join(save_dir, 'model.pt'))
                    logging.info(f"Saved model in {save_dir}")

                best_val_metrics = val_metrics
                best_val_string = val_string
                counter = 0
            else:
                counter += 1
                if counter == args.patience and epoch > args.min_epochs:
                    logging.info("Early stopping")
                    break

        lr_scheduler.step()
        # if self.hyperbolic and len(self.args.hyp_vars) != 0:
        #     hyperbolic_lr_scheduler.step()
        th.cuda.empty_cache()
    # best_dev, best_test = model.report_best()  ##implement later
    

    logging.info("Optimization Finished!")
    logging.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    # if not best_test_metrics:
    #     model.eval()
    #     best_emb = model.encode(data['features'], data['adj_train_norm'])
    #     best_test_metrics = model.compute_metrics(best_emb, data, 'test')
    logging.info(" ".join(["Val set results:", best_val_string]))
    logging.info(" ".join(["Test set results:", best_test_string]))

    if args.save:
        np.save(os.path.join(save_dir, 'embeddings.npy'), best_emb.cpu().detach().numpy())
        if hasattr(model.encoder, 'att_adj'):
            filename = os.path.join(save_dir, args.dataset + '_att_adj.p')
            pickle.dump(model.encoder.att_adj.cpu().to_dense(), open(filename, 'wb'))
            print('Dumped attention adj: ' + filename)

        json.dump(vars(args_to_save), open(os.path.join(save_dir, 'config.json'), 'w'))
        torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))
        logging.info(f"Saved model in {save_dir}") ### do we need to save logs or what???
    # return best_val_metrics,best_test_metrics
    print(best_val_metrics)
    print(best_val_metrics['loss'],'val mets')
    return best_val_metrics['loss']


def run_data_batch(data,model):
    # print('ALL')
    # print([m for m in model.parameters()])
    data_new={}
    for k,vals in data.items():
        data_new[k]=vals
        # for x, val in data.items():
        if torch.is_tensor(data_new[k]):
            data_new[k] = data_new[k].to(model.args.device)
    data_new['adj_mat']  = data_new['adj_mat'].to_sparse()
    data_new['adj_train_norm']  =  data_new['adj_mat']
    model.args.frech_B_list=[]

    for gid in data_new['graph_id']:
        frechet_B_list=[]

        if gid in model.args.frech_B_dict:
            # print('reusable')

            frech_B=model.args.frech_B_dict[gid]
        elif model.args.use_virtual:
            frech_B=91

        else:
            frech_B=frechet_B(gid)
        # print(frech_B)
        model.args.frech_B_list.append(frech_B)

    # assert -1 not in data_i['edges']
    # print(model.args.frech_B_list)
    edges_false_dict ={'train':{}}
    split='train'

    # data_i['false_dict']=edges_false_dict[split]
    embeddings = model.encode(data_new['features'].to(model.args.device), data_new['adj_mat'].to(model.args.device))
    return embeddings, data_new

# def get_data_list(data):
#     num_data=len(data['features'])
#     for i in range(num_data):


   
def run_data_single(data, index,model,skip_embedding=False,no_grad=False):

    data_i={}
    # print(data.shape,'DATA SHAPE')
    # print('index??')
    for k,vals in data.items():
        data_i[k]=vals[index]
        # for x, val in data.items():
        if torch.is_tensor(data_i[k]):
            data_i[k] = data_i[k].to(model.args.device)

    data_i['adj_mat']  = data_i['adj_mat'].to_sparse()
    data_i['adj_train_norm']  =  data_i['adj_mat']

    if data_i['graph_id'] in model.args.frech_B_dict:
        # print('reusable')
        model.args.frechet_B=model.args.frech_B_dict[data_i['graph_id']]
    elif model.args.use_virtual:
        model.args.frechet_B=91
    else:
        model.args.frechet_B=frechet_B(data_i['adj_train_norm'])
        
        model.args.frech_B_dict[data_i['graph_id']]=model.args.frechet_B

    # assert -1 not in data_i['edges']
    # edges_false_dict ={'train':{}}
    # split='train'

    # data_i['false_dict']=edges_false_dict[split]
    if skip_embedding:
        return None,data_i
    # print('SINGLE RUN \n\n\n')
    # print([m for m in model.parameters()]) 
    if no_grad:
        with th.no_grad():
            embeddings = model.encode(data_i['features'].to(model.args.device), data_i['adj_mat'].to(model.args.device))
    else:
        embeddings = model.encode(data_i['features'].to(model.args.device), data_i['adj_mat'].to(model.args.device))
    # train_metrics = model.compute_metrics(embeddings, data_i, 'train')
    # # train_metrics['loss'].backward()  #### NOT YET!

    # # loss.backward(retain_graph=True)
    # if args.grad_clip is not None:
    #     max_norm = float(args.grad_clip)
    #     all_params = list(model.parameters())
    #     for param in all_params:
    #         torch.nn.utils.clip_grad_norm_(param, max_norm)
    # optimizer.step()
    # model.update_epoch_stats(train_metrics, 'train')         
    # if model.epoch_stats['num_graphs'] % 23 ==0:
    # # if True:
    #     avg_stats,stat_string = model.report_epoch_stats()
    #     print(stat_string)
    return embeddings,data_i

def evaluate(epoch, data_loader, prefix, model,freeze=True):
    # print('EVALUATE \n\n\n\n\n ')
    
    initial_params=checkpoint_params(model)
    model.eval()
    setattr(model.args,'currently_training',False)
    # if freeze:
    #     model.freeze()
    # else:
    #     print('model not frozen!')
    #     model.unfreeze()
    # model.freeze()
    with torch.no_grad():
        # print('EVAL GOING IN')
        # print([m for m in model.parameters()]) 
        model.reset_epoch_stats(epoch, prefix)
        # print('EVAL GOING IN')
        # print([m for m in model.parameters()]) 
        # print(len(data_loader),'len' )
        for i, data in enumerate(data_loader):
            data['adj_mat']  =  data['adj_mat'].to_dense()
            # if len(data.shape)==3:
                # assert data.shape[-1]==1
            embeddings,data_i=run_data_single(data, 0,model,no_grad=freeze)
            # embeddings = model.encode(data['features'], data['adj_mat']) ### don't forget this still just gets one set of embeddings
            metrics = model.compute_metrics(embeddings, data_i, prefix)
            model.update_epoch_stats(metrics, prefix)
        epoch_stats,stat_string = model.report_epoch_stats()

    all_change,any_change=has_model_changed(model,initial_params)
    assert all_change==any_change==False,'MODEL SHOULD NOT CHANGE IN EVAL??'

    # setattr(model.args,'currently_training',True)
    return epoch_stats,embeddings,stat_string


if __name__ == '__main__':
    args = parser.parse_args()
    train(args)
    # hyperbolicty_sam