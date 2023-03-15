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
from models.base_models import NCModel, LPModel
from utils.data_utils import load_data
from utils.train_utils import get_dir_name, format_metrics
from utils.dataloader_utils import load_data_graphalg,load_dataset
import copy
from diff_frech_mean.frechet_agg import frechet_B

def train(args):
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
    # if 
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
    args.feat_dim = args.num_feature
    # train_loader, dev_loader, test_loader =load_data_graphalg(args)
    if args.task == 'nc':
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
    # Model and optimizer
    model = Model(args).to(args.device)
    logging.info(str(model))

    optimizer_full = getattr(optimizers, args.optimizer)(params=filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                                    weight_decay=args.weight_decay)

    # model.freeze()
    model.dc.unfreeze()
    print(model.dc)
    print(model.dc.t,'t')

    print(model.dc.parameters())
    # sssssjsb
    # if --fermi_freezeone', type=str, default='Niether', choices=['Niether','k', 't'])
    if args.fermi_use=='full':
            optimizer_fermi = getattr(optimizers, args.optimizer)(params=model.dc.parameters(), lr=args.lr,
                                                        weight_decay=args.weight_decay)
    elif args.fermi_use=='k':
            optimizer_fermi = getattr(optimizers, args.optimizer)(params=model.dc.r, lr=args.fermi_lr,
                                                        weight_decay=args.weight_decay)
    elif args.fermi_use=='t':
            optimizer_fermi = getattr(optimizers, args.optimizer)(params=model.dc.t, lr=args.fermi_lr,
                                                        weight_decay=args.weight_decay)
    model.dc.freeze()

    optimizer=optimizer_full
    # optimizer = getattr(optimizers, args.optimizer)(params=model.parameters(), lr=args.lr,
                                                    # weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(args.lr_reduce_freq),
        gamma=float(args.gamma)
    )
    # lr_scheduler_fermi = torch.optim.lr_scheduler.StepLR(
    #     optimizer_fermi,
    #     step_size=int(args.lr_reduce_freq),
    #     gamma=float(args.gamma)
    # )
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
        # if show_fermi_post:
            # print('FERMI OPTIM')
            # print(model.dc.r.item(),model.dc.t.item(),'FERMI epoch:',epoch)
            # model.freeze()
            # model.dc.unfreeze()
        show_fermi_post=False
        if (fermi_freq>0) and ((epoch+1) % fermi_freq==0):
            # print('FERMI OPTIM')
            print(model.dc.r.item(),model.dc.t.item(),'FERMI (adjusting) epoch:',epoch)
            model.freeze()
            model.dc.unfreeze()
            optimizer= optimizer_fermi
            # lr = 
            show_fermi_post=True

        else:
            # if show_fermi_post:
            print(model.dc.r.item(),model.dc.t.item(),'FERMI (non-adjusting) epoch:',epoch,'LR: ',lr_scheduler.get_last_lr())
            model.unfreeze()
            model.dc.freeze()
            optimizer= optimizer_full

        print(model.c,'MODEL C')

        model.train()
        model.reset_epoch_stats(epoch, 'train')
        setattr(args,'currently_training',True)
        print('reset')
        t = time.time()
        print(len(train_loader),'train length')
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()


            # print(i,len(train_loader))
            # for 

            for x, val in data.items():
                if torch.is_tensor(data[x]):
                    data[x] = data[x].to(args.device)
                    # data[x]
            # print(data['graph_id'])
            # data['features']  = data['features'][0]
            # data['edges']  = data['edges'][0]
            # data['edges_false']  = data['edges_false'][0]
            # data['labels']  = data['labels'][0]
            # data['adj_prob']  = data['adj_prob'][0]
            # if len(data['edges_false'][0])==0:
            #     continue  ## one protein has no flarnking negative edge. just fully connected 

            # data['adj_mat']  =  data['adj_mat'].to_dense()[0].to_sparse()
            data['adj_mat']  =  data['adj_mat'].to_dense()
            # data['adj_train_norm']  =  data['adj_mat']

            # if data['graph_id'][0] in args.frech_B_dict:
            #     # print('reusable')
            #     args.frechet_B=args.frech_B_dict[data['graph_id'][0]]
            # else:
            #     args.frechet_B=frechet_B(data['adj_train_norm'])
                
            #     args.frech_B_dict[data['graph_id'][0]]=args.frechet_B
            # # data['adj_mat']  =  data['adj_mat'].to_sparse()
            # edges_false = data['edges_false']
            # # print(edges_false,'false edge ')
            # edges_false_dict ={'train':{}}
            # split='train'

            # data['false_dict']=edges_false_dict[split]
            # print('new item')
            # print(data['adj_mat'])
            # print(edges_false.shape,'fasle')
            # print(data['adj_mat'].shape)
            batch_size=data['features'].shape[0]
            embeddings_list=[]
            data_list=[]
            # print(data,'DATA')
            # print(data['features'].shape,'DATA SHAPE!!')
            # ssksk

            for i in range(batch_size):
                embeddings,data_i=run_data_single(data, i,model)
                # _,data_i=run_data_single(data, i,model,skip_embedding=True)
                embeddings_list.append(embeddings)
                data_list.append(data_i)
            # print(len(data_list),'LEN DATA LIS')
            # optimizer.zero_grad()
            # embeddings_list,_= run_data_batch(data,model)
            # embeddings=embeddings_list[0]  ### just for saving?? get rid of this shit

            # run_data_batch, return stacked embedding (ie. embedding_list and processed data ie. data_list)


            train_metrics=model.compute_metrics_multiple(embeddings_list,data_list,'train')
            train_metrics['loss'].backward()


            if model.c<.1:
                print('Low model:')
                print(model.c)
            if model.c<.02:
                print('CRITICALLY LOW MODEL!!')
                print(model.c)
                model.c=torch.clip(model.c,0.02)


            # print(train_metrics['loss'],'BIG LOSS')
            # batch_losses=torch.zeros((batch_size))
            # embeddings = model.encode(data['features'].to(args.device), data['adj_mat'].to(args.device))
            # print(embeddings.shape,embeddings.type(),'output!!!')
            
            # train_metrics = model.compute_metrics(embeddings, data, 'train')
            # train_metrics=model.compute_metrics_multiple([embeddings],[data],'train')

            ##update epoch metrics--- 
            # model.update_epoch_stats(embeddings, data, 'train') 
            # if not use_batch:
            #     try:
            #         train_metrics['loss'].backward()
            #     except:
            #         model.compute_metrics(embeddings, data, 'train',verbose=True)
            # else:
            #     print(batch_loss,'batch!')
            #     if batch_count==0:
            #         batch_loss=train_metrics['loss']
            #     else:
            #         batch_loss+=train_metrics['loss']
                
            #     print(batch_count,'count')
            #     batch_losses[batch_count]=train_metrics['loss']
            #     print(batch_losses,'losses')

            #     batch_count+=1
                
            #     print(batch_count,batch_size,'bathc count')
            #     if batch_count==batch_size:
            #         total_batch_loss=torch.mean(batch_losses)
            #         # total_batch_loss=batch_loss/float(batch_size)
            #         print(total_batch_loss,'TOTS LOSS')

            #         # total_batch_loss.backward(retain_graph=True)
            #         batch_count=0
            #         batch_loss=0
            #         batch_losses=torch.zeros((batch_size))


            # loss.backward(retain_graph=True)
            if args.grad_clip is not None:
                max_norm = float(args.grad_clip)
                all_params = list(model.parameters())
                for param in all_params:
                    torch.nn.utils.clip_grad_norm_(param, max_norm)
            optimizer.step()
            # optimizer.zero_grad()
            # lr_scheduler.step() Naughty naughty
            # if self.hyperbolic and len(self.args.hyp_vars) != 0:
            #     hyperbolic_optimizer.step()
            # if self.args.is_regression and self.args.metric == "mae":
            #     loss = th.sqrt(loss)
            # ##check all this w/ batch situation/
            model.update_epoch_stats(train_metrics, 'train')         
            if model.epoch_stats['num_graphs'] % 23 <batch_size:
            # if True:
                avg_stats,stat_string = model.report_epoch_stats()
                print(stat_string)
                # break
                # break

                # break
            #     evaluate(epoch, dev_loader, 'dev', model)

            # if model.epoch_stats['num_graphs'] % 10 ==0:
            #     evaluate(epoch, dev_loader, 'dev', model)
            # break
        # print('broekn!')
        # continue
        # lr_scheduler.step()
        # if batch_loss>0:
            # batch_loss.backward()
        epoch_stats,stat_string = model.report_epoch_stats()
        # evaluate(epoch, dev_loader, 'dev', model)
        # if (epoch + 1) % args.log_freq == 0:
        #     logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1),
        #                            'lr: {}'.format(lr_scheduler.get_lr()[0]),
        #                            format_metrics(train_metrics, 'train'),
        #                            'time: {:.4f}s'.format(time.time() - t)
        #                            ]))
        # if (epoch + 1) % args.log_freq == 0:
        if True:
            logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1),
                                   'lr: {}'.format(lr_scheduler.get_last_lr()[0]),
                                   stat_string,
                                   'time: {:.4f}s'.format(time.time() - t)
                                   ]))

        # if (epoch + 1) % args.eval_freq == 0:
        if True:
            print('training epoch done in ', 'time: {:.4f}s'.format(time.time() - t))
            val_metrics,val_embeddings,val_string = evaluate(epoch, dev_loader, 'dev', model,freeze=True)
            test_metrics,test_embeddings,test_string = evaluate(epoch, test_loader, 'test', model,freeze=False)
            # print(val_metrics,val_embeddings ,'val')
            # print(test_metrics,test_embeddings,'test')
            # if (epoch + 1) % args.log_freq == 0:
            logging.info(" ".join(['Dev','Epoch: {:04d}'.format(epoch + 1), val_string]))
            logging.info(" ".join(['Test','Epoch: {:04d}'.format(epoch + 1), test_string]))
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

        # break

        # if self.args.is_regression and not self.early_stop.step(dev_loss, test_loss, epoch):        
        #     break
        # elif not self.args.is_regression and not self.early_stop.step(dev_acc, test_acc, epoch):
        #     break
        # if self.early_stop.is_improved or epoch==0: ### feels like this should be done in report_epoch_stats
        #     print('Saving new model epoch {}'.format(epoch))
        #     self.save_model()

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
    edges_false_dict ={'train':{}}
    split='train'

    data_i['false_dict']=edges_false_dict[split]
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
        for i, data in enumerate(data_loader):
            # for x, val in data.items():
                # if torch.is_tensor(data[x]):
                    # data[x] = data[x].to(model.args.device)


            # data['features']  = data['features'][0]
            # data['edges']  = data['edges'][0]
            # data['edges_false']  = data['edges_false'][0]
            # data['adj_mat']  =  data['adj_mat'].to_dense()[0].to_sparse()
            # data['adj_train_norm']  =  data['adj_mat']
            # data['labels']  = data['labels'][0]
            # edges_false = data['edges_false']
            # # print(edges_false,'false edge ')
            # split = 'train'
            # edges_false_dict={'train':{}}
            # for n1,n2 in edges_false:
            #     n1= n1.item()
            #     n2=n2.item()
            #     # print(n1,'AH')
            #     if n1 not in edges_false_dict[split]:
            #         edges_false_dict[split][n1]=set([])
            #     if n2 not in edges_false_dict[split]:
            #         edges_false_dict[split][n2]=set([])
            #     edges_false_dict[split][n2].add(n1)
            #     edges_false_dict[split][n1].add(n2)
            # data['false_dict']=edges_false_dict[split]

            # if data['graph_id'][0] in model.args.frech_B_dict:
            #     # print('reusable')
            #     model.args.frechet_B=model.args.frech_B_dict[data['graph_id'][0]]
            # else:
            #     model.args.frechet_B=frechet_B(data['adj_train_norm'])
                
            #     model.args.frech_B_dict[data['graph_id'][0]]=model.args.frechet_B

            # data['adj_mat']  =  data['adj_mat'].to_sparse()
            # print('eval')
            # print(data['features'].shape,'features  type 3')
            # print(data['adj_mat'].shape,'norm type 3')
            # print(len(data['edges_false']),'false edges')
            # if len(data['edges_false'])==0:
                # continue  ## one protein has no flarnking negative edge. just fully connected 
            data['adj_mat']  =  data['adj_mat'].to_dense()
            embeddings,data_i=run_data_single(data, 0,model,no_grad=False)
            # embeddings = model.encode(data['features'], data['adj_mat']) ### don't forget this still just gets one set of embeddings
            metrics = model.compute_metrics(embeddings, data_i, prefix)
            model.update_epoch_stats(metrics, prefix)
        epoch_stats,stat_string = model.report_epoch_stats()
    return epoch_stats,embeddings,stat_string


if __name__ == '__main__':
    args = parser.parse_args()
    train(args)
    # hyperbolicty_sam
