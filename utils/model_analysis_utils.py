
import os
import pandas as pd
import numpy as np
import torch
from utils.dataloader_utils import load_data_graphalg,load_dataset,load_meg_averages
from matplotlib import pyplot as plt
def save_embeddings_all(trial_dir,first=0,last=-1,save_special=False,overwrite=False):
    ### gotta be automatic nums!

    #     folders=[f for f in os.listdir(trial_dir) if os.path.isdir(f)]
    folders=[f for f in os.listdir(trial_dir) if ('.pt' not in f) and ('.csv' not in f)]
    print('FOLDERS')
    print(folders)
    #     sjlh
    # first=True
    folders=folders[first:]
    if last>0:
        folders=folders[:last]
    for f in folders:

        
        model_dir = os.path.join(trial_dir,str(f))
        if not os.path.exists(os.path.join(model_dir,'model.pt')):
            continue
        print(model_dir)
        save_embeddings(model_dir,save_special=save_special,overwrite=overwrite)
        #         sdd


def save_embeddings(model_dir,save_special=False,overwrite=False):
    if not os.path.join(model_dir,'final_model.pth'):
        print('model not finished training, skipping embeddings')
        return
    save_dir = model_dir
    scan_info_outpath = os.path.join(save_dir,'scan_info.csv')
    embeddings_outdir = os.path.join(save_dir,'embeddings')

    if os.path.exists(scan_info_outpath) and os.path.exists(embeddings_outdir) :
        print('Embeddings already exists in {}'.format(scan_info_outpath))
        if overwrite:
            print('Overwriting Embeddings!')
        else:
            return

    model_name='model'
    model_path = os.path.join(model_dir,"{}.pt".format(model_name))  ### args should be saved with the model
    model=load_model(model_dir)



    if save_special:
        embed_average_all_splits(model,save_dir=model_dir)
    # model.get_d
    setattr(model.args,'refresh_data',1)
    evaluate_inductive(model,['train','dev','test'],save_embeddings=True,save_dir=save_dir)
    
def load_model(model_dir):
    model_name ='model'
    model_path = os.path.join(model_dir,"{}.pt".format(model_name))  ### args should be saved with the model
    config_path = os.path.join(model_dir,'config.json')  ### args should be saved with the model
    # print(config_path,'CONFI')
    out_embedding_path = model_dir
    model = torch.load(model_path,map_location=torch.device('cpu'))
    print(model.args)
    try:
        model.args
    except:
        print(config_path,'CONFIGURATION')
        print(os.path.exists(config_path),'DO WE EXIST')
        with open(config_path, 'r') as j:
             model_config = json.loads(j.read())

        print('not a real model')
        print(model_config,'CRAZY MODEL eh>')
        set_params = default_config['set_params']

        args = parse_default_args(set_params) 
        Model = LPModel
        for k,v in model_config.items():
            print(k,v)
            if hasattr(args,k):
                setattr(args,k,v)
        print(args,'ARGUING')
        args.feat_dim = args.num_feature
        # args.r
        model = Model(args).to(args.device)

    return model

def create_embeddings(model,data_loader_dicts,scan_info_outpath,embeddings_outdir):
    if os.path.exists(embeddings_outdir):

        os.makedirs(embeddings_outdir,exist_ok=True)

    node_id_col='RoiID'
    dim_order = ['x','y','z','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o']
    dim_cols =dim_order[:model.args.output_dim]
    embedding_cols= [node_id_col]+dim_cols ### do we need anything else?

    scan_info_df=pd.DataFrame(columns=['graph_id','train','test','val','label','save_location'])
    all_embeddings=[]
    scan_data=[]
    seen_nodes=set([])
    # need embedding columns
    with torch.no_grad():
        model.eval()
        embeddings_df= pd.DataFrame(columns=embedding_cols) ##['x']
        # model.reset_epoch_stats(1000, split)
        first=True
        running_mAP=0
        running_mean_rank=0
        count=0

        for split,data_loader in data_loader_dicts:
            # data_loader= split_to_loader[split]
            model.reset_epoch_stats(1000, 'start')
            for i, data in enumerate(data_loader):
                for x, val in data.items():
                    if torch.is_tensor(data[x]):
                        data[x] = data[x].to(model.args.device)
                data['features']  = data['features'][0]
                print(data['features'].shape,'SHAPE!!')
                data['edges']  = data['edges'][0]
                data['edges_false']  = data['edges_false'][0]
                data['adj_mat']  =  data['adj_mat'].to_dense()[0].to_sparse()
                use_node_list = 'node_to_indx' in data.keys()

                if model.args.double_precision>0:
                    data['features']=data['features'].double()
                else:
                    print(model.args.double_precision,'PREC AGAIN')
                embeddings = model.encode(data['features'], data['adj_mat'] ) #### to add into data-> data['nodeIDs']. may seem like overkill but I won't let us get fucked up again
                pos_scores = model.decode(embeddings, data['edges'])
                neg_scores = model.decode(embeddings, data['edges_false'])

                node_num=data['labels'].shape[1]
                if not use_node_list:
                    embeddings_df[dim_cols]=embeddings[:node_num,:].detach().numpy()
                    embeddings_df[node_id_col]= [str(i)+"_r" for i in range(node_num)]
                else:
                    nodes=[]
                    indxs = []
                    for n,indy in data['node_to_indx'].items():
                        nodes.append(n)
                        indxs.append(indy.item())
                    embeddings_df[dim_cols]=embeddings[indxs,:].detach().numpy()
                    embeddings_df[node_id_col]= nodes

                graph_id = data['graph_id'][0]
                if graph_id not in seen_nodes:
                    seen_nodes.add(graph_id)
                else:
                    continue
                output_file= str(graph_id)+"_embeddings.csv"
                output_path = os.path.join(embeddings_outdir,output_file)
                # add all here or something?
                train=1 if split=='train' else 0
                test=1 if split=='test' else 0
                val=1 if split=='val' else 0
                label=data['labels'][0][0].item()
                save_path=output_path
            
                scan_info_row = [graph_id,train,test,val,label,save_path]
                scan_data.append(scan_info_row)

                if embeddings_outdir=='':
                    raise Exception("CANNOT SAVE TO NO PATH")
                print(output_path,'SAVE EMBEDDINGS!')

                embeddings_df.to_csv(output_path,index=False)
                all_embeddings.append(embeddings_df)

    scan_info_df= pd.DataFrame(data=scan_data ,columns=['graph_id','train','test','val','label','save_location'])
    scan_info_df=scan_info_df.set_index('graph_id',drop=False)
    print(scan_info_outpath)
    scan_info_df.to_csv(scan_info_outpath)
    return scan_info_df, all_embeddings
def evaluate_inductive(model,splits=['train','test','dev'],save_embeddings=True,save_dir=''):
	"""
	why cant this happen automatically after training?
	like within the same training script and everything so that we don't have to reload data etc?

	start, small.
	functionalize it here, then see how many args you need, how much is precomputed!
	"""
	model.args.device='cpu'
	# model.args=change_threshold(model.args,.35)
	train_loader, dev_loader, test_loader =load_dataset(model.args,'graph')
	node_id_col='RoiID'
	split_to_loader={'test':test_loader,'train':train_loader,'dev':dev_loader}
	# data_loaders= [split_to_loader[s] for s in splits]
	data_loaders= [(k,v) for k,v in split_to_loader.items() if k in splits]
	print(data_loaders)
	for k in data_loaders:
		print(k,'ss')
	model.eval()

	# scan_info_df=pd.DataFrame(columns=['graph_id','train','test','val','label','save_location']) ### graph ids, train/test/val, label, SAVE LOCATION
	scan_info_outpath = os.path.join(save_dir,'scan_info.csv')
	embeddings_outdir = os.path.join(save_dir,'embeddings')

	if save_embeddings and (embeddings_outdir==''):
	    raise Exception("Cannot save to no path")
	if save_embeddings and not os.path.exists(embeddings_outdir):
	    os.makedirs(embeddings_outdir)

	create_embeddings(model,data_loaders,scan_info_outpath,embeddings_outdir)



def create_res_df_study(study_dir,save_df=True,train_only=True):
### loss, roc, epoch #, gamma, final lr, 
    columns=['dir','loss','roc','gamma','c','lr-reduce-freq','lr','best_epoch']
    data_dict={c:[] for c in columns}
    # model_list=[os.path.join(study_dir,m) for m in os.listdir(study_dir) if ('.pt' not in m) and ('.csv' not in m)]
    model_list=[os.path.join(study_dir,m) for m in os.listdir(study_dir) if ('.pth' not in m) and ('.csv' not in m) and ('.json' not in m)]
    model_name ='model'
    random_num=str(np.random.randint(1000))
    for m in model_list:
        model_path = os.path.join(m,"{}.pt".format(model_name))  ### args should be saved with the model

        # model = th.load(model_path,map_location=torch.device('cpu'))
        model = load_model(m)
        loss= plot_loss(model,stat_to_plot='loss',show=True,train_only=False)
        roc= plot_loss(model,stat_to_plot='roc',show=False,train_only=train_only)
        data_dict['c'].append(model.c.item())
        data_dict['loss'].append(loss)
        data_dict['roc'].append(roc)
        data_dict['gamma'].append(model.args.gamma)
        data_dict['lr'].append(model.args.lr)
        data_dict['best_epoch'].append('')
        # data_dict['lr-reduce-freq'].append(getattr(model.args,'lr-reduce-freq'))
        data_dict['lr-reduce-freq'].append(model.args.lr_reduce_freq)
        data_dict['dir'].append(m)
    outdir=os.path.join(study_dir,'result_df'+random_num+'.csv')
    print(data_dict,'DATA DICT')
    res_df=pd.DataFrame.from_dict(data_dict)
    if save_df:
        res_df.to_csv(outdir)
    return res_df
        # data_dict['final_lr'].append(model.args.gamma)
        
def plot_loss_dir(model_list,stat_to_plot='roc',train_only=False):
    model_name ='model'
    print(model_list,'MODELS')

    
    for m in model_list:
        # if ''
        
        model_path = os.path.join(m,"{}.pt".format(model_name))  ### args should be saved with the model
        print(m)
        # model_list = os.path.join(m,)
        model = th.load(model_path,map_location=torch.device('cpu'))
        plot_loss(model,stat_to_plot=stat_to_plot,show=False,train_only=train_only)

    plt.legend()
    plt.show()



def plot_loss(model,stat_to_plot='roc',show=True,train_only=False):
    ### consider setting train back one.
    """
    to do:
        argmax of epoch
        plotting models seperately
        plotting models on avg

    """

    if stat_to_plot in ('roc'):
        optimize = 'max'
    else:
        optimize = 'min'
    loss_type = 'MSE' if model.args.use_weighted_loss else 'BCE'
    str_stat = stat_to_plot if stat_to_plot!= 'loss' else loss_type
    stats=model.metrics_tracker

    print(stats['train'])
    train_losses=[t[stat_to_plot] for t in stats['train']]
    dev_losses=[t[stat_to_plot] for t in stats['dev']]
    best_train = min(train_losses) if optimize=='min' else max(train_losses)
    best_dev = min(dev_losses) if optimize=='min' else max(dev_losses)
    str_dev=str(best_dev)[:5]
    str_train=str(best_train)[:5]



    plt.plot(train_losses,label='train')
    if not train_only:
            plt.plot(dev_losses,label='validation')

    if show:
        plt.title(stat_to_plot+'\n train: '+str(str_train)+'\n valid: '+str(str_dev))
        plt.legend()
        plt.show()
    return best_train,best_dev