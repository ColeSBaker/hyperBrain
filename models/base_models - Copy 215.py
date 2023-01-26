"""Data utils functions for pre-processing and data loading."""
import os
import pickle as pkl
import sys
# from config_debug import parser
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd
from scipy.linalg import block_diag
from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, utils ## causing naming issues


# def collate_fn()
def load_data(args, datapath):
    if args.task == 'nc':
        data = load_data_nc(args.dataset, args.use_feats, datapath, args.split_seed)
        #### adding this in
        adj = data['adj_train']
        adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = mask_edges(
                adj, args.val_prop, args.test_prop, args.split_seed
        )
        data['adj_train'] = adj_train
        data['train_edges'], data['train_edges_false'] = train_edges, train_edges_false
        data['val_edges'], data['val_edges_false'] = val_edges, val_edges_false
        data['test_edges'], data['test_edges_false'] = test_edges, test_edges_false
    else:
        data = load_data_lp(args.dataset, args.use_feats, datapath)
        adj = data['adj_train']
        if args.task == 'lp':
            adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = mask_edges(
                    adj, args.val_prop, args.test_prop, args.split_seed,data['override_edges']
            )
            data['adj_train'] = adj_train
            data['train_edges'], data['train_edges_false'] = train_edges, train_edges_false
            data['val_edges'], data['val_edges_false'] = val_edges, val_edges_false
            data['test_edges'], data['test_edges_false'] = test_edges, test_edges_false
    data['adj_train_norm'], data['features'] = process(
            data['adj_train'], data['features'], args.normalize_adj, args.normalize_feats
    )
    if args.dataset == 'airport':
        data['features'] = augment(data['adj_train'], data['features'])
    return data

def create_graph(data):
    cols = ['edges','train_edges','val_edges','test_edges']
    cols = ['train_edges']
    node_num = data['features'].shape[0] ### make node feature
    # print(data['features'].shape,'data shape')
    # print(data.keys())
    # edge_num = len(graph['edges'])
    nx_graph =nx.Graph()
    nx_graph.add_nodes_from([i for i in range(node_num)])
    # print()/
    for e in cols:
        # print(data.keys(),e in data.keys(),'data')
        if e not in data.keys():

            continue
        edges = data[e]
        print(edges)
        print(len(edges))
        for e in edges:
            # print(e)
            # print(e[0].item())
            # print(e[1].data)
            if e[0].item()!=e[1].item():
                # print()
                nx_graph.add_edge(e[0].item(),e[1].item())
    return nx_graph


def load_data_inductive(args,datapath):
    if args.task == 'gc':
        data = load_data_gc_inductive(args.dataset, args.use_feats, datapath, args.split_seed)
    else:
        data = load_data_lp_inductive(args.dataset, args.use_feats, datapath, args.split_seed)

# ############### FEATURES PROCESSING ####################################


def process(adj, features, normalize_adj, normalize_feats):
    if sp.isspmatrix(features):
        features = np.array(features.todense())
    if normalize_feats:
        features = normalize(features)
    features = torch.Tensor(features)
    if normalize_adj:
        adj = normalize(adj + sp.eye(adj.shape[0]))
    # print(adj,'ADJ IN PROCESS')
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    # print(adj,'ADJ After PROCESS')
    return adj, features


def normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def augment(adj, features, normalize_feats=True):
    deg = np.squeeze(np.sum(adj, axis=0).astype(int))
    deg[deg > 5] = 5
    deg_onehot = torch.tensor(np.eye(6)[deg], dtype=torch.float).squeeze()
    const_f = torch.ones(features.size(0), 1)
    features = torch.cat((features, deg_onehot, const_f), dim=1)
    return features

def one_hot_vec(length, pos):
    vec = [0] * length
    vec[pos] = 1
    return vec

def one_hot_from_adj(adj):
    one_hot_matrx = [one_hot_vec(adj.shape[0]) for i in range(len(adj.shape[0]))]
    return one_hot_matrx



# ############### DATA SPLITS #####################################################


def mask_edges(adj, val_prop, test_prop, seed,override_edges=[]): ### do it the same as here, but with no val(?) put in synthetic? BUT WILL STILL SLOW IT DOWN

    np.random.seed(seed)  # get tp edges
    x, y = sp.triu(adj).nonzero()
    pos_edges = np.array(list(zip(x, y)))
    print(pos_edges.shape,'pos_edges')
    print(adj.shape,'ADJ SHAPE')
    np.random.shuffle(pos_edges)
    # get tn edges
    print(pos_edges.shape,"POSITIVE")
    if len(override_edges)>0:
        pos_edges=np.array(override_edges) ### as written, this will naively split emails from same two people
        print(pos_edges.shape,'OVERRIDDEN')
    x, y = sp.triu(sp.csr_matrix(1. - adj.toarray())).nonzero()

    neg_edges = np.array(list(zip(x, y)))
    print(neg_edges.shape,'neg_edges')
    np.random.shuffle(neg_edges)

    m_pos = len(pos_edges)
    n_val = int(m_pos * val_prop)
    n_test = int(m_pos * test_prop)
    val_edges, test_edges, train_edges = pos_edges[:n_val], pos_edges[n_val:n_test + n_val], pos_edges[n_test + n_val:]
    print(train_edges.shape)


    val_edges_false, test_edges_false = neg_edges[:n_val], neg_edges[n_val:n_test + n_val]
    print(train_edges.shape,'train edge shape')
    train_edges_false = np.concatenate([neg_edges, val_edges, test_edges], axis=0)
    adj_train = sp.csr_matrix((np.ones(train_edges.shape[0]), (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    print(adj_train.shape)
    print(adj_train.T.shape)
    adj_train = adj_train + adj_train.T

    return adj_train, torch.LongTensor(train_edges.astype(int)), torch.LongTensor(train_edges_false.astype(int)), torch.LongTensor(val_edges.astype(int)), \
           torch.LongTensor(val_edges_false.astype(int)), torch.LongTensor(test_edges.astype(int)), torch.LongTensor(
            test_edges_false.astype(int))  



def split_data(labels, val_prop, test_prop, seed):
    np.random.seed(seed)
    nb_nodes = labels.shape[0]
    all_idx = np.arange(nb_nodes)
    pos_idx = labels.nonzero()[0]
    neg_idx = (1. - labels).nonzero()[0]
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    pos_idx = pos_idx.tolist()
    neg_idx = neg_idx.tolist()
    nb_pos_neg = min(len(pos_idx), len(neg_idx))
    nb_val = round(val_prop * nb_pos_neg)
    nb_test = round(test_prop * nb_pos_neg)
    idx_val_pos, idx_test_pos, idx_train_pos = pos_idx[:nb_val], pos_idx[nb_val:nb_val + nb_test], pos_idx[
                                                                                                   nb_val + nb_test:]
    idx_val_neg, idx_test_neg, idx_train_neg = neg_idx[:nb_val], neg_idx[nb_val:nb_val + nb_test], neg_idx[
                                                                                                   nb_val + nb_test:]
    return idx_val_pos + idx_val_neg, idx_test_pos + idx_test_neg, idx_train_pos + idx_train_neg


def bin_feat(feat, bins):
    digitized = np.digitize(feat, bins)
    return digitized - digitized.min()


# ############### LINK PREDICTION DATA LOADERS ####################################


def load_data_lp(dataset, use_feats, data_path):
    override_edges=False
    if dataset in ['cora', 'pubmed']:
        adj, features = load_citation_data(dataset, use_feats, data_path)[:2]
    # elif dataset == 'disease_lp':
    elif dataset in ('disease_lp','disease_nc'):
        adj, features = load_synthetic_data(dataset, use_feats, data_path)[:2]
    elif dataset == 'airport':
        adj, features = load_data_airport(dataset, data_path, return_label=False)
    elif dataset=='MEG':
        adj,features = load_data_MEG(dataset, data_path, return_label=False)
    elif dataset=='enron':
        adj,features,pos_edges = load_data_enron(dataset, data_path)
        # override_edges=True
    else:
        raise FileNotFoundError('Dataset {} is not supported.'.format(dataset))
    if override_edges:
        data = {'adj_train': adj, 'features': features,'override_edges':pos_edges}
    else:
        data = {'adj_train': adj, 'features': features,'override_edges':[]}
    return data

def load_data_lp_inductive(dataset, use_feats, data_path):
    if dataset in ['graphalg']:
        train_loader,val_loader,test_loader = load_data_graphalg(dataset, data_path)
    else:
        raise FileNotFoundError('Dataset {} is not supported for inductive.'.format(dataset))
    return train_loader,val_loader,test_loader


# ############### NODE CLASSIFICATION DATA LOADERS ####################################


def load_data_nc(dataset, use_feats, data_path, split_seed):
    if dataset in ['cora', 'pubmed']:
        adj, features, labels, idx_train, idx_val, idx_test = load_citation_data(
            dataset, use_feats, data_path, split_seed
        )
    else:
        # if dataset == 'disease_nc':
        if dataset in ('disease_nc','disease_lp'):
            adj, features, labels = load_synthetic_data(dataset, use_feats, data_path)
            val_prop, test_prop = 0.10, 0.60
        elif dataset == 'airport':
            adj, features, labels = load_data_airport(dataset, data_path, return_label=True)
            val_prop, test_prop = 0.15, 0.15

        elif dataset == 'MEG':
            adj, features, labels = load_data_MEG(dataset, data_path, return_label=True)
            val_prop, test_prop = 0.15, 0.15            
        else:
            raise FileNotFoundError('Dataset {} is not supported.'.format(dataset))
        idx_val, idx_test, idx_train = split_data(labels, val_prop, test_prop, seed=split_seed)

    labels = torch.LongTensor(labels)
    data = {'adj_train': adj, 'features': features, 'labels': labels, 'idx_train': idx_train, 'idx_val': idx_val, 'idx_test': idx_test}
    return data

################ GRAPH CLASSIFICATION DATA LOADERS ####################################
def load_data_gc_inductive(dataset, use_feats, data_path, split_seed):
    if dataset in ['graphalg']:
        train_loader,val_loader,test_loader = load_data_graphalg(dataset, data_path)
    else:
        raise FileNotFoundError('Dataset {} is not supported for inductive.'.format(dataset))
    return train_loader,val_loader,test_loader

# ############### DATASETS ####################################


def load_citation_data(dataset_str, use_feats, data_path, split_seed=None):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(os.path.join(data_path, "ind.{}.{}".format(dataset_str, names[i])), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(os.path.join(data_path, "ind.{}.test.index".format(dataset_str)))
    test_idx_range = np.sort(test_idx_reorder)

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = np.argmax(labels, 1)

    idx_test = test_idx_range.tolist()
    idx_train = list(range(len(y)))
    idx_val = range(len(y), len(y) + 500)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    if not use_feats:
        features = sp.eye(adj.shape[0])
    return adj, features, labels, idx_train, idx_val, idx_test


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_synthetic_data(dataset_str, use_feats, data_path):
    object_to_idx = {}
    idx_counter = 0
    edges = []
    with open(os.path.join(data_path, "{}.edges.csv".format(dataset_str)), 'r') as f:
        all_edges = f.readlines()
    for line in all_edges:
        n1, n2 = line.rstrip().split(',')
        if n1 in object_to_idx:
            i = object_to_idx[n1]
        else:
            i = idx_counter
            object_to_idx[n1] = i
            idx_counter += 1
        if n2 in object_to_idx:
            j = object_to_idx[n2]
        else:
            j = idx_counter
            object_to_idx[n2] = j
            idx_counter += 1

        # print(n1,i,'n1 i')
        # print(n2,j,'n2 j')
        edges.append((i, j))
    adj = np.zeros((len(object_to_idx), len(object_to_idx)))
    for i, j in edges:
        adj[i, j] = 1.  # comment this line for directed adjacency matrix
        adj[j, i] = 1.
    if use_feats:
        features = sp.load_npz(os.path.join(data_path, "{}.feats.npz".format(dataset_str)))
    else:
        features = sp.eye(adj.shape[0])
    # print(adj.shape)
    # print(sp.csr_matrix(adj))
    # ssss
    labels = np.load(os.path.join(data_path, "{}.labels.npy".format(dataset_str)))
    return sp.csr_matrix(adj), features, labels

def load_data_enron(dataset_str,data_path,use_weights=0):
    with open(os.path.join(data_path,'emails.txt')) as f:
        lines = f.readlines()
    nx_graph = nx.Graph()
    all_edges=[]
    weight={}

        
    for l in lines:
        body = l.split("\n")[0]
        # print(body,'body')

        items = body.split(' ')
        # print(items)
        e=tuple(items[1:])
        all_edges.append(list(e))
        if e in weight:
            weight[e]=weight[e]+1
        else:
            weight[e]=1
    num_nodes = max(list(weight.keys()))
    # print(len(weight.keys()),num_nodes,'len keys vs max id')
    for n in range(len(num_nodes)):    
        nx_graph.add_node(n)
    for e,w in weight.items():
        if use_weights>0:
            nx_graph.add_edge(int(e[0]),int(e[1]),weight=float(w))
        else:
            nx_graph.add_edge(int(e[0]),int(e[1]),weight=float(1))

    adj = nx.adjacency_matrix(nx_graph)
    features=sp.csr_matrix(adj).tolil()
    # print(features.shape,"FEATH CSHAP")
    return adj,features,all_edges

def load_data_MEG(dataset_str, data_path,return_label=False):
    # num_brains,NUM_BANDS,Num_Metric,NUM_ROIS,NUM_ROIS
    BAND_TO_INDEX = {'theta':0,'alpha':1,'beta1':2,'beta2':3,'beta':4,'gamma':5}
    METRIC_TO_INDEX = {'plv':0,'ciplv':1}
    block = False  ## incase we want one big adj matrix
    patients_to_use= 3
    use_feats=True
    adj_threshold = .33
    band_adj = BAND_TO_INDEX['alpha']
    metric_adj = METRIC_TO_INDEX['plv']
    scan_indx = 4
    dataset = np.load(os.path.join(data_path, "{}.ROIs.npy".format(dataset_str)))
    clinical = pd.read_csv(os.path.join(data_path, "{}.clinical.csv".format(dataset_str)))
    num_rois = dataset.shape[4]

    if block:
        scans = dataset[:,band_adj,metric_adj]
        adj = block_diag(*[scans[i] for i in range(patients_to_use)])
    else:
        single_scan = dataset[scan_indx,band_adj,metric_adj]
        adj = np.where(single_scan>adj_threshold,1,0)
    if use_feats:
        band_feat = BAND_TO_INDEX['alpha']
        metric_feat = METRIC_TO_INDEX['ciplv']
        feats = dataset[scan_indx,band_feat,metric_feat]  ### need to do feats here?
    else:

        one_hot = np.identity(num_rois)
        if block:
            feats =np.vstack([one_hot for i in range(patients_to_use)])

            print(feats)
        else:
            feats=one_hot

    # print(nx.convert_matrix.from_numpy_matrix(adj))
    # print('after convert_matrix')
    adj = nx.adjacency_matrix(nx.convert_matrix.from_numpy_matrix(adj))
    # print(adj,'adhhhhj')
    # sdff
    if return_label:
        return sp.csr_matrix(adj),feats,_
    else:
        return sp.csr_matrix(adj),feats




def load_data_airport(dataset_str, data_path, return_label=False):
    graph = pkl.load(open(os.path.join(data_path, dataset_str + '.p'), 'rb'))
    adj = nx.adjacency_matrix(graph)
    # print(adj,'adj')
    features = np.array([graph.nodes[u]['feat'] for u in graph.nodes()])
    if return_label:
        label_idx = 4
        labels = features[:, label_idx]
        features = features[:, :label_idx]
        labels = bin_feat(labels, bins=[7.0/7, 8.0/7, 9.0/7])
        return sp.csr_matrix(adj), features, labels
    else:
        return sp.csr_matrix(adj), features

# def load_dataloaders(args):

