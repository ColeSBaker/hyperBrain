"""Graph encoders."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import manifolds
from layers.att_layers import GraphAttentionLayer
import layers.hyp_layers as hyp_layers
from layers.layers import GraphConvolution, Linear, get_dim_act
import utils.math_utils as pmath

from Norm.norm import Norm,RiemannianGroupNorm ## added by Cole
from Norm.norm import RiemannianGroupNorm as RiemannianBatchNorm
class Encoder(nn.Module):
    """
    Encoder abstract class.
    """

    def __init__(self, c):
        super(Encoder, self).__init__()
        self.c = c

    def encode(self, x, adj):
        if self.encode_graph:
            input = (x, adj)
            output, _ = self.layers.forward(input)
        else:
            output = self.layers.forward(x)
        return output

class MLP(Encoder):
    """
    Multi-layer perceptron.
    """

    def __init__(self, c, args):
        super(MLP, self).__init__(c)
        assert args.num_layers > 0
        dims, acts = get_dim_act(args)
        layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            layers.append(Linear(in_dim, out_dim, args.dropout, act, args.bias))
        self.layers = nn.Sequential(*layers)
        self.encode_graph = False


class HNN(Encoder):
    """
    Hyperbolic Neural Networks.
    """

    def __init__(self, c, args):
        super(HNN, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        assert args.num_layers > 1
        # dims, acts, _ = hyp_layers.get_dim_act_curv(args)
        dims, acts, self.curvatures,use_acts = hyp_layers.get_dim_act_curv(args)

        hnn_layers = []
        print(dims,'dims')
        print(acts,'acts')
        print(use_acts,'uise acts')
        use_norm = True if hasattr(args,'use_norm') and args.use_norm>0 else False
        norm_type='bn'
        self.curvatures.append(self.c)
        for i in range(len(dims) - 1): ### implement on here ?
            print(norm_type)
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            norm = Norm(norm_type, out_dim) if ((i<len(dims)-2) and (use_norm)) else None
            # norm = Norm('gn', out_dim) if i<len(dims)-2 else None ## can't be batching output@
            act = acts[i]
            use_act=use_acts[i]
            hnn_layers.append(
                    hyp_layers.HNNLayer(
                            self.manifold, in_dim, out_dim, self.c, args.dropout, act, args.bias,use_act=use_act,norm=norm,args=args)
            )
        self.layers = nn.Sequential(*hnn_layers)
        self.encode_graph = False

    def encode(self, x, adj):
        x_hyp = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(x, self.c), c=self.c), c=self.c)
        return super(HNN, self).encode(x_hyp, adj)

class GCN(Encoder):
    """
    Graph Convolution Networks.
    """

    def __init__(self, c, args):
        super(GCN, self).__init__(c)
        assert args.num_layers > 0
        dims, acts,use_acts = get_dim_act(args)
        gc_layers = []
        print(dims,'dims')
        print(acts,'acts')
        print(use_acts,'uise acts')
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            use_act=use_acts[i]
            gc_layers.append(GraphConvolution(in_dim, out_dim, args.dropout, act, args.bias,use_act=use_act))
        self.layers = nn.Sequential(*gc_layers)
        self.encode_graph = True


class HGCN(Encoder):
    """
    Hyperbolic-GCN.
    """

    def __init__(self, c, args):
        super(HGCN, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        assert args.num_layers > 1
        dims, acts, self.curvatures,use_acts = hyp_layers.get_dim_act_curv(args)
        print(dims,'dims')
        print(acts,'acts')
        print(use_acts,'uise acts')
        self.curvatures.append(self.c)
        hgc_layers = []
        # Norm(norm_type, self.args.embed_size)
        use_frechet_agg = True if (hasattr(args,'use_frechet_agg') and args.use_frechet_agg>0) else False
        use_output_agg = args.output_agg if (hasattr(args,'output_agg')) else True 
        # use_frechet_agg=False
        use_norm = True if hasattr(args,'use_norm') and args.use_norm>0 else False
        hyp_act = True if hasattr(args,'hyp_act') and args.hyp_act else False

        if hyp_act:
            assert args.act not in ('leaky_relu', 'elu', 'selu')
        # norm_type='bn'
        norm_type='gn'
        # norm_type='rbn'
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            print(norm_type,'NORMAL TYPE')
            norm = Norm(norm_type, args,out_dim) if ((i<len(dims)-2) and (use_norm)) else None ## can't be batching output@
            use_agg = True if ((i<len(dims)-2) or (use_output_agg)) else False ## can't be batching output@

            print(use_agg,'USE AGG')

            # norm=None
            act = acts[i]

            use_act=use_acts[i]
            hgc_layers.append(
                    hyp_layers.HyperbolicGraphConvolution(
                            self.manifold, in_dim, out_dim, c_in, c_out,
                             args.dropout, act, args.bias, args.use_att, args.local_agg,use_frechet_agg,use_act=use_act,norm=norm,args=args,use_agg=use_agg
                    )
            )
        # sdd
        # jsjsj
        self.layers_list=hgc_layers
        self.layers = nn.Sequential(*hgc_layers)
        self.encode_graph = True

    def encode(self, x, adj):
        x_tan = self.manifold.proj_tan0(x, self.curvatures[0])
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])
        return super(HGCN, self).encode(x_hyp, adj)


class GAT(Encoder):
    """
    Graph Attention Networks.
    """

    def __init__(self, c, args):
        super(GAT, self).__init__(c)
        assert args.num_layers > 0
        dims, acts = get_dim_act(args)
        gat_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            assert dims[i + 1] % args.n_heads == 0
            out_dim = dims[i + 1] // args.n_heads
            concat = True
            gat_layers.append(
                    GraphAttentionLayer(in_dim, out_dim, args.dropout, act, args.alpha, args.n_heads, concat))
        self.layers = nn.Sequential(*gat_layers)
        self.encode_graph = True


class Shallow(Encoder):
    """
    Shallow Embedding method.
    Learns embeddings or loads pretrained embeddings and uses an MLP for classification.
    """

    def __init__(self, c, args):
        super(Shallow, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        self.use_feats = args.use_feats
        weights = torch.Tensor(args.n_nodes, args.dim)
        if not args.pretrained_embeddings:
            weights = self.manifold.init_weights(weights, self.c)
            trainable = True
        else:
            weights = torch.Tensor(np.load(args.pretrained_embeddings))
            assert weights.shape[0] == args.n_nodes, "The embeddings you passed seem to be for another dataset."
            trainable = False
        self.lt = manifolds.ManifoldParameter(weights, trainable, self.manifold, self.c)
        self.all_nodes = torch.LongTensor(list(range(args.n_nodes)))
        layers = []
        if args.pretrained_embeddings is not None and args.num_layers > 0:
            # MLP layers after pre-trained embeddings
            dims, acts = get_dim_act(args)
            if self.use_feats:
                dims[0] = args.feat_dim + weights.shape[1]
            else:
                dims[0] = weights.shape[1]
            for i in range(len(dims) - 1):
                in_dim, out_dim = dims[i], dims[i + 1]
                act = acts[i]
                layers.append(Linear(in_dim, out_dim, args.dropout, act, args.bias))
        self.layers = nn.Sequential(*layers)
        self.encode_graph = False

    def encode(self, x, adj):
        h = self.lt[self.all_nodes, :]
        if self.use_feats:
            h = torch.cat((h, x), 1)
        return super(Shallow, self).encode(h, adj)