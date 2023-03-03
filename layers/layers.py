"""Euclidean layers."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


def get_dim_act(args):
    """
    Helper function to get dimension and activation at every layer.
    :param args:
    :return:
    """
    if not args.act:
        act = lambda x: x
    else:
        act = getattr(F, args.act)

    acts = [act] * (args.num_layers - 1)
    dims = [args.feat_dim] + ([args.dim] * (args.num_layers - 1))
    use_acts = [True]* (args.num_layers - 1)
    print(dims,'dims')
    if args.task in ['lp', 'rec']:
        dims += [args.dim]
        acts += [act]
        use_acts +=[True]

        n_curvatures = args.num_layers
    else:
        n_curvatures = args.num_layers - 1
    ### add our stuff
    if hasattr(args,'output_dim') and args.output_dim>0:
        dims[-1]=args.output_dim
    else:
        dims[-1]= args.dim

    if hasattr(args,'output_act'): ## if has arg, change to last to final function, regardless of whether we added extra dim or not
        if args.output_act not in ('None',None):
            acts[-1]= getattr(F, args.output_act) 
            use_acts[-1]=True
        else:
            acts[-1]=act
            use_acts[-1]=False
    else:
        acts[-1]=act
        use_acts[-1]=True


    return dims, acts,use_acts


class GraphConvolution(Module):
    """
    Simple GCN layer.
    """

    def __init__(self, in_features, out_features, dropout, act, use_bias,use_act):
        super(GraphConvolution, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.act = act
        self.use_act=use_act
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, input):
        x, adj = input

        # print(x,'XXX input')
        print(x.shape)
        print(torch.norm(x,dim=1),'input notm')

        hidden = self.linear.forward(x)
        # print(torch.norm(hidden,dim=1),'forward norm')
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        if adj.is_sparse:
            support = torch.spmm(adj, hidden)
        else:
            support = torch.mm(adj, hidden)
        if self.use_act:
            support = self.act(support)
        output = (support, adj)
        print(torch.norm(support,dim=1),'output norm')
        return output

    def extra_repr(self):
        return 'input_dim={}, output_dim={}'.format(
                self.in_features, self.out_features
        )


class Linear(Module):
    """
    Simple Linear layer with dropout.
    """

    def __init__(self, in_features, out_features, dropout, act, use_bias):
        super(Linear, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.act = act

    def forward(self, x):
        hidden = self.linear.forward(x)
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        out = self.act(hidden)
        return out


class FermiDiracDecoder(Module):
    """Fermi Dirac to compute edge probabilities based on distances."""

    def __init__(self, r, t):
        super(FermiDiracDecoder, self).__init__()
        # self.r = r
        # self.t = t
        # self.r = torch.tensor(float(r))
        # print
        self.r= torch.tensor(float(r))
        self.t = torch.tensor(float(t)) 
        self.r = nn.Parameter(torch.tensor(float(r)))
        self.t = nn.Parameter(torch.tensor(float(t)))
        self.freeze()            
        # self.weight = nn.Parameter(torch.ones(hidden_dim))
            # self.bias = nn.Parameter(torch.zeros(hidden_dim))

    def freeze(self):
        self.r.requires_grad=False
        self.t.requires_grad=False

    def unfreeze(self):
        self.r.requires_grad=True
        self.t.requires_grad=True
    def forward(self, dist):
        ### i believe these are square distances in practice
        try:
            max_clamp = 88
            probs = 1. / (torch.exp(  torch.clamp( ((dist - self.r) / self.t),max=max_clamp))+ 1.0)
            diff = (dist - self.r) ## somehow goes to zero faster with lower learning rate?
        except Exception as e:
            print('BAD PASS TO FermiDiracDecoder')
            print(dist,'prbably 0 or massive')
            print('probably from embedding that exceeds radius 1?')
            print((dist - self.r),'dist')
            print((torch.exp((dist - self.r) / self.t)))
            raise e
        return probs

