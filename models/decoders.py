"""Graph decoders."""
import manifolds
import torch.nn as nn
import torch.nn.functional as F

from layers.att_layers import GraphAttentionLayer
from layers.layers import GraphConvolution, Linear
import utils.math_utils as pmath

class Decoder(nn.Module):
    """
    Decoder abstract class for node classification tasks.
    """

    def __init__(self, c):
        super(Decoder, self).__init__()
        self.c = c

    def decode(self, x, adj):
        if self.decode_adj:
            input = (x, adj)
            probs, _ = self.cls.forward(input)
        else:
            probs = self.cls.forward(x)
        return probs


class GCNDecoder(Decoder):
    """
    Graph Convolution Decoder.
    """

    def __init__(self, c, args):
        super(GCNDecoder, self).__init__(c)
        act = lambda x: x
        self.cls = GraphConvolution(args.dim, args.n_classes, args.dropout, act, args.bias)
        self.decode_adj = True


class GATDecoder(Decoder):
    """
    Graph Attention Decoder.
    """

    def __init__(self, c, args):
        super(GATDecoder, self).__init__(c)
        self.cls = GraphAttentionLayer(args.dim, args.n_classes, args.dropout, F.elu, args.alpha, 1, True)
        self.decode_adj = True


class LinearDecoder(Decoder):
    """
    MLP Decoder for Hyperbolic/Euclidean node classification models.
    """

    def __init__(self, c, args):
        super(LinearDecoder, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        self.input_dim = args.dim if not hasattr(args,'output_dim') else args.output_dim 
        self.output_dim = args.n_classes
        self.bias = args.bias
        self.cls = Linear(self.input_dim, self.output_dim, args.dropout, pmath.identity, self.bias)
        # self.cls = Linear(self.input_dim, self.output_dim, args.dropout, lambda x: x, self.bias) pmath
        self.decode_adj = False

    def decode(self, x, adj):
        #### HOW DO WE NORMALIZE SO THAT THE NORM OF A VECTOR IS CLAMPED???
        # print(min(x[:,0]),max(x[:,0]),'X man')
        # print(min(x[:,1]),max(x[:,1]),'X man')
        # print(x.isnan().any(),'X nan')
        # print(self.c,'C MONEY')
        h = self.manifold.proj_tan0(self.manifold.logmap0(x, c=self.c), c=self.c)

        # print(min(h[:,0]),max(h[:,0]),'H man')
        # print(min(h[:,1]),max(h[:,1]),'H man')
        # print(h.isnan().any(),'H nan')
        # print('mapped')
        return super(LinearDecoder, self).decode(h, adj)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, c={}'.format(
                self.input_dim, self.output_dim, self.bias, self.c
        )


model2decoder = {
    'GCN': GCNDecoder,
    'GAT': GATDecoder,
    'HNN': LinearDecoder,
    'HGCN': LinearDecoder,
    'MLP': LinearDecoder,
    'Shallow': LinearDecoder,
}

