import torch
import torch.nn as nn
from diff_frech_mean.frechetmean import Poincare as Frechet_Poincare
from diff_frech_mean.frechetmean.frechet import frechet_mean
from diff_frech_mean.riemannian_batch_norm import RiemannianBatchNorm
# from diff_frech_mean.frechet_agg import frechet_mean
class Norm(nn.Module):

    def __init__(self, norm_type, args,hidden_dim=64, print_info=None):
        super(Norm, self).__init__()
        # assert norm_type in ['bn', 'ln', 'gn', None]
        self.norm = None
        self.norm_type=norm_type
        self.print_info = print_info
        self.hidden_dim=hidden_dim
        # self.manifold=manifold
        self.c=args.c
        self.args=args
        self.frech_man = Frechet_Poincare(-self.c)

        if norm_type == 'bn':
            # print(norm_type,'BATCH NORM IN USE')
            print(hidden_dim,'HIDDEN DIM!')
            
            self.norm = nn.BatchNorm1d(hidden_dim)
            self.norm_hyp=False ### does normalization take place in hyperbolic space
        elif norm_type == 'gn':
            self.norm = norm_type
            self.weight = nn.Parameter(torch.ones(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))

            self.mean_scale = nn.Parameter(torch.ones(hidden_dim))
            self.norm_hyp=False
        elif norm_type=='rbn':

            if self.c==None:
                raise Exception('CANNOT HAVE TUNABLE C FOR BATCH NORM!!! yet....')
            self.norm=RiemannianBatchNorm(hidden_dim, self.frech_man,self.args) ## args added by cole
            # self.norm=RiemannianBatchNorm(hidden_dim, self.frech_man)
            self.norm_hyp=True


        

    # def forward(self, graph, tensor, print_=False):
    def forward(self, tensor, print_=False):
        #### mmmmmmmm you could be onto something here with adding the frechet mean to this...
        #### will take some extra thinking cap work... 
        #### gonna have to understand frechet means a little better ;)


        # if self.norm is not None and type(self.norm) != str:
        if self.norm_type=='bn':
            # batch norm
            print(tensor.shape,'batch norm in shape')
            # print(self.norm(tensor.))
            swapped=False
            if self.hidden_dim==tensor.shape[1]:
                print('matches first')
            elif self.hidden_dim==tensor.shape[2]:
                print('matches second')
                print('need to change')
                swapped=True

            tensor=torch.swapaxes(tensor, 1, 2) 
            norm_out=self.norm(tensor)
            print(norm_out.shape,'batch norm out shape')
            if swapped:
                print('swapping out')
                norm_out=torch.swapaxes(norm_out, 1, 2)
            print(norm_out.shape,'final batch out')

            return norm_out
        elif self.norm is None:
            return tensor
        elif self.norm_type=='rbn':
            return self.norm(tensor)

        elif self.norm_type=='gn':
            num_nodes = tensor.shape[0]
            # batch_list = graph.batch_num_nodes
            batch_list = [num_nodes]
            batch_size = len(batch_list)
            batch_list = torch.Tensor(batch_list).long().to(tensor.device)
            batch_index = torch.arange(batch_size).to(tensor.device).repeat_interleave(batch_list)
            batch_index = batch_index.view((-1,) + (1,) * (tensor.dim() - 1)).expand_as(tensor)
            mean = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
            mean = mean.scatter_add_(0, batch_index, tensor)
            mean = (mean.T / batch_list).T
            mean = mean.repeat_interleave(batch_list, dim=0) ### should give 
            sub = tensor - mean * self.mean_scale

            std = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
            std = std.scatter_add_(0, batch_index, sub.pow(2))
            std = ((std.T / batch_list).T + 1e-6).sqrt()
            std = std.repeat_interleave(batch_list, dim=0)
            # print(std,'STANDARD')
            return self.weight * sub / std + self.bias  ### obviously gonna be tricky dealing with shit like this


class RiemannianGroupNorm(nn.Module):
    def __init__(self, dim, manifold=Frechet_Poincare()):
        super(RiemannianGroupNorm, self).__init__()
        self.man = manifold

        self.mean = nn.Parameter(self.man.zero_tan(self.man.dim_to_sh(dim)))
        self.var = nn.Parameter(torch.tensor(1.0))

        # statistics
        self.running_mean = None
        self.running_var = None
        self.updates = 0

    def forward(self, x, training=True, momentum=0.9,frechet_B=None):
        ### this is working as is because we don't have batches??-- but re
        num_nodes = x.shape[0]
        on_manifold = self.man.exp0(self.mean)

        # frechet mean, use iterative and don't batch (only need to compute one mean)
        # print(x.shape,'x shape')
        # print(x.T.shape,'t shape')
        input_mean = frechet_mean(x, self.man,frechet_B) ## mean per dimension, across node... fromo 
        input_var = self.man.frechet_variance(x, input_mean)
        # print(input_mean.shape,'MEAN SHAPE')
        # print(input_mean.shape,'VAR SHAPE')
        # transport input from current mean to learned mean
        input_logm = self.man.transp(
            input_mean,
            on_manifold,
            self.man.log(input_mean, x),
        )
        
        # re-scaling
        # print(input_logm.shape,'logm')
        input_logm = (self.var / (input_var + 1e-6)).sqrt() * input_logm

        # project back
        output = self.man.exp(on_manifold.unsqueeze(-2), input_logm)

        return output
