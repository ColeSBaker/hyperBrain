import torch
import torch.nn as nn
import matho

from frechetmean.manifolds import Poincare, Lorentz
from frechetmean.frechet import frechet_mean


class RiemannianBatchNorm(nn.Module):
    def __init__(self, dim, manifold):
        super(RiemannianBatchNorm, self).__init__()
        self.man = manifold

        self.mean = nn.Parameter(self.man.zero_tan(self.man.dim_to_sh(dim)))
        self.var = nn.Parameter(torch.tensor(1.0))

        # statistics
        self.running_mean = None
        self.running_var = None
        self.updates = 0

    def forward(self, x, training=True, momentum=0.9):
        num_nodes = tensor.shape[0]
        on_manifold = self.man.exp0(self.mean)

        # frechet mean, use iterative and don't batch (only need to compute one mean)
        print(x.shape,'x shape')
        print(x.T.shape,'t shape')
        input_mean = frechet_mean(x, self.man) ## mean per dimension, across node... fromo 
        input_var = self.man.frechet_variance(x, input_mean)

        print(input_mean.shape,'MEAN SHAPE')
        print(input_mean.shape,'VAR SHAPE')
        
        input_mean = frechet_mean(x.T, self.man) ## mean per dimension, across node... fromo 
        input_var = self.man.frechet_variance(x.T, input_mean)

        print(input_mean.shape,'MEAN SHAPE')
        print(input_mean.shape,'VAR SHAPE')
        # transport input from current mean to learned mean
        input_logm = self.man.transp(
            input_mean,
            on_manifold,
            self.man.log(input_mean, x),
        )

        # re-scaling
        input_logm = (self.var / (input_var + 1e-6)).sqrt() * input_logm

        # project back
        output = self.man.exp(on_manifold.unsqueeze(-2), input_logm)

        self.updates += 1
        if self.running_mean is None:
            self.running_mean = input_mean
        else:
            self.running_mean = self.man.exp(
                self.running_mean,
                (1 - momentum) * self.man.log(self.running_mean, input_mean)
            )
        if self.running_var is None:
            self.running_var = input_var
        else:
            self.running_var = (
                1 - 1 / self.updates
            ) * self.running_var + input_var / self.updates


        num_nodes = tensor.shape[0]
        # batch_list = graph.batch_num_nodes
        batch_list = [num_nodes]
        batch_size = len(batch_list)
        batch_list = torch.Tensor(batch_list).long().to(tensor.device)
        batch_index = torch.arange(batch_size).to(tensor.device).repeat_interleave(batch_list)
        batch_index = batch_index.view((-1,) + (1,) * (tensor.dim() - 1)).expand_as(tensor)

        print(batch_index,'BATCH IT')
        print(tensor.shape,'tensor shaps')
        print(*tensor.shape[1:])
        mean = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        mean = mean.scatter_add_(0, batch_index, tensor)
        print(mean,'calc1')
        print(batch_list,'list it')
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0) ### should give 
        print(mean.shape,'MEAN')
        print(torch.sum(tensor,dim=1),'actual')
        print(mean,'calc2')
        sub = tensor - mean * self.mean_scale

        print(self.mean_scale,'mean scale')
        std = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        std = std.scatter_add_(0, batch_index, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-6).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)
        return output
