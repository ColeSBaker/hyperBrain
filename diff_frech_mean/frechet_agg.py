import torch
import torch.nn.functional as F
from diff_frech_mean.frechetmean.frechet import frechet_mean


def frechet_agg(x, adj, man,B=None):
    """
    Compute Frechet Graph Aggregation

    Args
    ----
        x (tensor): batched tensor of hyperbolic values. Batch size is size of adjacency matrix.
        adj (tensor): sparse coalesced adjacency matrix
        man (Manifold): hyperbolic manifold
    """


    indices = adj.coalesce().indices().transpose(-1, -2)
    n, d = x.shape
    if not B:
        B = frechet_B(adj)

    batched_tensor = []
    weight_tensor = []
    for i in range(n):
        si = indices[indices[:, 0] == i, -1]
        # print(x[si],'')
        batched_tensor.append(F.pad(x[si], (0, 0, 0, B - len(si))))
        # weight_tensor.append(F.pad(torch.ones_like(si), (0, B - len(si))))
        weight_tensor.append(F.pad(adj.coalesce().values()[si], (0, B - len(si))))

    batched_tensor = torch.stack(batched_tensor)
    # weight_tensor = torch.stack(weight_tensor) ## original.. but throughs error with gradients.
    weight_tensor = torch.stack(weight_tensor).float() 
    frech_mean=frechet_mean(batched_tensor, man, w=weight_tensor)
    # print(frech_mean,'FRECH MEAN')
    return frech_mean


def frechet_B(adj):

    indices = adj.coalesce().indices().transpose(-1, -2)
    degs=torch.count_nonzero(adj.to_dense(),dim=1)
    B=max(degs).int()
    ## easy precalc for transductive, barely any more effort for inductive
    # B = max([sum(indices[:, 0] == i) for i in range(n)]) ### we can probably calculate this?  ### highest degree?.... calculate one for large graphs
    print(B,'B')
    return B
