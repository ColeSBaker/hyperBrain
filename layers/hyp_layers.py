import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.module import Module

from layers.att_layers import DenseAtt
from Norm.norm import Norm,RiemannianGroupNorm ## added by Cole
from diff_frech_mean.frechetmean import Poincare as Frechet_Poincare
from diff_frech_mean.frechet_agg import frechet_agg
def get_dim_act_curv(args):
    """
    Helper function to get dimension and activation at every layer.
    :param args:bias
    :return:
    """
    if not args.act:
        # act = lambda x: x
        act=None
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

    print(dims,'dims after')
    if args.c is None:
        # create list of trainable curvature parameters
        curvatures = [nn.Parameter(torch.Tensor([1.])) for _ in range(n_curvatures)]
    else:
        # fixed curvature
        curvatures = [torch.tensor([args.c]) for _ in range(n_curvatures)]
        if not args.cuda == -1:
            curvatures = [curv.to(args.device) for curv in curvatures]
    return dims, acts, curvatures,use_acts


class HNNLayer(nn.Module):
    """
    Hyperbolic neural networks layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, act, use_bias, use_act=True,norm=None,args=None):
        super(HNNLayer, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c, dropout, use_bias)
        self.hyp_act = HypAct(manifold, c, c, act,norm=norm)
        self.use_act=use_act

    def forward(self, x):
        h = self.linear.forward(x)
        if self.use_act: ## careful w/ c = None, bc the activation projects from c in to c out
            h = self.hyp_act.forward(h)
        return h


class HyperbolicGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """



    def __init__(self, manifold, in_features, out_features, c_in, c_out, dropout, act, use_bias, use_att, local_agg,
        use_frechet_mean, use_act=True,norm=None,args=None,use_agg=True):
        super(HyperbolicGraphConvolution, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c_in, dropout, use_bias)
        # print*use_bias
        ### YOOOOOO implement the hyperbolic aggregation here!!!!!! 
        ### also... seems ideal for hyperbolic graphnorm!!!! 
        self.agg = HypAgg(manifold, c_in, out_features, dropout, use_att, local_agg,use_frechet_mean,use_agg=use_agg) 
        self.agg.args=args

        if use_act:
            self.hyp_act = HypAct(manifold, c_in, c_out, act,norm=norm)
        self.use_act=use_act
        # self.most_recent={'in':None,'out':None}
        # self.c_in=c_cin
        # self.
        # self.in_features

    def forward(self, input):
        x, adj = input
        # print(x.shape,'New layer!')
        # print(x.min(),x.max(),'forward, act=',self.use_act)
        # print(x.dtype,'insides')
        self.most_recent={}
        self.most_recent['in']=x
        h = self.linear.forward(x) ### it's all in the agg?
        h_singles=torch.zeros_like(h)
        if len(x.shape)>2:
            for i in range(x.shape[0]): ### size of batch
                h_i=self.linear.forward(x[i])
                h_singles[i]=h_i

        # h = self.agg.forward(h, adj)
            h=h_singles
        h_singles=torch.zeros_like(h)
        if len(x.shape)>2:
            for i in range(x.shape[0]): ### size of batch
                # h_i=self.linear.forward(x[i])
                h_i = self.agg.forward(h[i], adj[i])
                h_singles[i]=h_i
                # print(h_i,'single')
                # print(h[i],'fu;;')
                # print(h_i==h[i],'EQUALS')

            h=h_singles
        else:
            h = self.agg.forward(h, adj)
        h_singles=torch.zeros_like(h)

        # print(h.min(),h.max(),'after linear agg=',self.use_act)
        if self.use_act: ## careful w/ c = None, bc the activation projects from c in to c out
            # 
            if len(x.shape)>2:
                for i in range(x.shape[0]): ### size of batch
                    # h_i=self.linear.forward(x[i])
                    h_i=self.hyp_act.forward(h[i])
                    h_singles[i]=h_i
                    # print(h_i,'single')
                    # print(h[i],'fu;;')
                    # print(h_i==h[i],'EQUALS')
                h=h_singles
            else:
                h = self.hyp_act.forward(h)
            
            # print(h.min(),h.max(),'after activation act=',self.use_act)
        # else:
            # print("NO ACTIVATION WITH OUT REPRESENTATION")
        # print(x.min(),x.max(),'after agg ', c_in,c_out, 'act=',self.use_act)
        # print(h.shape,'OUT LAYER')
        output = h, adj

        self.most_recent['out']=h
        return output


class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, use_bias):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()
        # self.most_recent={'in':None,'out':None}
        
        
    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):
        self.most_recent={}
        self.most_recent['in']=x
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)


        mv = self.manifold.mobius_matvec(drop_weight, x, self.c) ### just doe
        res = self.manifold.proj(mv, self.c)
        if self.use_bias:
            # print("USE BIAS")
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        self.most_recent['out']=res
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )


class HypAgg(Module):
    """
    Hyperbolic aggregation layer.
    """

    def __init__(self, manifold, c, in_features, dropout, use_att, local_agg,use_frechet_agg,use_agg=True):
        super(HypAgg, self).__init__()
        self.manifold = manifold
        self.c = c

        self.in_features = in_features
        self.dropout = dropout
        self.local_agg = local_agg
        self.use_att = use_att
        self.use_agg=use_agg
        self.use_frechet_agg=use_frechet_agg
        print('frechet agg',use_frechet_agg)
        if self.use_att:
            self.att = DenseAtt(in_features, dropout)
        if self.use_frechet_agg:
            # print('using again')
            self.frechet_agg_man = Frechet_Poincare()
        # self.most_recent={'in':None,'out':None}

    def forward(self, x, adj):
        self.most_recent={}
        self.most_recent['in']=x
        # if not self.use_agg:
        #     self.most_recent['out']=x
        #     # print('ignored!!!')
        #     return x
        # print('not ignored')
        x_tangent = self.manifold.logmap0(x, c=self.c)
        if self.use_att:
            if self.local_agg:
                x_local_tangent = []
                for i in range(x.size(0)):
                    x_local_tangent.append(self.manifold.logmap(x[i], x, c=self.c))
                x_local_tangent = torch.stack(x_local_tangent, dim=0)
                adj_att = self.att(x_tangent, adj)
                att_rep = adj_att.unsqueeze(-1) * x_local_tangent
                support_t = torch.sum(adj_att.unsqueeze(-1) * x_local_tangent, dim=1)
                output = self.manifold.proj(self.manifold.expmap(x, support_t, c=self.c), c=self.c)
                return output
            else:
                adj_att = self.att(x_tangent, adj)
                support_t = torch.matmul(adj_att, x_tangent)

        elif self.use_frechet_agg and len(x.shape)<=2:
            # if batch_calc:
            if hasattr(self.args,'frechet_B'): ### to save a bunch of time
                frechet_B=self.args.frechet_B
            else:
                frechet_B=None
            self.frechet_agg_man = Frechet_Poincare(-self.c)
            # if len(x.shape)>2:
            
            output=frechet_agg(x=x,adj=adj,man = self.frechet_agg_man,B=frechet_B)

            self.most_recent['out']=output
            return output

        elif self.use_frechet_agg:
            self.frechet_agg_man = Frechet_Poincare(-self.c)
            output=torch.zeros_like(x)
            # output_oneper=torch.zeros_like(x)
            # print(x.shape)

            for i in range(x.shape[0]): ### size of batch
                frech_B=self.args.frech_B_list[i]
                # print(frech_B)
                output_i=frechet_agg(x[i],adj[i], man=self.frechet_agg_man,B=frech_B)
                output[i]=output_i
                # output_oneper[i]=proj

            return output
            # for frech_B in self.args.frechet_B_ist:


        else:

            # print('nothing')
            support_t = torch.spmm(adj, x_tangent)
        output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)
        return output


    def extra_repr(self):
        return 'c={}'.format(self.c)


class HypAct(Module):
    """
    Hyperbolic activation layer.
    """

    def __init__(self, manifold, c_in, c_out, act,norm=None,hyp_act=False):
        super(HypAct, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act
        # self.hyp_act=hyp_act
        # if not norm:

        self.use_norm=False if not norm else True
        # self.use_norm=False if not norm else True
        self.norm=norm

        # self.most_recent={'in':None,'out':None}
        # self.most_recent['in':x]
        # self.most_recent['out':h]
        # print(self.use_norm)
        # print(self.norm,'NORMAN')
        # else

# norm_type, self.args.embed_size
    def forward(self, x,use_act=True,use_batch=True,frechet_B=None): ## if need be, we can make act]]\\\
        # print(x.mean(),x.min(),'input')
        self.most_recent={}
        self.most_recent['in']=x

        norm_hyp=False

        if not use_act:
            return x

        if (self.act==None) and (self.use_norm) and (self.norm.norm_hyp): #### no need to every log transport!!!
            # print('no transports fools')
            kudtddmhdgmhd
            output=self.norm(x)
            self.most_recent['out']=output
            return output
        # if self.hyp_act:
        #     output=self.act(x)

        #     return output

        xt  = self.manifold.logmap0(x, c=self.c_in)
        xt_logmap=xt

        if self.act==None:
            pass
        else: ### probably should put the act like this
            xt = self.act(xt)

        if self.use_norm and not self.norm.norm_hyp: ### Norm Causing gradient issues, unclear if directly or indirectly.. further investigation required
            # xt=self.norm(xt,frechet_B=frechet_B)
            # xt=self.norm(xt,frechet_B=frechet_B)
            xt=self.norm(xt)
                
        # for i in range(len(x)):
            # print(x[i],'X',xt_logmap[i],xt[i],'OUTPUT')

        xt = self.manifold.proj_tan0(xt, c=self.c_out)

        if self.use_norm and self.norm.norm_hyp: ### Norm Causing gradient issues, unclear if directly or indirectly.. further investigation required
            xt=self.norm(self.manifold.expmap0(xt, c=self.c_out),frechet_B)
            # xt= self.norm()
            proj=self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)
            output=self.norm(proj)
            self.most_recent['out']=output
            return output #### what does manifold projection do??

        # print(output,'output!!')


        output=self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)
        self.most_recent['out']=output
        return output



        # # print(xt.mean(),xt.min(),'after log')
        # # if use_batch
        # # print(xt.shape,'xt')
        # if use_act:
        #     # if use_batch:
        #     # xt = self.norm(xt)

        #         # print(xt.mean(),xt.min(),xt.std(),'after norm')
        #     xt = self.act(xt)
        #     if self.use_norm: ### Norm Causing gradient issues, unclear if directly or indirectly.. further investigation required
        #         # print(xt.mean(),xt.min(),xt.std(),'before norm')
        #         # print(xt.mean(axis=0).shape)  # we want # embedding 
        #         # print(xt.mean(axis=1).shape)
        #         # print()

        #         xt=self.norm(xt)
        # # print(xt.mean(),xt.min(),'after act')
        # # if self.act:  -- in future for keeping everything consistent
        # #     xt = self.act(self.manifold.logmap0(x, c=self.c_in))
        # # else:
        # #     xt = self.manifold.logmap0(x, c=self.c_in)
        # xt = self.manifold.proj_tan0(xt, c=self.c_out)
        # return self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )