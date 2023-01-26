import numpy as np
# from HypHC_master.utils.lca import hyp_lca
from manifolds.poincare import PoincareBall
import torch
def norm(x, axis=None):
    return np.linalg.norm(x, axis=axis)

#-------------------------
#----- Poincaré Disk -----
#-------------------------

# NOTE: POSSIBLE ISSUE WITH DIFFERENT WAYS TO SPECIFY MINKOWSKI DOT PRODUCT
# arbritray sign gives different signatures (+, +, +, -), (+, -, -, -)
    
# distance in poincare disk
pcball=PoincareBall()
def poincare_dist(u, v, c,eps=1e-5):


    if not ((c<1) or (c>1)):
        d = 1 + 2 * norm(u-v)**2 / ((1 - norm(u)**2) * (1 - norm(v)**2) + eps)

        dist=np.arccosh(d)
        # print(dist,'disty ')
        # print("SHOULD NOT BE HERE")
    else:

        dist_sq=pcball.sqdist(torch.from_numpy(u.astype(float)),torch.from_numpy(v.astype(float)),c=c)
        dist=dist_sq**(1/2)
        # print(dist,'DISTANCE before')
        dist=dist.item()
        # print(dist,'DISTANCE')
    return dist

def dist_to_prob_fd(dist,t,r):

    prob= 1/(np.exp((dist - r) / t)+ 1.0)
    # print(dist,prob)
    return prob
def fd_prob(u,v,t,r):
    dist=poincare_dist(u,v)
    return dist_to_prob_fd(dist,t,r)

# compute symmetric poincare distance matrix
def poincare_distances(embedding,c,use_fd=False,full=False):
    n = embedding.shape[0]
    dist_matrix = np.zeros((n, n))
    use_fd=False
    # fd_r=1.88
    # fd_t=1.1
    for i in range(n):
        for j in range(i+1, n):
            # if use_fd:
                # dist_matrix[i][j] = fd_prob(embedding[i], embedding[j],fd_t,fd_r)
            # else:
            # if 
            dist_matrix[i][j] = poincare_dist(embedding[i], embedding[j],c)
            if full:
                dist_matrix[j][i] = poincare_dist(embedding[i], embedding[j],c)
    return dist_matrix

# def euc_dist(u,v):
    # return math.dist(u-v)

def hyp_lca_all(embedding):
    raise Exception('not doing LCA anymore in this repo')
    n = embedding.shape[0]
    dim = embedding.shape[1]
    # print(embedding,'EMBEDDING')
    coord_matrix = np.zeros((n, n,dim))
    rad_matrix = np.zeros((n, n))
    print(n,"N")
    for i in range(n):
        a = torch.from_numpy(embedding[i])
        for j in range(i+1, n):
            b = torch.from_numpy(embedding[j])
            # print(a,'A SHOULD BE TORCH?')
            coord_matrix[i][j],rad_matrix[i][j] = hyp_lca(a, b)
    return coord_matrix,rad_matrix    
# convert array from poincare disk to hyperboloid
def poincare_pts_to_hyperboloid(Y, eps=1e-6, metric='lorentz'):
    ## haven't needed it yet- should work just like before
    # print()
    dim=Y.shape[1]
    mink_pts = np.zeros((Y.shape[0], dim+1))
    r = norm(Y, axis=1)
    if metric == 'minkowski':
        mink_pts[:, 0] = 2/(1 - r**2 + eps) * (1 + r**2)/2
        for i in range(dim):
            mink_pts[:,i+1] = 2/(1 - r**2 + eps) * Y[:,i]        
        # mink_pts[:, 1] = 2/(1 - r**2 + eps) * Y[:, 0]
        # mink_pts[:, 2] = 2/(1 - r**2 + eps) * Y[:, 1]
    else:
        for i in range(dim):
            mink_pts[:,i] = 2/(1 - r**2 + eps) * Y[:,i]
        mink_pts[:,dim] = 2/(1 - r**2 + eps) * (1 + r**2)/2
        # mink_pts[:, 0] = 2/(1 - r**2 + eps) * Y[:, 0]
        # mink_pts[:, 1] = 2/(1 - r**2 + eps) * Y[:, 1]
        # mink_pts[:, 2] = 2/(1 - r**2 + eps) * (1 + r**2)/2
    return mink_pts

# convert single point to hyperboloid
def poincare_pt_to_hyperboloid(y, eps=1e-6, metric='lorentz'):
    # print(y,'Y for hyperboloid')
    dim=y.shape[0]

    mink_pt = np.zeros((dim+1, ))
    r = norm(y)
    if metric == 'minkowski':
        mink_pt[0] = 2/(1 - r**2 + eps) * (1 + r**2)/2
        for i in range(dim):
            mink_pt[i+1] = 2/(1 - r**2 + eps) * y[i]
    else:
        for i in range(dim):
            mink_pt[i] = 2/(1 - r**2 + eps) * y[i]
        mink_pt[dim] = 2/(1 - r**2 + eps) * (1 + r**2)/2
    # print(mink_pt.shape,'MINK SHAPE')
    return mink_pt

#------------------------------
#----- Hyperboloid Model ------
#------------------------------

# NOTE: POSSIBLE ISSUE WITH DIFFERENT WAYS TO SPECIFY MINKOWSKI DOT PRODUCT
# arbritray sign gives different signatures (+, +, +, -), (+, -, -, -)

# define hyperboloid bilinear form
def hyperboloid_dot(u, v):
    return np.dot(u[:-1], v[:-1]) - u[-1]*v[-1]

# define alternate minkowski/hyperboloid bilinear form
def minkowski_dot(u, v):
    return u[0]*v[0] - np.dot(u[1:], v[1:]) 

# hyperboloid distance function
def hyperboloid_dist(u, v, eps=1e-6, metric='lorentz'):
    if metric == 'minkowski':
        dist = np.arccosh(-1*minkowski_dot(u, v))
    else:
        dist = np.arccosh(-1*hyperboloid_dot(u, v))
    if np.isnan(dist):
        #print('Hyperboloid dist returned nan value')
        return eps
    else:
        return dist

# compute symmetric hyperboloid distance matrix
def hyperboloid_distances(embedding):
    n = embedding.shape[0]
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist_matrix[i][j] = hyperboloid_dist(embedding[i], embedding[j])
    return dist_matrix

# convert array to poincare disk
def hyperboloid_pts_to_poincare(X, eps=1e-6, metric='lorentz'):
    ### here we go
    dim=X.shape[1]-1
    poincare_pts = np.zeros((X.shape[0], X.shape[1]-1))
    if metric == 'minkowski':
        for i in range(dim):
            poincare_pts[:, i] = X[:, i+1] / ((X[:, i]+1) + eps)
        # poincare_pts[:, 0] = X[:, 1] / ((X[:, 0]+1) + eps)
        # poincare_pts[:, 1] = X[:, 2] / ((X[:, 0]+1) + eps)
    else:
        for i in range(dim):
            poincare_pts[:, i] = X[:, i] / ((X[:, dim]+1) + eps)
        # poincare_pts[:, 0] = X[:, 0] / ((X[:, 2]+1) + eps)
        # poincare_pts[:, 1] = X[:, 1] / ((X[:, 2]+1) + eps)
    return poincare_pts

# project within disk
def proj(theta,eps=0.1):
    if norm(theta) >= 1:
        theta = theta/norm(theta) - eps
    return theta

# convert single point to poincare
def hyperboloid_pt_to_poincare(x, eps=1e-6, metric='lorentz'):
    dim=x.shape[0]-1
    poincare_pt = np.zeros((dim))

    if metric == 'minkowski':
        for i in range(dim):
            poincare_pts[i] = x[i+1] / ((x[ i]+1) + eps)
        # poincare_pt[0] = x[1] / ((x[0]+1) + eps)
        # poincare_pt[1] = x[2] / ((x[0]+1) + eps)
    else:
        for i in range(dim):
            poincare_pt[i] = x[i] / ((x[dim]+1) + eps)
        # poincare_pt[0] = x[0] / ((x[2]+1) + eps)
        # poincare_pt[1] = x[1] / ((x[2]+1) + eps)
    return proj(poincare_pt)
    
# helper function to generate samples
def generate_data(n, radius=0.7, hyperboloid=False):
    theta = np.random.uniform(0, 2*np.pi, n)
    u = np.random.uniform(0, radius, n)
    r = np.sqrt(u)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    init_data = np.hstack((x.reshape(-1,1), y.reshape(-1,1)))
    if hyperboloid:
        return poincare_pts_to_hyperboloid(init_data)
    else:
        return init_data