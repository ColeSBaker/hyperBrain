# import libraries
import numpy as np
import matplotlib.pyplot as plt
import sys
# my_path = 'C:\\Users\\dreww\\Desktop\\hyperbolic-learning\\utils' # path to utils.py 
# sys.path.append(my_path)
from hyperbolic_learning_master.utils.utils import *

#----------------------------------------------------------
#----- Frechet Mean Optimization in Hyperboloid Model -----
#----------------------------------------------------------

def exp_map(v, theta, eps=1e-6):
    # v: tangent vector in minkowski space
    # theta: parameter vector in hyperboloid with centroid coordinates
    # project vector v from tangent minkowski space -> hyperboloid
    return np.cosh(norm(v))*theta + np.sinh(norm(v)) * v / (norm(v) + eps)

def minkowski_distance_gradient(u, v):
    # u,v in hyperboloid
    # returns gradient with respect to u
    return -1*(hyperboloid_dot(u,v)**2 - 1)**-1/2 * v

def minkowski_loss_gradient(theta, X):
    # X : array with points in hyperboloid cluster
    # theta: parameter vector in hyperboloid with centroid coordinates
    # returns gradient vector

    # print(X.shape,'array shape')
    # print(theta.shape,'centroid shape')
    distances = np.array([-1*hyperboloid_dist(theta, x) for x in X]).reshape(-1,1)
    distance_grads = np.array([minkowski_distance_gradient(theta, x) for x in X])
    grad_loss = 2*np.mean(distances*distance_grads, axis=0)
    if np.isnan(grad_loss).any():
        print('Hyperboloid dist returned nan value')
        # return eps
        return .00001
    else:
        return grad_loss

def project_to_tangent(theta, minkowski_grad):
    # grad: gradient vector in ambient Minkowski space
    # theta: parameter vector in hyperboloid with centroid coordinates
    # projects to hyperboloid gradient in tangent space
    return minkowski_grad + hyperboloid_dot(theta, minkowski_grad)*theta

def update_theta(theta, hyperboloid_grad, alpha=0.1):
    # theta: parameter vector in hyperboloid with centroid coordinates
    return exp_map(-1*alpha*hyperboloid_grad, theta)

def frechet_loss(theta, X):
    s = X.shape[0]
    dist_sq = np.array([hyperboloid_dist(theta, x)**2 for x in X])
    return np.sum(dist_sq) / s

def compute_mean(theta, X, num_rounds = 10, alpha=0.3, tol = 1e-4, verbose=False):
    centr_pt = theta.copy()
    centr_pts = []
    losses = []
    # print(theta.shape,'THETA')
    # print(X.shape,'X SHAPE')
    for i in range(num_rounds):
        gradient_loss = minkowski_loss_gradient(centr_pt, X)
        tangent_v = project_to_tangent(centr_pt, -gradient_loss)
        centr_pt = update_theta(centr_pt, tangent_v, alpha=alpha)
        centr_pts.append(centr_pt)
        losses.append(frechet_loss(centr_pt, X))
        if verbose:
            print('Epoch ' + str(i+1) + ' complete')
            print('Loss: ', frechet_loss(centr_pt, X))
            print('\n')
    return centr_pt

def compute_mean_only(theta, X, num_rounds = 10, alpha=0.3, tol = 1e-4, verbose=False):
    centr_pt = theta.copy()
    centr_pts = []
    losses = []
    # print(theta.shape,'THETA')
    # print(X.shape,'X SHAPE')
    for i in range(num_rounds):
        gradient_loss = minkowski_loss_gradient(centr_pt, X)
        tangent_v = project_to_tangent(centr_pt, -gradient_loss)
        centr_pt = update_theta(centr_pt, tangent_v, alpha=alpha)
        centr_pts.append(centr_pt)
        losses.append(frechet_loss(centr_pt, X))
        if verbose:
            print('Epoch ' + str(i+1) + ' complete')
            print('Loss: ', frechet_loss(centr_pt, X))
            print('\n')
    return centr_pt

#-----------------------------------------------
#----- Hyperbolic K-Means Clustering Model -----
#-----------------------------------------------

class HyperbolicKMeans():
    """
    Perform K-Means clustering in hyperbolic space. Applies gradient descent in
    the hyperboloid model to iteratively compute Fréchet means, and the Poincaré disk
    model for visualization.
    
    API design is modeled on the standard scikit-learn Classifier API
    """
    
    def __init__(self,n_clusters=6,max_iter=300,tol=1e-8,verbose=False,dims=2):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.verbose =  verbose
        self.labels = None
        self.cluster_centers_ = None
        self.dims=dims
        
    def init_centroids(self, radius=0.3):
        # randomly sample starting points on small uniform ball 
        ## wouldn't be thaaaat hard to do arbitrary dims
        theta = np.random.uniform(0, 2*np.pi, self.n_clusters)
        if self.dims >2:
            phi = np.random.uniform(0, 2*np.pi, self.n_clusters)
        else:
            phi = np.full_like(theta,np.pi/2) ### should put phi on the axis and leave x,y untouched

        angles=[np.random.uniform(0, np.pi, self.n_clusters) for angle in range(self.dims-2)]+[theta]
        u = np.random.uniform(0, radius, self.n_clusters)
        r = np.sqrt(u)

        running_sin =1  ### x_n = r* cos(phi_n)* prod(sin(phi_i)) all i<n
        cartesians =[]
        for i in range(len(angles)): ## will cover all but final coordinate
            ang_i=angles[i]
            x_i=r*running_sin*np.cos(ang_i)
            cartesians.append(x_i)
            running_sin*=np.sin(ang_i)

        final_dim = r*running_sin
        cartesians.append(final_dim)

        
        x_ = r * np.cos(theta)
        y_ = r * np.sin(theta)

        x = r * np.cos(theta)*np.sin(phi)
        y = r * np.sin(theta)*np.sin(phi)
        z = r* np.cos(phi)
        # print(x,'x')
        # print(x_,'x_')
        # # print(self.dims,"SEKF DIMS")

        cartesians=np.array(cartesians).T
        # print(cartesians.shape,'CARTESIAN COORDS')
        if self.dims==2:
            for i in range(len(x)):
                assert x[i]==x_[i]
            centers = np.hstack((x.reshape(-1,1), y.reshape(-1,1)))
        else:
            centers = np.hstack((x.reshape(-1,1), y.reshape(-1,1),z.reshape(-1,1)))
        # print(centers.shape)
        #     # assert x==x_
        # # if cen
        # centers = np.hstack((x.reshape(-1,1), y.reshape(-1,1)))
        # print(centers.shape,'cetner shape')

        # print(cartesians.shape,'OG CENTROID SHAPE')
        self.centroids = cartesians
        
    def init_assign(self, labels=None):
        # cluster assignments as indicator matrix
        assignments = np.zeros((self.n_samples, self.n_clusters))
        for i in range(self.n_samples):
            if labels is not None:
                # assign to classes with ground truth input labels
                assignments[i][labels[i]] = 1
            else:
                # randomly initialize each binary vector
                j = np.random.randint(0, self.n_clusters)
                assignments[i][j] = 1
        self.assignments = assignments
        
    def update_centroids(self, X):
        """Updates centroids with Fréchet means in Hyperboloid model
        Parameters
        ----------
        X : array of shape (n_samples, dim) with input data.
        First convert X to hyperboloid points
        """
        dim = X.shape[1]
        # print(dim,'dimensions')
        # print(X.shape,'x shape')
        new_centroids = np.empty((self.n_clusters, dim)) 
        # print(new_centroids.shape,'NEW SHAPE')
        H = poincare_pts_to_hyperboloid(X)
        # print(H.shape,'INIT X SHAPE')  
        for i in range(self.n_clusters):
            if np.sum(self.assignments[:, i] ==1) == 0: ## no objects assigned to this centroid
                new_centroids[i] = self.centroids[i]
                # print(new_centroids[i].shape,'NEW SHAPE FOR NO ASS')
            else:
                # find subset of observations in cluster
                H_k = H[self.assignments[:, i] ==1]
                # print(H_k.shape,'PLAIN SHAPE')
                theta_k = poincare_pt_to_hyperboloid(self.centroids[i])
                # solve for frechet mean
                # print(theta_k.shape,'theta shape')
                # print(H_k.shape,'hk shaoe')
                fmean_k = compute_mean(theta_k, H_k, alpha=0.1)
                # convert back to Poincare disk
                new_centroids[i] = hyperboloid_pt_to_poincare(fmean_k)
        self.centroids = new_centroids
        self.cluster_centers_=self.centroids
        
    def cluster_var(self, X, return_all=False):
        n = self.centroids.shape[0]
        var_C = []
        for i in range(n):
            var_C.append(np.mean(np.array([poincare_dist(self.centroids[i], x) for x in X])))
        if return_all:
            self.variances = var_C
        else:
            self.variances = np.sort(var_C)[-2]
    
    def fit(self, X, y=None, max_epochs=40, verbose=False):
        """
        Fit the K centroids from X, and return the class assignments by nearest centroid
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        max_epochs: maximum number of gradient descent iterations
        verbose: optionally print training scores
        """
        
        # make sure X within poincaré ball
        #X = Normalizer().fit_transform(X)
        if (norm(X, axis=1) > 1).any():
            X = X / (np.max(norm(X, axis=1)))
        
        # initialize random centroids and assignments
        self.n_samples = X.shape[0]
        self.init_centroids()
        
        if y is not None:
            self.init_assign(y)
            self.update_centroids(X)
        else:
            self.init_assign()
        
        # loop through the assignment and update steps
        for j in range(max_epochs):
            self.inertia_ = 0
            self.update_centroids(X)
            for i in range(self.n_samples):
                # zero out current cluster assignment
                self.assignments[i, :] = np.zeros((1, self.n_clusters))
                # find closest centroid (in Poincare disk)
                centroid_distances = np.array([poincare_dist(X[i], centroid) for centroid in self.centroids])
                cx = np.argmin(centroid_distances)
                self.inertia_ += centroid_distances[cx]**2
                self.assignments[i][cx] = 1
            if verbose:
                print('Epoch ' + str(j) + ' complete')
                print(self.centroids)
        self.labels = np.argmax(self.assignments, axis=1)
        self.cluster_var(X)
        return
    
    def predict(self, X):
        """
        Predict class labels for given data points by nearest centroid rule
        Parameters
        ----------
        X : array, shape (n_samples, n_features). Observations to be assigned to the
        class represented by the nearest centroid.
        """
        # zero out current cluster assignment
        n = X.shape[0]
        labels = np.zeros((n, self.n_clusters))
        # find closest centroid (in Poincare disk)
        for i in range(n):
            centroid_distances = np.array([poincare_dist(X[i], centroid) for centroid in self.centroids])
            cx = np.argmin(centroid_distances)
            labels[i][cx] = 1
        return labels

    def transform(self, X,use_origin=False):
        """
        Predict class labels for given data points by nearest centroid rule
        Parameters
        ----------
        X : array, shape (n_samples, n_features). Observations to be assigned to the
        class represented by the nearest centroid.
        """
        # zero out current cluster assignment
        n = X.shape[0]
        n_dims = X.shape[1]
        origin=np.zeros(n_dims)
        centroids= self.centroids ## Cole added

        if use_origin: ## cole added
            centroids=centroids.tolist()
            centroids.append(origin)
            centroids=np.array(centroids)            
        n_clusters=len(centroids)
        transformed_coords = np.zeros((n, n_clusters))

        # find closest centroid (in Poincare disk)
        for i in range(n):
            centroid_distances = np.array([poincare_dist(X[i], centroid) for centroid in centroids])
            transformed_coords[i]=centroid_distances


        return transformed_coords

#------------------------------------------
#----- Visualization helper functions -----
#------------------------------------------

def dist_squared(x, y, axis=None):
    return np.sum((x - y)**2, axis=axis)

# plot clustering results in poincare disk
def plot_clusters(emb, labels, centroids, edge_list, title=None, height=8, width=8,
                  add_labels=False, label_dict=None, plot_frac=1, edge_frac=1, label_frac=0.001,label_names=None,save_file=''
                  , add_sublabel=False,sublabels=np.array([])):
    # Note: parameter 'emb' expects data frame with node ids and coords
    emb.columns = ['node', 'x', 'y']
    n_clusters = len(centroids)
    plt.figure(figsize=(width,height))
    plt.xlim([-1.0,1.0])
    plt.ylim([-1.0,1.0])
    ax = plt.gca()
    circ = plt.Circle((0, 0), radius=1, edgecolor='black', facecolor='None', linewidth=3, alpha=0.5)
    ax.add_patch(circ)
    fig = plt.gcf()
    fig.set_size_inches(8, 8,forward=True)

    if add_sublabel:
        assert len(labels)==len(sublabels)

    
    # set colormap
    if n_clusters <= 12:
        # colors = ['b', 'r', 'g', 'y', 'm', 'c', 'k', 'silver', 'lime', 'skyblue', 'maroon', 'darkorange']
        colors = ['b', 'r', 'g', 'y', 'm', 'c', 'silver', 'lime', 'skyblue', 'maroon', 'darkorange','k']
    elif 12 < n_clusters <= 20:
        colors = [i for i in plt.cm.get_cmap('tab20').colors]
    else:
        cmap = plt.cm.get_cmap(name='viridis')
        colors = cmap(np.linspace(0, 1, n_clusters))

    edge_colors = ['b','w','darkorange']
    sublabel_type='edgecolor'
    n_sublabels= len(sublabels.unique())
    # sublabel_type='marker'
    
    # plot embedding coordinates and centroids
    emb_data = np.array(emb.iloc[:, 1:3])
    handles=[]
    names = []
    for i in range(n_clusters):
        # print(label_names)
        if len(label_names[(labels[:, i] == 1)].values)==0:
            continue
        if 'N/A' in label_names[(labels[:, i] == 1)].values[0]:
            continue


        print(label_names[(labels[:, i] == 1)])
        print()
        print(label_names[(labels[:, i] == 1)].values[0])
        # assert 
        ### need to make sure all label names are the same
        # if label_names is not None: ## do this later..

        # if add_sublabel:
        #     for s in range(len(n_sublabels)):
        #         if sublabel_type=='edgecolors':
        #                 ax.scatter(emb_data[(labels[:, i] == 1)&(labels[:, i] == 1), 0], emb_data[(labels[:, i] == 1), 1],
                         # color = colors[i], alpha=0.8, edgecolors='w', linewidth=2, s=250,label=label_names[(labels[:, i] == 1)].values[0])
        ax.scatter(emb_data[(labels[:, i] == 1), 0], emb_data[(labels[:, i] == 1), 1],
                             color = colors[i], alpha=0.8, edgecolors='w', linewidth=2, s=250,label=label_names[(labels[:, i] == 1)].values[0])
        # handle, = plt.scatter(emb_data[(labels[:, i] == 1), 0], emb_data[(labels[:, i] == 1), 1],
                             # color = colors[i], alpha=0.8, edgecolors='w', linewidth=2, s=250)
        # plt.scatter(centroids[i, 0], centroids[i, 1], s=750, color = colors[i],
                    # edgecolor='black', linewidth=2, marker='*');
        # handles.append(handle)
        # names.append(label_names[(labels[:, i] == 1)].values[0])
    ax.legend(loc='upper right')
    # plot edges
    for i in range(int(len(edge_list) * edge_frac)):
        x1 = emb.loc[(emb.iloc[:, 0] == edge_list[i][0]), ['x', 'y']].values[0]
        x2 = emb.loc[(emb.node == edge_list[i][1]), ['x', 'y']].values[0]
        _ = plt.plot([x1[0], x2[0]], [x1[1], x2[1]], '--', c='black', linewidth=1, alpha=0.35)
    
    # add labels to embeddings
    if add_labels and label_dict != None:
        plt.grid('off')
        plt.axis('off')
        embed_vals = np.array(list(label_dict.values()))
        keys = list(label_dict.keys())
        # set threshhold to limit plotting labels too close together
        min_dist_2 = label_frac * max(embed_vals.max(axis=0) - embed_vals.min(axis=0)) ** 2
        labeled_vals = np.array([2*embed_vals.max(axis=0)])
        n = int(plot_frac*len(embed_vals))

        for i in np.random.permutation(len(embed_vals))[:n]:
            if np.min(dist_squared(embed_vals[i], labeled_vals, axis=1)) < min_dist_2:
                continue
            else:
                # props = dict(boxstyle='round', lw=2, edgecolor='black', alpha=0.35)
                # _ = ax.text(embed_vals[i][0], embed_vals[i][1]+0.02, s=keys[i].split('.')[0],
                #             size=10, fontsize=12, verticalalignment='top', bbox=props)

                props = dict(boxstyle='round', lw=2, edgecolor='black', alpha=0.35)
                try:
                    _ = ax.text(embed_vals[i][0], embed_vals[i][1]+0.02, s=keys[i].split('.')[0],
                            size=10, verticalalignment='top', bbox=props)
                except:
                    _ = ax.text(embed_vals[i][0], embed_vals[i][1]+0.02, s=keys[i],
                                size=10, verticalalignment='top', bbox=props)

                labeled_vals = np.vstack((labeled_vals, embed_vals[i]))
    if title != None:
        plt.suptitle('Hyperbolic K-Means - ' + title, size=16);
    if save_file!='':
        plt.savefig(save_file)
    else:
        plt.show();        
