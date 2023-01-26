import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')
# %matplotlib inline
import networkx as nx
import sys
import os


# import modules within repository
my_path = r'C:\Users\Cole S Baker\Desktop\Thesis\Thesis\hgcn\hyperbolic-learning-master\utils' # path to utils folder
sys.path.append(my_path)
from utils import *
# from poincare_embeddings import *
# from poincare_embedding import 
from embed import train_embeddings, load_embeddings,load_embeddings_npy, evaluate_model,mean_average_precision
from hkmeans import HyperbolicKMeans, plot_clusters

# ignore warnings
import warnings
warnings.filterwarnings('ignore');

# display multiple outputs within a cell
# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all";

data_path = r"C:\Users\Cole S Baker\Desktop\Thesis\Thesis\hgcn\hyperbolic-learning-master\data\disease\disease_lp.edges.csv"


disease = pd.read_csv(data_path)
print(disease['0'])

print('Total unique nodes: ', len(np.unique(list(disease['1'].values) + list(disease['1'].values))))
relations = [[disease['0'][i], disease['1'][i]] for i in range(len(disease))]
edge_list=relations


# load embedding vectors


their_path=r'C:\Users\Cole S Baker\Desktop\Thesis\Thesis\hgcn\hyperbolic-learning-master\data\disease\disease_pc_embeddings'
my_path_2d = r"C:\Users\Cole S Baker\Desktop\Thesis\Thesis\hgcn\hyperbolic-learning-master\data\disease\disease_id_HGCN.npy"
my_path_2d= r"C:\Users\Cole S Baker\Desktop\Thesis\Thesis\hgcn\logs\disease_lp\lp\2022_2_18\15\embeddings.npy"
# my_path_2d = r"C:\Users\Cole S Baker\Desktop\Thesis\Thesis\hgcn\logs\disease_lp\lp\2022_2_16\25\embeddings.npy")
# my_path_2d = r"C:\Users\Cole S Baker\Desktop\Thesis\Thesis\hgcn\study\disease_lp\lp\2022_2_15\10\14\embeddings.npy" ## 2 dim-2 output_dim no identity
# my_path_2d = r"C:\Users\Cole S Baker\Desktop\Thesis\Thesis\hgcn\logs\disease_lp\lp\2022_2_16\27\embeddings.npy" ## 2 dim-2 output_dim no identity same but all trani)
# my_path_2d = r"models_cole\enron_vectors_distance.npy")


plot_emb_path = their_path
cluster_emb_path = their_path

# plot_emb_path = my_path_2d
# cluster_emb_path = their_path






emb = load_embeddings_npy(plot_emb_path)  if ".npy" in plot_emb_path else load_embeddings(plot_emb_path) ## for names purposes
plot_emb = emb
cluster_emb = load_embeddings_npy(cluster_emb_path)  if ".npy" in cluster_emb_path else load_embeddings(cluster_emb_path) ## for names purposes
# emb = load_embeddings(their_path)  ## for names purposes
print(emb,'eddings')
# print(emb)

# ed
cluster_emb_data = np.array(cluster_emb.iloc[:, 1:3])
plot_emb_data = np.array(plot_emb.iloc[:, 1:3])



# get node attributes
# names = employees['first'].values + ' ' + employees['last'].values
names = emb['node'].astype(int)


enron_dict = {}
for i in range(emb.shape[0]):
    enron_dict[names[i]] = plot_emb_data[i]

print('lets cluster')

print(cluster_emb_data.shape,'this one')
# fit hyperbolic kmeans clusters to embedding vectors
hkmeans = HyperbolicKMeans(n_clusters=6)
hkmeans.fit(cluster_emb_data, max_epochs=10)


print('time for the mets!')
# ranks,avg_precision = mean_average_precision(relations,enron_dict)


# print(np.mean(ranks),'rank')
# print(np.mean(avg_precision))
# plot results
plot_clusters(emb, labels = hkmeans.assignments, centroids = hkmeans.centroids, edge_list=edge_list,
              add_labels=True, label_dict=enron_dict, label_frac = 0.001, edge_frac = 0.01, width=12, height=12, title='Disease')

# plot_clusters(emb, labels = hkmeans.assignments, centroids = hkmeans.centroids, edge_list=edge_list,
#               add_labels=True, label_frac = 0.001, edge_frac = 0.01, width=12, height=12, title='Enron Emails')