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
my_path = r'C:\Users\Cole S Baker\Desktop\Thesis\Thesis\hgcn\hyperbolic-learning-master\utils' # path to utils folder  ## geez fix this so its consistent
sys.path.append(my_path)

# print(sys.path)
print(os.getcwd())
from utils import *
# from utils import *
# from poincare_embeddings import *
# from poincare_embedding import 
from embed import train_embeddings, load_embeddings,load_embeddings_npy, evaluate_model,mean_average_precision
from hkmeans import HyperbolicKMeans, plot_clusters

# ignore warningscd
import warnings



def plot_clusters(relations_path,embeddings_path,cluster_path):
	## this doesn't give us much except how nice the plotting is and the clusters.. but will we even use the clusters?
	warnings.filterwarnings('ignore');
	data_path=relations_path
	plot_emb_path = embeddings_path
	cluster_emb_path=cluster_path

	# display multiple outputs within a cell
	# from IPython.core.interactiveshell import InteractiveShell
	# InteractiveShell.ast_node_interactivity = "all";

	data_path = r"C:\Users\Cole S Baker\Desktop\Thesis\Thesis\hgcn\data\MEG\35\relations\2relations.csv"
	# emb_path = 

	meg = pd.read_csv(data_path)
	print(disease['0'])

	print('Total unique nodes: ', len(np.unique(list(meg['1'].values) + list(meg['1'].values))))
	relations = [[meg['0'][i], meg['1'][i]] for i in range(len(meg))]
	edge_list=relations


	plot_emb_path = their_path
	cluster_emb_path = their_path



	emb = load_embeddings_npy(plot_emb_path)  if ".npy" in plot_emb_path else load_embeddings(plot_emb_path) ## for names purposes
	plot_emb = emb
	cluster_emb = load_embeddings_npy(cluster_emb_path)  if ".npy" in cluster_emb_path else load_embeddings(cluster_emb_path) ## for names purposes
	# emb = load_embeddings(their_path)  ## for names purposes

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
	# plot results
	plot_clusters(emb, labels = hkmeans.assignments, centroids = hkmeans.centroids, edge_list=edge_list,
	              add_labels=True, label_dict=enron_dict, label_frac = 0.001, edge_frac = 0.01, width=12, height=12, title='MEG')

	# plot_clusters(emb, labels = hkmeans.assignments, centroids = hkmeans.centroids, edge_list=edge_list,
	#               add_labels=True, label_frac = 0.001, edge_frac = 0.01, width=12, height=12, title='Enron Emails')