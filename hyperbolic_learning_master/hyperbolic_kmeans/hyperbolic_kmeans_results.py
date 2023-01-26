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

from embed import train_embeddings, load_embeddings,load_embeddings_npy, evaluate_model,mean_average_precision
from hkmeans import HyperbolicKMeans, plot_clusters

# ignore warnings
import warnings
warnings.filterwarnings('ignore');

# display multiple outputs within a cell
# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all";


emails = pd.read_table(r'enron\emails.txt', delimiter=' ', header=None)
emails = emails[emails.iloc[:, 1] != emails.iloc[:, 2]].reset_index(drop=True)
emails.iloc[:, 1:].to_csv(r'enron\enron_relations.csv', index=False)
edge_list = []
for i in range(emails.shape[0]):
    edge_list.append(list(emails.iloc[i, 1:]))


employees = pd.read_csv('enron/employee_info.csv')
employees = employees.fillna('NA')

# load embedding vectors
# C:\Users\Cole S Baker\Desktop\Thesis\Thesis\hgcn\logs\enron\lp\2022_2_16\92"

their_path=r'models\enron_vectors'
my_path_2d = r"models_cole\enron_vectors_2d_trainall.npy"
# my_path_2d = r"C:\Users\Cole S Baker\Desktop\Thesis\Thesis\hgcn\logs\enron\lp\2022_2_17\9\embeddings.npy"
my_path_2d = r"C:\Users\Cole S Baker\Desktop\Thesis\Thesis\hgcn\study\enron\lp\2022_2_15\6\89\embeddings.npy"
# my_path_2d = r"C:\Users\Cole S Baker\Desktop\Thesis\Thesis\hgcn\logs\enron\lp\2022_2_18\3\embeddings.npy")


plot_emb_path = their_path
cluster_emb_path = their_path

# plot_emb_path = my_path_2d
# cluster_emb_path = my_path_2d






emb = load_embeddings_npy(plot_emb_path)  if ".npy" in plot_emb_path else load_embeddings(plot_emb_path) ## for names purposes
# emb = load_embeddings(their_path)  ## for names purposes

# print(emb)

# ed

if ".npy" in plot_emb_path:

	plot_emb= np.load(plot_emb_path)
	plot_emb_data=plot_emb
else:
	plot_emb = load_embeddings(plot_emb_path)
	plot_emb_data = np.array(emb.iloc[:, 1:3])

if ".npy" in cluster_emb_path:
	print('my clusters!!')
	cluster_emb= np.load(cluster_emb_path)
	cluster_emb_data=cluster_emb
else:
	print('their clusters!!')
	cluster_emb = load_embeddings(cluster_emb_path)
	cluster_emb_data = np.array(emb.iloc[:, 1:3])
# emb_data.shape

# get node attributes
names = employees['first'].values + ' ' + employees['last'].values
names = names + ' (' + employees['title'] + ')'
# names = emb['node'].astype(int)

enron_dict = {}
for i in range(emb.shape[0]):
    enron_dict[names[i]] = plot_emb_data[i]

print('lets cluster')

# fit hyperbolic kmeans clusters to embedding vectors
hkmeans = HyperbolicKMeans(n_clusters=6)
hkmeans.fit(cluster_emb_data, max_epochs=10)
# ranks,avg_precision = mean_average_precision(edge_list,enron_dict)
# print(np.mean(ranks),'rank')
# print(np.mean(avg_precision))
# plot results
plot_clusters(emb, labels = hkmeans.assignments, centroids = hkmeans.centroids, edge_list=edge_list,
              add_labels=True, label_dict=enron_dict, label_frac = 0.001, edge_frac = 0.01, width=12, height=12, title='Enron Emails')