import os
from utils.embedding_utils import *
from classifiers.CustomClassifiers import AggregateClassifier
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

BAND_TO_INDEX = {'theta':0,'alpha':1,'beta1':2,'beta2':3,'beta':4,'gamma':5}
METRIC_TO_INDEX = {'plv':0,'ciplv':1}

TEMPLATE_PATH=os.path.join(r"C:\Users\Cole S Baker\Desktop\Thesis\Thesis\hgcn\data\MEG","AALtemplate_Feb9_wNetPcts.csv")
CLINICAL_PATH=os.path.join(os.getcwd(),'data/MEG/MEG.clinical.csv')
RAW_SCAN_PATH=os.path.join(os.getcwd(),'data/MEG/MEG.ROIs.npy')
    # # parser.add_argument('--raw_clinical_file', type=str, default=os.path.join(os.getcwd(),'data/MEG/MEG.clinical.csv'))
    # parser.add_argument('--raw_scan_file', type=str, default=os.path.join(os.getcwd(),'data/MEG/MEG.ROIs.npy'))
    # # a=
    # parser.add_argument('--raw_atlas_file', type=str, default=os.path.join(os.getcwd(),'data/MEG/AALtemplate.csv'))
class EmbAnalysis():

	def __init__(self,root,dims,suffix='.csv'):
		self.scan_info_path= os.path.join(root,'scan_info.csv')
		self.root=root
		self.emb_root = os.path.join(root,'embeddings')
		self.scan_info = pd.read_csv(self.scan_info_path)
		self.dims=dims

		self.label_options = {'FN':'Functional Net','SOB':'SOB','FN_SOB':'Func Net SOB','All':'All'} 

		# self.stat_dict={'FN':{},'FN_SOB':{},'SOB':{}}

		self.set_template_df(TEMPLATE_PATH)
		self.set_clinical_df(CLINICAL_PATH)

		self.stat_dict={}
		# self.template_df=template_df
		self.suffix=suffix
		self.hkmeans=None
		self.target=['diagnosis']
		self.base_model=SVC
		self.feature_dict={}

		self.graph_labels = self.clinical_df['diagnosis'].values
		self.scan_ids = self.clinical_df['Scan Index'].values

		self.cluster_stats=['wc','bc','rc','hkc']
		self.node_stats=['pw','node_coords','node_rad']
		self.node_stats=['pw','node_rad']
		self.already_stacked_nodes=False
		# self.stacked_nodes=None
		self.hkmeans_log=[] ### so we can keep track

		self.add_weighted_raw()
		# self.X_weight

	# def add_

	def add_features(self,X,name,cluster_type=''):
		### need to stack yer X's

		print(X,'Xs')

		### FIGURE OUT HOW TO HANDLE NODE COORDS ANOTHER TIME
		X_comb,y,pat_to_indx = combine_patient_feats(X,self.graph_labels,self.clinical_df) ### NO CONDITIONALS
    
		if cluster_type=='':

			self.feature_dict[name]={'data':X_comb,'pat_to_indx':pat_to_indx}
		else:
			if cluster_type not in self.feature_dict:
				self.feature_dict[cluster_type]={}
			self.feature_dict[cluster_type][name]={'data':X_comb,'pat_to_indx':pat_to_indx}

	def stack_embeddings(self):
		self.stacked_nodes=stack_embeddings(self.emb_root,self.scan_ids,self.template_df,suffix='.csv',dims=self.dims)
		dim_cols=['x','y','z']
		self.node_coords= np.array([np.array(d[dim_cols]) for d in self.stacked_nodes])
		self.already_stacked_nodes=True

	def calculate_centroids(self,k,add_origin=True,plot=False):
		hyp_km_node_full,hyp_km_node_ind,self.hkmeans=hyp_cluster_full(self.node_coords,k=k,add_origin_centroid=add_origin,plot_cents=plot)

		self.add_features(hyp_km_node_ind,'hk_node')
		self.add_features(hyp_km_node_full,'hk_node_mean')

		self.hkmeans_log.append(self.hkmeans)


	def run_analysis(self,label='FN',net_t=.25,bin_rank_threshold=False,use_binary=False,weighted=False,override=True,run_hk_if_none=True,k_if_none=5):
		label_to_use=self.label_options[label]
		if run_hk_if_none and len(self.hkmeans_log)==0:
			if not self.already_stacked_nodes:
				self.stack_embeddings()
			self.calculate_centroids(k=k_if_none)

		if label in self.stat_dict and not override:
			print('Analysis Previously Complete')
			return self.stat_dict[label]
		stat_dict = embedding_analysis(self.emb_root,self.scan_ids,label_to_use,net_threshold=net_t,
		                               template_df=self.template_df,
		                               bin_rank_threshold=bin_rank_threshold,weighted=weighted,use_binary=use_binary,suffix=self.suffix,dims=self.dims,hkmeans_full=self.hkmeans)

		self.stat_dict[label]=stat_dict  ## this naming is so numb

	def add_analysis_stats(self,label):
		self.cluster_stats=['wc','bc','rc','hkc']
		self.node_stats=['pw','node_rad']

		stat_dict=self.stat_dict[label]
		for cs in self.cluster_stats:
			self.add_features(stat_dict[cs]['stat_df'],name=cs,cluster_type=label)
		for ns in self.node_stats:
			self.add_features(stat_dict[ns]['stat_df'],name=ns) ### not cluster specific

	def add_weighted_raw(self):
		scans=self.scan_data
		X_weight = np.array([a[np.triu_indices(a.shape[0])] for a in scans])
		self.add_features(X_weight,'X_weight')

	def set_raw_scans(self,new_scan_path,metric='plv',band='beta'):
		print('setting new scans: ',new_scan_path)
		band_adj = BAND_TO_INDEX[band]
		metric_adj = METRIC_TO_INDEX[metric]
		raw_scans=np.load(new_scan_path)
		self.scan_data = raw_scans[:,band_adj,metric_adj] ### this is not changed by any hyperparamters

	def set_target(self,target):

		self.target=target

	def set_model(self,model):  ### makes easier to create multiple identical classifiers quickly
		self.model=model

	def set_clinical_df(self,clincal_path):
		print('Are you sure you want to change clinical path?')
		self.clinical_df = pd.read_csv(clincal_path)

	def set_template_df(self,new_path):
		self.template_df = pd.read_csv(new_path)
		self.template_df= self.template_df.apply(lambda row: meg_roi_network_assigments(row,combine_dmn=False),axis=1)

	# def 
	def create_classifier(self,feature_list=[],cluster_types=['FN'],fit_method='scan_mean',predict_method='same',target=None,base_model=None):
		""" will concatenate all items in feature_list
		options are: raw,c_rad,c_btw,c_within,node_rad,node_polar,c_polar,node_cent,c_cent
		target- what is y feature. if None, will use current default, dido for base_model
		cluster_types- if feature list has cluster stats, what kind of cluster ('FN','FN_SOB','SOB','All')
						### must have run that analysis before creating said classifier. will add all cluster features for all cluster types 

		"""

		cluster_feats=set(['c_rad','c_btw','c_within','c_polar','c_cent'])
		cluster_feats=self.cluster_stats
		if target is None:
			target=self.target
		if base_model is None:
			base_model=self.base_model


		X_list=[]

		pat_to_indx=None
		first=True

		for f in feature_list:
			print(f,'F')
			print(cluster_feats,'cluster feats')
			if f in cluster_feats:
				for ct in cluster_types:
					assert ct in self.feature_dict
					assert f in self.feature_dict[ct]
					X_list.append(self.feature_dict[ct][f]['data'])
			else:
				assert f in self.feature_dict
				X_list.append(self.feature_dict[f]['data'])

				pat_to_indx_f=self.feature_dict[f]['pat_to_indx']
				if first:
					pat_to_indx=pat_to_indx_f
					first=False

				else:
					assert pat_to_indx_f==pat_to_indx

		features = np.hstack(X_list)
		ys=self.graph_labels

		ac = AggregateClassifier(self.base_model,fit_method=fit_method,predict_method=predict_method)
		embC = EmbeddingClassifier(ac,features,ys,pat_to_indx)

		return embC

class EmbeddingClassifier(BaseEstimator,ClassifierMixin):
	def __init__(self,AggClassifier,features,ys,pat_to_indx,feature_list=[]):
		### we need patientid/patientindx/
		self.features=features
		self.feature_list=feature_list## just for bookkeeping
		self.ys=ys
		self.AggClassifier=AggClassifier
		self.base_model=self.AggClassifier.base_model
		self.pat_to_indx=pat_to_indx



	def X_to_indices(self,X):
		ex = X[0,0] ## assuming 2dims
		print(ex,'EX')
		if type(ex)==int:
			indices=X
		elif type(ex)==str:
			indices=[self.pat_to_indx[p[0]] for p in X]
		return np.array(indices)

	def indices_to_y(self,indices):
		y=self.ys[indices]
		return y

	def X_to_y(self,X):
		### useful for cross val, not needing to put in work for giving correct label
		indices=self.X_to_indices
		y=self.indices_to_y(indices)
		return y

	def fit(self,X,y):
		### we really don't need y here
		# use_indi

		indices= self.X_to_indices(X)
		print(len(indices),'ind len')
		print(X.shape,'X shape')
		print(y.shape,'y shape')
		try:
			if y==None:
				y_true=self.ys[indices]
			else:
				y_true=y
		except:
			y_true=y

		
		# print(self.features.shape,'FEATURES')
		# print(indices.shape,'Indicotomy')
		# print(indices)


		X_features=self.features[indices]

		self.classes_ = unique_labels(y)
		 
		self.X_ = X
		# self.y_ = y
		# self.y_min = np.min(y)
		return self.AggClassifier.fit(X_features,y_true)

	def predict(self,X):
		indices= self.X_to_indices(X)
		# print(indices,'INDIECE')
		# print(self.features,'FEATS OF CLAY')
		X_features=self.features[indices]
		return self.AggClassifier.predict(X_features)

	def predict_proba(self,X):
		indices= self.X_to_indices(X)
		X_features=self.features[indices]
		return self.AggClassifier.predict_proba(X_features)

	def set_params(self,params):
		### we can just pass these along to the AggClassifier
		self.AggClassifier.set_params(**params)







		# assert len(clu)

# 		label=label_options['FN']
# ##### Need to check new angle analysis with previous way
# stat_dict_fn = embedding_analysis(emb_root,scan_ids,label,net_threshold=.25,
#                                template_df=template_df,
#                                bin_rank_threshold=10,use_binary=False,suffix='.csv',dims=3,hkmeans_full=hkmeans_full))