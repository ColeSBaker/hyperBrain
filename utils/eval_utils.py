from sklearn.metrics import average_precision_score, accuracy_score, f1_score,roc_auc_score
from sklearn import metrics
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import torch.nn.functional as F
# from sklearn.metrics import roc_auc_score, average_precision_score
# from utils.eval_utils import acc_f1,binary_acc,get_calibration_metrics,draw_reliability_graph
def acc_f1(output, labels, average='binary'):
    preds = output.max(1)[1].type_as(labels)
    if preds.is_cuda:
        preds = preds.cpu()
        labels = labels.cpu()
    accuracy = accuracy_score(preds, labels)
    f1 = f1_score(preds, labels, average=average)
    return accuracy, f1

def binary_acc(output,labels):
	# labels=np.array(labels)
	correct=0
	for i in range(len(labels)):
		p = output[i]
		l = labels[i]
		# print(p,l)
		# print(p is int, l is float)
		# print(type(p), type(l))
		# print(type(p)==int, type(l)==np.__name__)
		# print(int(p),int(l),int(p)==int(l))
		# if type(l)==nd.array():
    # print(l,p)
		if type(l)==float:
			
			if len(l)>1:
				raise Exception('why are we getting a list')
			l=l[0]
		try:
			if round(p)==round(l):

				correct+=1
		except:
			if round(p)==round(l[0]):

				correct+=1			
# 					if round(p)==round(l):
# 				correct+=1
# Hyperboloid		
	acc = float(correct)/float(len(labels))
	# correct= np.where() 
	# print(acc,'acc')
	return acc
def bin_to_two_class(preds):
	new = np.zeros((preds.shape[0],2))
	new[:,0]=preds 
	new[:,1]=1-preds
	return new 

class IdentityEstimator():
  def init(self):
    # self.X=X
    pass

  def predict(self,X):
    return X

  def fit(self,X,y):
    return
  # def scoring
def run_scaling(data,model,one_side=True):

  ### splitting data
  split='val'
  node_num = len(data['features'])
  adj_mat = data['adj_mat'] if model.is_inductive else data['adj_train_norm']
  embeddings = model.encode(data['features'], adj_mat)
  edges_val=data['val_edges'].cpu().detach().data.numpy()
  edges_false_val= data['val_edges_false'].cpu().detach().data.numpy()
  edges=data['train_edges'].cpu().detach().data.numpy()
  edges_false=data['train_edges_false'].cpu().detach().data.numpy()
  edges_false= edges_false[np.random.randint(0, len(edges_false),len(edges))]
  pos_scores = model.decode(embeddings, edges) ### how to we match pos_scores to neg scores?
  neg_scores = model.decode(embeddings, edges_false) 

  pos_scores_val = model.decode(embeddings, edges_val) ### how to we match pos_scores to neg scores?
  neg_scores_val = model.decode(embeddings, edges_false_val) 

  labels = np.array([1] * pos_scores.shape[0] + [0] * neg_scores.shape[0])
  preds_original = np.array(list(pos_scores.data.cpu().numpy()) + list(neg_scores.data.cpu().numpy()))

  labels_val = np.array([1] * pos_scores_val.shape[0] + [0] * neg_scores_val.shape[0])
  preds_original_val = np.array(list(pos_scores_val.data.cpu().numpy()) + list(neg_scores_val.data.cpu().numpy()))

  X_calib = preds_original[:,None]
  y_calib = labels[:,None]

  X_calib_val = preds_original_val[:,None]
  y_calib_val = labels_val[:,None]

  i = LogisticRegression()
  # i = IdentityEstimator()
  i.fit(X_calib,y_calib)

  calibrated_clf=CalibratedClassifierCV( base_estimator=i,cv="prefit",method='isotonic')
  # calibrated_clf=CalibratedClassifierCV( base_estimator=i,cv="prefit")
  # CalibratedClassifierCV( base_estimator=model,cv="prefit")
  calibrated_clf.fit(X_calib, y_calib)
  preds_calibrated = calibrated_clf.predict_proba(X_calib_val)
  pos_pred = preds_calibrated[:,1]
  print(labels.shape,preds_original.shape,'shapes')
  # loss = F.binary_cross_entropy(y_calib, X_calib) ### is this average or sum??
  roc = roc_auc_score(labels_val, preds_original_val)
  ap = average_precision_score(labels_val, preds_original_val)
  acc = binary_acc(labels_val,preds_original_val)
  print("YO WHAT")
  print("UNCALIBRATED:",roc,'roc',roc,'loss',ap,'prec',acc,'acc')
  # loss = F.binary_cross_entropy(labels, pos_pred) ### is this average or sum??
  roc = roc_auc_score(labels_val, pos_pred)
  ap = average_precision_score(labels_val, pos_pred)
  acc = binary_acc(labels_val,pos_pred)
  print("CALIBRATED:",roc,'roc',roc,'loss',ap,'prec',acc,'acc')

  # print('roc',loss,'loss',ap,'prec',acc,'acc')
  # print(preds_calibrated[1:],'oye')
  # print(calibrated_clf.get_params(),'params')
  draw_reliability_graph(labels_val,preds_original_val,one_side=False)
  draw_reliability_graph(labels_val,pos_pred,one_side=False)

  draw_reliability_graph(labels_val,preds_original_val,one_side=True)
  draw_reliability_graph(labels_val,pos_pred,one_side=True)
# def calc_bins(labels,preds,c=0):
#   # Assign each prediction to a bin
#   num_bins = 10

#   preds = np.array(preds)
#   if c==0:
#     preds = 1-preds
#   labels = np.array(labels).astype(int)
#   preds = preds[labels==c]
#   print(preds,'preds!!!')
#   labels = labels[labels==c]
#   bins = np.linspace(0.1, 1, num_bins)
#   binned = np.digitize(preds, bins)
#   print(binned,'bins!!')

#   # Save the accuracy, confidence and size of each bin
#   bin_accs = np.zeros(num_bins)
#   bin_confs = np.zeros(num_bins)
#   bin_sizes = np.zeros(num_bins)

#   for b in range(num_bins):
#     bin_sizes[b] = len(preds[binned == b])
#     print(b,bin_sizes[b])
#     if bin_sizes[b] > 0:
#       # bin_accs[b] = (labels_oneh[binned==bin]).sum() / bin_sizes[bin]
#       bin_accs[b] = binary_acc(labels[binned==b],preds[binned==b]) ### what do we e
#       bin_confs[b] = (preds[binned==b]).sum() / bin_sizes[b]
#   print(bin_sizes)
#   return bins, binned, bin_accs, bin_confs, bin_sizes

def calc_bins(labels,preds,one_side=True,num_class=2):
  # Assign each prediction to a bin
  if one_side==False:
    return calc_bins_twoside(labels,preds)
  num_bins =10
  one_side=False
  bins = np.linspace(0.1, 1, num_bins)
  preds=np.array(preds)
  confs = preds if one_side else np.max(bin_to_two_class(preds),axis=1)
  binned = np.digitize(confs,bins)

  labels=np.array(labels)

  # Save the accuracy, confidence and size of each bin
  eps=.00000001
  bin_accs = np.zeros((num_bins,num_class))
  bin_confs = np.zeros((num_bins,num_class))

  
  bin_sizes = np.zeros((num_bins,num_class))
  for n in range(num_class):
    for b in range(num_bins):
      confs_spec = confs[(binned==b)&(labels==n)]
      labels_spec = labels[(binned==b)&(labels==n)]
      preds_spec = preds[(binned==b)&(labels==n)] ### not necissarily a high prediction, just a high pred for one group
      bin_sizes[b,n] = len(confs_spec)
      if bin_sizes[b,n] > eps:
        bin_accs[b,n] = binary_acc(labels_spec,preds_spec) ### what do we e
        bin_confs[b,n] = (confs_spec).sum() / bin_sizes[b,n]  ### so then this becomes 1-preds ?  ### do abs value here?

  bin_sizes_full=np.sum(bin_sizes,axis=1)
  bin_accs=((bin_accs[:,0]*bin_sizes[:,0])+(bin_accs[:,1]*bin_sizes[:,1]))/(bin_sizes_full+eps)

  bin_confs=((bin_confs[:,0]*bin_sizes[:,0])+(bin_confs[:,1]*bin_sizes[:,1]))/(bin_sizes_full+eps)
  return bins, binned, bin_accs, bin_confs, bin_sizes_full

def calc_bins_twoside(labels,preds):
  num_bins =10
  start = 1/num_bins
  eps=.00000001
  bins = np.linspace(start, 1, num_bins)
  preds=np.array(preds)
  confs = np.max(bin_to_two_class(preds),axis=1)
  binned = np.digitize(preds,bins) ## we bin by prediction strength, NOT confidence
  bin_accs = np.zeros((num_bins))
  bin_confs = np.zeros((num_bins))
  bin_sizes = np.zeros((num_bins))
  labels=np.array(labels)
  for b in range(num_bins):
    # print(bins[b])
    # print(binned[b*20:b*20+20])
    # print(preds[b*20:b*20+20])
    confs_spec = confs[(binned==b)]
    labels_spec = labels[(binned==b)]
    preds_spec = preds[(binned==b)] ### not necissarily a high prediction, just a high pred for one group
    bin_sizes[b] = len(confs_spec)
    if bin_sizes[b] > eps:
      bin_accs[b] = binary_acc(labels_spec,preds_spec) ### what do we e
      bin_confs[b] = (confs_spec).sum() / bin_sizes[b]  ### so then this becomes 1-preds ?  ### do abs value here?
  return bins, binned, bin_accs, bin_confs, bin_sizes
def get_calibration_metrics(labels,preds,one_side=False):   ## can be done all at once or in bits....

  ECE = 0
  MCE = 0
  bins, _, bin_accs, bin_confs, bin_sizes = calc_bins(labels,preds,one_side=one_side)

  for i in range(len(bins)):
    abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
    ECE += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif
    MCE = max(MCE, abs_conf_dif)

  return ECE, MCE


def draw_reliability_graph(labels,preds,one_side=False):  ## needs save_dir, just give this metrics so we can plot whatever? ## or just recalc
  print('reliability')
  ECE, MCE = get_calibration_metrics(labels,preds,one_side=one_side)
  bins, _, bin_accs, _, bin_sizes = calc_bins(labels,preds,one_side=one_side)


  fpr, tpr, thresholds = metrics.roc_curve(labels, preds)
  roc_auc = metrics.auc(fpr, tpr)
  display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                    estimator_name='example estimator')
  # display.plot()
  # plt.show()

  fig = plt.figure(figsize=(8, 8))
  ax = fig.gca()

  # x/y limits
  ax.set_xlim(0, 1.05)
  ax.set_ylim(0, 1)

  # x/y labels
  plt.xlabel('Confidence')
  plt.ylabel('Accuracy')

  # Create grid
  ax.set_axisbelow(True) 
  ax.grid(color='gray', linestyle='dashed')

  # Error bars
  num_bins=len(bins)
  print(bins)
  print(bins[num_bins//2:])
  print(bins[num_bins//2:]+bins[num_bins//2:])
  # print(len(bins[num_bins//2:]+bins[num_bins//2:]))

  double_bins = np.concatenate((bins[num_bins//2:][::-1],bins[num_bins//2:]))

  print(bins,'BINS!!')
  print(double_bins,'double binned')
  # if one_side:
  #   plt.bar(bins, bins,  width=0.1, alpha=0.3, edgecolor='black', color='r', hatch='\\')
  # else:

  #   plt.bar(bins,double_bins,  width=0.1, alpha=0.3, edgecolor='black', color='r', hatch='\\')
    # plt.bar(bins[:num_bins],bins[num_bins:])
  # Draw bars and identity line
  plt.bar(bins, bin_accs, width=0.1, alpha=1, edgecolor='black', color='b')
  if one_side:
    plt.plot([0,1],[0,1], '--', color='gray', linewidth=2)
  else:

    plt.plot([0,.5],[1,.5], '--', color='gray', linewidth=2)
    plt.plot([.5,1],[.5,1], '--', color='gray', linewidth=2)

  for i,b in enumerate(bins):
      size= bin_sizes[i]
      print(b-.1,-2,size,'hey!')
      plt.text(b, 0, str(int(size)), color='black', fontweight='bold',fontsize=10)

  plt.text(1,1,str("ROC: "+str(int(roc_auc*100))), color='black', fontweight='bold',fontsize=10)

  # Equally spaced axes
  plt.gca().set_aspect('equal', adjustable='box')

  # ECE and MCE legend
  ECE_patch = mpatches.Patch(color='green', label='ECE = {:.2f}%'.format(ECE*100))
  # MCE_patch = mpatches.Patch(color='red', label='MCE = {:.2f}%'.format(MCE*100))
  # plt.legend(handles=[ECE_patch, MCE_patch])
  plt.legend(handles=[ECE_patch])
  print(bin_sizes,'bin sizes')
  plt.show()
  


  # plt.savefig('calibrated_network.png', bbox_inches='tight')

#draw_reliability_graph(preds)
