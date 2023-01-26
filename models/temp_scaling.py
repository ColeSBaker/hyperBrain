import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm as tqdm


def T_scaling(logits, args):
  temperature = args.get('temperature', None)
  return torch.div(logits, temperature)


def run_scaling(model,data):
		### gonnna need to do this properly
	### splitting data
	split='val'
    node_num = len(data['features'])
    adj_mat = data['adj_mat'] if is_inductive else data['adj_train_norm']
    embeddings = model.encode(data['features'], adj_mat)
    edges=data[f'{split}_edges'].cpu().detach().data.numpy()
    edges_false= data[f'{split}_edges_false'].cpu().detach().data.numpy() 
    if split=='train':
        # print()
        edges_false = edges_false[np.random.randint(0, len(edges_false),len(edges))]

    if not hasattr(model,'task') or model.task=='lp':
    # if False:
            pos_scores = model.decode(embeddings, edges) ### how to we match pos_scores to neg scores?
            neg_scores = model.decode(embeddings, edges_false)
    else:
            pos_scores = model.decide_lp(embeddings, edges) ### how to we match pos_scores to neg scores?
            neg_scores = model.decide_lp(embeddings, edges_false)   


    labels = [1] * pos_scores.shape[0] + [0] * neg_scores.shape[0]
    preds_original = np.array(list(pos_scores.data.cpu().numpy()) + list(neg_scores.data.cpu().numpy()))
	X_calib = preds_original
	y_calib = labels

	CalibratedClassifierCV( base_estimator=model,cv="prefit")
	# CalibratedClassifierCV( base_estimator=model,cv="prefit")
	calibrated_clf.fit(X_calib, y_calib)
	preds_calibrated = calibrated_clf(X_calib)
    loss = F.binary_cross_entropy(labels, preds_original) ### is this average or sum??
    roc = roc_auc_score(labels, preds_original)
    ap = average_precision_score(labels, preds_original)
    acc = binary_acc(labels,preds_original)
    print("YO WHAT")
    print("UNCALIBRATED:",roc,'roc',loss,'loss',ap,'prec',acc,'acc')
    loss = F.binary_cross_entropy(labels, preds_calibrated) ### is this average or sum??
    roc = roc_auc_score(labels, preds_calibrated)
    ap = average_precision_score(labels, preds_calibrated)
    acc = binary_acc(labels,preds_calibrated)
    print("CALIBRATED:",roc,'roc',loss,'loss',ap,'prec',acc,'acc')

    print('roc',loss,'loss',ap,'prec',acc,'acc')
    dfff
	draw_reliability_graph(labels,preds_original)
	draw_reliability_graph(labels,preds_calibrated)

# def run_scaling(model,data_or_valloader):
# 	temperature = nn.Parameter(torch.ones(1).cuda())
# 	args = {'temperature': temperature}
# 	criterion = F.binary_cross_entropy()

# 	# Removing strong_wolfe line search results in jump after 50 epochs
# 	optimizer = optim.LBFGS([temperature], lr=0.001, max_iter=10000, line_search_fn='strong_wolfe')

# 	logits_list = []
# 	labels_list = []
# 	temps = []
# 	losses = []

# 	# if model.is_inductive:  ## do this later
# 	# 	for i, data in enumerate(tqdm(data_or_valloader, 0)):
# 	# 	    images, labels = data[0].to(device), data[1].to(device)

# 	# 	    net.eval()
# 	# 	    with torch.no_grad():
# 	# 	      logits_list.append(net(images))
# 	# 	      labels_list.append(labels)

# 	# Create tensors
# 	split='val'
#     node_num = len(data['features'])
#     adj_mat = data['adj_mat'] if is_inductive else data['adj_train_norm']
#     embeddings = model.encode(data['features'], adj_mat)
#     edges=data[f'{split}_edges'].cpu().detach().data.numpy()
#     edges_false= data[f'{split}_edges_false'].cpu().detach().data.numpy() 
#     if split=='train':
#         # print()
#         edges_false = edges_false[np.random.randint(0, len(edges_false),len(edges))]

#     if not hasattr(model,'task') or model.task=='lp':
#     # if False:
#             pos_scores = model.decode(embeddings, edges) ### how to we match pos_scores to neg scores?
#             neg_scores = model.decode(embeddings, edges_false)
#     else:
#             pos_scores = model.decide_lp(embeddings, edges) ### how to we match pos_scores to neg scores?
#             neg_scores = model.decide_lp(embeddings, edges_false)            
#     labels = [1] * pos_scores.shape[0] + [0] * neg_scores.shape[0]
#     preds = np.array(list(pos_scores.data.cpu().numpy()) + list(neg_scores.data.cpu().numpy()))
# 	logits_list = preds
# 	labels_list = labels

# 	def _eval():
# 	  loss =  F.binary_cross_entropy((T_scaling(logits_list, args), labels_list))
# 	  loss.backward()
# 	  temps.append(temperature.item())
# 	  losses.append(loss)
# 	  return loss


# 	optimizer.step(_eval)

# 	print('Final T_scaling factor: {:.2f}'.format(temperature.item()))

# 	plt.subplot(121)
# 	plt.plot(list(range(len(temps))), temps)

# 	plt.subplot(122)
# 	plt.plot(list(range(len(losses))), losses)
# 	plt.show()


# 	preds_original, _ = test()
# 	preds_calibrated, _ = test(T_scaling, temperature=temperature)

# 	draw_reliability_graph(labels,preds_original)
# 	draw_reliability_graph(labels,preds_calibrated)