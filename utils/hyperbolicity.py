import os
import pickle as pkl
import sys
import time
# from datetime import time
import networkx as nx
import numpy as np
from tqdm import tqdm
from utils.data_utils import load_data_lp


def hyperbolicity_sample(G, num_samples=50000): ## uh oh don't fuck with this, go back and add og copy
    curr_time = time.time()
    hyps = []
    
    total1 = 0
    total2 = 0
    dyk1 = time.time()
    # paths = nx.all_pairs_dijkstra_path_length(G, cutoff=None, weight='weight')


    # paths = nx.all_pairs_dijkstra_path_length(G, cutoff=None,weight='None')
    # paths = nx.all_pairs_shortest_path_length(G, cutoff=None)
    # all_pairs_shortest_path_length
    # print('all paths')
    # pdict= dict(paths)
    # print('dicted')
    # print(paths)
    # paths=dict(paths)
    # print('paths now a dict')
    # print(len([e for e in G.edges()]),'how many edges?')
    # print(len([e for e in G.nodes()]),'how many nodes?')
    # print(len([ c for c in nx.connected_components(G)]),'num components')
    # print(len(max(nx.connected_components(G), key=len)),'largest components')
    # print(time.time()-dyk1,'all shortest')
    # total1 +=(time.time()-dyk1)
    # # print('finished')

    pdict = {}
    # # pdict=paths
    # # # print(paths[0])
    # # # print(paths[0][2])
    # # print(len(paths),'so many paths')

    # print

    # print(len(paths),'path length!')
    # for p in paths: ## this loop may slow everything enough that its not worth it to precompute
    # #     # print(p[0],'o 0 ')
    # #     # print(p[1],'p1')
    #     pdict[p[0]]=p[1]
    #     if len(pdict.keys())%500==500:
    #         print(len(pdict.keys()))

        # pdict[p[1]]=p[0]
    # print([p for p in paths])
    # print('outof p dict')
    # s
    fails=0
    for i in tqdm(range(num_samples)):
        # curr_time = time.time()
        node_tuple = np.random.choice(G.nodes(), 4, replace=False)
        s = []
        # print(i,'sampled?')
        # if i%4000 ==50:
        #     # print(float(fails)/i,'PCT FAILS AFTER ',i)
        #     try:
        #         print('hyp so far: ',max(hyps))
        #     except:
        #         print('nothing good so far')
        try:
            # # norm1 = time.time()
            d01 = nx.shortest_path_length(G, source=node_tuple[0], target=node_tuple[1], weight=None) ### this all from Chami og. replaced by doing all shortest paths first and memoizing
            d23 = nx.shortest_path_length(G, source=node_tuple[2], target=node_tuple[3], weight=None)
            d02 = nx.shortest_path_length(G, source=node_tuple[0], target=node_tuple[2], weight=None)
            d13 = nx.shortest_path_length(G, source=node_tuple[1], target=node_tuple[3], weight=None)
            d03 = nx.shortest_path_length(G, source=node_tuple[0], target=node_tuple[3], weight=None)
            d12 = nx.shortest_path_length(G, source=node_tuple[1], target=node_tuple[2], weight=None)
            # norm1 = time.time()
            # d01 = nx.shortest_path_length(G, source=node_tuple[0], target=node_tuple[1], weight='weight')
            # d23 = nx.shortest_path_length(G, source=node_tuple[2], target=node_tuple[3], weight='weight')
            # d02 = nx.shortest_path_length(G, source=node_tuple[0], target=node_tuple[2], weight='weight')
            # d13 = nx.shortest_path_length(G, source=node_tuple[1], target=node_tuple[3], weight='weight')
            # d03 = nx.shortest_path_length(G, source=node_tuple[0], target=node_tuple[3], weight='weight')
            # d12 = nx.shortest_path_length(G, source=node_tuple[1], target=node_tuple[2], weight='weight')
            # # print(time.time()-norm1,'4 dyk')
            # total2 += (time.time()-norm1)
            # print(d01,d23,d02,d13,d03,d12,'beofre')
            after=time.time()
            # d01 = pdict[node_tuple[0]][node_tuple[1]]
            # d23 = pdict[node_tuple[2]][node_tuple[3]]
            # d02 = pdict[node_tuple[0]][node_tuple[2]]
            # d13 = pdict[node_tuple[1]][node_tuple[3]]
            # d03 = pdict[node_tuple[0]][node_tuple[3]]
            # d12 = pdict[node_tuple[1]][node_tuple[2]]
            # print(time.time()-after,'recall')
            total1 +=(time.time()-after)
            # print(d01,d23,d02,d13,d03,d12,'after')
            s.append(d01 + d23)
            s.append(d02 + d13)
            s.append(d03 + d12)
            s.sort()
            hyps.append((s[-1] - s[-2]) / 2)
            # print(d01,d02,d12,d13,d23,'shortest paths should')
        except Exception as e:
            fails+=1
            # print(e)
            # print(node_tuple)
            # print('disconnected comp')
            # print('disconnected')
            # sfg
            continue

    print('Time for hyp: ', time.time() - curr_time)
    # print(total1,'our way')
    # print(total2,'theirs')
    if len(hyps)==0:
        return hyperbolicity_sample(G,num_samples=10000)
    print('hyperbolicity',max(hyps))

    return max(hyps)

# def 


if __name__ == '__main__':
    dataset = 'pubmed'
    data_path = os.path.join(os.getcwd()+'\\data', dataset)
    data = load_data_lp(dataset, use_feats=False, data_path=data_path)
    graph = nx.from_scipy_sparse_matrix(data['adj_train'])
    print('Computing hyperbolicity', graph.number_of_nodes(), graph.number_of_edges())
    hyp = hyperbolicity_sample(graph)
    print('Hyp: ', hyp)

