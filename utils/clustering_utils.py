
from sklearn.decomposition import PCA

from scipy.cluster.hierarchy import inconsistent
#k=2
#fcluster(Z, k, criterion='maxclust')
int_to_marker=['o','^','s','p','H','d','*','8','1','2','3','4']
pad=100
for p in range(pad):
    int_to_marker+=['.']
def ints_to_marker(int_list,max_int=10):
    markers = ["v",'^','<','>','1','2','3','4','8']
    markers = ['o','^','s','p','P','h','8','*','1','2','3','4']
    # pad=100
    # for p in pad:
    #     markers+=['.']
    # print()
    marker_list= [markers[min(i,max_int)] for i in int_list]

    return marker_list
    
class Dendrogram(): ###goddammit this should be one super dendrogram class, oneway takes a linkage matrix the other takes other shit fuck!
    def __init__(self,data,linkage_matrix,seperate_adj=False,adj_mat=None,merge_method='weighted',data_reduction={'name':'pca','dim':2}):
        if seperate_adj:
            self.adj_mat=adj_mat
        else:
            self.adj_mat=data
        self.data=data

        self.linkage_matrix=linkage_matrix
        self.N= data.shape[0] ## double check this
        self.nodes=[]
        self.lines=[]
        self.tree= nx.DiGraph()
        self.frechet_man = Frechet_Poincare()
#         print(data.shape,'DATA SAHPE')
        for i in range(data.shape[0]):
            node = self.create_node(i)

            self.nodes.append(node)
        for l in linkage_matrix:
            self.add_line(l,merge_method)
            
        self.full_data=np.array([n.coords for n in self.nodes])


        # print(self.full_data)
        self.node_df=self.create_df()
        self.plot_coords=self.get_plot_coords_df(self.full_data,data_reduction)
        # for n in self.nodes:
            # print(n.node_id,'node_id',n.coords)
            # print(self.full_data[n.node_id],'coord_df og')
            # print(self.plot_coords[n.node_id],'coord_df')
        # print(self.plot_coords.shape,'DATAREDICTIOM')
        self.edges=self.get_edges()

        self.lca_dict=self.make_lca_dict()
        self.desc_dict=self.make_desc_dict()
        self.trip_subtree_dict=self.make_triple_subtree_dict()
        # print()
#         print(self.data,'DATA')
#         print(self.full_data,'FULL DATA')
#         print(self.full_data.shape,'should be same dim shape but w/ nodes+lines')

    def make_triple_subtree_dict(self):
        leafs = [l.node_id for l in self.nodes if l.is_leaf]
        lca_dict=self.lca_dict
        desc_dict=self.desc_dict
        subtree_dict={}
        for i in range(len(leafs)):
            Li = leafs[i]
            for j in range(i+1,len(leafs)):
                Lj=leafs[j]
                lca_ij=lca_dict[(Li,Lj)]
                for k in range(j+1,len(leafs)):
                    Lk=leafs[k]
                    lca_ik=lca_dict[(Li,Lk)]
                    lca_jk=lca_dict[(Lj,Lk)]
        #             print()
                    if min(lca_ik,lca_jk,lca_ij)<150:
                        pass

                    C_jk_i= lca_jk in desc_dict[lca_ik]
                    C_jk_i2= lca_jk in desc_dict[lca_ij]

                    C_ik_j= lca_ik in desc_dict[lca_ij]
                    C_ik_j2= lca_ik in desc_dict[lca_jk]

                    C_ij_k= lca_ij in desc_dict[lca_ik]
                    C_ij_k2= lca_ij in desc_dict[lca_jk]
                    assert C_jk_i==C_jk_i2
                    assert C_ik_j==C_ik_j2
                    assert C_ij_k==C_ij_k2
                    subtree_dict[(j,k,i)]=0
                    subtree_dict[(k,j,i)]=0
                    subtree_dict[(i,k,j)]=0
                    subtree_dict[(k,i,j)]=0
                    subtree_dict[(i,j,k)]=0
                    subtree_dict[(j,i,k)]=0              
                    if C_jk_i:
                        subtree_dict[(j,k,i)]=1
                        subtree_dict[(k,j,i)]=1
                    if C_ik_j:
                        subtree_dict[(i,k,j)]=1
                        subtree_dict[(k,i,j)]=1
                    if C_ij_k:
                        subtree_dict[(i,j,k)]=1
                        subtree_dict[(j,i,k)]=1   
        return subtree_dict        
    def make_desc_dict(self):
        desc_dict={}
        for n in self.nodes:
            desc_dict[n.node_id]=set([d for d in n.child_nodes])
        return desc_dict
    def make_lca_dict(self):
        lca_dict={}
        T = self.tree
        for node_tup,lca in nx.tree_all_pairs_lowest_common_ancestor(T):
            if node_tup[0]>90 or node_tup[1]>90:
                continue
            if node_tup[0]==node_tup[1]:
                continue
            tup1=node_tup
            tup2=(node_tup[1],node_tup[0])                                     
            lca_dict[tup1]=lca
            lca_dict[tup2]=lca
        return lca_dict       
    def score(self,score_type='tree_samp'):
        if score_type=='tree_samp':
            score = tree_sampling_divergence(self.adj_mat, self.linkage_matrix)
        elif score_type=='dasguspta_score':
            score = dasgupta_score(self.adj_mat, self.linkage_matrix)
            # score = score/
        elif score_type=='dasguspta_cost':
            score = dasgupta_cost(self.adj_mat, self.linkage_matrix)
            score=score/(self.adj_mat.mean()*100)
        else:
            raise Exception('Invalid choice: {}, \
                            options are {}'.format((score_type,
                                                    ('tree_samp','dasguspta_score','dasguspta_score'))))
        return score
    
    def create_node(self,node_id,add_to_tree=True):
        coords = self.data[node_id]
#         print(node_id,'NODE')
#         print(coords,'NODAL COORd')
        child_nodes=[]
        # child_nodes=[node_id] ## oh come on this is fucked
        node=Node(node_id,direct_children=[],
                  child_nodes=child_nodes,child_leafs=[coords],layer=0,coords=coords) ### may be best 
                                                ##to leave self out of child_nodes
        # print(node,'node?')
        if add_to_tree:
            self.tree.add_node(node.node_id)

        return node
    
    def add_line(self,line,merge_method='weighted',add_to_tree=True):
#         print(line,'LINE')
        assert (int(line[0])*10)==((line[0])*10)
        node1=self.nodes[int(line[0])]
        node2=self.nodes[int(line[1])]

        # print(node1,node2,'NODES')

        distance=line[2]
        num_leaf_nodes=line[3]
        self.lines.append(line)
        new_node=self.merge_nodes(node1,node2,coord_merge_method=merge_method)
        # if add_to_tree:
        #     # self.tree.add_edge(node1.node_id,node2.node_id)
        #     self.tree.add_edge(new_node.node_id,node2.node_id)
        #     self.tree.add_edge(new_node.node_id,node1.node_id)
        assert num_leaf_nodes==new_node.weight
#         print(new_node.node_id)
#         print(len(self.nodes))
        assert len(self.nodes)==(new_node.node_id)
        assert new_node.weight<=self.N
        self.nodes.append(new_node)
    
    def merge_nodes(self,node1,node2,coord_merge_method='weighted',lca_coords=None,node_id=-1,add_to_tree=True):
        ### weighted takes average of coords of the node and weights by # of nodes that make that node up
        ## I believe valid for euclidean
        ## lca_coords gives new coord if merge_method=hyp_lca
        new_layer= max(node1.layer,node2.layer)+1
#         new_id = len(self.lines)+self.N-1
        new_id = node_id if node_id>0 else len(self.lines)+self.N-1
        child_nodes = node1.child_nodes+node2.child_nodes+[node1.node_id,node2.node_id]
        # print(node1.node_id,node2.node_id,'who are the parents?')
        # print(node1.child_nodes,node2.child_nodes,'parents first')
        # print(child_nodes,'okay children')
        child_leafs = node1.child_leafs+node2.child_leafs
        if coord_merge_method=='weighted':
            # print('weighted')
            to_merge=np.array([node1.coords*node1.weight,node2.coords*node2.weight]).T
            coords = np.sum(to_merge,axis=1)/(node1.weight+node2.weight)      
            to_merge_full=np.array([l for l in child_leafs]).T
            coords_full = np.mean(to_merge_full,axis=1)

        elif coord_merge_method=='hyp_lca':
            # print('hyp_lca ............')
            # lca_coords==None
#                 print(lca_coords==None,'NO LCA COORDS')
            # print('')
            # print(node1.coords,node2.coords,'NODE COORDS')
            lca_coords,_=hyp_lca(torch.from_numpy(node1.coords)
                                          ,torch.from_numpy(node2.coords),return_numpy=True)

            # print(lca_coords,'NEED THESE TO MATCH')
            # try:
#                 lca_coords==None
# #                 print(lca_coords==None,'NO LCA COORDS')
#                 # print('')
#                 lca_coords,_=hyp_lca(torch.from_numpy(node1.coords)
#                                               ,torch.from_numpy(node2.coords),return_numpy=True)
#                 # print(lca_coords,'get through?')
#             except:

#                 # pass
# #                 print('truth value of blahb blah blah')
#                 print('lca coords already exist')
#                 pass

            coords=lca_coords

        elif coord_merge_method=='hyp_lca_centroid':
            # print(child_leafs)
#             leaf_nodes=node1.child_lefas
            leafs= torch.Tensor(child_leafs)
            coords=frechet_mean(leafs, self.frechet_man,return_numpy=True)
        else:
            print('huh coords')
            to_merge=np.array([l.coords for l in child_leafs]).T
            coords = np.mean(to_merge,axis=1)    

        direct_children=[node1,node2]       
        new_node=Node(new_id,direct_children,child_nodes,child_leafs,new_layer,coords)
        if add_to_tree:
            self.tree.add_node(new_node.node_id)
            self.tree.add_edge(new_node.node_id,node1.node_id)
            self.tree.add_edge(new_node.node_id,node2.node_id)
        # print(new_node.coords,'ARE WE DONE YUET?',new_node.node_id)
        # print([d.coords for d in new_node.direct_children],'direct children')
        node1.parent_node=new_node
        node2.parent_node=new_node
        node1.has_parent=True
        node2.has_parent=True
        return new_node
#     def get_clusters(self,metric='maxclust',t=5,depth=4):
#     def get_clusters(self,metric='maxclust',t=5,depth=4):
    def get_clusters(self,metric='distance',t=1.5,depth=4):
        if metric!='inconsistency' and depth!=4:
            print('DEPTH UNNECESSARY IF NOT USING INCONSITENCY')
        clusters=fcluster(self.linkage_matrix,t,criterion=metric)
        
        # print(len(clusters),'how many clusters')
        #### would be sick to do internal nodes as well
        return clusters
#         if metric==k:clus
#             if max_d>0:
#                 print('IGNORING MAX D')
#         elif max_d>0:
            
#         elif:
#             raise Exception('must have either k or d')
    def create_df(self):
        columns=['node_id','is_leaf','layer']
        data= [[n.node_id,n.is_leaf,n.layer] for n in self.nodes]
        data_df = pd.DataFrame(columns=columns,data=data)
        # print(data_df)
        return data_df
        
        
        
    def get_plot_coords_df(self,coord_df,data_reduction={'name':'pca','dim':2}):
#         if self.has_reduction:
#             pca
        
        if data_reduction['name']=='pca':
            pca = PCA(n_components=data_reduction['dim'])
            data_plot = pca.fit_transform(coord_df)
        else:
            data_plot=coord_df
        # print(data_plot,"DATA PLOT")


        return data_plot
    
    def get_edges(self):
        data_to_plot = self.plot_coords
        edges=[]
        # for n in self.nodes:
            # if not n.has_parent:
                # continue
            # edges.append([data_to_plot[n.node_id],data_to_plot[n.parent_node.node_id]])

        for n in self.nodes:
            if len(n.direct_children)<2:
                continue
            n1=n.direct_children[0]
            n2=n.direct_children[1]
            edges.append([data_to_plot[n1.node_id],data_to_plot[n2.node_id]])


        edges=np.array(edges)
        return edges
#         print(edges.shape)

    def plot_cluster_progression(self,hyp_edges=True):
        # fig, ax = plt.subplots()
        data_to_plot = self.plot_coords
        leafs=data_to_plot[self.node_df['is_leaf']==True]
        print(len(leafs),'how many leafs?')
        # plt.scatter(leafs[:,0],leafs[:,1],c='gray',alpha=.1)

        for n_id in range(len(self.nodes)):

            n=self.nodes[n_id]


            if n.is_leaf:
                continue

            print(n.node_id,n.coords,self.plot_coords[n.node_id],'NEW NODE')

            fig, ax = plt.subplots()
            ax.scatter(leafs[:,0],leafs[:,1],c='gray',alpha=.5)
            # internal_so_far = data_to_plot[[ch.node_index for ch in self.]]
            # print(n.child_leafs,'CHILD LEAFS')
            # print(n.child_nodes,'node children')
            # print([ch.node_index for ch in n.child_leafs])
            # print([ch.node_index for ch in n.child_leafs if ch not in n.direct_children])
            child_node_ids=[ch for ch in n.child_nodes if ((self.nodes[ch] not in n.direct_children) and (self.nodes[ch].is_leaf))]
            child_coords = data_to_plot[child_node_ids]
            direct_child_coords= data_to_plot[[ch.node_id for ch in n.direct_children ]] ## highest priority
            internal_child_node_ids=[ch for ch in n.child_nodes if ((self.nodes[ch] not in n.direct_children) and (not self.nodes[ch].is_leaf))]
            internal_child_coords = data_to_plot[internal_child_node_ids] ## all these may overlap, so just go exclusive to non exclusive or do ifs and elses
            #### should be split by color for the two direct_child_coords
            e = direct_child_coords
            viz.plot_geodesic(e[0],e[1],ax=ax)

            for ic_id in n.child_nodes:
                ic=self.nodes[ic_id]
                if len(ic.direct_children)<2:
                    continue
                # if ic in n.direct_children
                # if self.nodes[]
                dc=ic.direct_children
                viz.plot_geodesic(dc[0],dc[1],ax=ax,ls='--',alpha=.5,linewidth=.8)

            ax.scatter(direct_child_coords[:,0],direct_child_coords[:,1],)
            if len(internal_child_coords>0):
                ax.scatter(internal_child_coords[:,0],internal_child_coords[:,1],alpha=.4,marker='.',color='blue')
            if len(child_coords)>0:
                ax.scatter(child_coords[:,0],child_coords[:,1],alpha=.4,color='blue')

            # print(n.coords,'n parent')
            # print(direct_child_coords,'direct_children')
            # circ = plt.Circle((0, 0), radius=1, edgecolor='black', facecolor='None', linewidth=3, alpha=0.5)
            # ax.add_patch(circ)
           
            
            ax.set_aspect('equal')
            ax.scatter(n.coords[0],n.coords[1],marker='^',color='green')
            plt.show()

            # if n_id>94:
                # break




    def plot_nodes(self,color_metric='cluster',
                   cluster_metric='maxclust',cluster_t=5,max_layer=2,hyp_edges=False):
        pad_color='orange'
        pad=50
        colors=['b', 'r', 'g', 'y', 'm', 'c', 'k', 'silver', 'lime', 'skyblue', 'maroon', 'darkorange']+[pad_color]*pad

        # hyp_edges=False
#         colors = cm.rainbow(np.linspace(0, 1, 14))
        ## take p argument, that
        data_to_plot = self.plot_coords

        # print(Dat)
        leafs=data_to_plot[self.node_df['is_leaf']==True]
#         internal_df = 
        internals = data_to_plot[self.node_df['is_leaf']==False]
        internal_layers = self.node_df[self.node_df['is_leaf']==False]['layer']
#         print(data_to_plot,'PLOTTING')
#         int_layer_markers=ints_to_marker(internal_layers,max_plot=6)
        if data_to_plot.shape[1]>2:
            print("TOO MANY DIMS, USING FIRST 2")
        if color_metric=='cluster':
#             p
            clusters=self.get_clusters(cluster_metric,cluster_t)
            # print(np.max(clusters),'NUM CLUSTERS')
            colors=[colors[c] for c in clusters]
            
        else:
            raise Exception('COLORS')
        ax = plt.gca()
        plt.scatter(leafs[:,0],leafs[:,1],c=colors,alpha=.5)

        print(internal_layers.unique(),'how manx??')
        
        for l in internal_layers.unique():
            if l>max_layer:
                continue
            to_plot = data_to_plot[self.node_df['layer']==l]
            mark=int_to_marker[l]
            plt.scatter(to_plot[:,0],to_plot[:,1],marker=mark,c='blue')
            
        if hyp_edges:
            for e in self.edges:
                # print(e[:,0],'edge one?')
                # viz.plot_geodesic(e[:,0],e[:,1],ax=ax)
                viz.plot_geodesic(e[0],e[1],ax=ax)
                
        else:
            for e in self.edges:
                plt.plot(e[:,0],e[:,1],c='blue',linestyle='dashed',linewidth=1)
        fig = plt.gcf()
        circ = plt.Circle((0, 0), radius=1, edgecolor='black', facecolor='None', linewidth=3, alpha=0.5)
        ax.add_patch(circ)
       
        
        ax.set_aspect('equal')
        ax.set_xlim(xmin=-1.1,xmax=1.1)
        ax.set_xbound(lower=-1.1, upper=1.1)
        ax.set_ylim(ymin=-1.1,ymax=1.1)
        ax.set_ybound(lower=-1.1, upper=1.1)
   
        plt.show()

    # def
        
class Node():
    def __init__(self,node_id,direct_children,child_nodes,child_leafs,layer,coords):
        self.dims=coords.shape[0]
        origin = np.zeros((self.dims))
        self.node_id=node_id
        self.direct_children=direct_children
        self.child_nodes=child_nodes
        self.child_leafs=child_leafs
        self.layer=layer  ### maybe add layers for p situations....
        self.coords=coords
        self.hyp_radius= poincare_dist(coords,origin)
        self.weight=len(child_leafs)
        self.is_leaf=True if self.layer==0 else False
        self.has_parent=False
        self.parent_node=None
#         print(self.direct_children,'direct')
        # print(node_id,[c.node_id for c in self.direct_children])
        
        
    ### node needs id, merge function, coords, children
        

class HypClustering():
    def __init__(self, embeddings,metric='deepestLCA',method='deepestLCA'):
        """
        method- what will we be taking LCA of/ what will represent merged nodes
            -cumLCA: merged nodes will be represented by their LCA
            -centroid: merged nodes will be represented by the hyperbolic centroid of all of the leaf nodes
            -averageIndy: merged nodes will be represented by all it's nodes individually. 
                        The distance between clusters will be the average LCA height of all nodes
                        This is equivalent to agglomerativeClustering where similarity is LCA
            -singleIndy: will implement alg from Chami, order nodes by deepest LCA, 
                        cluster the trees they belong to
        
        metric- 'deepestLCA' or Hyperbolic distance.. but right now just using deepestLCA
        """
#         print(embedding,'EMBEDDING')
        self.hyp_lca,self.hyp_lca_rad=hyp_lca_all(embeddings)
#         self.node_rads = 
        self.embeddings=embeddings
        self.coords=self.embeddings ## maybe change so embeddings has more info?
        self.metric=metric
        self.merge_method=method
        self.merge_heap=[]
        self.N=embeddings.shape[0]
        self.tree= nx.DiGraph()
        self.nodes=[self.create_node(n) for n in range(self.N)]
        # print('radiii')
        # for n in self.nodes:
            # print(n.node_id)
            # print(np.linalg.norm(n.coords),'eucl rad')
            # print(n.hyp_radius,'HYP RADIUS')
        self.forget_nodes=set([]) ### for some algs, we want to forget nodes, ie forget them if they're popped
        self.lines=[]
        self.frechet_man = Frechet_Poincare()
        self.origin=np.zeros((self.embeddings.shape[1]))
        for i in range(self.N):    
            for j in range(i+1,self.N):
                lca_rad= -self.hyp_lca_rad[i,j] ## give us deepest lca
                heappush(self.merge_heap,(lca_rad,i,j))
        self.lca_dict,self.lca_rad_dict=self.create_lca_dicts()
        self.clustered=False
        self.link_mat=None
        self.dendro_obj=None
#         self.merge_method=met
        
    def get_dendro(self,ground_adj):
        if not self.clustered:
            raise Exception('CANT MAKE DENDRO IF HAVENT CLUSTERED')
        dendi=Dendrogram(self.coords,self.link_mat,
                         seperate_adj=True,adj_mat=ground_adj,merge_method=self.merge_method,
                        data_reduction={'name':'hyperbolic'})
        self.dendro_obj=dendi
        return dendi
    def plot_dendrogram(self,p=5,truncate_mode='level'):
        dendrogram(self.link_mat, truncate_mode=truncate_mode, p=p)
        plt.show()
    def plot_nodes(self,cluster_t=5,max_layer=2,hyp_edges=True):
        if not self.dendro_obj:
            raise Exception('Bro you should fix all this')
        print(max_layer,'maximum layer')
        self.dendro_obj.plot_nodes(cluster_t=cluster_t,max_layer=max_layer,hyp_edges=hyp_edges)
        plt.show()
    def plot_cluster_progression(self):
        self.dendro_obj.plot_cluster_progression(hyp_edges=True)
    def run_cluster(self):
        self.lines=[]
        min_dist=False
        first=True
        while len(self.merge_heap)>0:
#             print(len(self.merge_heap),'MERGE LEN')
#             print(len(self.forget_nodes),'Already forgotten')
            deepest_lca,n1,n2 = heappop(self.merge_heap)
            if n1 in self.forget_nodes:
#                 print(n1,'FORGET ABOUT IT')
                continue
            if n2 in self.forget_nodes:
#                 print(n2,'FORGET ABOUT IT')
                continue
            lca_coords=self.lca_dict[n1][n2]        
            # print('one time')
            merged=self.merge_nodes(self.nodes[n1],self.nodes[n2],
                                    coord_merge_method=self.merge_method,
                                    lca_coords=lca_coords,node_id=len(self.nodes))
            distance=deepest_lca
            # print(n1,n2,'lca_rad',deepest_lca,'lca_coords',lca_coords,)
            
            
            min_rad=min(self.nodes[n1].hyp_radius,self.nodes[n2].hyp_radius)
            if abs(distance)>(min_rad+.0001) and self.merge_method=='hyp_lca':
                print('what the hell')
                print(abs(distance),'lcarad',self.nodes[n1].hyp_radius,self.nodes[n2].hyp_radius,'rad 1 and 2')
                raise Exception('what the hells going on here')
            new_line=[n1,n2,distance,merged.weight]
            self.lines.append(new_line)
            self.nodes.append(merged)
            self.forget_nodes.add(n1)
            self.forget_nodes.add(n2)
            cluster_as_lca=self.merge_method=='hyp_lca_centroid'
            # print('before updat')
            self.update_lcas(merged,cluster_as_lca)## adds new lcas to lca_dicts, and those to merge_heaps

        # print('out?')
        self.clustered=True
        self.link_mat=np.array(self.lines)
        # print(self.link_mat[:,2])
        # print(self.link_mat[:,2].min())
        self.link_mat[:,2]=self.link_mat[:,2]-self.link_mat[:,2].min()
        return self.link_mat
    def update_lcas(self,node,cluster_as_lca=False):
        node_coords=node.coords
        self.lca_dict[node.node_id]={}
        self.lca_rad_dict[node.node_id]={}
        lcas_to_add=[] ## must do this so we don't modify looping list
#         print(len(self.merge_heap),'HEAP LENGTH BEFORE')
#         for n3 in self.merge_heap:
        for n3 in self.nodes:
            if n3 in self.forget_nodes:
#                 print('skipping node n3') ## not completely necissary but speeds things up
                continue 
            n3_coords= n3.coords
            if n3.node_id==node.node_id:
#                 print('same guy?')
                continue
            if cluster_as_lca:
#                 new_lca,new_lca_rad = self.two_node_frechet_mean(node,n3)
                new_lca,new_lca_rad = self.two_node_frechet_mean_grad(node,n3)
#                 print(new_lca_rad,'OG RAD')
                new_lca_rad=-new_lca_rad
#                 print(new_lca_rad,'neg RAD')
                
            else:
                new_lca,new_lca_rad = hyp_lca(torch.from_numpy(node.coords)
                                              ,torch.from_numpy(n3_coords),return_numpy=True)
                new_lca_rad=-new_lca_rad[0]
#             new_lca=new_lca[0]
            self.lca_dict[node.node_id][n3.node_id]=new_lca
            self.lca_dict[n3.node_id][node.node_id]=new_lca
            self.lca_rad_dict[node.node_id][n3.node_id]=new_lca_rad
            self.lca_rad_dict[n3.node_id][node.node_id]=new_lca_rad
            
            heappush(self.merge_heap,(new_lca_rad,node.node_id,n3.node_id))
#             print(len(self.merge_heap),'ADDED ONE W/',node.node_id)
#         print(len(self.merge_heap),'HEAP LENGTH AFTER')

    def merge_nodes(self,node1,node2,coord_merge_method='weighted',lca_coords=None,node_id=-1,add_to_tree=True):
        ### weighted takes average of coords of the node and weights by # of nodes that make that node up
        ## I believe valid for euclidean
        ## lca_coords gives new coord if merge_method=hyp_lca
        
        new_layer= max(node1.layer,node2.layer)+1
#         new_id = len(self.lines)+self.N-1
        new_id = node_id if node_id>0 else len(self.lines)+self.N-1
        child_nodes = node1.child_nodes+node2.child_nodes
        child_leafs = node1.child_leafs+node2.child_leafs
        # print(coord_merge_method,'MERGE METHOD')
        if coord_merge_method=='weighted':
            to_merge=np.array([node1.coords*node1.weight,node2.coords*node2.weight]).T
            coords = np.sum(to_merge,axis=1)/(node1.weight+node2.weight)      
            to_merge_full=np.array([l for l in child_leafs]).T
            coords_full = np.mean(to_merge_full,axis=1)
        elif coord_merge_method=='hyp_lca':
            try:
                lca_coords==None
#                 print(lca_coords==None,'NO LCA COORDS')
                lca_coords=hyp_lca(node1.coords,node2.coords)
            except:
                pass
#                 print('truth value of blahb blah blah')
#                 print('lca coords already exist').
            coords=lca_coords
        elif coord_merge_method=='hyp_lca_centroid':
            ### do we need an initial theta?
            ### could we actually use the hyp_lca as initial theta?
#             theta_k = poincare_pt_to_hyperboloid(self.centroids[i])
#             fmean_k = compute_mean(theta_k, H_k, alpha=0.1)
#             new_centroids[i] = hyperboloid_pt_to_poincare(fmean_k)
            ### should this be the centroid of all nodes? or just the leaf nodes?
                      ### probably just the leaf nodes.
#             print(child_leafs)
#             leaf_nodes=node1.child_lefas
            coords,_ = self.two_node_frechet_mean_grad(node1,node2)
#             leafs= torch.Tensor(child_leafs)

#             coords_calc=frechet_mean(leafs, self.frechet_man,return_numpy=True)
#             print(coords,'grad')
#             print(coords_calc,'calculated')
        else:
            to_merge=np.array([l.coords for l in child_leafs]).T
            coords = np.mean(to_merge,axis=1)          
        direct_children=[node1,node2]       
        new_node=Node(new_id,direct_children,child_nodes,child_leafs,new_layer,coords)
        if add_to_tree:
            self.tree.add_node(new_node.node_id)
            self.tree.add_edge(new_node.node_id,node1.node_id)
            self.tree.add_edge(new_node.node_id,node2.node_id)
        node1.parent_node=new_node
        node2.parent_node=new_node
        node1.has_parent=True
        node2.has_parent=True


        
#         print(node1.coords,'node1')
#         print(node2.coords,'node2')
#         print(new_node.coords,'new_node')
#         print([d.coords for d in new_node.direct_children],'COORDS')
        return new_node
    def create_node(self,node_id,add_to_tree=True):
        coords = self.coords[node_id]
#         print(node_id,'NODE')
#         print(coords,'NODAL COORd')
        node=Node(node_id,direct_children=[],
                  child_nodes=[node_id],child_leafs=[coords],layer=0,coords=coords) ### may be best 
                                                ##to leave self out of child_nodes
        if add_to_tree:
            self.tree.add_node(node.node_id)

        return node
    
    def create_lca_dicts(self):
        lca_dict={}
        lca_rad_dict={}
        for k in range(self.N):
            lca_dict[k]={}
            lca_rad_dict[k]={}
            for t in range(self.N):
                if k==t: ## could be an infinity situation to guarantee value nmw
                    continue
                b = max(k,t)
                a= min(k,t)
#                 print(a,b,'AB')
#                 print(self.hyp_lca.shape,'FULL SHAPE')
                
                lca_dict[k][t]=self.hyp_lca[a,b]
                lca_rad_dict[k][t]=-self.hyp_lca_rad[a,b]
        return lca_dict,lca_rad_dict
    
    def get_score(self,score_type='dasgupta_score'):
        return self.dendro_obj.score(score_type)
    
    def two_node_frechet_mean_grad(self,node1,node2):
        theta,_=hyp_lca(torch.from_numpy(node1.coords)
                                              ,torch.from_numpy(node2.coords),return_numpy=True)
        child_leafs=node1.child_leafs+node2.child_leafs
#         leafs= torch.Tensor(child_leafs)
        leafs=np.array(child_leafs)
        coords=compute_mean(theta=theta,X=leafs)
        rad = poincare_dist(coords,self.origin)
        return coords,rad
    
    def two_node_frechet_mean(self,node1,node2):
#         print(self.origin.shape,'origin shape ')
        child_leafs=node1.child_leafs+node2.child_leafs
        leafs= torch.Tensor(child_leafs)
        coords=frechet_mean(leafs, self.frechet_man,return_numpy=True)
        rad = poincare_dist(coords,self.origin)
        if rad<0:
            raise Exception('RADIUS SHOULDNt BE <0:')
        return coords,rad

def dasgupta_wang(W,T_dict):
    ## W is the similarity matrix (ie. plv mat or whatever)
    ## T_dict takes any triple of leaf nodes (i,j,k), and tells you if i,j are deepest of ij,jk,ik should be subtracted
                ### T_dict should have permutations (i,j,k)=(j,i,k)
    second_term_up=0
    second_term_low=0
    second_term_true=0
    first_term=0
    n_first=0
    n_second=0
#     n_triples
    for i in range(len(W)):
        # for j in range(i+1,len(adj)):
        for j in range(i+1,len(W)):
            if i==j:
                continue
            w_ij=W[i,j]
            first_term+=w_ij
            n_first+=1
            # for k in range(len(adj)):
            for k in range(j+1,len(W)):
                if k in (i,j):
                    continue

                tij_k=T_dict[(i,j,k)]
                tik_j=T_dict[(i,k,j)]
                tjk_i=T_dict[(k,j,i)]
                assert(tij_k+tik_j+tjk_i)==1

                w_ik=W[i,k]
                w_jk=W[j,k]

                w_nojk = w_ij+w_ik
                w_noik = w_ij+w_jk
                w_noij = w_ik+w_jk

                # true= -1*(tjk_i*w_jk+tij_k*w_ij+tik_j*w_ik)
                true= max(tjk_i*w_nojk,
                    tik_j*w_noik,tij_k*w_noij) #### Wang's Dasgupta-- cancel out the weight of the deepest pair
                                ### if that pair is the biggest weight, true=second_term_low=min(2 options)
#                 print(t1,t2,t3,'t1,t2,t3')
                second_term_low+=min(w_nojk,w_noik,w_noij)
                second_term_up+=max(w_nojk,w_noik,w_noij)
                second_term_true+=true
                n_second+=1
    n_triples=n_second
    # n_second=1
    # n_first=1
    lb = (second_term_low)/n_second+(2*first_term)/n_first
    ub = (second_term_up)/n_second+(2*first_term)/n_first
    cost = (second_term_true)/n_second+(2*first_term)/n_first

    lb = (second_term_low)+(2*first_term)
    ub = (second_term_up)+(2*first_term)
    cost= (second_term_true)+(2*first_term)
    # print(n_triples,'how many triples?')
    ratio=ub/lb
    lb/=n_first
    ub/=n_first
    cost/=n_first
#     lb/=n_triples
#     ub/=n_triples
    # print(2*first_term,'FIRST TERM')
    return cost,ub,lb, ratio

def upper_lower_bounds(adj):
    second_term_up=0
    second_term_low=0
    first_term=0
    n_first=0
    n_second=0
#     n_triples
    for i in range(len(adj)):
        # for j in range(i+1,len(adj)):
        for j in range(i+1,len(adj)):
            if i==j:
                continue
            w_ij=adj[i,j]
            first_term+=w_ij
            n_first+=1
            # for k in range(len(adj)):
            for k in range(j+1,len(adj)):
                if k in (i,j):
                    continue
                w_ik=adj[i,k]
                w_jk=adj[j,k]
                t1 = w_ij+w_ik
                t2=w_ij+w_jk
                t3 = w_ik+w_jk
#                 print(t1,t2,t3,'t1,t2,t3')
                second_term_low+=min(t1,t2,t3)
                second_term_up+=max(t1,t2,t3)
                n_second+=1
    n_triples=n_second
    n_second=1
    n_first=1
    lb = (second_term_low)/n_second+(2*first_term)/n_first
    ub = (second_term_up)/n_second+(2*first_term)/n_first
    lb = (second_term_low)+(2*first_term)
    ub = (second_term_up)+(2*first_term)
    ratio=ub/lb
#     lb/=n_triples
#     ub/=n_triples
#     lb/=n_triples
#     ub/=n_triples
    return ub,lb, ratio
                



def plot_clusters(node_coords,hkmeans,p_show=1):
    colors=['b', 'r', 'g', 'y', 'm', 'c', 'k', 'silver', 'lime', 'skyblue', 'maroon', 'darkorange']
    if len(hkmeans.centroids)>len(colors):
        colors = [i for i in plt.cm.get_cmap('tab20').colors]
        
    cluster_labels = np.array([colors[np.argmax(l)] for l in hkmeans.predict(node_coords)])
    to_show= np.random.choice(node_coords.shape[0], size=int(node_coords.shape[0]*p_show), replace=False)
    centroids=hkmeans.centroids
    centroid_ids = np.array([colors[int(l)] for l in np.arange(centroids.shape[0])] )
    
    
#     plt.figure(figsize=(width,height))
    plt.xlim([-1.0,1.0])
    plt.ylim([-1.0,1.0])
    ax = plt.gca()
    circ = plt.Circle((0, 0), radius=1, edgecolor='black', facecolor='None', linewidth=3, alpha=0.5)
    ax.add_patch(circ)
    fig = plt.gcf()
    fig.set_size_inches(8, 8,forward=True)
    ax.scatter(node_coords[:,0][to_show],node_coords[:,1][to_show],color = cluster_labels[to_show])
    
    plt.scatter(centroids[:, 0], centroids[:, 1], s=750, color = centroid_ids, linewidth=2, marker='*');
    plt.show()
#     ax.scatter(node_coords[:,1])
def hyp_cluster_full(node_coords,k=20,add_origin_centroid=False,plot_cents=True):
    """
    node_coords should be nscans x nodes x dims
    """
    n_graphs,n_nodes,n_dims=node_coords.shape
    flat_coords =node_coords.reshape(-1,node_coords.shape[2])
    
    print(node_coords.shape,'OG SHAP')
    print(flat_coords.shape,'FLATTENED SHAPE')
    hkmeans = HyperbolicKMeans(n_clusters=k,dims=n_dims)
    hkmeans.fit(flat_coords, max_epochs=10)  
#     print(n)
    
    k_out = k+1 if add_origin_centroid else k
    node_centroid_dists=np.zeros((n_graphs,n_nodes,k_out))
    graph_centroid_dists=np.zeros((n_graphs,k_out))
    if plot_cents:
        plot_clusters(flat_coords,hkmeans,p_show=.2)
    
    for g in range(n_graphs):
        g_nodes = node_coords[g]
#         print(g_nodes.shape,'node shape')
        node_dist = hkmeans.transform(g_nodes,use_origin=add_origin_centroid) ##node x k  ### could be a good spot to insert origin?
#         print(node_dist.shape,'node dist shape')
        graph_dist = np.mean(node_dist,axis=0) ## k,
#         print(graph_dist.shape,'graph shape')
        node_centroid_dists[g]=node_dist
        graph_centroid_dists[g]= graph_dist 
        
        
        
    ### add some plotting
    
    return graph_centroid_dists,node_centroid_dists,hkmeans