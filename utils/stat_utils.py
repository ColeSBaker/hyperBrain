
import pandas as pd
import numpy as np
import pingouin as pg
from statsmodels.stats.multitest import fdrcorrection
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
plt.style.use('seaborn')
import seaborn as sns
import os
def standard_anova(data,columns,clinical_df):
    # planning on mixing with mixed_anova
    df = pd.DataFrame(data)
    stat_columns=df.columns
    df[columns]=np.array(clinical_df[columns])

    first=True
    for c in stat_columns:
        df[c]=pd.to_numeric(df[c])
        res = pg.anova(dv=c, between=columns,data=df)
        res['metric']=c
        if first:
            combined_results=res
            first=False
        else:
            combined_results=pd.concat([combined_results,res],axis=0)
    
    # alternative could do label only before the error correction...
    # but for now we're still sig
    combined_results=combined_results.reset_index()
    fdr=fdrcorrection(pvals=combined_results['p-unc'].values,alpha=.05)    
    combined_results[['Significant','q_value']]= np.array([fdr[0],fdr[1]]).T
    combined_results['Significant']=fdr[0]
    combined_results['q_value']=fdr[1]
    combined_results['p_value']=combined_results['p-unc']
    
    # label_only=combined_results[combined_results['Source']=='label']
    # label_only=label_only.reset_index()
    # fdr=fdrcorrection(pvals=label_only['p-unc'].values,alpha=.05) 
    # label_only[['Significant_alt','q_value_alt']]= np.array([fdr[0],fdr[1]]).T
    return combined_results

def mixed_anova(data,split_labels,clinical_df):
    """
    should be to pass in arg to make purely repeated measure
    """
    df = pd.DataFrame(data)
    stat_columns=df.columns
    df['label']=split_labels
    df['time_labels']=np.array(clinical_df['Pre'])
    df['ID']=np.array(clinical_df['ID'])

    first=True
    for c in stat_columns:
        df[c]=pd.to_numeric(df[c])
        res = pg.mixed_anova(dv=c, between='label',within='time_labels',subject='ID', data=df)
        res['metric']=c
        if first:
            combined_results=res
            first=False
        else:
            combined_results=pd.concat([combined_results,res],axis=0)
    
    # alternative could do label only before the error correction...
    # but for now we're still sig
    combined_results=combined_results.reset_index()
    print(combined_results,'COMBINED RESULTS')

    fdr=fdrcorrection(pvals=combined_results['p-unc'].values,alpha=.05)    
    combined_results[['Significant','q_value']]= np.array([fdr[0],fdr[1]]).T
    combined_results['Significant']=fdr[0]
    combined_results['q_value']=fdr[1]
    combined_results['p_value']=combined_results['p-unc']
    
    label_only=combined_results[combined_results['Source']=='label']
    label_only=label_only.reset_index()
    time_only=combined_results[combined_results['Source']=='time_labels']
    time_only=time_only.reset_index()
    # fdr=fdrcorrection(pvals=label_only['p-unc'].values,alpha=.05) 
    # label_only[['Significant_alt','q_value_alt']]= np.array([fdr[0],fdr[1]]).T
    return combined_results,label_only,time_only


# def welch_ttest(entity,g1, g2): 
def welch_from_columns(g1,g2,equal_var=False):
        t, p = stats.ttest_ind(g1, g2, equal_var = equal_var) 
        
        return [p,t]
    
def welch_from_columns_onepop(g1,popmean=0):
    t, p = stats.ttest_1samp(g1, popmean = popmean) 
    return [p,t]

def conf_interval(pop,conf=.95):
    mean = pop.mean()
    z_score = conf_to_z(conf)
    SE = pop.std()/(pop.shape[0]**(1/2))
    offset = z_score*SE
#     print((pop.shape[0]**(1/2)),'popshape')
#     print(offset)
    conf_int = (mean-offset,mean+offset)
    return conf_int
    
def conf_to_z(conf):
    st.norm.ppf(1-(1-0.95)/2)
    return st.norm.ppf(1-(1-conf)/2)

def welch_ttest(df,split_labels,equal_var=False): 
    ## Welch-Satterthwaite Degrees of Freedom ##
    ### lost our node information... this is dangerous?
    # print(df,'DF')
    split_options = np.array(list(set(list(split_labels)))).astype(int)
    split_labels=np.array(split_labels).astype(int)
    split_options.sort()
    # print(split_options,'split options')
    # print(split_labels)
    # eeffe
    split_0 = df[split_labels==split_options[0]]
    if len(split_options)==1:
    
        data = np.array([[r]+welch_from_columns_onepop(split_0[:,r],popmean=0) for r in range(df.shape[1])])
        dataframe = pd.DataFrame(data=data,columns=['welch_ttest','p_value','t_value'])
    else:
        split_1 = df[split_labels==split_options[1]]
        data = np.array([[r]+welch_from_columns(split_0[:,r],split_1[:,r],equal_var=equal_var) for r in range(df.shape[1])])
    dataframe = pd.DataFrame(data=data,columns=['welch_ttest','p_value','t_value'])
    #### this is where we drop rsns
#     dataframe['conf1']=
    
    # print(dataframe,'dedd')
    fdr_alpha=.05
    fdr= fdrcorrection(dataframe['p_value'].values,alpha=fdr_alpha)
    print("ALPHA AT {}".format(fdr_alpha))
    
#     print(np.array(fdr))
    dataframe[['Significant','q_value']]= np.array([fdr[0],fdr[1]]).T
    return dataframe

# print(conf_to_z(.95))
def remove_rsn(stat_df,labels ,ignore=''):
    include=[]
    for l in labels:

        rsn_split=l.split('_')
        if len(rsn_split)<2:
            include.append(True)

        elif rsn_split[0]==ignore:
            include.append(False)
        elif (len(rsn_split)==2) and (rsn_split[1]==ignore):
            # print('REMOVE')
            include.append(False)
        else:
            # print('TRUTH')
            include.append(True)
    include=np.array(include)
    # print(stat_df.shape,'STAT SHAPE')
    # print(include.shape,'include SHAPE')
    include_df=stat_df[:,np.array(include)]
    include_labels=np.array(labels)[np.array(include)]

    # print('include_labe')
    return include_df,include_labels

def metric_analysis(graph_by_metric_df,entity_names,graph_labels,metric_col=[],y_axis='',analyze_time=False,
                    plot_title='',graph_label_names=['Healthy Control','SCI'],sort_vals=False,max_plot=100,column_name=[]
                   ,print_pval=True,print_qval=True,df_save_path='',ignore_rsn='',group_analysis=True,clinical_df=None,first_stat=None,plot_save_path=''):
    """
    graph_by_metric_df- should be df w/ flattened metric for every graph ie. 90x 8000 for 90 graphs w/ 8000 metrics
    entity_labels-the name we're gonna use for plotting each of these. if one metric per node, entity is node 1-90, if cross node should be 90*89/2
    #graph_labels. graphx1, whatever we want to group by
    column_name- whatever we want to label the y_axis ie. the name of the metric. Doesn't have to correspond to anything
    


    """
    # sig_df = welch_ttest(eval_df,graph_labels).sort_values(by='p_value')

    # graph_by_metric_df
    # print(graph_by_metric_df,'GRAPH LABELS PRE')

    # graph_by_metric_df,entity_names=remove_rsn(graph_by_metric_df,entity_names,ignore_rsn)
    # print(graph_by_metric_df,'GRAPH LABELS POST')

    # raise Exception('GAME PLAN---- functionalize plotting, so that you can split eval_df into 2,')
    if metric_col:
        pass
    elif graph_label_names== ['Healthy Control', 'SCD']:
        metric_col='diagnosis'
    elif graph_label_names== ['CogTr', 'Control']:
        metric_col='CogTr'
    elif graph_label_names== ['Post', 'Pre']:
        metric_col='Pre'
    else:
        skip_val=True


    if metric_col:

        val_col=np.array(clinical_df[metric_col])
        val_col=val_col-val_col.min()
        clinical_df[metric_col]=val_col
        print(val_col,'val col')
        print(graph_labels,'graph met')
        assert np.sum(np.abs((val_col-graph_labels)))==0
        print('ALL EQUAL!')



    eval_df=graph_by_metric_df

    # savepath_df=os.path.join(save_dir,column_name[0]+'_df.csv')

    # print(eval_df,'EVALUATION DF')
    # print(eval_df.shape)
    # sss
    # print(entity_names,'ENTITY NAMES')
#     print(eval_df,'evals')
    # sig_df = welch_ttest(eval_df,graph_labels).sort_values(by='p_value')  
    # sig_df = welch_ttest(eval_df,graph_labels).sort_values(by='p_value')  

    print(graph_labels,'LABELS')
    # if False:
    if clinical_df is not None and len(clinical_df['ID'].unique())*2<=len(clinical_df['ID']):
        repeated=True
        print('measures are repeated, doing mixed anaylsis instead of ttest')
        sig_df_combined,sig_df_label,sig_df_time = mixed_anova(eval_df,graph_labels,clinical_df)
        sig_df_list=[sig_df_label,sig_df_time]
        graph_label_name_list=[graph_label_names,['Post','Pre']]
        # this needs to be a flarnking variable!!
        sig_df=sig_df_list[0]
        graph_label_names=graph_label_name_list[0]
        graph_labels=clinical_df['Pre'].values
        graph_labels=clinical_df[metric_col].values

        print(graph_labels,'GRAPH LABELS')
        # zll
        # if analyze_time:
        #     sig_df=sig_df_list[1]
        #     graph_label_names=graph_label_name_list[1]
        # else:
        #     sig_df=sig_df_list[0]
        #     graph_label_names=graph_label_name_list[0]
    else:
        repeated=False
        sig_df = welch_ttest(eval_df,graph_labels)
        sig_df_combined=sig_df
        sig_df_list=[sig_df]
        graph_label_name_list=[graph_label_names]


    print(sig_df_combined,'SIG NF COMBINED')
    # ekeke

    print(sig_df,'SIG DF')
    print(sig_df.shape,'SIG DF')
    # sksks
    
    n_plot=min(sig_df.shape[0],max_plot)
    keepers = list(sig_df[:n_plot].index.astype(int))
# 
    # print(sig_df[:max_plot],'shoudl be top')
    # print(keepers,'keeps!')
    use_full_names=True
    full_entity_names=[]
    str_qs=[]
    str_ps=[]
    
    

    for i in sig_df.index.astype(int):
        ### full sorted list of new names.. 
        
        new_name=entity_names[i]
        
        p =sig_df[sig_df.index==i]['p_value'].values[0]
        q =sig_df[sig_df.index==i]['q_value'].values[0]
    
        if p<.0001:
           str_p =np.format_float_scientific(p, precision = 0, exp_digits=1)
        elif p<.001:
            str_p=str(p)[1:6]
            
        else:
            str_p=str(p)[1:5]
#         str_q='q-value: '+str(q)[1:5]
#         str_p='p-value: '+str_p
       
        str_p='p: '+str_p
        
        if q<.0001:
           str_q =np.format_float_scientific(q, precision = 0, exp_digits=1)
        elif q<.001:
            str_q=str(q)[1:6]
            
        else:
            str_q=str(q)[1:5]
         
        str_q='q: '+str_q
            
        if q<.05:
            str_q+='*'
        if p<.05:
            str_p+='*'
#         str_q='q: '+str(q)[1:5] ,
        mean_str=get_group_mu_str(eval_df[:,i],graph_labels)
        add_mean_str=False
        new_name = entity_names[i]+'\n'+str_p
        if True:
            new_name+='\n'+str_q
        if add_mean_str:
            new_name+='\n'+mean_str
#         if q!=p:
#             new_name+='\n'+str_q
        full_entity_names.append(new_name)
        str_qs.append(str_q)
        str_ps.append(str_p)

    show_data=np.array([entity_names,str_ps,str_qs]).T
    show_df=pd.DataFrame(columns=[['StatName','p_value','q_value']],data=show_data)
    if df_save_path:
        sig_df_combined.to_csv(df_save_path)

    if sort_vals:
#         eval_df=eval_df[eval_df['entity'].isin(keepers)]
        eval_df=eval_df[:,keepers]
        
        if use_full_names:  ## these will have already been sorted... you're all twisted
            entity_names=full_entity_names[:n_plot]
        else:
            entity_names=[entity_names[i] for i in keepers]
    
    eval_df = pd.DataFrame(np.ravel(eval_df), columns=y_axis)  ### ADD BIG ASS STARS
    num_labels = eval_df.shape[0]/graph_labels.shape[0]
    assert num_labels - int(num_labels) == 0 ### make sure this makes a whole number
    eval_df['label'] = np.repeat(graph_labels,num_labels)
    eval_df['label'] = eval_df.label.apply(lambda x: graph_label_names[int(x)])
    print('done with label')
    eval_df['network'] = np.tile(entity_names, graph_labels.shape[0]) ### where the names, pvals etc. gets set
    print(eval_df.shape)
    print('eval df shape')
    print('tile up')
#     print(eval_df['network'])
#     print(eval_df['network'].shape,'what is this mystery?')
    sns.set(style="whitegrid")
# ax = sns.boxplot(x=tips["total_bill"], whis=[5, 95])
    cis= np.array([[-0.35, 0.50] for i in range(len(keepers))])
    ax = sns.boxplot(x="network", y=y_axis[0], hue="label", data=eval_df, palette="pastel",whis=1.,notch=True,bootstrap=1000,showfliers = False)
#     ax = sns.swarmplot(x="network", y=column_name[0], hue="label", dta=eval_df)
    print('bout to show')
    plt.title(plot_title, size=16)
    if plot_save_path:
        print("SAVING")
        print(plot_save_path)
        plt.savefig(plot_save_path)

        # plt.save()
    plt.show();
    # lkjfg
# def plot_analysis()

    # lkjfg
# def plot_analysis()

def get_group_mu_str(eval_df,graph_labels):
    label_options=list(set(graph_labels))
    label_options.sort()
    # print(label_options)
    # print(eval_df,'EVAL DF')
    metr_str=''
    for o in label_options:

        sub_df=eval_df[graph_labels==o]
        # print(sub_df.mean(),'mean')
        # metr_str+='\n'+o+' '+str(sub_df.mean())
        metr_str+='\n'+str(o)+' '+ np.format_float_positional(np.median(sub_df),precision=3)
        metr_str+='\n'+ np.format_float_positional(sub_df.std(),precision=3)
        metr_str+='\n'+ str(sub_df.shape[0])

    return metr_str
def plot_boxplot(eval_df,entity_names,graph_labels,plot_title, column_name='',graph_label_names=[''],max_to_plot=-1):
    eval_df = pd.DataFrame(np.ravel(eval_df), columns=column_name)
    num_labels = eval_df.shape[0]/graph_labels.shape[0]
    assert num_labels - int(num_labels) == 0 ### make sure this makes a whole number
    eval_df['label'] = np.repeat(graph_labels,num_labels)
    
    # print(graph_labels,'graph labels')
    # print(num_labels,'num labels')
    # print(graph_label_names,'label names')
    # print(eval_df.label.unique(),'UNIQUE LABLES')

    eval_df['label'] = eval_df.label.apply(lambda x: graph_label_names[int(x)])
    eval_df['network'] = np.tile(entity_names, graph_labels.shape[0])
#     print(eval_df['network'])
#     print(eval_df['network'].shape,'what is this mystery?')
    sns.set(style="whitegrid")
# ax = sns.boxplot(x=tips["total_bill"], whis=[5, 95])
    # if max_to_plot>
    # cis= np.array([[-0.35, 0.50] for i in range(len(keepers))])
    sns.stripplot(x="network", y=column_name[0], hue="label", data=eval_df, palette="pastel")
    # ax = sns.boxplot(x="network", y=column_name[0], hue="label", data=eval_df, palette="pastel",whis=1.,notch=True,bootstrap=1000)
#     ax = sns.swarmplot(x="network", y=column_name[0], hue="label", dta=eval_df)
    plt.title(plot_title, size=16)
    plt.show();

def stack_angular(stat_df):
    angles=['theta','phi','alpha','beta']  ### need to expand to more angles
    angle_feats=['sin_theta']+['cos_'+a for a in angles]
    angle_feats=[af for af in angle_feats if af in stat_df[0][0].columns]
    thetasin_df=[]
    thetacos_df=[]
    phi_df=[]
    radius_df=[]
    all_angle_df=[]
    angle_feat_dict={af:[] for af in angle_feats}


    for s in range(len(stat_df)): ### TO AUTOMATE FOR MULTIPLE DIMS IF YOU WANT!!
        clust_list_all_ang=[]
        for af in angle_feats:
            clust_list_af=[]
            for c in stat_df[s]:
                clust_list_af.append(c[af].values)
                clust_list_all_ang.append(c[af].values)
            angle_feat_dict[af].append(clust_list_af)
        all_angle_df.append(clust_list_all_ang)
    for af,vals in angle_feat_dict.items():
        # print(np.array(vals).shape,'OG ANGLE SHAPE')
        angle_feat_dict[af]=np.array(vals)[:,:,0]


    for s in range(len(stat_df)):
            clus_list=[]
            for c in stat_df[s]:
        #         print(c['sin_theta'])
                clus_list.append(c['hyp_r'].values)

            radius_df.append(clus_list)
    # for s in range(len(stat_df)):
    #     clus_list=[]
    #     for c in stat_df[s]:
    # #         print(c['sin_theta'])
    #         clus_list.append(c['sin_theta'].values)
    #     thetasin_df.append(clus_list)
        
    # thetacos_df=[]
    # for s in range(len(stat_df)):
    #     clus_list=[]
    #     for c in stat_df[s]:
    # #         print(c['sin_theta'])
    #         clus_list.append(c['cos_theta'].values)
    #     thetacos_df.append(clus_list)

    # # for s in range(len(stat_df)):
    # #         clus_list=[]
    # #         for c in stat_df[s]:
    # #     #         print(c['sin_theta'])
    # #             clus_list.append(c['cos_phi'].values)
    # #         phi_df.append(clus_list)

    # for s in range(len(stat_df)):
    #         clus_list=[]
    #         for c in stat_df[s]:
    #     #         print(c['sin_theta'])
    #             clus_list.append(c['hyp_r'].values)

    #         radius_df.append(clus_list)

    # thetasin_df=np.array(thetasin_df)[:,:,0]
    # thetacos_df=np.array(thetacos_df)[:,:,0]
    # phi_df=np.array(phi_df)[:,:,0]
    radius_df=np.array(radius_df)[:,:,0]
    all_angle_df=np.array(all_angle_df)[:,:,0]
    # print(radius_df,'RADIUS DUH')
    return radius_df,angle_feat_dict,all_angle_df
    # return radius_df,thetasin_df,thetacos_df,""
    # return radius_df,thetasin_df,thetacos_df,phi_df

def digest_stat_dict(stat_dict,useable_ids):
    wc = stat_dict['wc']['stat_df'][useable_ids]
    wc_labels = stat_dict['wc']['labels']
    bc = stat_dict['bc']['stat_df'][useable_ids]
    bc_labels = stat_dict['bc']['labels']
    rc = stat_dict['rc']['stat_df'][useable_ids]
    rc_labels = stat_dict['rc']['labels']
    node = stat_dict['node_rad']['stat_df'][useable_ids]
    node_labels = stat_dict['node_rad']['labels']
    pw = stat_dict['pw']['stat_df'][useable_ids]
    pw_labels = stat_dict['pw']['labels']

    hkc=stat_dict['hkc']['stat_df']
    # radius_df,thetasin_df,thetacos_df,_=stack_angular(hkc)
    radius_df,angle_feat_dict,all_angle_df=stack_angular(hkc)
    radius_df=radius_df[useable_ids]
    all_angle_df=all_angle_df[useable_ids]
    for af,vals in angle_feat_dict.items():
        angle_feat_dict[af]=np.array(vals)[useable_ids]
    # thetasin_df=thetasin_df[useable_ids]
    # thetacos_df=thetacos_df[useable_ids]
    return wc,bc,rc,node,pw,radius_df,angle_feat_dict,all_angle_df,wc_labels,bc_labels,rc_labels,node_labels,pw_labels

def digest_stat_dict_prob(stat_dict,useable_ids):
    wc_prob = stat_dict['wc_prob']['stat_df'][useable_ids]
    wc_labels = stat_dict['wc']['labels']
    bc_prob = stat_dict['bc_prob']['stat_df'][useable_ids]
    bc_labels = stat_dict['bc']['labels']
    rc_prob = stat_dict['rad_prob']['stat_df'][useable_ids]
    rc_labels = stat_dict['rc']['labels']
    node_prob = stat_dict['node_rad_prob']['stat_df'][useable_ids]
    node_labels = stat_dict['node_rad_prob']['labels']
    pw_prob = stat_dict['pw_prob']['stat_df'][useable_ids]
    pw_labels = stat_dict['pw_prob']['labels']
    return wc_prob,bc_prob,rc_prob,node_prob,pw_prob,wc_labels,bc_labels,rc_labels,node_labels,pw_labels

def drop_rsn_metrics(stat_dict,ignore_abrevs=set(['SMN'])):
    hjhhke

def slice_metrics(stat_dict,metric_col,clinical_df,conditionals,use_difference=False,
    use_second_stat=False,stat_dict2=None,save_dir='',ignore_rsn=''):

    # print()
    # ignore_rsn=''
    group_analysis=True
    if use_second_stat:
        assert metric_col=='band'
    if metric_col in ('diagnosis','diagnosis_inv'):
        group_labels = ['Healthy Control', 'SCD']
    if metric_col == 'CogTr':
        # group_labels=['Control','CogTr']
        group_labels=['CogTr','Control']
    if metric_col == 'Pre':
        group_labels=['Post','Pre']
    if metric_col in ('age','age_dyn','diagnosis_age','diagnosis_train'):
        group_labels=['Age']
        group_analysis=False
    if metric_col=='diagnosis_train_cat':
        group_labels= ['HC Train', 'HC No Train','SCD Train', 'SCD No Train']
        # group_labels= ['HC Train','SCD Train', 'HC No Train', 'SCD No Train']
        # group_labels= ['HC Train','SCD Train']
    if metric_col=='band':
        assert use_second_stat
        group_labels=['Alpha','Gamma']
    if metric_col=='constant':
        group_labels=['All']

    # side_X=clinical_df['side']

    # if use_second_stat:
        # group_labels
    


    # patient_ids=clinical_df[conditionals]['ID'].values
    # conditionals=True

    # print(stat_dict)

    if use_difference:
        clinical_df=clinical_df.sort_values('Scan Index')
        # useable_1=clinical_df[conditionals&(clinical_df.Pre==1)].sort_values('ID')
        # useable_2=clinical_df[conditionals&(clinical_df.Pre==0)].sort_values('ID')

        useable_1=clinical_df[conditionals&(clinical_df.Pre==1)]
        useable_2=clinical_df[conditionals&(clinical_df.Pre==0)]

        scan_labels = clinical_df[conditionals&(clinical_df.Pre==0)][metric_col].values  #### better be the same!
        clinical_use=clinical_df[conditionals&(clinical_df.Pre==0)]
        pats1=useable_1['ID'].values.tolist()
        pats2=useable_2['ID'].values.tolist()
        print(pats1)
        print(pats2)
        assert pats1==pats2
        useable_ids1=useable_1['Scan Index'].values
        useable_ids2=useable_2['Scan Index'].values
        print('still need to change since we switched stack ang')
        wc1,bc1,rc1,node1,pw1,radius_df1,angle_feat_dict1,all_angle_df1,wc_labels,bc_labels,rc_labels,node_labels,pw_labels=digest_stat_dict(stat_dict,useable_ids1)
        thetasin_df1=angle_feat_dict1['sin_theta']
        thetacos_df1=angle_feat_dict1['cos_theta']
        wc2,bc2,rc2,node2,pw2,radius_df2,angle_feat_dict2,all_angle_df2,wc_labels2,bc_labels2,rc_labels2,node_labels2,pw_labels2=digest_stat_dict(stat_dict,useable_ids2)

        thetasin_df2=angle_feat_dict2['sin_theta']
        thetacos_df2=angle_feat_dict2['cos_theta']
        wc =wc2-wc1
        bc = bc2-bc1
        rc = rc2-rc1

        node = node2-node1
        pw = pw2-pw1
        radius_df=radius_df2-radius_df1
        thetasin_df=thetasin_df2-thetasin_df1
        thetacos_df=thetacos_df2-thetacos_df1

        wc_prob1,bc_prob1,rc_prob1,node_prob1,pw_prob1,wc_labels,bc_labels,rc_labels,node_labels,pw_labels= digest_stat_dict_prob(stat_dict,useable_ids1)
        wc_prob2,bc_prob2,rc_prob2,node_prob2,pw_prob2,wc_labels,bc_labels,rc_labels,node_labels,pw_labels= digest_stat_dict_prob(stat_dict,useable_ids2)

        wc_prob =wc_prob2-wc_prob1
        bc_prob = bc_prob2-bc_prob1
        rc_prob = rc_prob2-rc_prob1

        node_prob = node_prob2-node_prob1
        pw_prob = pw_prob2-pw_prob1

        # hkc=hkc1-hkc2


    elif use_second_stat:
        useable_both=clinical_df[conditionals]['Scan Index'].values
        clinical_use=clinical_df[conditionals]

        scan_labels = np.array([0 for i in range(len(useable_both))]+[1 for i in range(len(useable_both))])  #### better be the same!

        # pats1=useable_both['ID'].values.tolist()

        # assert pats1==pats2
        wc1,bc1,rc1,node1,pw1,radius_df1,angle_feat_dict1,all_angle_df1,wc_labels,bc_labels,rc_labels,node_labels,pw_labels=digest_stat_dict(stat_dict,useable_both)
        thetasin_df1=angle_feat_dict1['sin_theta']
        thetacos_df1=angle_feat_dict1['cos_theta']

        wc2,bc2,rc2,node2,pw2,radius_df2,angle_feat_dict2,all_angle_df2,wc_labels2,bc_labels2,rc_labels2,node_labels2,pw_labels2=digest_stat_dict(stat_dict2,useable_both)
        thetasin_df2=angle_feat_dict2['sin_theta']
        thetacos_df2=angle_feat_dict2['cos_theta']

        # wc,bc,rc,node,pw,radius_df,thetasin_df,thetacos_df,wc_labels,bc_labels,rc_labels,node_labels,pw_labels=digest_stat_dict_prob(stat_dict,useable_ids)
        wc_prob1,bc_prob1,rc_prob1,node_prob1,pw_prob1,wc_labels,bc_labels,rc_labels,node_labels,pw_labels= digest_stat_dict_prob(stat_dict,useable_both)
        wc_prob2,bc_prob2,rc_prob2,node_prob2,pw_prob2,wc_labels,bc_labels,rc_labels,node_labels,pw_labels= digest_stat_dict_prob(stat_dict2,useable_both)
        # print('hello>>>')

        wc =np.vstack([wc1,wc2])
        bc = np.vstack([bc1,bc2])
        rc =np.vstack([rc1,rc2])
        node = np.vstack([node1,node2])
        pw = np.vstack([pw1,pw2])

        wc_prob =np.vstack([wc_prob1,wc_prob2])
        bc_prob = np.vstack([bc_prob1,bc_prob2])
        rc_prob =np.vstack([rc_prob1,rc_prob2])
        node_prob = np.vstack([node_prob1,node_prob2])
        pw_prob = np.vstack([pw_prob1,pw_prob2])
        print(wc.shape,'should be twice?')
        # hkc=hkc1-hkc2
        radius_df=radius_df1-radius_df2
        thetasin_df=thetasin_df1-thetasin_df2
        thetacos_df=thetacos_df1-thetacos_df2
    else:    
        useable_ids = clinical_df[conditionals]['Scan Index'].values
        clinical_use=clinical_df[conditionals]
        # useable_ids=[0,1,2,3,4]
        # print(useable_ids)
        scan_labels = clinical_df[conditionals][metric_col].values
        wc,bc,rc,node,pw,radius_df,angle_feat_dict,all_angle_df,wc_labels,bc_labels,rc_labels,node_labels,pw_labels=digest_stat_dict(stat_dict,useable_ids)
        wc1=None
        bc1=None
        rc1=None
        wc_prob1=None
        bc_prob1=None
        rc_prob1=None
        thetasin_df=angle_feat_dict['sin_theta']
        thetacos_df=angle_feat_dict['cos_theta']
        # wc,bc,rc,node,pw,radius_df,thetasin_df,thetacos_df,wc_labels,bc_labels,rc_labels,node_labels,pw_labels=digest_stat_dict(stat_dict,useable_ids)
        wc_prob,bc_prob,rc_prob,node_prob,pw_prob,wc_labels,bc_labels,rc_labels,node_labels,pw_labels= digest_stat_dict_prob(stat_dict,useable_ids)
        # print('hello>>>')
        # print(wc_prob)
    # phi_df=phi_df[usable_ids]

    # print(phi_df.shape)
    # sss

    print(wc.shape)
    print(wc_labels)

    # reoih

    print(scan_labels,'SCAN 1')
    if min(scan_labels)>0:
        scan_labels=scan_labels-1
    print(scan_labels,'SCAN 2!')
    # if scan_labels.max()-scan_labels.min()>30:
        ### if non-catagorical, normalize!
        # print(scan_labels)
        # scan_labels=StandardScaler().fit_transform(scan_labels.reshape(-1, 1)).T[0]
        # scan_labels=sk_normalize(scan_labels.reshape(1, -1) ,axis=1)
        # print(scan_labels)

    
#     print(len(node))
#     print(node_labels,'node labels')
# #     labels = clinical_df[usable_ids][metric_col]

#     print(wc.shape)
#     print(wc_labels)
#     print(scan_labels)
    
#     # print(node,'NDODE NODE NODE')
#     print(scan_labels.shape,'SCAN SHAPE')

    # sss
    # kjg
    # print(wc,'WC')
    # aaa

    print('coh')
    metric_analysis(wc,wc_labels,scan_labels,column_name=['Cluster Cohesion'],
                    plot_title='',graph_label_names=group_labels,sort_vals=True,max_plot=8,save_dir=save_dir,ignore_rsn=ignore_rsn,group_analysis=group_analysis,clinical_df=clinical_use,first_stat=wc1)
    print('rad')
    metric_analysis(rc,rc_labels, scan_labels,column_name=['Cluster Radius from Origin'],
                    plot_title='',graph_label_names=group_labels,sort_vals=True,max_plot=8,save_dir=save_dir,ignore_rsn=ignore_rsn,group_analysis=group_analysis,clinical_df=clinical_use,first_stat=rc1)

    
    # ddd
    metric_analysis(node,node_labels,scan_labels,column_name=['Node Radius from Origin'],
                    plot_title='',graph_label_names=group_labels,sort_vals=True,max_plot=8,save_dir=save_dir,ignore_rsn=ignore_rsn,group_analysis=group_analysis,clinical_df=clinical_use)
    print('btw clust')
    metric_analysis(bc,bc_labels,scan_labels,column_name=['Dist Btw Clusters'],
                    plot_title='',graph_label_names=group_labels,sort_vals=True,max_plot=6,save_dir=save_dir,ignore_rsn=ignore_rsn,group_analysis=group_analysis,clinical_df=clinical_use,first_stat=bc1)
    metric_analysis(pw,pw_labels,scan_labels,column_name=['Dist Btw Node'],
                    plot_title='',graph_label_names=group_labels,sort_vals=True,max_plot=8,save_dir=save_dir,ignore_rsn=ignore_rsn,group_analysis=group_analysis,clinical_df=clinical_use,first_stat=bc1)

    if 'wc_prob' in stat_dict:
        print('coh prob')

        metric_analysis(wc_prob,wc_labels,scan_labels,column_name=['Cluster Cohesion Prob'],
                        plot_title='',graph_label_names=group_labels,sort_vals=True,max_plot=8,save_dir=save_dir,ignore_rsn=ignore_rsn,group_analysis=group_analysis,clinical_df=clinical_use,first_stat=wc_prob1)
        print('rad prob')
        metric_analysis(rc_prob,rc_labels, scan_labels,column_name=['Cluster Radius from Origin Prob'],
                        plot_title='',graph_label_names=group_labels,sort_vals=True,max_plot=8,save_dir=save_dir,ignore_rsn=ignore_rsn,group_analysis=group_analysis,clinical_df=clinical_use,first_stat=rc_prob1)

        metric_analysis(node_prob,node_labels,scan_labels,column_name=['Node Radius from Origin Prob'],
                        plot_title='',graph_label_names=group_labels,sort_vals=True,max_plot=8,save_dir=save_dir,ignore_rsn=ignore_rsn,group_analysis=group_analysis,clinical_df=clinical_use,)

        metric_analysis(bc_prob,bc_labels,scan_labels,column_name=['Dist Btw Clusters Prob'],
                        plot_title='',graph_label_names=group_labels,sort_vals=True,max_plot=6,save_dir=save_dir,ignore_rsn=ignore_rsn,group_analysis=group_analysis,clinical_df=clinical_use,first_stat=bc_prob1)
        metric_analysis(pw_prob,pw_labels,scan_labels,column_name=['Dist Btw Node Prob'],
                        plot_title='',graph_label_names=group_labels,sort_vals=True,max_plot=8,save_dir=save_dir,ignore_rsn=ignore_rsn,group_analysis=group_analysis)





def digest_stat_df(stat_df,stat_type,prob=False):
    print(c for c in stat_df.columns)
    
    cols_to_use=[c for c in stat_df.columns if stat_type.lower() in c.lower()]
    if prob:
        cols_to_use=[c for c in cols_to_use if 'prob' in c.lower()]
    else:
        cols_to_use=[c for c in cols_to_use if 'prob' not in c.lower()]
    return np.array(stat_df[cols_to_use]),cols_to_use

def slice_metric_df(stat_df_full,label_col,conditionals='',title='',plot_save_path='',df_save_path=''):
    if conditionals:
        conditional_df=stat_df_full[conditionals]
    else:
        conditional_df=stat_df_full
    if len(conditional_df['ID'].unique())*2<=len(conditional_df['ID']):
        repeated=True
    else:
        repeated=False
    print(conditional_df.shape,'CONDITIONAL DF')
    scan_labels = conditional_df[label_col].values
    print(scan_labels,"SCAN LABELS")
    print(conditional_df['ID'],'ID LABELS')
    
    if min(scan_labels)>0:
        scan_labels=scan_labels-1  
        
    if label_col in ('diagnosis','diagnosis_inv'):
        group_labels = ['Healthy Control', 'SCD']
    elif label_col == 'CogTr':
        # group_labels=['Control','CogTr']
        group_labels=['CogTr','Control']
    elif label_col == 'Pre':
        group_labels=['Post','Pre']
    else:
        raise Exception('Unknown metric col: {}'.format(metric_col))
        
    cluster_labels=["pDMN","aDMN","DAN","FPN","VN","VAN","SN","SMN"]
        
    rc,rc_labels=digest_stat_df(conditional_df,stat_type='rad')
    wc,wc_labels=digest_stat_df(conditional_df,stat_type='coh')
    bc,bc_labels=digest_stat_df(conditional_df,stat_type='btw')
    
    age=np.array(conditional_df[['age']])
    wc=wc.astype(float)

    val_col=np.array(stat_df_full[label_col])
    val_col=val_col-val_col.min()
#     print(val_col,'val col')
#     print(scan_labels,'graph met')
#     assert np.sum(np.abs((val_col-scan_labels)))==0
#     print('ALL EQUAL!')

    print(wc_labels,'WC LABELS')

#     metric_analysis(wc,cluster_labels,scan_labels,column_name=label_col,y_axis=['Cluster Cohesion'],plot_title=title,
#                     plot_save_path=plot_save_path+'_C',
#                     graph_label_names=group_labels,sort_vals=True,max_plot=8,clinical_df=conditional_df)
    print('rad')
    
    print(label_col,'label col')

    if repeated:
        metric_analysis(rc,cluster_labels, scan_labels,column_name=label_col,y_axis=['Cluster Radius from Origin'],
                    analyze_time=True,
                    plot_save_path=plot_save_path+'_R_time',df_save_path=df_save_path+'R.csv',
                    plot_title=title,graph_label_names=group_labels,sort_vals=True,max_plot=8,clinical_df=conditional_df)
    else:
        metric_analysis(rc,cluster_labels, scan_labels,column_name=label_col,y_axis=['Cluster Radius from Origin'],
                        plot_save_path=plot_save_path+'_R',df_save_path=df_save_path+'R.csv',
                        plot_title=title,graph_label_names=group_labels,sort_vals=True,max_plot=8,clinical_df=conditional_df)
#     print('btw clust')
#     metric_analysis(bc,bc_labels,scan_labels,y_axis=['Dist Btw Clusters'],column_name=['Dist Btw Clusters'],
#                     plot_save_path=plot_save_path+'_D',
#                     plot_title=title,graph_label_names=group_labels,sort_vals=True,max_plot=6,clinical_df=conditional_df)
#     metric_analysis(age,['age'], scan_labels,column_name=label_col,y_axis=['age'],
#                     plot_save_path=plot_save_path+'_A',df_save_path=df_save_path+'A.csv',
#                     plot_title=title,graph_label_names=group_labels,sort_vals=True,max_plot=1,clinical_df=conditional_df)

    

def run_anova(emb_stats,output_dir,title='',label_col='diagnosis',cond_dict={}):
    """
    emb_stats either path or pandas dataframe
    """
    #     embedding_dir=r'C:\Users\coleb\OneDrive\Desktop\Fall 2021\Neuro\hyperBrain\study\meg\ds\pats_CogTr1\L3\HGCN_full_findc_id_dp'
    #     emb_stat_paths=[os.path.join(embedding_dir,f) for f in os.listdir(embedding_dir) if 'embedding_stats_' in f]
    #     print(emb_stats,'STAT PATH')
    #     if
    if type(emb_stats)==str:
        stat_df=pd.read_csv(emb_stats)
    else:
        stat_df=deepcopy(emb_stats)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plot_save_path=os.path.join(output_dir,'anova_plot'+title)
    df_save_path=os.path.join(output_dir,'anova_df'+title)
    if not title:
        title='RM ANOVA analysis'
    print(stat_df.shape,'STAT DICT BEFORE')
    stat_df_before=stat_df
    for condition,val in cond_dict.items():
        stat_df=stat_df[stat_df[condition]==val]
        title='{}_{}{}'.format(title,condition,val)
    #         title=title+'_'+condition+'=='+val
    #     print(cond_dict)
    #     print(title)
    print(stat_df_before.shape,'BEFORE HAND')
    print(stat_df.shape,'STAT SHAPE AFTER COND')
    if not stat_df.shape[0]:
        print(stat_df_before,'WHAT HAPPENED TO EEVRYTHINGN')

    #     assert False
    #     return

    slice_metric_df(stat_df,label_col=label_col,title=title,plot_save_path=plot_save_path,df_save_path=df_save_path)