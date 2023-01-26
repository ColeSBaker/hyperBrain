
import os
def welch_from_columns(g1,g2,equal_var=False):
        t, p = stats.ttest_ind(g1, g2, equal_var = equal_var) 
        
        return [p,t]
    
def welch_from_columns_onepop(g1,popmean=0):
    t, p = stats.ttest_1samp(g1, popmean = popmean) 
    return [p,t]


def welch_ttest(df,split_labels,equal_var=False): 
    ## Welch-Satterthwaite Degrees of Freedom ##
    ### lost our node information... this is dangerous?
    split_options = np.array(list(set(list(split_labels)))).astype(int)
    split_labels=np.array(split_labels).astype(int)
    split_options.sort()
    print(split_options,'split options')
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

def metric_analysis(graph_by_metric_df,entity_names,graph_labels,column_name='',
                    plot_title='',graph_label_names=['Healthy Control','SCI'],sort_vals=False,max_plot=100
                   ,print_pval=True,print_qval=True,save_dir='',ignore_rsn='',group_analysis=True,clinical_df=None,first_stat=None):
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
    eval_df=graph_by_metric_df

    savepath_df=os.path.join(save_dir,column_name[0]+'_df.csv')

    # print(eval_df,'EVALUATION DF')
    # print(eval_df.shape)
    # sss
    # print(entity_names,'ENTITY NAMES')
#     print(eval_df,'evals')
    # sig_df = welch_ttest(eval_df,graph_labels).sort_values(by='p_value')  
    # sig_df = welch_ttest(eval_df,graph_labels).sort_values(by='p_value')  

    # print(graph_labels,'LABELS')
    
    sig_df = welch_ttest(eval_df,graph_labels)

    # print(sig_df,'SIG DG')
    # print(graph_labels,'GRAPH LABELS!')

    # sig_df_anova=anova_test(eval_df,graph_labels,side_X=clinical_df,first_stat=first_stat)

    # if not group_analysis:
        # return sig_df,sig_df_anova
    # ggdgd
    # print(eval_df,'EVAL DF')
    # ktgluig
    
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
        add_mean_str=True
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
    
#         print(new_name)
        
#         print(new_name,str_p,str_q)
#     err 
    # print(entity_names,'entity_names')
    # print(str_ps,'string ps')
    # print(str_qs,'string qs')
    show_data=np.array([entity_names,str_ps,str_qs]).T
    show_df=pd.DataFrame(columns=[['StatName','p_value','q_value']],data=show_data)
    if save_dir!='':
        show_df.to_csv(savepath_df)
        show_df=pd.read_csv(savepath_df)


    print(show_df,'SHOW DF?')
    # print(show_df.sort_values(by='p_value') )
    # ig_df = welch_ttest(eval_df,graph_labels).sort_values(by='p_value') 

    
    if sort_vals:
#         eval_df=eval_df[eval_df['entity'].isin(keepers)]
        eval_df=eval_df[:,keepers]
        
        if use_full_names:  ## these will have already been sorted... you're all twisted
            entity_names=full_entity_names[:n_plot]
        else:
            entity_names=[entity_names[i] for i in keepers]
    
#     for i in range(max_plot):
#     if add_strings:
        
        
    eval_df = pd.DataFrame(np.ravel(eval_df), columns=column_name)  ### ADD BIG ASS STARS
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
    ax = sns.boxplot(x="network", y=column_name[0], hue="label", data=eval_df, palette="pastel",whis=1.,notch=True,bootstrap=1000,showfliers = False)
#     ax = sns.swarmplot(x="network", y=column_name[0], hue="label", dta=eval_df)
    print('bout to show')
    plt.title(plot_title, size=16)
    plt.show();