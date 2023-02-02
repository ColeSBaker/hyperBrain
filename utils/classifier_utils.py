    
def run_combos_list(model_type,feature_list,names=[],times=2,
    use_agg=False,agg_fit=None,agg_pred=None,
    hyp_search=True,passed_params=None,y_comb=[],
    use_nested=True):
# nested_params={'use_nested':True,'inner_loo':True,'inner_splits':5}):
    ##linear only matters for svc
    def execute(X,y):
        if use_agg:
            assert agg_fit is not None
            assert agg_pred is not None
            if agg_fit=='split_sample' and agg_pred=='same':
                ## BRO WHAT???? THIS IS CHEATING ISN"T IT????
                temp_model=AggregateClassifier(model_type,fit_method=agg_fit,predict_method=agg_pred)
                X,y=temp_model.reform_inputs(X,y) ### only use temp model for this reform
                
                
            print(y.shape,'y?')
            print(X.shape,'X')
            print(passed_params,'PASS')
            # X_norm = MinMaxScaler().fit_transform(X)
            # res=make_classifier_agg(X,y,base_model=model_type,
            #             fit_method=agg_fit,predict_method=agg_pred,
            #                         hyp_search=hyp_search,passed_params=passed_params)
            if use_nested:
                res=make_classifier_nested(X,y,base_model=model_type,
                        fit_method=agg_fit,predict_method=agg_pred,
                                    hyp_search=hyp_search,passed_params=passed_params)   
            else:
                res=make_classifier_agg(X,y,base_model=model_type,
                        fit_method=agg_fit,predict_method=agg_pred,
                                    hyp_search=hyp_search,passed_params=passed_params)
            print(res,'res')
            return res
        else:
            return make_classifier(X,y=y,model=model_type)

#     print()
    f1_list=np.zeros((len(feature_list),times))
    roc_list=np.zeros((len(feature_list),times))
    for t in range(times):
        
        for i in range(len(feature_list)):
            feats=feature_list[i]
            res=execute(feats,y=y_comb)
            f1_list[i,t]=res[0]
            roc_list[i,t]=res[1]
#             f1_list[i,t]=i
#             roc_list[i,t]=i
    
    f1_means=f1_list.mean(axis=1)
    roc_means=roc_list.mean(axis=1)
    for j in range(len(f1_means)):
        print(f1_means[j],roc_means[j],names[j])

    
def run_combos(model_type,times=2,use_agg=False,agg_fit=None,agg_pred=None,hyp_search=True,passed_params=None):
    ##linear only matters for svc
    def execute(X,y):
        if use_agg:
            assert agg_fit is not None
            assert agg_pred is not None
            if agg_fit=='split_sample' and agg_pred=='same':
                ## BRO WHAT???? THIS IS CHEATING ISN"T IT????
                temp_model=AggregateClassifier(model_type,fit_method=agg_fit,predict_method=agg_pred)
                X,y=temp_model.reform_inputs(X,y) ### only use temp model for this reform
                
                
            print(y.shape,'y?')
            print(passed_params,'PASS')

            res=make_classifier_agg(X,y,base_model=model_type,
                        fit_method=agg_fit,predict_method=agg_pred,
                                    hyp_search=hyp_search,passed_params=passed_params)   
            print(res,'res')
            return res
        else:
            return make_classifier(X,y=y,model=model_type)
    weighted_f1s=[]
    X_wtncs=[]
    X_wtccs=[]
    X_wtnccs=[]
    X_nccs=[]
    X_wradwithcluss = []
    X_wradbetcluss = []
    X_wradbetwitcluss=[]
    
    weighted_roc=[]
    X_wtncs_roc=[]
    X_wtccs_roc=[]
    X_wtnccs_roc=[]
    X_nccs_roc=[]
    X_wradwithcluss_roc = []
    X_wradbetcluss_roc = []
    X_wradbetwitcluss_roc=[]
    print(y_comb)
    print(y_comb.mean(),y_comb.std(),'HOW MANY OF EACH?')

#     print()
    for t in range(times):
        res=execute(X_weight_comb,y=y_comb, )
        weighted_f1s.append(res[0])
        weighted_roc.append(res[1])
        
        res=execute(X_rawstack,y=y_comb, )  ### stack raw
        X_wtncs.append(res[0])
        X_wtncs_roc.append(res[1])
        
        res=execute(X_allemb,y=y_comb, )  ### other w/   ### node+cluster only
        X_wtnccs.append(res[0])
        X_wtnccs_roc.append(res[1])
        
        res=execute(X_wradbetwitclus,y=y_comb, )  ## otgre w/ all clust
        X_wtccs.append(res[0])
        X_wtccs_roc.append(res[1])  ## ours,other w/ all clust
        
#         X_wradbetwitclus=np.hstack((X_weight_g_comb,x_cc_comb,X_wc_use_comb,X_bc_use_comb))
# X_wradwitclus=np.hstack((X_weight_comb,X_weight_g_comb,x_cc_comb,X_wc_use_comb,X_bc_use_comb))
        res=execute(X_ncwit,y=y_comb, ) ## clust only
        X_wradbetcluss.append(res[0])
        X_wradbetcluss_roc.append(res[1])
        
        res=execute(X_wradwitclus,y=y_comb, ) ## alpha w/ cluster
        X_wradbetwitcluss.append(res[0])
        X_wradbetwitcluss_roc.append(res[1])
#         break

#         X_nodekm_comb,y_comb = combine_patient_feats(X_node_km_dist,graph_labels,clinical_df,conditional=conditionals)
# X_fnkm_comb
#         X_wtccs.append( execute(X_wtcc,y=y_comb, ))
        
#         X_nccs.append(execute(X_ncc,y=y_comb,
#                                           ))
# #         X_wradwithcluss.append(execute(X_wradwitclus,y=y_comb,
# #                                           ))
# #         X_wradbetcluss.append(make_classifier(X_wradbetclus,y=y_comb,
# #                                           model=model_type))
#         X_wradbetcluss.append(execute(X_wfnkm,y=y_comb,))
#         X_wradbetwitcluss.append(execute(X_wbetclus,y=y_comb))
        
#     print(X_wfnkm.shape,'FN SHAPEL')
#     print(X_wnodekm.shape,'FN SHAPEL should be bulk of before')
#     print(X_weight_comb.shape,'weighted')
    print(np.array(weighted_f1s).mean(),np.array(weighted_roc).mean(),'weighted')
    print(np.array(X_wtncs).mean(),np.array(X_wtncs_roc).mean(),'raw stack')
    print(np.array(X_wtnccs).mean(),np.array(X_wtnccs_roc).mean(),'w+node+cluster__rn node avg cent')
    print(np.array(X_wtccs).mean(),np.array(X_wtccs_roc).mean(),'all cluster stuff')
    print(np.array(X_nccs).mean(),np.array(X_nccs_roc).mean(),'node+cluster')
    print(np.array(X_wradwithcluss).mean(),np.array(X_wradwithcluss_roc).mean(),'Weight and all clus')
    print(np.array(X_wradbetcluss).mean(),np.array(X_wradbetcluss_roc).mean(),'node + cluster info')
    print(np.array(X_wradbetwitcluss).mean(),np.array(X_wradbetwitcluss_roc).mean(),'weight+Cluster rad')


    

base_param_grid_svc = [
#   {'C': [.2,.5,1, 10, 1000], 'kernel': ['linear']},
  {'C': [.3,1, 10,100], 'gamma': [1,.5,.1], 'kernel': ['rbf'],'class_weight':['balanced']},
    {'C': [.3,1, 10, 200], 'degree': [2,3], 'kernel': ['poly'],'class_weight':['balanced']},
 ]
# base_param_grid_svc = [
#   # {'C': [.1,.2,.5,1, 10,100, 1000], 'kernel': ['linear'],'class_weight':['balanced']},
#   {'C': [.1,.2,.5,1, 10,100, 1000], 'degree': [2,3,4], 'kernel': ['poly'],'class_weight':['balanced']},
#   {'C': [.1,.2,.5,1, 10,100, 1000], 'gamma': [1,.5,.1,0.01,0.001], 'kernel': ['rbf'],'class_weight':['balanced']},
# #     {'C': [.1,.3, 1000], 'gamma': [0.01,0.001, 0.0001], 'kernel': ['rbf'],'class_weight':[None,'balanced']},
#  ]

base_param_grid_linear=[
  {'C': [.2,.5,1,2,3,5, 10,100,1000],'penalty':['l1'], 'solver': ['liblinear']},
    {'C': [.2,.5,1,2,3,5, 10,100,1000],'penalty':['l2'], 'solver': ['lbfgs']},
 ]
base_param_grid_linear=[
  {'C': [.5,1,5, 10,100,1000],'penalty':['l1'], 'solver': ['liblinear'],'class_weight':['balanced']},
    {'C': [.5,1,5, 10,100,1000],'penalty':['l2'], 'solver': ['lbfgs'],'class_weight':['balanced']},
 ]
# base_param_grid_linear=[
  # {'C': [1000],'penalty':['l1'], 'solver': ['liblinear']},
    # {'C': [5],'penalty':['l2'], 'solver': ['lbfgs']},
 # ]
# base_param_grid_linear=[
  # {'C': [1,2,3,5, 10,100,1000],'penalty':['l1'], 'solver': ['liblinear']} ]
# 
base_param_grid_rfc=[
  {'sampling_method':['uniform'],'objective':['binary:logistic'],'eval_metric':['logloss']},
 ]

# n_estimators=100, *, criterion='gini', max_depth=None, min_samples_split=2
# base_param_grid_linear=[
#   {'C': [.2,.5,1,2,3,5, 10,100,1000],'penalty':['l1'], 'solver': ['liblinear']},
#     {'C': [.2,.5,1,2,3,5, 10,100,1000],'penalty':['l2'], 'solver': ['lbfgs']},
#  ]


def agg_params_grid(base_model,fit_method='concatenate',predict_method='same'):   
    agg_params={'base_model':base_model,'fit_method':fit_method,'predict_method':predict_method}
    full_param_list=[]
    base_param_list=base_param_grid_svc if base_model==SVC else agg_param_grid_linear
    for p in base_param_grid_svc:
        merged_dict={**p,**agg_params}
        full_param_list.append(merged_dict)
    return full_param_list
        # # params={'base_model':SVC,'fit_method':'split_sample','base_model_params':svc_params}
# params={'base_model':SVC,'fit_method':'split_sample'}

# full_params = {**params,**svc_params}

# def run_grid_search(X,y,model,parameters,scoring="roc_auc",cv=5,p_test=.2):
# def run_grid_search(X,y,model,parameters,scoring="roc_auc",cv=5,p_test=.2):
def run_grid_search(X,y,model,parameters,scoring="f1_macro",cv=5,p_test=.2):
#     splitter = StratifiedKFold(n_splits=cv,shuffle=True)
#     splitter = StratifiedKFold(n_splits=cv)
#     cv=10
    # splitter = StratifiedShuffleSplit(n_splits=cv, test_size=p_test)
    # splitter= RepeatedStratifiedKFold(n_splits=8, n_repeats=2)
    splitter= RepeatedStratifiedKFold(n_splits=5)  ### like to use big splits for this so that     
    #                                                     ## the removal of one item in nested doesn't change hyperparameters drastic st. we find the marginally worst set
    #                                                     ## of params for that particular set
    # splitter = LeaveOneOut()
    # clf = GridSearchCV(model, parameters,scoring=scoring,cv=splitter)
    # print(splitter,'splitter!!')
    # print(model,'MODEL')

    clf = GridSearchCV(model, parameters,scoring=scoring,cv=splitter)
    clf.fit(X, y)
    clf_df =pd.DataFrame(clf.cv_results_)
    clf_df=clf_df.sort_values(by=['rank_test_score'])

    # print(clf_df,'CLF DF')
    # print(cv,'what is going on?')
    # print(clf.cv,'GET CV')
    # pd.set_option('display.max_colwidth', None)
#     with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    # print(clf_df.iloc[:3][['params','mean_test_score','rank_test_score']])
    
    return clf
    
def make_classifier_nested(X,y,base_model,fit_method='concatenate',
                        predict_method='same',hyp_search=True, passed_params=None):
    
        X_clone= deepcopy(X)
        model = AggregateClassifier(base_model=base_model
                                    ,fit_method=fit_method,predict_method=predict_method)

                                    
        params = base_param_grid_svc if base_model==SVC else base_param_grid_linear
        print(X.shape,'SHAPPE  UP')
        if base_model==SVC:
            params=base_param_grid_svc
        elif base_model==LogisticRegression:
            params=base_param_grid_linear
        else:
            params=base_param_grid_rfc

        # gs = run_grid_search(deepcopy(X),y,model,params,cv=8)
        # gs_df = pd.DataFrame(gs.cv_results_)
        # best = gs_df[gs_df['rank_test_score']==1]

        # best_params = best['params']
        # f1=gs.best_score_
        # print(f1,gs.n_splits_,'num_split?')
        # best_model = gs.best_estimator_  
        # best_params_full = gs.best_params_  

        # print(best_params,'BEST PARAMS??')
        # print()
#             best_
        # best_model = AggregateClassifier(base_model=base_model
        #                             ,fit_method=fit_method,predict_method=predict_method
        #                                 ,base_model_params=best_params)
#         use_params=best_params
        


        # outer_splitter = StratifiedKFold(n_splits=10,shuffle=True)
        # inner_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=np.random.randint(0,40))
        # outer_splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=np.random.randint(0,40))
        outer_splitter = LeaveOneOut()
# 
        prediction=np.zeros_like(y)
        probs=np.zeros_like(y)
        y_vals=[]
        probs=[]
        prediction=[]
        current_start=0
        ## to test this, first take the hyperparam search out of the loop and do it like nonest
        for train_index, test_index in outer_splitter.split(X, y):
            print(len(probs),'NUM PROBS')
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            num_test=y_test.shape[0]
            current_end=current_start+num_test
            # assert prediction[current_start:current_end].sum()==0 ## double check we arent overwriting anything

            gs = run_grid_search(deepcopy(X_train),y_train,model,params)

            gs_df = pd.DataFrame(gs.cv_results_)
            best = gs_df[gs_df['rank_test_score']==1]
            best_params = best['params']
            f1=gs.best_score_
            print(f1,gs.n_splits_,'num_split?')
            best_model = gs.best_estimator_  
            best_params = gs.best_params_  
            # best_params=best_params_full

            print(best_params,'BEST PARAMS??')
            # print(f1,gs.n_splits_,'num_split?')
            print()
#             best_
            best_model_sub = AggregateClassifier(base_model=base_model
                                        ,fit_method=fit_method,predict_method=predict_method
                                            ,base_model_params=best_params)

            best_model_sub.fit(X_train,y_train)
            # best_model.fit(X_train,y_train)
            test_pred=best_model_sub.predict(X_test)
            test_prob=best_model_sub.predict_proba(X_test)
            # print(test_prob.shape)
            if base_model==SVC:
                print('SVC')
                roc_prob=test_prob
            else:
                print('LIN DOG')
                roc_prob=test_prob[:,1]

            # prediction[current_start:current_end]=test_pred
            probs.extend(roc_prob)
            prediction.extend(test_pred)
            y_vals.extend(y_test)
            current_start=current_end
            if len(probs)>0:
                try:
                    cur_roc = roc_auc_score(y_vals, probs)
                    print(cur_roc,'Current ROC')
                except:
                    pass
        probs=np.array(probs)
        prediction=np.array(prediction)
        y_vals=np.array(y_vals)

        print(prediction)
        print(prediction.shape)
        print(y.shape,'Y I AOUGHTA')

        print(probs)
        print(probs.shape)




        
        # if base_model==SVC:
        #    roc_prob=proba
        # else:
        #     roc_prob=proba[:,1]

        f1 = f1_score(y_vals, prediction, average='macro')
        print(classification_report(y_vals, prediction))
        print(accuracy_score(y_vals, prediction),'accuracy')
#         pred_roc = best_model.
        roc = roc_auc_score(y_vals, probs)
        # f1=.7
        print(roc,'roc')
        return f1,roc


def make_classifier_agg(X,y,base_model,fit_method='concatenate',
                        predict_method='same',hyp_search=True, passed_params=None):
    
        X_clone= deepcopy(X)
        model = AggregateClassifier(base_model=base_model
                                    ,fit_method=fit_method,predict_method=predict_method)
                                    
        params = base_param_grid_svc if base_model==SVC else base_param_grid_linear
        if base_model==SVC:
            params=base_param_grid_svc
        elif base_model==LogisticRegression:
            params=base_param_grid_linear
        else:
            params=base_param_grid_rfc
#         params= agg_params_grid(base_model,fit_method='fit_method','predict_method'=predict_method) 
        if hyp_search:
            gs = run_grid_search(deepcopy(X),y,model,params,cv=8)

            gs_df = pd.DataFrame(gs.cv_results_)
            best = gs_df[gs_df['rank_test_score']==1]

            best_params = best['params']
            f1=gs.best_score_
            print(f1,gs.n_splits_,'num_split?')
            best_model = gs.best_estimator_  
            best_params = gs.best_params_  

            print(best_params,'BEST PARAMS??')
            print()
#             best_
            best_model = AggregateClassifier(base_model=base_model
                                        ,fit_method=fit_method,predict_method=predict_method
                                            ,base_model_params=best_params)
            use_params=best_params

        else:
#             passed_params['probability']=False
            best_model=AggregateClassifier(base_model=base_model
                                        ,fit_method=fit_method,predict_method=predict_method
                                            ,base_model_params=passed_params)
            use_params=passed_params
            
#         use_params_proba=use_params
#         if base_model=='SVC':
#             use_params_proba['probability']=True
#         best_model_proba=AggregateClassifier(base_model=base_model
#                                     ,fit_method=fit_method,predict_method=predict_method
#                                         ,base_model_params=use_params_proba)

        
#         splitter = StratifiedKFold(n_splits=5,shuffle=True)
        splitter = LeaveOneOut()
#         plitter = StratifiedShuffleSplit(n_splits=20, test_size=.2)

#         print(best_model,"BEST MODEL ")
#         print(use_params,'BEST PARAMS')
#         print(base_model,'base model>')
#         print(best_model.base_model,'BASE_MODEL')
        print('NEW!')
        print(y,'EXAMPLE Y')
        print(y.shape,'EXAMPLE Y')
        print(X.shape,'new')
        print(y.shape,'clone')
        
#         best

        print(X.shape,' X shape???sss')
        print(splitter)
#         y=y-y.min()
        print(y,'YYY')
        prediction = cross_val_predict(best_model,deepcopy(X) , y, cv=splitter,method='predict')   
        
#         best_mo
        
        proba =cross_val_predict(best_model,deepcopy(X) , y, cv=splitter,method='predict_proba') 
    
#         print(proba.shape,'what?')
        
        if base_model==SVC:
           roc_prob=proba
        else:
            roc_prob=proba[:,1]
#         print(prediction,'PREDs')
#         print(proba,'PROBA')
#         roc_prob=proba
#         print(prediction.shape,'huh') 
#         print(y.shape,'y')
#         print(pre)
#         f1 = f1_score(y, pred, average='macro')
        f1 = f1_score(y, prediction, average='macro')
        print(classification_report(y, prediction))
        print(accuracy_score(y, prediction),'accuracy')
#         pred_roc = best_model.
        roc = roc_auc_score(y, roc_prob)
        print(roc,'roc')
        return f1,roc
def make_classifier(X,y,model=SVC,class_weight='balanced',hyp_search=True):
    if model in (RFC, XGBClassifier):
        hyp_search=False
#     if hyp_search:
    if hyp_search:
#         if model
        params = base_param_grid_svc if model==SVC else base_param_grid_linear
        gs = run_grid_search(X,y,model(),params,cv=5)
        gs_df = pd.DataFrame(gs.cv_results_)
        best = gs_df[gs_df['rank_test_score']==1]
        best_params = best['params']
        f1=gs.best_score_
        print(f1,gs.n_splits_,'num_split?')
        best_model = gs.best_estimator_  
        
#         print(best_model.coefs_,'COEFFIENCIETS')
#         print(np.max((best_model.coefs_),'MAX COEFF'))
        pred = cross_val_predict(model(best_params), X, y, cv=5)
        pred = cross_val_predict(best_model, X, y, cv=5)
        print(classification_report(y, pred))
    else:
        if model==XGBClassifier:
            pred = cross_val_predict(model(), X, y, cv=5)
        else:
            pred = cross_val_predict(model(class_weight=class_weight,bootstrap=True,
                                      n_estimators=400), X, y, cv=5)
        
#         f1 = f1_score(y, pred, average='weighted')
        f1 = f1_score(y, pred, average='macro')
        print(classification_report(y, pred))
        print(accuracy_score(y, pred),'accuracy')
        roc = roc_auc_score(y, pred)
        print(roc,'roc')
    return f1,roc