import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest,RFE
from sklearn.feature_selection import chi2
class EnsembleEmbeddingClassifier(BaseEstimator,ClassifierMixin):
    def __init__(self):
        pass


# class 

class AggregateClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self, base_model,base_model_params={},fit_method='scan_mean',predict_method='same',temp_model=LogisticRegression(C=1000)):

        """
        fit_method- how we want to handle multiple scans, take mean, first,second, first-diff, first-second,split all samples scan for scan
        predict_method- same will do same method as fit. 'prediction_mean' will take mean of two predictions. only works for 'split_sample'
        temp_model=LogisticRegression(C=1000)

        """

        if 'probability' in base_model_params:
            if base_model_params['probability']!=True:
                raise Exception('NEED PROBA')
        elif base_model==SVC:
            base_model_params['probability']=True


        self.temp_model=temp_model
        self.normalize=False

        self.use_feature_selection=False
        self.feature_selection_method=SelectKBest
        # self.select_features_n=-1
        self.select_features_prop=.9
        # print(base_model,'BASE /\MODEL')
        # print(base_model_params,'BASE MODEL PARAMS')

        # print(base_model_params)
        # print(base_model_params,'BASE')
        self.base_model_params=base_model_params
        self.base_model=base_model

        # self.
        self.base_clf=base_model()
        # print(base_model_params,'model params')
        # print(self.base_model_params)
        self.base_clf.set_params(**base_model_params)

        # print(self.base_clf,'BASE CLF')
        
        self.fit_method = fit_method
        self.predict_method = predict_method

        self.fit_methods_allowed = set(['scan_mean','use_first','use_second','use_diff','concatenate','split_sample'])
        self.predict_methods_allowed= set(['same','prediction_mean','use_second'])

        self.y_min=None
        assert fit_method in self.fit_methods_allowed
        assert predict_method in self.predict_methods_allowed
        if self.predict_method=='prediction_mean':
            if self.fit_method!='split_sample':
                print('Mismatched fit/prediction: cannot tale prediction mean of ',fit_method)
                print('Will use "same" for prediction')
                self.predict_method='same'
                raise Exception('Mismat')

# def nr
    def create_feature_selector(self,X,y):
        # self.feature_selector=SelectKBest(chi2,k=int(X.shape[1]*self.select_features_prop))
        self.feature_selector=RFE(self.base_clf,n_features_to_select=.9)
        self.feature_selector.fit(X, y)
        X_new=self.feature_selector.transform(X)
        return self.feature_selector,X_new
    def feature_selection_transform(self,X):
        return self.feature_selector.transform(X)
    def reform_y(self,y,fit=True):
        # if 

        if (self.fit_method!='split_sample') or (fit==False and self.predict_method=='prediction_mean'):
            ## all other methods keep same y
            if len(y.shape)>1 and y.shape[1]>1:
                # print(y.shape,'y shape')
                assert np.all(y[:,0]==y[:,1])
                y=y[:,0]  
            else:
                y=y
        else: #### why can't we just keep same y?
            ### need twice as many ys for split sample
            if len(y.shape)>1 and y.shape[1]>1:
                y= np.concatenate((y[:,0],y[:,1]))
                # y=y[:,0]  
            else:
                y= np.concatenate((y,y))       
            
        # print(y.shape,'YYYY')
        if len(y.shape)>1 and y.shape[1]==1:
            y=y[:,0]
        # print(y.shape,'YYYY')
        return y

    def reform_X(self,X,prefit=True):
        # assert len(X.shape)==3
        if len(X.shape)<3:
            # print('no reform necissary')
            return X
        X_1=np.array(X[:,:,0])
        X_2=np.array(X[:,:,1])
        if self.fit_method=='scan_mean':
            X = np.mean(X,axis=2)
        elif self.fit_method=='use_first':
            X=X_1
        elif self.fit_method=='use_second':
            X=X_2
        elif self.fit_method=='use_diff':
            X_d=X_1-X_2
            X = np.concatenate((X_1,X_d),axis=1) 
            # print(X_d[0],'X D')
            # print(X.shape,'X final shape')
        elif self.fit_method=='concatenate':

            X=np.concatenate((X_1,X_2),axis=1)     ## reshape is cleaner+ extends beyond two scans dummy
        elif self.fit_method=='split_sample':
            # print(X_1.shape,'1 time')
            X = np.concatenate((X_1,X_2))
            # print(X.shape,'1 time')
        else:
            raise Exception()

        if self.normalize:
            if prefit:
                # print(prefit,'PREFIT')
                X=self.normalizer.transform(X)
            else:
                pass

        if self.use_feature_selection:

            
            if prefit:
                # print(X.shape,'BEFORE')
                X=self.feature_selection_transform(X)
                # print(X.shape,'AFTER')
            else:
                pass
                # print('Need to prefit our scalers and feature selectors')

        # print(X.shape,'X SHAPE')
        # if self.normalize
        return X

    def reform_inputs(self,X,y,prefit=True):
        assert len(X.shape)==3
        if len(X.shape)<3:
            return X,y
        X,y=self.reform_X(X,prefit=prefit),self.reform_y(y)
        if self.normalize:
            self.normalizer=MinMaxScaler()
            X=self.normalizer.fit_transform(X,y)
        if self.use_feature_selection:
            _,X=self.create_feature_selector(X,y)


        return X,y

    def predict_mean(self,X,mean_axis):
        if mean_axis==2:
            predictions=np.array([self.get_proba(X[:,:,ai]) for ai in range(X.shape[mean_axis])])
        if mean_axis==1:
            predictions=np.array([self.base_clf.get_proba(X[:,ai]) for ai in range(X.shape[mean_axis])])
        # print(predictions.shape,'pred shape')
        # print(predictions,"PREDS")
        mean_predict_proba= np.mean(predictions,axis=0)
        # print(mean_predict_proba,"MEANNY")

        # print(mean_predict_proba.shape,'MEAN SHAPE')
        # mean_predict=np.argmax(mean_predict_proba,axis=1)
        # print(mean_predict,'mean qd?')
        return mean_predict_proba

    def fit(self, X, y):
        ### WHY DON't WE SCALE?

        # Check that X and y have correct shape
        # print(X.shape,'X SHAPE before')
        # print(X.shape,'X SHAPE After')
        if len(X.shape)>2:
            # print('reshape')F
            X,y=self.reform_inputs(X,y,prefit=False)

        # print(X.shape,'X')
        # print(y.shape,'y')

        # print(X.shape,'X SHAPE After')
        # print(X,'BEOFRE')
        X, y = check_X_y(X, y)

        # print(X,'After')
        # print()
        self.base_clf.fit(X=X,y=y)
        if self.temp_model is not None:
            temp_pred = self.base_clf.predict_proba(X)
            # print(temp_pred,'before')
            temp_pred = self.get_proba(X)
            # print(temp_pred,'after')

            self.temp_model.fit(X=temp_pred,y=y)
            # print()
            # print(self.temp_model.predict_proba(temp_pred) ,'final_temps')
            # print(self.temp_model.coef_,'coeffs')
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y
        self.y_min = np.min(y)
        # print(self.y_min,'MINIMANIA')
        # Return the classifier
        return self

    def predict(self, X):
        ### met

        # Check is fit had been called
        check_is_fitted(self)


        y_prob=self.predict_proba(X)
        y=self.proba_to_pred(y_prob)

        if self.predict_method!='same':
            pass
        else:
            X=self.reform_X(X)

            # y_prob = self.get_proba(X)
            # y=self.proba_to_pred(y_prob)
            y_theirs=self.base_clf.predict(X)

        return y

    def predict_ret_ys(self,X,y):
        if len(X.shape)>2:
            new_ys=self.reform_y(y,fit=False) 
        preds=self.predict(X)
        return preds,new_ys  



    def get_proba(self,X):
        if self.base_model==SVC:
            return self.base_clf.decision_function(X)[:,None]
            # return self.base_clf.decision_function(X)[None,:]

        # elif self.base_model==LogisticRegression:
            # return self.base_clf.predict_proba(X)
        else:
            return self.base_clf.predict_proba(X)
    def proba_to_pred(self,proba):
        if self.base_model==SVC:
            y=np.where(proba>0,1+self.y_min,self.y_min).T[0]
            return y
        # elif self.base_model==LogisticRegression:
        else:
            return np.argmax(proba,axis=1)+self.y_min  
    def predict_proba(self,X):

        check_is_fitted(self)
        if self.predict_method=='same':
            X=self.reform_X(X)
            # X=MinMaxScaler(X)

            y_prob=self.get_proba(X)
            # y_prob= self.base_clf.predict_proba(X)
            X = check_array(X)

        elif self.predict_method=='use_second':
            X=X[:,:,0]
            y_prob=self.get_proba(X)
            X = check_array(X)
        else:
        # Input validation
            # print(X.shape,'X MEAN')
            X=X
            y_prob= self.predict_mean(X,mean_axis=2)

        
        # print(y,'Y')
        # print(X.shape,'X')
        # print(self.X_.shape,'X_')
        return y_prob

    def set_params(self,**params):
        self.base_clf.set_params(**params)
        # print(self.base_clf,'ADJUSTED CLF')
        return self

    # def scoring(self)

class AggregateClassifierGrid(AggregateClassifier,BaseEstimator,ClassifierMixin):
    def __init__(self,params={},fit_method='scan_mean',predict_method='same'):
        if 'base_model' in params:
            base_model=params['base_model']
            del params['base_model']
        else:
            raise Exception('NEED BASE MODEL')

        if 'fit_method' in params:
            fit_method=params['fit_method']
            del params['fit_method']
        if 'predict_method' in params:
            predict_method=params['predict_method']
            del params['predict_method']

        # if 

        base_model_params=params  ## all leftover params are for the base model
        # print(base_model_params,'basic')
        AggregateClassifier.__init__(self,base_model=base_model,base_model_params=base_model_params,fit_method=fit_method,predict_method=predict_method)

    # def set_params()


# X_dims = (90,50,2)
# y_dims=(90,1)

# X=np.random.randint(low=0,high=2,size=X_dims)
# # # y_1=np.random.randint(low=0,high=2,size=y_dims[0])
# y=np.zeros(y_dims)

# ones=[1,4,5,6,87,8,6,10,14,15,26,46,45]
# # y[2]=1
# # # y[10]=1

# for i in ones:
#     y[i]=1
# # # print(y)
# # # # ac = AggregateClassifier(base_model=SVC,fit_method='use_second')

# # # # # ac.fit(X,y)
# svc_params={'probability':True,'C':.5}
# # # params={'base_model':SVC,'fit_method':'split_sample','base_model_params':svc_params}
# # params={'base_model':SVC,'fit_method':'split_sample'}

# # full_params = {**params,**svc_params}
# # print(params,'para')
# # print(full_params,'fullly')


# # ac = AggregateClassifier(base_model=SVC,fit_method='split_sample',predict_method='same')
# # print(ac.base_clf)
# # ac.set_params(svc_params)
# # print(ac.base_clf)
# # ac()
# # ac = AggregateClassifierGrid(full_params)
# # # # ac = AggregateClassifier(base_model=SVC,fit_method='split_sample',predict_method='same',base_model_params=params)
# # ac.fit(X,y)
# # pred= ac.predict(X)
# # # # # print(pred,'AH')
# # # ac = AggregateClassifier(**params)
# # # ac.fit(X,y)
# # # ac.predict(X)

