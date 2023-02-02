import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

class EnsembleEmbeddingClassifier(BaseEstimator,ClassifierMixin):
    def __init__(self):
        pass


# class 

class AggregateClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self, base_model,base_model_params={},fit_method='scan_mean',predict_method='same'):

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
        self.base_model_params=base_model_params
        self.base_model=base_model
        self.base_clf=base_model()
        self.base_clf.set_params(**base_model_params)
        self.fit_method = fit_method
        self.predict_method = predict_method

        self.fit_methods_allowed = set(['scan_mean','use_first','use_second','use_diff','concatenate','split_sample'])
        self.predict_methods_allowed= set(['same','prediction_mean'])

        self.y_min=None
        assert fit_method in self.fit_methods_allowed
        assert predict_method in self.predict_methods_allowed
        if self.predict_method=='prediction_mean':
            if self.fit_method!='split_sample':
                print('Mismatched fit/prediction: cannot tale prediction mean of ',fit_method)
                print('Will use "same" for prediction')
                self.predict_method='same'
                raise Exception('Mismat')

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

    def reform_X(self,X):
        # assert len(X.shape)==3
        if len(X.shape)<3:
            # print('no reform necissary')
            return X
        X_1=np.array(X[:,:,0])
        X_2=np.array(X[:,:,1])
        if self.fit_method=='scan_mean':
            X = np.mean(X,axis=2)

        elif self.fit_method=='use_one':
            if self.which_one is None:
                raise Exception('NEED TO PICK WHICH ONE')
            X=X[:,:,self.which_one]
            X=X_2
        elif self.fit_method=='use_diff':
            X_d=X_1-X_2
            X = np.concatenate((X_1,X_d),axis=1) 
        elif self.fit_method=='concatenate':

            X=np.concatenate((X_1,X_2),axis=1)     ## reshape is cleaner+ extends beyond two scans dummy
        elif self.fit_method=='split_sample':
            X=np.concatenate([X[:,:,i]] for i in range(X.shape[0]))

        else:
            raise Exception()

        # print(X.shape,'X SHAPE')
        return X

    def reform_inputs(self,X,y):
        assert len(X.shape)==3
        if len(X.shape)<3:
            return X,y
        X,y=self.reform_X(X),self.reform_y(y)
        return X,y

    def predict_mean(self,X,mean_axis):
        if mean_axis==2:
            predictions=np.array([self.get_proba(X[:,:,ai]) for ai in range(X.shape[mean_axis])])
        if mean_axis==1:
            predictions=np.array([self.base_clf.get_proba(X[:,ai]) for ai in range(X.shape[mean_axis])])
        mean_predict_proba= np.mean(predictions,axis=0)
        return mean_predict_proba

    def fit(self, X, y):

        # Check that X and y have correct shape
        # print(X.shape,'X SHAPE before')
        # print(X.shape,'X SHAPE After')
        if len(X.shape)>2:
            # print('reshape')
            X,y=self.reform_inputs(X,y)
        X, y = check_X_y(X, y)
        self.base_clf.fit(X=X,y=y)
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
            y_prob=self.get_proba(X)
            # y_prob= self.base_clf.predict_proba(X)
            X = check_array(X)
        else:
        # Input validation
            # print(X.shape,'X MEAN')
            y_prob= self.predict_mean(X,mean_axis=2)

    
        return y_prob

    def set_params(self,**params):
        self.base_clf.set_params(**params)
        # print(self.base_clf,'ADJUSTED CLF')
        return self

    # def scoring(self)
