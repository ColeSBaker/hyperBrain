import pandas as pd
import numpy as np
from statsmodels.stats.multitest import fdrcorrection

filename=r'share_data/CG1_PRE2R.csv'
sig_df=pd.read_csv(filename)
print(sig_df)


def fdr_correction(p_col,sig_df,method):
    """
    method: Which method to use for FDR correction. {'i', 'indep', 'p', 'poscorr'}
    all refer to fdr_bh (Benjamini/Hochberg for independent or positively correlated tests).
    {'n', 'negcorr'} both refer to fdr_by (Benjamini/Yekutieli for general or negatively correlated tests). 
    Defaults to 'indep'.
    """
    fdr=fdrcorrection(pvals=sig_df[p_col].values,alpha=.05,method=method)    
    sig_df[['Significant','q_value']]= np.array([fdr[0],fdr[1]]).T
    return sig_df

corrected=fdr_correction(p_col='p-value',sig_df=sig_df,method='indep')[['p-value','q_value','Significant']]
print(corrected)