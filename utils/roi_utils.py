
import numpy as npy
import pandas as pd

DMN=['l PCC', 'r PCC', 'l prec', 'r prec', 'l par', 'r par', 'l ITG', 'r ITG', 'l parahip', 'r parahip', 'l SFG', 'r SFG', 'ant med PFC', 'vent med PFC' ]
pDMN=['l PCC', 'r PCC', 'l prec', 'r prec', 'l par', 'r par', 'l ITG', 'r ITG', 'l parahip', 'r parahip']
aDMN=[ 'l SFG', 'r SFG', 'ant med PFC', 'vent med PFC' ]
DAN=[ 'l lat FC', 'r lat FC', 'l post intrapar', 'r post intrapar', 'l ant intrapar', 'r ant intrapar' ]
FPN=[ 'r intrap Sulcus', 'l intrap Sulcus', 'r FC', 'l FC', 'r precun', 'l precun', 'middl Cingulate', 'r inf pariet', 'l inf parie', 'r dIPFC', 'l dIPFC' ]
VN=[ 'l prim vis', 'r prim vis', 'r lingual', 'l lingual', 'r cuneus', 'l cuneus', 'l mid occ', 'r mid occ', 'l inf occ', 'r inf occ' ]
VAN=[ 'r precentral sulc', 'r mid FG', 'r ant insula', 'r temppar junc', 'r sup temp sulc', 'l precentral sulc', 'l mid FG', 'l ant insula', 'l temppar junc', 'l sup temp sulc' ]
SN=[ 'l fronto insular', 'r fronto insular', 'ACC', 'l amyg', 'r amyg' ]
SMN=[ 'l sensorimotor prim', 'r sensorimotor prim', 'l SMA', 'r SMA' ]

RSN_TO_LEN={'DMN':len(DMN),'pDMN':len(pDMN),'aDMN':len(aDMN),'DAN':len(DAN),
                        'FPN':len(FPN),'VN':len(VN),'VAN':len(VAN),'SN':len(SN),'SMN':len(SMN)}

def meg_roi_network_assigments(row,combine_dmn=True,network_thresh=0,
    rank_thresh_inclusive=8,use_edge_atlas=False,use_binary=False,
    balanced=False):

    ### takes in df that for each ROI has the pct of its sources that belong to a given subnetwork
    ### this is where an edge based system could work?

    # if use_nets == use_SOB:
    #     if use_SOB:
    #         raise Exception('Both network and SOB are true')
    #     if not use_SOB:
    #         raise Exception('Both network and SOB are False')



    if 'Left' in row['Names']:
        sob_assignment=1
        sob_assignment_name='L'
    elif 'Right' in row['Names']:
        sob_assignment=0
        sob_assignment_name='R'
    else:
        raise Exception('NIETHER RIGHT NOR LEFT IN ', str(row['Name']))

    row['SOB']=sob_assignment
    row['SOB Name']=sob_assignment_name

    networks= ["DMN","pDMN","aDMN","DAN","FPN","VN","VAN","SN","SMN"]
    network_to_full = {'DMN':'Default Mode Net','pDMN':'Posterior DMN','aDMN':'Anterior DMN','DAN':'Dorsal Att Net' , 'FPN':'Frontoparietal Net', 'VN':'Visual Net','VAN':'Ventral Att Net', 'SN':'Salience Net','SMN':'Sensorimotor Net' }
    if combine_dmn:
        networks = ["DMN","DAN","FPN","VN","VAN","SN","SMN"]
    else:
        networks = ["pDMN","aDMN","DAN","FPN","VN","VAN","SN","SMN"]
    null_idx = len(networks)

    pct_or_thresh_name='thresh' if use_binary else 'pct'

    bal_add=' Bal' if balanced else ''
    network_cols = [n+ " Pct"+bal_add for n in networks]
    rank_cols = [n+ " Rank"+bal_add for n in networks]
    network_bin_cols = [n+ " Bin" for n in networks]
    pcts = row[network_cols]

    
    adaptive_offset=0 ### to account for us having more rois than them
    # else:
        # adaptive_thresh=True
        
    if rank_thresh_inclusive>0 and not use_binary:
        print('cannot use rank_thresh without binary')
        print('using pct only, making rank thresh -1')
        rank_thresh_inclusive=-1
    for i in range(len(network_bin_cols)):
        # print(rank_thresh_inclusive,'RANK THRESH')
        if rank_thresh_inclusive=='adaptive':
            assert use_binary
            bin_thresh=RSN_TO_LEN[networks[i]]+adaptive_offset
            # print(networks[i],bin_thresh)
            row[network_bin_cols[i]]= 1 if row[rank_cols[i]]<=bin_thresh else 0

        elif rank_thresh_inclusive>0:
            assert use_binary
            bin_assigns=np.where((row[rank_cols]<=rank_thresh_inclusive),1,0)
            row[network_bin_cols[i]]=bin_assigns[i] if pcts[i]>.03 else 0
        else:
            # assert 
            row[network_bin_cols[i]]= 1 if pcts[i]>=network_thresh else 0
    ass_vals=row[network_bin_cols]  ### will change if binary

    if ass_vals.values.max()<= network_thresh:  ### here's the network thresh..... works w/ binary bc 1 or 0 will either be above or below
        # print('nothing!')
        row['Functional Net']=null_idx
        row['Functional Net Name']='N/A'
    else:

        assignment = ass_vals.values.argmax()
        # print(assignment)
        assignments_name = network_to_full[networks[assignment]]
        # print(assignments_name,'name dat ass')

        row['Functional Net']=int(assignment)
        row['Functional Net Name']=assignments_name

    row['Func Net SOB']= (row['SOB']*(null_idx+1))+row['Functional Net']
    row['Func Net SOB Name']= row['SOB Name']+' '+row['Functional Net Name']
    # row['ROI Name']=
    return row