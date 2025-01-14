import json
import os.path as osp
from sklearn.model_selection import KFold
import glob
import numpy as np
import torch
import lifelines.utils.concordance as LUC
import random
import pickle

STAGE = {'TCGA-KIRC':{'Stage I': 1, 'Stage II': 2, 'Stage III': 3, 'Stage IV': 4},
         'TCGA-LUSC':{'Stage I': 1, 'Stage IA': 2, 'Stage IB': 3, 'Stage II': 4, 'Stage IIA': 5, 'Stage IIB': 6, 'Stage III': 7, 'Stage IIIA': 8, 'Stage IIIB': 9, 'Stage IV': 10},
         'TCGA-LUAD':{'Stage I': 1, 'Stage IA': 2, 'Stage IB': 3, 'Stage II': 4, 'Stage IIA': 5, 'Stage IIB': 6, 'Stage IIIA': 7, 'Stage IIIB': 8, 'Stage IV': 9},
         'TCGA-UCEC':{'Stage I': 1, 'Stage IA': 2, 'Stage IB': 3, 'Stage II': 4, 'Stage IIA': 5, 'Stage IIB': 6, 'Stage III': 7, 'Stage IIIA': 8, 'Stage IIIB': 9, 'Stage IV': 10}}
STAGE_T = {'TCGA-KIRC':{'T1':1,'T1a':2,'T1b':3,'T2':4,'T2a':5,'T2b':6,'T3':7,'T3a':8,'T3b':9,'T3c':10,'T4':11},
           'TCGA-LUSC':{'T1':1,'T1a':2,'T1b':3,'T2':4,'T2a':5,'T2b':6,'T3':7,'T3a':8,'T4':9},
           'TCGA-LUAD':{'T1':1,'T1a':2,'T1b':3,'T2':4,'T2a':5,'T2b':6,'T3':7,'T4':8, 'TX':9},
           'TCGA-UCEC':{'T1':1,'T1a':2,'T1b':3,'T2':4,'T2a':5,'T2b':6,'T3':7,'T4':8}}
STAGE_M = {'TCGA-KIRC':{'M0':1,'M1':2,'MX':3},
           'TCGA-LUSC':{'M0':1,'M1':2,'M1a':3,'M1b':4,'MX':5},
           'TCGA-LUAD':{'M0':1,'M1':2,'M1a':3,'M1b':4,'MX':5},
           'TCGA-UCEC':{'M0':1,'M1':2,'M1a':3,'M1b':4,'MX':5}}
STAGE_N = {'TCGA-KIRC':{'N0':1,'N1':2,'NX':3},
           'TCGA-LUSC':{'N0':1,'N1':2,'N2':3,'N3':4,'NX':5},
           'TCGA-LUAD':{'N0':1,'N1':2,'N2':3,'N3':4,'NX':5},
           'TCGA-UCEC':{'N0':1,'N1':2,'N2':3,'N3':4,'NX':5}}
GENDER = {'male':0,'female':1}
def get_WSI_sample_list(WSI_info_list,centers,patch_ft_dir,WSI_patch_coor_dir,clinical=False):
    with open(WSI_info_list, 'r') as fp:
        lbls = json.load(fp)
    if WSI_patch_coor_dir is not None:
        all_coor_list = []
        if isinstance(centers,list):
            for center in centers:
                all_coor_list.extend(glob.glob(osp.join(WSI_patch_coor_dir.format(center), '*_coors.pkl')))
        else:
            all_coor_list.extend(glob.glob(osp.join(WSI_patch_coor_dir.format(centers), '*_coors.pkl')))
        
        coor_dict = {}
        def get_id(_dir):
            tmp_dir = _dir.split('/')[-1][:-10]
            return tmp_dir
        for _dir in all_coor_list:
            id = get_id(_dir)
            coor_dict[id] = _dir
    if patch_ft_dir is not None:
        all_ft_list = []
        if isinstance(centers,list):
            for center in centers:
                all_ft_list.extend(glob.glob(osp.join(patch_ft_dir.format(center), '*_fts.npy')))
        else:
            all_ft_list.extend(glob.glob(osp.join(patch_ft_dir.format(centers), '*_fts.npy')))
        

        
    all_dict = {}
    survival_time_max = 0
    survival_time_min = None
    none_list = []
    if 'TCGA' in WSI_info_list:
        for patient in lbls:
            image_id = patient['diagnoses'][0]['submitter_id'].split('_')[0]
            all_dict[image_id] = {}
            if clinical:
                all_dict[image_id]['gender'] = GENDER[patient['demographic']['gender']]
                if patient['demographic']["age_at_index"] is not None:
                    all_dict[image_id]['age'] = (patient['demographic']["age_at_index"]+5) // 5
                else:
                    all_dict[image_id]['age'] = 0
                
                if "ajcc_pathologic_stage" in patient["diagnoses"][0].keys():
                    all_dict[image_id]['stage'] = STAGE[centers[0]][patient["diagnoses"][0]["ajcc_pathologic_stage"]]
                else:
                    all_dict[image_id]['stage'] = 0
                if "ajcc_pathologic_t" in patient["diagnoses"][0].keys():
                    all_dict[image_id]['t'] = STAGE_T[centers[0]][patient["diagnoses"][0]["ajcc_pathologic_t"]]
                else:
                    all_dict[image_id]['t'] = 0
                if "ajcc_pathologic_m" in patient["diagnoses"][0].keys():
                    all_dict[image_id]['m'] = STAGE_M[centers[0]][patient["diagnoses"][0]["ajcc_pathologic_m"]]
                else:
                    all_dict[image_id]['m'] = 0
                if "ajcc_pathologic_n" in patient["diagnoses"][0].keys():
                    all_dict[image_id]['n'] = STAGE_N[centers[0]][patient["diagnoses"][0]["ajcc_pathologic_n"]]
                else:
                    all_dict[image_id]['n'] = 0

            if 'days_to_death' in patient['demographic'].keys():
                time = int(patient['demographic']['days_to_death'])
                all_dict[image_id]['status'] = int(1)
            else:
                try:
                    time = int(patient['diagnoses'][0]['days_to_last_follow_up'])
                    all_dict[image_id]['status'] = int(0)
                except:
                    del all_dict[image_id]
                    continue
            all_dict[image_id]['survival_time'] = time
            

            #filter low survival time
            if time<=7:
                del all_dict[image_id]
                continue


            if str(image_id) in coor_dict.keys():
                    all_dict[image_id]['patch_coors'] = coor_dict[str(image_id)]
            else:
                del all_dict[image_id]
                none_list.append(image_id)
                print('no coor_dir     '+image_id)
            survival_time_max = survival_time_max \
                    if survival_time_max > time else time
            if survival_time_min is None:
                survival_time_min = time
            else:
                survival_time_min = survival_time_min \
                    if survival_time_min < time else time
    
    if patch_ft_dir is not None:
        def get_id(_dir):
            tmp_dir = _dir.split('/')[-1][:-8]
            return tmp_dir
        for _dir in all_ft_list:
            id = get_id(_dir)
            if id in all_dict.keys():
                all_dict[id]['ft_dir']=_dir


    return all_dict, survival_time_max, survival_time_min

def get_n_fold_data_list(data_dict,n_fold,random_seed,clinical=False):
    censored_keys = []
    uncensored_keys = []
    for key in data_dict.keys():
        if data_dict[key]['status'] == 1:
            uncensored_keys.append(key)
        else:
            censored_keys.append(key)
    print("censored length {}".format(len(censored_keys)))
    print("uncensored length {}".format(len(uncensored_keys)))

    n_fold_uncensored_train_list = []
    n_fold_uncensored_val_list = []
    n_fold_censored_train_list = []
    n_fold_censored_val_list = []
    n_fold_train_list = []
    n_fold_val_list = []
    kf = KFold(n_splits=n_fold, shuffle=True, random_state=random_seed) #random_seed
    for train_idx, val_idx in kf.split(uncensored_keys):
        train_keys = [uncensored_keys[i] for i in train_idx]
        val_keys = [uncensored_keys[i] for i in val_idx]

        train_data_dict = {key: data_dict[key] for key in train_keys}
        val_data_dict = {key: data_dict[key] for key in val_keys}
        n_fold_uncensored_train_list.append(train_data_dict)
        n_fold_uncensored_val_list.append(val_data_dict)

    for train_idx, val_idx in kf.split(censored_keys):
        train_keys = [censored_keys[i] for i in train_idx]
        val_keys = [censored_keys[i] for i in val_idx]

        train_data_dict = {key: data_dict[key] for key in train_keys}
        val_data_dict = {key: data_dict[key] for key in val_keys}
        n_fold_censored_train_list.append(train_data_dict)
        n_fold_censored_val_list.append(val_data_dict)

    for i in range(n_fold):
        n_fold_train_list.append(dict(n_fold_censored_train_list[i],**n_fold_uncensored_train_list[i]))
        n_fold_val_list.append(dict(n_fold_censored_val_list[i],**n_fold_uncensored_val_list[i]))

    print()

    return n_fold_train_list, n_fold_val_list

def sort_survival_time(gt_survival_time,pre_risk,censore, output_fts,patch_ft=None,coors=None):
    ix = torch.argsort(gt_survival_time, dim= 0, descending=True)#
    gt_survival_time = gt_survival_time[ix]
    pre_risk = pre_risk[ix]
    censore = censore[ix]
    output_fts = output_fts[ix]
    if patch_ft is not None:
        patch_ft = patch_ft[ix]
        coors = coors[ix]
        return gt_survival_time,pre_risk,censore,output_fts,patch_ft,coors
    return gt_survival_time,pre_risk,censore,output_fts

def accuracytest(survivals, risk, censors):
    survlist = []
    risklist = []
    censorlist = []

    for riskval in risk:
        # riskval = -riskval
        risklist.append(riskval.cpu().detach().item())

    for censorval in censors:
        censorlist.append(censorval.cpu().detach().item())

    for surval in survivals:
        # surval = -surval
        survlist.append(surval.cpu().detach().item())

    C_value = LUC.concordance_index(survlist, risklist, censorlist)

    return C_value

