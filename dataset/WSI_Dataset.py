from torch.utils.data import Dataset
import torch
import numpy as np
import pickle


class SlidePatch(Dataset):
    def __init__(self, data_dict: dict, survival_time_max, survival_time_min,clinical=False):
        super().__init__()
        self.st_max = float(survival_time_max)
        self.st_min = float(survival_time_min)
        self.count = 0
        self.id_list = list(data_dict.keys())
        self.data_dict = data_dict
        self.clinical = clinical

    def __getitem__(self, idx: int):
        id = self.id_list[idx]
        self.count += 1
        fts = torch.tensor(np.load(self.data_dict[id]['ft_dir'])).float() 
        ti = torch.tensor(self.data_dict[id]['survival_time']).float()
        survival_time = ti/365  # self.st_max #(self.st_min * (self.st_max - ti))/ (ti * (self.st_max - self.st_min)) #ti/self.st_max #  /self.st_max #
        status = torch.tensor(self.data_dict[id]['status'])
        with open(self.data_dict[id]['patch_coors'], 'rb') as f:
            coors = pickle.load(f)
            coors = torch.Tensor(coors)

        if self.clinical:
            stage = self.data_dict[id]['stage']
            stage_t = self.data_dict[id]['t']
            stage_m = self.data_dict[id]['m']
            stage_n = self.data_dict[id]['n']
            age = self.data_dict[id]['age']
            gender = self.data_dict[id]['gender']
            attr = torch.tensor([stage, stage_t, stage_m, stage_n, age, gender])#
            return fts, survival_time, status, coors, attr, id
        else:
            return fts, survival_time, status, coors, id

    def __len__(self) -> int:
        return len(self.id_list)
