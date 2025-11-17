# -*- coding:utf-8 -*-
'''
@Author: ZYP
@contact: 3137168510@qq.com
@Time: 2024/2/11 0:21
@version: 1.0
@File: data.py
'''

import torch
from torch.utils.data import Dataset

class create_point(Dataset):
    def __init__(self,N_pde,N_bound,N_ini,N_real,device):
        self.device = device
        self.len_pde = N_pde
        self.len_bound = N_bound
        self.len_ini = N_ini
        self.len_real = N_real
        self.rand_point(N_pde,N_bound,N_ini,N_real)

    def rand_point(self,N_pde,N_bound,N_ini,N_real):
        self.pde_x = 2 * torch.rand(N_pde,1,dtype=torch.float32) - 1 # -1 < x < 1
        self.pde_t = torch.rand_like(self.pde_x, dtype=torch.float32)
        self.ini_x = 2 * torch.rand(N_ini,1, dtype=torch.float32) - 1 # -1 < x < 1
        self.ini_t = torch.zeros_like(self.ini_x, dtype=torch.float32)
        self.real_x = 2 * torch.rand(N_real,1, dtype=torch.float32) - 1 # -1 < x < 1
        self.real_t = torch.rand_like(self.real_x, dtype=torch.float32)
        self.bound_x1 = torch.ones(N_bound // 2,1,dtype=torch.float32) # x = 1
        self.bound_xm1 = torch.full_like(self.bound_x1,-1, dtype=torch.float32)  # x = -1
        self.bound_tx1 = torch.rand_like(self.bound_x1, dtype=torch.float32)
        self.bound_txm1 = torch.rand_like(self.bound_xm1,  dtype=torch.float32)

    def __len__(self):
        return max(self.len_pde, self.len_bound, self.len_ini, self.len_real)

    def __getitem__(self, idx):
        # 为每类获取 index，此处采用循环索引方式以循环采样每种类型的点
        idx_bound = idx % (self.len_bound // 2)
        idx_ini = idx % self.len_ini
        idx_pde = idx % self.len_pde

        kargs = {
            'pde_x': self.pde_x[idx_pde],
            'pde_t': self.pde_t[idx_pde],
            'ini_x': self.ini_x[idx_ini],
            'ini_t': self.ini_t[idx_ini],
            'bound_x1': self.bound_x1[idx_bound],
            'bound_tx1': self.bound_tx1[idx_bound],
            'bound_xm1': self.bound_xm1[idx_bound],
            'bound_txm1': self.bound_txm1[idx_bound],
        }

        return kargs