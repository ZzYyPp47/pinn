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

class create_point(object):
    def __init__(self,delta,device):
        self.device = device
        self.rand_point(delta)

    def rand_point(self,delta):
        x = torch.arange(-1, 1 + delta, delta, requires_grad=True,dtype=torch.float32,device=self.device)  # 不包含最后一项
        t = torch.arange(-1, 1 + delta, delta, requires_grad=True,dtype=torch.float32,device=self.device)
        grid_x, grid_t = torch.meshgrid(x, t)
        self.pde_x = grid_x.reshape(-1, 1)
        self.pde_t = grid_t.reshape(-1, 1)
        self.attention = torch.ones_like(self.pde_x,requires_grad=True,dtype=torch.float32,device=self.device)


    def __len__(self):
        return len(self.len_pde)
        # return max(self.len_pde, self.len_bound, self.len_ini)

    def __getitem__(self, idx):
        # 为每类获取 index，此处采用循环索引方式以循环采样每种类型的点
        # idx_bound = idx % self.len_bound
        # idx_ini = idx % self.len_ini
        idx_pde = idx % self.len_pde

        return {
            'pde': (self.pde_x[idx_pde], self.pde_t[idx_pde]),
            # 'bound': (self.bound_x1[idx_bound], self.bound_tx1[idx_bound], self.bound_xm1[idx_bound],
            #           self.bound_txm1[idx_bound]),
            # 'ini': (self.ini_x[idx_ini], self.ini_t[idx_ini])
        }
