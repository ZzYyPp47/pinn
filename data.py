# -*- coding:utf-8 -*-
'''
@Author: ZYP
@contact: 3137168510@qq.com
@Time: 2024/2/11 0:21
@version: 1.0
@File: data.py
'''

import torch

class create_point(object):
    def __init__(self,N_pde,N_bound,N_ini,N_real,device):
        self.device = device
        self.rand_point(N_pde,N_bound,N_ini,N_real)
    def rand_point(self,N_pde,N_bound,N_ini,N_real):
        self.pde_x = 2 * torch.rand(N_pde,1,requires_grad=True,dtype=torch.float32,device=self.device) - 1 # -1 < x < 1
        self.pde_t = torch.rand_like(self.pde_x,requires_grad=True, dtype=torch.float32,device=self.device)
        self.ini_x = 2 * torch.rand(N_ini,1,requires_grad=True, dtype=torch.float32,device=self.device) - 1 # -1 < x < 1
        self.ini_t = torch.zeros_like(self.ini_x,requires_grad=True, dtype=torch.float32,device=self.device)
        self.real_x = 2 * torch.rand(N_real,1,requires_grad=True, dtype=torch.float32,device=self.device) - 1 # -1 < x < 1
        self.real_t = torch.rand_like(self.real_x,requires_grad=True, dtype=torch.float32,device=self.device)
        self.bound_x1 = torch.ones(N_bound // 2,1,requires_grad=True,dtype=torch.float32,device=self.device) # x = 1
        self.bound_xm1 = torch.full_like(self.bound_x1,-1,requires_grad=True, dtype=torch.float32,device=self.device)  # x = -1
        self.bound_tx1 = torch.rand_like(self.bound_x1,requires_grad=True, dtype=torch.float32,device=self.device)
        self.bound_txm1 = torch.rand_like(self.bound_xm1, requires_grad=True, dtype=torch.float32,device=self.device)
