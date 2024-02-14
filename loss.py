# -*- coding:utf-8 -*-
'''
@Author: ZYP
@contact: 3137168510@qq.com
@Time: 2024/2/12 12:41
@version: 1.0
@File: loss.py
'''

import torch


class LossCompute:
    def __init__(self, model, loss_func, point, loss_weight):
        self.model = model
        self.loss_func = loss_func
        self.point = point
        self.loss_weight = loss_weight # 损失函数各项的权重分配

    # 计算梯度
    def gradient(self,func,var,order = 1):
        if order == 1:
            return torch.autograd.grad(func,var,grad_outputs=torch.ones_like(func),create_graph=True,only_inputs=True)[0]
        else:
            out = self.gradient(func,var)# 不要加order(以正常计算1阶导),否则会无限循环调用！
            return self.gradient(out,var,order - 1)

    def loss_pde(self):
        u = self.model(torch.cat([self.point.pde_x, self.point.pde_t], dim=1))
        u_t = self.gradient(u, self.point.pde_t)
        return self.loss_func(u_t + u, torch.zeros_like(u))

    def loss_bound(self):
        u_x1 = self.model(torch.cat([self.point.bound_x1, self.point.bound_tx1], dim=1))
        u_xm1 = self.model(torch.cat([self.point.bound_xm1, self.point.bound_txm1], dim=1))
        return self.loss_func(u_x1, torch.zeros_like(u_x1)) + self.loss_func(u_xm1, torch.zeros_like(u_xm1))

    def loss_ini(self):
        u = self.model(torch.cat([self.point.ini_x, self.point.ini_t], dim=1))
        return self.loss_func(u, torch.sin(torch.pi * self.point.ini_x))

    def loss_real(self):
        out = self.model(torch.cat([self.point.real_x, self.point.real_t], dim=1))
        return self.loss_func(out, torch.exp(-self.point.real_t) * torch.sin(torch.pi * self.point.real_x))

    def loss(self):
        return self.loss_weight[0]*self.loss_pde()+self.loss_weight[1]*self.loss_bound()+self.loss_weight[2]*self.loss_ini()+self.loss_weight[3]*self.loss_real()