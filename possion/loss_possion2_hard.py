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
        self.point = point if point is not None else PointContainer()
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
        u_tt = self.gradient(u, self.point.pde_t,2)
        u_xx = self.gradient(u, self.point.pde_x, 2)
        lapacian_u = u_xx + u_tt
        return self.loss_func(lapacian_u + (self.point.pde_t ** 2) * (self.point.pde_x ** 2), torch.zeros_like(u))

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
        return self.loss_pde()

    # 更新已存在的点
    def update_points(self, new_point_container):
        # 为每个属性进行检查和更新
        for point in ['pde_x', 'pde_t', 'ini_x', 'ini_t', 'bound_x1', 'bound_xm1', 'bound_tx1', 'bound_txm1']:
            new_point = getattr(new_point_container, point, None)
            if new_point is not None:
                current_point = getattr(self.point, point, None)
                if current_point is not None:
                    # 如果当前属性已存在，将新的点向量附加到旧的向量后面
                    updated_value = torch.cat((current_point, new_point), dim=0)  # 假设是沿着第0维连接
                    setattr(self.point, point, updated_value)
                else:
                    # 如果当前属性不存在，直接设置新向量
                    setattr(self.point, point, new_point)

# 为空point保留的初始化容器
class PointContainer:
    def __init__(self,pde_x = None,pde_t = None,ini_x = None,ini_t = None,bound_x1 = None,bound_xm1 = None,bound_tx1 = None,bound_txm1 = None):
        self.pde_x = pde_x if pde_x is not None else None
        self.pde_t = pde_t if pde_t is not None else None
        self.ini_x = ini_x if ini_x is not None else None
        self.ini_t = ini_t if ini_t is not None else None
        self.bound_x1 = bound_x1 if bound_x1 is not None else None
        self.bound_xm1 = bound_xm1 if bound_xm1 is not None else None
        self.bound_tx1 = bound_tx1 if bound_tx1 is not None else None
        self.bound_txm1 = bound_txm1 if bound_txm1 is not None else None