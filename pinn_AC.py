#-*- coding:utf-8 -*-
'''
@Author: ZYP
@contact: 3137168510@qq.com
@Time: 2024/1/25 15:26
@version: 1.0
@File: pinn.py
'''

# 导入必要的包
import torch
import time
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import datetime
from loss_AC import *


class pinn(object):
    # 初始化
    def __init__(self,name,total_epochs,tor,loss_func,loss_A_func,model,point,A_func,opt,opt_A,weights_init,device):
        self.total_epochs = total_epochs
        self.path = 'save/'+name+'.pth'# '_'+datetime.date.today().strftime('%Y-%m-%d,%H:%M:%S')+'.pth'
        self.tor = tor # loss的停止阈值
        self.device = device # 使用的设备
        self.Epochs_loss = [] # 记录loss
        self.weights_init = weights_init # 确定神经网络初始化的方式
        self.loss_func = loss_func # 损失函数(接口)
        self.model = model # 神经网络(接口)
        self.opt = opt # 优化器(接口)
        self.opt_A = opt_A # 优化器(接口)
        self.loss_computer = LossCompute(model, loss_func, loss_A_func,point,A_func)  # 损失计算(接口)
        print(self.model)

    # 参数初始化
    def weights_initer(self,model):
        if isinstance(model,nn.Conv2d):
            self.weights_init(model.weight.data)
            model.bias.data.zero_()
        elif isinstance(model,nn.Linear):
            self.weights_init(model.weight.data)
            model.bias.data.zero_()

    # 为LBFGS优化器准备的闭包函数
    def closure(self):
        self.opt.zero_grad()  # 清零梯度
        self.opt_A.zero_grad()  # 清零梯度信息
        Loss = self.loss_computer.loss()  # 计算损失
        self.Epochs_loss.append([self.Epochs_loss[-1][0] + 1, Loss.item()]) if self.Epochs_loss != [] else self.Epochs_loss.append([1, Loss.item()])
        print('loss:', Loss.item())
        Loss.backward() # 反向计算出梯度
        self.loss_computer.point.attention.grad *= -1  # 倒转梯度方向
        self.opt_A.step()  # 更新参数
        return Loss.item()

    # 训练模型
    def train_all(self):
        # start_time = time.time()
        self.model = self.model.to(self.device) # 将model移入相应设备
        self.model.apply(self.weights_initer)# 神经网络初始化
        self.model.train() # 启用训练模式
        print('start training,using seed:{}'.format(torch.initial_seed()))
        for epoch in range(self.total_epochs):
            self.opt.zero_grad() # 清零梯度信息
            self.opt_A.zero_grad()  # 清零梯度信息
            Loss = self.loss_computer.loss()
            self.Epochs_loss.append([epoch + 1,Loss.item()])
            Loss.backward() # 反向计算出梯度
            self.loss_computer.point.attention.grad *= -1 # 倒转梯度方向
            self.opt.step() # 更新参数
            self.opt_A.step()  # 更新参数
            if (epoch + 1)% 1000 == 0:
                print('Epoch:{}/{},Loss={}'.format(epoch + 1,self.total_epochs,Loss.item()))
            # if Loss <= self.tor:
            #     print('Epoch:{}/{},Loss={}<={}(given tolerate loss)'.format(epoch + 1,self.total_epochs,Loss.item(),self.tor))
            #     break
        # self.Epochs_loss = np.array(self.Epochs_loss)
        # state_dict = {"Arc": self.model,"seed": torch.initial_seed(), "model": self.model.state_dict(), "opt": self.opt.state_dict(), "loss": self.Epochs_loss}
        # torch.save(state_dict,self.path)
        # print('training terminated,saved to{}'.format(self.path))
        # end_time =time.time()
        # print('using times:{}s'.format(end_time-start_time))

    # 最基本的训练模块
    def train(self):
        self.opt.zero_grad()  # 清零梯度信息
        Loss = self.loss_computer.loss()
        Loss.backward()  # 反向计算出梯度
        self.opt.step()  # 更新参数
        return Loss.item()

     # 保存模型参数
    def save(self):
        state_dict = {"Arc": self.model,"seed": torch.initial_seed(), "model": self.model.state_dict(), "opt": self.opt.state_dict(), "loss": self.Epochs_loss}
        torch.save(state_dict,self.path)
        print('model saved to {}'.format(self.path))
