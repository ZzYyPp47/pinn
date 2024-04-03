# -*- coding:utf-8 -*-
'''
@Author: ZYP
@contact: 3137168510@qq.com
@Time: 2024/2/10 16:44
@version: 1.0
@File: CNN.py
'''
# Arc示例:
#     arc = [2,20,20,1]
# 以此Arc，脚本将自动创建一个具有4层的前馈神经网络，其中各层神经元个数分别为2,20,20,1

import torch
import torch.nn as nn

class FNN(nn.Module):
    def __init__(self,Arc,func,device,input_transform = None,output_transform = None):
        super(FNN, self).__init__()  # 调用父类的构造函数
        self.input_transform = input_transform # 输入特征转换
        self.output_transform = output_transform # 输出特征转换
        self.func = func # 定义激活函数
        self.Arc = Arc # 定义网络架构
        self.device = device
        self.model = self.create_model().to(self.device)
        # print(self.model)


    def create_model(self):
        layers = []
        for ii in range(len(self.Arc) - 1):
            layers.append(nn.Linear(self.Arc[ii], self.Arc[ii + 1]))
            if ii < len(self.Arc) - 2:  # 最后一层之前要加激活函数
                layers.append(self.func)
        return nn.Sequential(*layers)  # *将layers列表解包为独立变量依次传入sequential(不允许直接传入列表)


    def forward(self,x):
        if self.input_transform is not None:
            x = self.input_transform(x)
        out = self.model(x)
        if self.output_transform is not None:
            out = self.output_transform(out,x)
        return out