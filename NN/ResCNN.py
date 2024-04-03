# -*- coding:utf-8 -*-
'''
@Author: ZYP
@contact: 3137168510@qq.com
@Time: 2024/2/10 23:38
@version: 1.0
@File: ResFNN.py
'''

# 示例:
# ResCNN(2,5,nn.ReLU())
# 将自动创建一个具有5个残差块,残差块中特征层为2的残差卷积神经网络(共2+2*5=12层)，其中激活函数使用Relu

import torch
import torch.nn as nn
class ResCNN(nn.Module):
    def __init__(self,mid,num_blocks,func,input_transform = None,output_transform = None):
        super(ResCNN, self).__init__()  # 调用父类的构造函数
        self.func = func
        self.input_transform = input_transform # 输入特征转换
        self.output_transform = output_transform # 输出特征转换
        self.Conv_F = nn.Conv2d(1,mid,kernel_size=3,padding='same',stride=1,padding_mode='zeros')
        self.blocks = nn.ModuleList([nn.Sequential(
            nn.Conv2d(mid,mid,kernel_size=3,padding='same',stride=1,padding_mode='zeros'),
            self.func,
            nn.Conv2d(mid,mid,kernel_size=3,padding='same',stride=1,padding_mode='zeros'))
            for _ in range(num_blocks)])
        # 使用列表推理建立num_blocks个残差块，每个块都有两个卷积层和一个激活函数
        self.Conv_E = nn.Conv2d(mid,1,kernel_size=3,padding='same',stride=1,padding_mode='zeros')

    def forward(self,x):
        if self.input_transform is not None:
            x = self.input_transform(x)
        out = self.Conv_F(x)  # 首先，数据通过第一层卷积层
        for block in self.blocks:  # 然后数据依次通过每个残差块
            residual = out  # 在进入残差块之前，保存当前的模型输出，以便后面进行残差连接
            out = block(out)  # 数据通过残差块
            out += residual  # 残差连接：原模型的输出被加到经过残差块处理的数据上
            out = self.func(out)  # 结果进入预设的激活函数
        out = self.Conv_E(out)  # 最后，数据通过最后一层卷积层，得到最终输出
        if self.output_transform is not None:
            out = self.output_transform(out,x)
        return out