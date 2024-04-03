# -*- coding:utf-8 -*-
'''
@Author: ZYP
@contact: 3137168510@qq.com
@Time: 2024/2/10 18:04
@version: 1.0
@File: CNN.py
'''

import torch
import torch.nn as nn

# Arc示例:
#     arc = [
#     {'in': 1, 'out': 64, 'ker': 3},
#     {'in': 64, 'out': 128, 'ker': 3},
#     {'in': 128, 'out': 256, 'ker': 3},
# ]
# 以此Arc，脚本将自动创建一个具有3个卷积块(共6层)的卷积神经网络

class CNN(nn.Module):
    def __init__(self,Arc,pool_size,func,device,input_transform = None,output_transform = None):
        super(CNN, self).__init__()  # 调用父类的构造函数
        self.input_transform = input_transform # 输入特征转换
        self.output_transform = output_transform # 输出特征转换
        self.func = func # 定义激活函数
        self.Arc = Arc # 定义网络架构(这里Arc是一个字典)
        self.pool_size = pool_size
        self.device = device
        self.model = self.create_model().to(self.device)
        # print(self.model)

    def create_model(self):
        layers = []
        for layer in self.Arc:
            layers.append(
                nn.Conv2d(layer['in'],layer['out'],layer['ker'],padding='same',stride=1,padding_mode='zeros')
            )
            layers.append(self.func)
            layers.append(nn.MaxPool2d(self.pool_size,stride=1,padding='same')) # 注意padding='same'只有在步长为1时才可用
        return nn.Sequential(*layers)

    def forward(self,x):
        if self.input_transform is not None:
            x = self.input_transform(x)
        out = self.model(x)
        if self.output_transform is not None:
            out = self.output_transform(out,x)
        return out
