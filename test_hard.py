# -*- coding:utf-8 -*-
'''
@Author: ZYP
@contact: 3137168510@qq.com
@Time: 2024/2/11 21:59
@version: 1.0
@File: test.py
'''

import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
import random
from OtherFunctions import *
from pinn import *
from data import *
from NN.FNN import FNN
from NN.ResFNN import ResFNN

def test():
    # 选择设备
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    # device = "cpu"

    # 初始化
    seed = 0
    mid = 20 # Res网络隐藏层神经元个数
    num_blocks = 50 # 残差块个数
    Arc = [2,20,20,20,1] # 神经网络架构
    func = nn.Tanh() # 确定激活函数
    N_pde = 5000
    N_bound = 500
    N_ini = 500
    N_real = 100
    learning_rate = 0.001
    init_method = torch.nn.init.kaiming_uniform_ # 设置神经网络参数初始化方法
    name = 'FNN_hard' # 模型参数保存的名字
    load_path = 'save/'+name+'.pth' # 加载模型的路径
    total_epochs = 20000
    batch_size = 5000
    tor = 1e-6 # loss阈值
    loss_func = nn.MSELoss().to(device) # 确定损失计算函数
    loss_weight = [1,0,0,0] # loss各项权重(pde,bound,ini,real)

    # dataloader 设置
    dataloader_karg = {
        'shuffle': True,
        'pin_memory': True,
        'num_workers': os.cpu_count() // 8,
        'prefetch_factor': 2,
        'persistent_workers': True,
        'batch_size': batch_size
    }

    # 建立模型
    set_seed(seed)  # 设置确定的随机种子
    model = FNN(Arc,func,device,output_transform=func_out)
    point = create_point(N_pde,N_bound,N_ini,N_real,device)
    dataloader = DataLoader(point, **dataloader_karg)
    opt = torch.optim.Adam(params = model.parameters(),lr=learning_rate)
    pinn_demo = pinn(name,total_epochs,batch_size,tor,loss_func,model,dataloader,opt,init_method,loss_weight,device)
    # 务必确定model与point位于同一设备!!

    # 训练
    start_time = time.time()
    pinn_demo.train_all()
    # 初始化L-BFGS优化器
    pinn_demo.opt = torch.optim.LBFGS(pinn_demo.model.parameters(), history_size=100, tolerance_change=0, tolerance_grad=1e-09, max_iter=5000, max_eval=10000)
    num_epochs_lbfgs = 1
    print(f'\nnow using L_BFGS...')
    for epoch in range(num_epochs_lbfgs):
        pinn_demo.opt.step(pinn_demo.closure)  # 更新权重,注意不要加括号!因为传递的是函数本身而不是函数的返回值！
    end_time = time.time()
    print(f'\nusing times:{end_time - start_time}s')
    pinn_demo.save()

    # 画图
    draw(pinn_demo, load_path, device)

# 输出转换函数
def func_out(out,x):
    return (x[:,0:1] - 1) * (x[:,0:1] + 1) * x[:,1:2] * out + torch.sin(torch.pi * x[:,0:1])

if __name__ == '__main__':
    test()
