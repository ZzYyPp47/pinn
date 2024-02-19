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
import matplotlib as plt
import random
from pinn import *
from data import *
from NN.FNN import FNN
from NN.ResFNN import ResFNN

# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)  # 为CPU设置随机种子
    np.random.seed(seed)  # 为np设置随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
        torch.backends.cudnn.deterministic = True  # 使用确定性算法

# 加载并测试...
def test():
    # 初始化
    seed = 0
    mid = 20 # Res网络隐藏层神经元个数
    num_blocks = 50 # 残差块个数
    Arc = [2,20,20,1] # 神经网络架构
    func = nn.ReLU() # 确定激活函数
    N_pde = 5000
    N_bound = 5000
    N_ini = 5000
    N_real = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 自动选择可用设备,优先GPU
    learning_rate = 0.001
    init_method = torch.nn.init.kaiming_uniform_ # 设置神经网络参数初始化方法
    name = 'FNN' # 模型参数保存的名字
    load_path = 'save/'+name+'.pth' # 加载模型的路径
    total_epochs = 6000
    tor = 0.0001 # loss阈值
    loss_func = nn.MSELoss().to(device) # 确定损失计算函数
    loss_weight = [1,3,5,0] # loss各项权重(pde,bound,ini,real)

    # 建立模型
    set_seed(seed)  # 设置确定的随机种子
    model = FNN(Arc,func,device)
    point = create_point(N_pde,N_bound,N_ini,N_real,device)
    opt = torch.optim.Adam(params = model.parameters(),lr=learning_rate)
    weights_init = init_method # Kaiming初始化
    pinn_demo = pinn(name,total_epochs,tor,loss_func,model,point,opt,weights_init,loss_weight,device)
    # 务必确定model与point位于同一设备!!

    # 训练
    pinn_demo.train()

    # 加载并测试
    pinn_demo.model.to('cpu') # 卸载到cpu
    checkpoint = torch.load(load_path) # 加载模型
    print('loading from',load_path)
    pinn_demo.model.load_state_dict(checkpoint['model'])
    pinn_demo.opt.load_state_dict(checkpoint['opt'])
    pinn_demo.Epochs_loss = checkpoint['loss']
    # current_epoch = checkpoint["loss"][-1][0]
    # current_loss = checkpoint["loss"][-1][1]
    pinn_demo.model.eval() # 启用评估模式
    with torch.no_grad():
        x = torch.linspace(-1, 1, 100)
        t = torch.linspace(0, 1, 100)
        grid_x, grid_t = torch.meshgrid(x, t)
        mesh_x = grid_x.reshape(-1, 1)
        mesh_t = grid_t.reshape(-1, 1)
        y = (torch.exp(-mesh_t) * torch.sin(torch.pi * mesh_x)).reshape(grid_x.shape)
        pred = pinn_demo.model(torch.cat([mesh_x, mesh_t], dim=1)).reshape(grid_x.shape)
        N = 900 # 等高线密集程度

        plt.figure()
        plt.contourf(grid_t,grid_x,pred,N,cmap='jet')
        plt.colorbar()
        plt.title("pred")
        plt.xlabel("t")
        plt.ylabel("x")

        plt.figure()
        plt.contourf(grid_t,grid_x,y,N,cmap='jet')
        plt.colorbar()
        l1 = plt.scatter(pinn_demo.point.real_t.cpu(), pinn_demo.point.real_x.cpu(), marker='.', c='k',s=5)
        plt.legend(handles=[l1], labels=['real_point(if used)'], loc='best')
        plt.title("real")
        plt.xlabel("t")
        plt.ylabel("x")

        plt.figure()
        error = torch.abs(pred - y).reshape(grid_x.shape)
        plt.contourf(grid_t, grid_x, error,N,cmap='jet')
        plt.colorbar()
        plt.title("Abs error")
        plt.xlabel("t")
        plt.ylabel("x")

        plt.figure()
        plt.plot(pinn_demo.Epochs_loss[:, 0], pinn_demo.Epochs_loss[:, 1])
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title('losses with epochs')

        plt.show()


if __name__ == '__main__':
    test()
