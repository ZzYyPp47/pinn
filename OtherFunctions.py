# -*- coding:utf-8 -*-
'''
@Author: ZYP
@Contact: 3137168510@qq.com
@Time: 2025/11/17 16:46
@File: OtherFunctions.py
'''

import torch
import matplotlib.pyplot as plt
import numpy as np

# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)  # 为CPU设置随机种子
    np.random.seed(seed)  # 为np设置随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
        torch.backends.cudnn.deterministic = True  # 使用确定性算法
        torch.backends.cudnn.benchmark = True  # cudnn基准(使用卷积时可能影响结果)

# 画图辅助函数
def draw(pinn, load_path, device):
    # pinn.model.to('cpu') # 卸载到cpu
    checkpoint = torch.load(load_path,weights_only=False)  # 加载模型
    print('loading from {}'.format(load_path))
    pinn.model.load_state_dict(checkpoint['model'])
    pinn.opt.load_state_dict(checkpoint['opt'])
    pinn.Epochs_loss = checkpoint['loss']
    pinn.Epochs_loss = np.array(pinn.Epochs_loss)
    pinn.model.eval()  # 启用评估模式
    with torch.no_grad():
        x = torch.arange(-1, 1.002, 0.002, device=device)  # 不包含最后一项
        t = torch.arange(0, 1.001, 0.001, device=device)
        grid_x, grid_t = torch.meshgrid(x, t)
        mesh_x = grid_x.reshape(-1, 1)
        mesh_t = grid_t.reshape(-1, 1)
        pred = pinn.model(torch.cat([mesh_x, mesh_t], dim=1)).reshape(grid_x.shape)
        y = (torch.exp(-mesh_t) * torch.sin(torch.pi * mesh_x)).reshape(grid_x.shape)
        N = 500  # 等高线密集程度`

        total_relative_l2 = torch.norm(pred - y) / torch.norm(y)
        print('total_relative_l2 ={}'.format(total_relative_l2.item()))

        plt.figure()
        plt.contourf(grid_t.cpu(), grid_x.cpu(), pred.cpu(), N, cmap='jet')
        plt.colorbar()
        plt.title("pred")
        plt.xlabel("t")
        plt.ylabel("x")

        plt.figure()
        plt.contourf(grid_t.cpu(), grid_x.cpu(), y.cpu(), N, cmap='jet')
        plt.colorbar()
        data = pinn.dataloader.dataset
        l1 = plt.scatter(data.real_t, data.real_x, marker='.', c='k',s=5)
        plt.legend(handles=[l1], labels=['real_point(if used)'], loc='best')
        plt.title("real")
        plt.xlabel("t")
        plt.ylabel("x")

        plt.figure()
        abs_error = torch.abs(pred - y)
        plt.contourf(grid_t.cpu(), grid_x.cpu(), abs_error.cpu(), N, cmap='jet')
        plt.colorbar()
        plt.title("Abs error")
        plt.xlabel("t")
        plt.ylabel("x")

        plt.figure()
        plt.semilogy(pinn.Epochs_loss[:, 0], pinn.Epochs_loss[:, 1])
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title('losses with epochs')

        plt.show()
