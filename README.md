# 带有软注意力机制的PINN

## 简介

本项目在<[ZzYyPp47/pinn: 一个简易的模块化物理信息神经网络实现(PINN) (github.com)](https://github.com/ZzYyPp47/pinn)>基础上，复现了论文:

>```
>@article{mcclenny2023self,
>  title={Self-adaptive physics-informed neural networks},
>  author={McClenny, Levi D and Braga-Neto, Ulisses M},
>  journal={Journal of Computational Physics},
>  volume={474},
>  pages={111722},
>  year={2023},
>  publisher={Elsevier}
>}
>```

所提出的软注意力机制。

以下未标注`.py`文件的图片,可自行通过<[ZzYyPp47/pinn: 一个简易的模块化物理信息神经网络实现(PINN) (github.com)](https://github.com/ZzYyPp47/pinn)>复现

## Example

以下述示例PDE展示注意力机制的行为：
$$
\left\{
\begin{aligned}
&\frac{\partial u}{\partial t}=-u\\
&u(x=1,t)=0\\
&u(x=-1,t)=0\\
&u(x,t=0)=\sin (\pi x)
\end{aligned}
\right.
$$
精确解为$u=e^{-t}\sin (\pi x)$

硬边界即:
$$
\hat{u}(x,t)=(x-1)(x+1)tNN(x,t,\theta)+\sin (\pi x)
$$

具体来说, 注意力机制的基本思想是为每个子损失函数的采样点赋予动态权重, 并在每轮训练中自动更新这些权重, 从而在训练过程中实现更细粒度、更灵活的权重调节. 与传统的固定或比例性权重不同, 这些自适应权重可以根据实时训练反馈进行调整, 以更精确地反映当前各子损失函数的重要性和贡献度. 从而可以看作是"软注意力层"

将损失函数$\mathcal{L}$改写如下:
$$
\begin{equation}
    \mathcal{L}(\boldsymbol{\boldsymbol{\lambda}}_{pde}, \boldsymbol{\lambda}_{ini}, \boldsymbol{\lambda}_{bound})=\mathcal{L}_{pde}(\boldsymbol{\lambda}_{pde})+\mathcal{L}_{ini}(\boldsymbol{\lambda}_{ini})+\mathcal{L}_{bound}(\boldsymbol{\lambda}_{bound}). 
\end{equation}
$$

其中
$$
\begin{equation}
\boldsymbol{\lambda}_{pde}=
\left(
\begin{array}{c}
 \lambda_{1}^{pde}\\
 \lambda_{2}^{pde}\\
 \vdots\\
 \lambda_{N_{pde}}^{pde}\\
 \end{array}
 \right), 
 \quad
 \boldsymbol{\lambda}_{ini}=
\left(
\begin{array}{c}
  \lambda_{1}^{ini}\\
  \lambda_{1}^{ini}\\
 \vdots\\
  \lambda_{N_{ini}}^{ini}\\
 \end{array}
 \right), 
 \quad
 \boldsymbol{\lambda}_{bound}=
\left(
\begin{array}{c}
  \lambda_{1}^{b}\\
  \lambda_{2}^{b}\\
 \vdots\\
 \lambda_{N_{b}}^{b}\\
 \end{array}
 \right). 
\end{equation}
$$
是可训练的、分别对应于$PDE$采样点、边界条件采样点、初始条件采样点的非负权重矩阵. 而$\mathcal{L}_{pde}(\mathcal{L}_{pde}), $
$\mathcal{L}_{ini}(\mathcal{L}_{ini}), \mathcal{L}_{bound}(\mathcal{L}_{bound})$​​分别代表:
$$
\begin{equation*}
\left\{
\begin{aligned}
\mathcal{L}_{pde}(\boldsymbol{\lambda}_{pde})&:=\frac{1}{N_{pde}}\sum_{i=1}^{N_{pde}}\sigma(\lambda_{i}^{pde})\left|\mathcal{F}(x_i^{pde}, t_i^{pde})\right|^2, \\
\mathcal{L}_{ini}(\boldsymbol{\lambda}_{ini})&:=\frac{1}{N_{ini}}\sum_{i=1}^{N_{ini}}\sigma(\lambda_{i}^{ini})\left|\mathcal{N}_{\theta}(x_i^{ini}, 0)-x_i^{ini}\cos(\pi x_i^{ini})\right|^2, \\
\mathcal{L}_{bound}(\boldsymbol{\lambda}_{bound})&:=\frac{1}{N_{b}}\sum_{i=1}^{N_{b}}\sigma(\lambda_{i}^{b})\left(\left|\mathcal{N}_{\theta}(1, t_i^{b})-\mathcal{N}_{\theta}(-1, t_i^{b})\right|^2+\left|\frac{\partial\mathcal{N}_{\theta}}{\partial x}(1, t_i^{b})-\frac{\partial\mathcal{N}_{\theta}}{\partial x}(-1, t_i^{b})\right|^2\right). \\
\end{aligned}
\right. 
\end{equation*}
$$


为了进一步增强自适应权重层的效果, 文中引入了一个定义在$[0, +\infty)$上的非负、严格单调递增的连续函数$\sigma(\cdot)$, 用来放大权重的影响. 为方便叙述, 称这个函数为“激励函数”, 其作用是通过特定的数学变换, 使得权重能够在更广泛的范围内起作用, 从而更好地突出某些关键组分的重要性. 常用的激励函数包括多项式函数、指数函数和对数函数等. 不同的激励函数能够产生不同的效果, 通过选择合适的激励函数, 可以进一步加强模型的表现. 

这里各采样点的权重依照以下机制更新:
$$
\begin{equation}
    \left\{
    \begin{aligned}
\boldsymbol{\lambda}^{k+1}_{pde}&=\boldsymbol{\lambda}^{k}_{pde}+\rho_{pde}^{k}\nabla_{\boldsymbol{\lambda}_{pde}}\mathcal{L}(\boldsymbol{\lambda}^{k}_{pde}, \boldsymbol{\lambda}^{k}_{ini}, \boldsymbol{\lambda}^{k}_{bound}), \\
\boldsymbol{\lambda}^{k+1}_{ini}&=\boldsymbol{\lambda}^{k}_{ini}+\rho_{ini}^{k}\nabla_{\boldsymbol{\lambda}_{ini}}\mathcal{L}(\boldsymbol{\lambda}^{k}_{pde}, \boldsymbol{\lambda}^{k}_{ini}, \boldsymbol{\lambda}^{k}_{bound}), \\
\boldsymbol{\lambda}^{k+1}_{bound}&=\boldsymbol{\lambda}^{k}_{bound}+\rho_{bound}^{k}\nabla_{\boldsymbol{\lambda}_{bound}}\mathcal{L}(\boldsymbol{\lambda}^{k}_{pde}, \boldsymbol{\lambda}^{k}_{ini}, \boldsymbol{\lambda}^{k}_{bound}), \\   
    \end{aligned}
    \right. 
\end{equation}
$$

其中$\rho_{\mathcal{L}ambda}^{k},$ $\mathcal{L}ambda=pde,$ $ini,$ $bound$ 指各权重层分别在 $k$ 轮时的学习率, 而
$$
\begin{equation*}
\begin{aligned}
    \nabla_{\boldsymbol{\lambda}_{pde}}\mathcal{L}(\boldsymbol{\lambda}^{k}_{pde}, \boldsymbol{\lambda}^{k}_{ini}, \boldsymbol{\lambda}^{k}_{bound})&=
    \frac{1}{N_{pde}}
    \left(
    \begin{aligned}
        &\sigma'(\lambda_{1}^{pde, k})\left|\mathcal{F}(x_i^{pde}, t_i^{pde})\right|^2\\
        &\cdots \\
        &\sigma'(\lambda_{N_{pde}}^{pde, k})\left|\mathcal{F}(x_i^{pde}, t_i^{pde})\right|^2\\
    \end{aligned}
    \right), \\
        \nabla_{\boldsymbol{\lambda}_{ini}}\mathcal{L}(\boldsymbol{\lambda}^{k}_{pde}, \boldsymbol{\lambda}^{k}_{ini}, \boldsymbol{\lambda}^{k}_{bound})&=
    \frac{1}{N_{ini}}
    \left(
    \begin{aligned}
        &\sigma'(\lambda_{1}^{ini, k})\left|\mathcal{N}_{\theta}(x_i^{ini}, 0)-x_i^{ini}\cos(\pi x_i^{ini})\right|^2\\
        &\cdots \\
        &\sigma'(\lambda_{N_{ini}}^{ini, k})\left|\mathcal{N}_{\theta}(x_i^{ini}, 0)-x_i^{ini}\cos(\pi x_i^{ini})\right|^2\\
    \end{aligned}
    \right), \\ 
        \nabla_{\boldsymbol{\lambda}_{bound}}\mathcal{L}(\boldsymbol{\lambda}^{k}_{pde}, \boldsymbol{\lambda}^{k}_{ini}, \boldsymbol{\lambda}^{k}_{bound})&=
    \frac{1}{N_{b}}
    \left(
    \begin{aligned}
        &\sigma'(\lambda_{1}^{b, k})\left(\left|\mathcal{N}_{\theta}(1, t_i^{b})-\mathcal{N}_{\theta}(-1, t_i^{b})\right|^2+\left|\frac{\partial\mathcal{N}_{\theta}}{\partial x}(1, t_i^{b})-\frac{\partial\mathcal{N}_{\theta}}{\partial x}(-1, t_i^{b})\right|^2\right)\\
        & \cdots \\
        &\sigma'(\lambda_{N_{b}}^{b, k})\left(\left|\mathcal{N}_{\theta}(1, t_i^{b})-\mathcal{N}_{\theta}(-1, t_i^{b})\right|^2+\left|\frac{\partial\mathcal{N}_{\theta}}{\partial x}(1, t_i^{b})-\frac{\partial\mathcal{N}_{\theta}}{\partial x}(-1, t_i^{b})\right|^2\right)\\
    \end{aligned}
    \right). 
\end{aligned}
\end{equation*}
$$

由于$\sigma(\cdot)$是严格单调递增函数, 从而$\sigma'(\cdot)>0,$​​ 这就确保了任何关于权重层的更新都只会增加而不会减少. 

## 运行结果

PINN:

![1](D:\pythoncode\pinn\2.png)

$L_2误差为0.0023934857454150915,训练时间为71.6s$

带硬边界的PINN:

![1](D:\pythoncode\pinn\3.png)

$L_2误差为0.00065491913119331,训练时间100.9s$

带有注意力机制的硬边界PINN`test_hard.py`:

![1](figures\pinn_A_H.png)

$L_2误差为0.00015310250455513597,训练时间为135.1s$

## possion(三角函数)

直接运行`possion_hard.py`，程序将会以硬边界求解下述possion问题：
$$
\left\{
\begin{aligned}
-\Delta u &= 2\pi^2\sin\pi x \sin \pi y &(x,y) \ on \ \Omega \\
u &= 0&(x,y)\ on \ \partial \Omega \\
\Omega &= (-2,2) \times (-2,2)
\end{aligned}
\right.
$$
硬边界即:
$$
\hat{u}(x,t)=(x-2)(x+2)(y-2)(y+2)NN(x,y,\theta)
$$

## 运行结果

PINN:

![1](figures\possion_base.png)

$L_2误差为0.5102556943893433,训练时间为385.1s$

硬边界PINN:

![1](figures\possion_hard.png)

$L_2误差为0.00010067245602840558,训练时间为303.8s$

带有注意力机制的硬边界PINN`test_possion_hard.py`:

![1](figures\possion_attention.png)

$L_2误差为4.983487815479748e-05,训练时间为113.2s$

## possion(多项式)

直接运行`possion_hard.py`，程序将会以硬边界求解下述possion问题：
$$
\left\{
\begin{aligned}
-\Delta u &= x^2y^2 &(x,y) \ on \ \Omega \\
u &= 0&(x,y)\ on \ \partial \Omega \\
\Omega &= (-1,1) \times (-1,1)
\end{aligned}
\right.
$$
硬边界即:
$$
\hat{u}(x,t)=(x-1)(x+1)(y-1)(y+1)NN(x,y,\theta)
$$

## 运行结果

硬边界PINN:

![1](figures\possion2_hard.png)

$L_2误差为0.0003638958816223158,训练时间为195.1s$

带有注意力机制的硬边界PINN`test_possion2_hard.py`:

![1](figures\possion2_attention.png)

$L_2误差为0.00047229572998226877,训练时间为424.4s$

## Allen-Cahn

直接运行`AC_hard.py`，程序将会以硬边界求解下述Allen-Cahn问题：
$$
\left\{
\begin{aligned}
&u_t - 0.0001u_{xx}+5u^3-5u=0\\
&u(0,x)=x^2\cos(\pi x)\\
&u(t,-1)=u(t,1)=-1\\
&\Omega \times T=[-1,1]\times[0,1]
\end{aligned}
\right.
$$
硬边界即:
$$
\hat{u}(x,t)=(x-1)(x+1)tNN(x,t,\theta)+x^2\cos(\pi x)
$$

## 运行结果

RAR_PINN:

![1](figures\AC_RAR.png)

$L_2误差为0.02517542242777833,训练时间为1716.2s$

硬边界PINN:

![1](figures\AC_hard.png)

$L_2误差为0.013191858378527654,训练时间为615.0s$

带有注意力机制的硬边界PINN`AC_hard.py`:

![1](figures\AC_Attention.png)

$L_2误差为0.007096630355174227,训练时间为367.9s$

