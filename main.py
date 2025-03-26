import torch
import torch.nn as nn
import numpy as np


# 三维Maxwell专用神经网络
class MaxwellPINN(nn.Module):
    def __init__(self, layers):
        super(MaxwellPINN, self).__init__()
        self.net = nn.Sequential()
        for i in range(len(layers) - 2):
            self.net.append(nn.Linear(layers[i], layers[i + 1]))
            self.net.append(nn.Tanh())
        self.net.append(nn.Linear(layers[-2], layers[-1]))  # 输出6个分量(E,H)

    def forward(self, x):
        return self.net(x)  # 输出形状：[batch, 6]


# 三维Maxwell方程残差计算
class MaxwellResidual:
    def __init__(self, epsilon=1.0, mu=1.0):
        self.epsilon = epsilon
        self.mu = mu

    def compute_flux(self, E, H, n):
        """
        计算通量项（基于间断伽辽金方法）
        E: 电场 [batch, 3]
        H: 磁场 [batch, 3]
        n: 法向量 [batch, 3]
        """
        F_E = torch.cross(H, n, dim=1)
        F_H = -torch.cross(E, n, dim=1)
        return F_E, F_H

    def domain_residual(self, pinn, x, t):
        x.requires_grad = True
        t.requires_grad = True
        inputs = torch.cat([x, t], dim=1)
        EH = pinn(inputs)
        E = EH[:, 0:3]
        H = EH[:, 3:6]

        # 计算空间梯度
        curl_E = self.curl(E, x)
        curl_H = self.curl(H, x)

        # 时间导数
        E_t = torch.autograd.grad(E, t, torch.ones_like(E),
                                  create_graph=True)[0]
        H_t = torch.autograd.grad(H, t, torch.ones_like(H),
                                  create_graph=True)[0]

        # Maxwell方程残差
        res_E = self.epsilon * E_t - curl_H
        res_H = self.mu * H_t + curl_E
        return res_E, res_H

    def curl(self, F, x):
        """
        计算矢量场的旋度
        F: [batch, 3] 矢量场
        x: [batch, 3] 空间坐标
        """
        curl = torch.zeros_like(F)
        for i in range(3):
            grad = torch.autograd.grad(F[:, i], x,
                                       torch.ones_like(F[:, i]),
                                       create_graph=True,
                                       retain_graph=True)[0]
            curl[:, i] = grad[:, (i + 1) % 3] - grad[:, (i - 1) % 3]
        return curl


# 三维数据生成
def generate_3d_data(num=1000):
    # 初始条件
    x = torch.rand(num, 3) * 2 - 1  # 空间坐标[-1,1]^3
    t_ic = torch.zeros(num, 1)
    E_ic = torch.zeros(num, 3)
    H_ic = torch.zeros(num, 3)

    # 边界条件
    x_bc = torch.rand(num // 10, 3) * 2 - 1
    t_bc = torch.rand(num // 10, 1)

    # 界面采样（示例：平面z=0为界面）
    interface_mask = (x[:, 2] > -0.1) & (x[:, 2] < 0.1)
    x_interface = x[interface_mask]
    n_interface = torch.zeros_like(x_interface)
    n_interface[:, 2] = 1.0  # 法向量

    return {
        'ic': (x, t_ic, E_ic, H_ic),
        'bc': (x_bc, t_bc),
        'interface': (x_interface, n_interface)
    }


# 改进的训练函数
def train_dg_pinn():
    # 网络结构 [x,y,z,t] -> [E_x,E_y,E_z,H_x,H_y,H_z]
    pinn = MaxwellPINN([4, 64, 64, 64, 6])
    equation = MaxwellResidual()
    optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-4)

    data = generate_3d_data(5000)

    for epoch in range(10000):
        optimizer.zero_grad()

        # 域内残差
        x_pde = torch.rand(1000, 3) * 2 - 1
        t_pde = torch.rand(1000, 1)
        res_E, res_H = equation.domain_residual(pinn, x_pde, t_pde)
        loss_pde = torch.mean(res_E ** 2 + res_H ** 2)

        # 初始条件
        x_ic, t_ic, E_ic, H_ic = data['ic']
        EH_pred = pinn(torch.cat([x_ic, t_ic], 1))
        loss_ic = torch.mean((EH_pred[:, 0:3] - E_ic) ** 2 +
                             (EH_pred[:, 3:6] - H_ic) ** 2)

        # 界面通量条件
        x_int, n_int = data['interface']
        if len(x_int) > 0:
            t_int = torch.rand(len(x_int), 1)
            EH_int = pinn(torch.cat([x_int, t_int], 1))
            E_int = EH_int[:, 0:3]
            H_int = EH_int[:, 3:6]

            # 计算通量跳变
            F_E, F_H = equation.compute_flux(E_int, H_int, n_int)
            loss_flux = torch.mean(F_E ** 2 + F_H ** 2)
        else:
            loss_flux = 0.0

        total_loss = 1.0 * loss_pde + 1.0 * loss_ic + 0.1 * loss_flux
        total_loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Loss: {total_loss.item():.4e}")


# 主程序
if __name__ == "__main__":
    train_dg_pinn()
