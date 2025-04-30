import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.fft as fft
import math
from torch.nn import MSELoss
import numpy as np
import time
import torch.optim as optim
import torch.optim.lr_scheduler as LS
from math import floor, ceil
import nibabel as nib
import os
# def generate_dipole(shape, z_prjs=(0, 0, 1), vox=(1, 1, 1), shift=True):
#     if len(shape) == 5:
#         _, _, Nx, Ny, Nz = shape
#         FOVx, FOVy, FOVz = vox * torch.tensor([Nx, Ny, Nz], device=vox.device)
#     else:
#         Nx, Ny, Nz = shape
#         FOVx, FOVy, FOVz = vox * shape
#     x = torch.linspace(-Nx / 2, Nx / 2 - 1, Nx, device=vox.device)
#     y = torch.linspace(-Ny / 2, Ny / 2 - 1, Ny, device=vox.device)
#     z = torch.linspace(-Nz / 2, Nz / 2 - 1, Nz, device=vox.device)
#     kx, ky, kz = torch.meshgrid(x/FOVx, y/FOVy, z/FOVz)
#     D = 1 / 3 - (kx * z_prjs[0] + ky * z_prjs[1] + kz * z_prjs[2]) ** 2 / (kx ** 2 + ky ** 2 + kz ** 2)
#     D[floor(Nx / 2), floor(Ny / 2), floor(Nz / 2)] = 0
#     D = D if len(shape) == 3 else D.unsqueeze(0).unsqueeze(0)
#     return torch.fft.fftshift(D).to(vox.device) if shift else D.to(vox.device)

def generate_dipole_img(shape, z_prjs=(0, 0, 1), vox=(1, 1, 1)):
    # 获取设备信息
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vox = torch.tensor(vox, device=device)

    # all dimensions should be even
    if len(shape) == 5:
        _, _, Nx, Ny, Nz = shape
    else:
        Nx, Ny, Nz = shape

    x = torch.linspace(-Nx / 2, Nx / 2 - 1, Nx, device=device)
    y = torch.linspace(-Ny / 2, Ny / 2 - 1, Ny, device=device)
    z = torch.linspace(-Nz / 2, Nz / 2 - 1, Nz, device=device)

    x, y, z = torch.meshgrid(x, y, z, indexing="ij")

    x = x * vox[0]
    y = y * vox[1]
    z = z * vox[2]

    d = torch.prod(vox) * (3 * (x * z_prjs[0] + y * z_prjs[1] + z * z_prjs[2]) ** 2 - x ** 2 - y ** 2 - z ** 2) \
        / (4 * torch.tensor(math.pi, device=device) * (x ** 2 + y ** 2 + z ** 2) ** 2.5)

    d[torch.isnan(d)] = 0
    d = d if len(shape) == 3 else d.unsqueeze(0).unsqueeze(0)

    return torch.real(fft.fftn(fft.fftshift(d))).to(device)

def forward_field_calc(sus, z_prjs=(0, 0, 1), vox=(1, 1, 1), need_padding=True):
    """
    Calculate frequency offset field from susceptibility distribution.
    """
    device = sus.device
    vox = torch.tensor(vox, device=device)
    Nx, Ny, Nz = sus.shape[-3:]  # 获取最后三个维度的大小
    
    # 检查是否需要填充
    if need_padding:
        pad_size = [Nz // 2, Nz // 2, Ny // 2, Ny // 2, Nx // 2, Nx // 2]
        sus = F.pad(sus, pad_size)

    # 生成 dipole kernel 并确保其尺寸匹配
    D = generate_dipole_img(sus.shape[-3:], z_prjs, vox).to(device)

    # 计算频率偏移场
    field_fft = D * fft.fftn(sus, dim=(-3, -2, -1))
    field = torch.real(fft.ifftn(field_fft, dim=(-3, -2, -1)))

    # 如果需要，去除填充，确保输出形状与原始输入相同
    if need_padding:
        field = field[..., Nx//2: -Nx//2, Ny//2: -Ny//2, Nz//2: -Nz//2]

    return field


class LoTLoss(nn.Module):
    def __init__(self, vox = (1, 1, 1)):
        super(LoTLoss, self).__init__()
        vx, vy, vz = vox
        mid_x = 1 / (vx ** 2) 
        mid_y = 1 / (vy ** 2) 
        mid_z = 1 / (vz ** 2) 
        mid_v = -2 * (1 / (vx ** 2) + 1 / (vy ** 2) + 1 / (vz ** 2) )

        conv_op = [[[0, 0, 0],
                    [0, mid_z, 0],
                    [0, 0, 0]],

                   [[0, mid_x, 0],
                    [mid_y, mid_v, mid_y],
                    [0, mid_x, 0]],

                   [[0, 0, 0],
                    [0, mid_z, 0],
                    [0, 0, 0]], ]
        conv_op = np.array(conv_op)
        conv_op = torch.from_numpy(conv_op)
        conv_op = conv_op.float()
        conv_op = torch.unsqueeze(conv_op, 0)
        conv_op = torch.unsqueeze(conv_op, 0)

        self.kernel = nn.Parameter(conv_op, requires_grad= False)

    def forward(self, phi, lfs, mask, TE, B0, gamma = 267.52):
        
        ## mask: chi mask
        expPhi_r = torch.cos(phi)
        expPhi_i = torch.sin(phi)
        
        a_r = self.LG(expPhi_r, self.kernel)  ## first term. (delta(1j * phi)
        a_i = self.LG(expPhi_i, self.kernel)  

        ## b_r = a_r * expPhi_r + a_i * expPhi_i    ## first term  multiply the second term (exp(-1j * phi) = cos(phi) - j * sin(phi)))
        b_i = a_i * expPhi_r - a_r * expPhi_i

        b_i = b_i * mask
        ## normalization 

        TE = TE[:, None, None, None, None].repeat(1, 1, phi.shape[-3], phi.shape[-2], phi.shape[-1])

        b_i = -1 * b_i / (B0 * TE * gamma)  

        d_i = self.LG(lfs, self.kernel)  

        d_i = d_i * mask 

        return b_i, d_i

    def LG(self, tensor_image, weight):
        out = F.conv3d(tensor_image, weight, bias=None,stride=1,padding=1)  ## 3 * 3 kernel, padding 1 zeros. 

        h, w, d = out.shape[2], out.shape[3], out.shape[4]
        out[:, :, [0, h-1], :,:] = 0
        out[:, :, :, [0, w-1],:] = 0
        out[:, :, :, :, [0, d-1]] = 0
        return out