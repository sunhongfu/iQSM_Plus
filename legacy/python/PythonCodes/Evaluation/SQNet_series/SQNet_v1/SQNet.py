import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchinfo import summary

"""
Unet 编码器:使用3D卷积编码器提取 QSM,R2',Lfs 的多尺度特征。
QSM 编码器：将 QSM 图像编码为两个引导向量，分别为 Chi-pos 和 Chi-neg 的分离提供引导。
级联交互组：每组包含 SRU 和LeFF归一化,用于逐步分离 Chi-pos 和 Chi-neg 特征。
图像恢复层:两个独立的3D解码器将分离的特征映射到三维图像空间,生成 Chi-pos 和 Chi-neg。
"""

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# Unet 图像编码器 输入QSM ， R2‘ , Lfs
class ImageEncoder3D(nn.Module):
    def __init__(self):
        super(ImageEncoder3D, self).__init__()

        self.enc1 = ConvBlock(3, 64)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool3d(2)

    def forward(self, x):
        feature_maps = []
        
        enc1 = self.enc1(x)
        feature_maps.append(enc1)
        
        enc2 = self.enc2(self.pool1(enc1))
        feature_maps.append(enc2)

        enc3 = self.enc3(self.pool2(enc2))
        feature_maps.append(enc3)
        
        fvin = self.pool3(enc3)

        return fvin, feature_maps

# QSM 编码器 - 生成两个引导向量
class QSMEncoder3D(nn.Module):
    def __init__(self):
        super(QSMEncoder3D, self).__init__()
        
        # 用于生成Chi-pos的引导向量
        self.fc_pos = nn.Sequential(
            ConvBlock(1, 64),
            nn.MaxPool3d(2),
            ConvBlock(64, 128),
            nn.MaxPool3d(2),
            ConvBlock(128, 256),
            nn.MaxPool3d(2)
        )

        # 用于生成Chi-neg的引导向量
        self.fc_neg = nn.Sequential(
            ConvBlock(1, 64),
            nn.MaxPool3d(2),
            ConvBlock(64, 128),
            nn.MaxPool3d(2),
            ConvBlock(128, 256),
            nn.MaxPool3d(2)
        )

    def forward(self, x):
        
        # 生成引导向量
        guide_vector_pos = self.fc_pos(x) 
        guide_vector_neg = self.fc_neg(x) 
        
        return guide_vector_pos, guide_vector_neg

class ConvSRU(nn.Module):
    def __init__(self,channels=256):
        super(ConvSRU, self).__init__()

        self.update_gate = ConvBlock(channels, channels)
        self.out_gate = ConvBlock(channels, channels)
        # self.bottleneck = ConvBlock(256, 512)

    def forward(self, fvin, guide_vector):
        update = torch.sigmoid(self.update_gate(guide_vector))

        out_inputs = torch.tanh(self.out_gate(guide_vector))
        h_new = fvin * (1 - update) + out_inputs * update
        # h_new = self.bottleneck(h_new)
        return h_new

class InteractionGroup3D(nn.Module):
    def __init__(self, num_leff_blocks=2):
        super(InteractionGroup3D, self).__init__()
        
        # 自适应全局交互模块 SRU
        self.SRU = ConvSRU()
        
        # LeFF 层用于特征细化
        self.leff_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(256, 256, kernel_size=1),
                nn.ReLU(),
                nn.Conv3d(256, 256, kernel_size=3, padding=1)
            ) for _ in range(num_leff_blocks)
        ])

    def forward(self, fvin, guide_vector):
        # SRU 模块生成交互特征
        f_interaction = self.SRU(fvin, guide_vector)
        
        # LeFF 块进行特征细化
        for leff in self.leff_blocks:
            f_interaction = leff(f_interaction) + f_interaction
        
        return f_interaction

#对于Chi-pos和Chi-neg各自的特征的形成
class CascadeInteractionModule3D(nn.Module):
    def __init__(self, num_groups=2, num_leff_blocks=2):
        super(CascadeInteractionModule3D, self).__init__()
        
        # 前 N 个级联交互组用于生成 Chi-pos
        self.interaction_groups_pos = nn.ModuleList([InteractionGroup3D(num_leff_blocks) for _ in range(num_groups)])
        
        # 后 N 个级联交互组用于生成 Chi-neg
        self.interaction_groups_neg = nn.ModuleList([InteractionGroup3D(num_leff_blocks) for _ in range(num_groups)])
        
        # 1x1 卷积用于通道数的调整
        self.concat_conv = nn.Conv3d(512, 256, kernel_size=1)
        

    def forward(self, fvin, guide_vector_pos, guide_vector_neg):
        # 前 N 个级联交互组逐步生成 F_chi_pos
        f_chi_pos = fvin
        for group in self.interaction_groups_pos:
            f_chi_pos = group(f_chi_pos, guide_vector_pos)
        
        # 将 FVin 和 F_chi_pos 拼接后通过 1x1 卷积进行通道数调整
        f_concat = torch.cat([fvin, f_chi_pos], dim=1)
        f_concat = self.concat_conv(f_concat)
        
        # 后 N 个级联交互组逐步生成 F_chi_neg
        f_chi_neg = f_concat
        for group in self.interaction_groups_neg:
            f_chi_neg = group(f_chi_neg, guide_vector_neg)

        return f_chi_pos, f_chi_neg

# 图像恢复层
class ImageDecoder3D(nn.Module):
    def __init__(self):
        super(ImageDecoder3D, self).__init__()
        
        # Decoder
        self.upconv3 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(512, 256)
        self.upconv2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256, 128)
        self.upconv1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64)
        # Output layer
        self.conv_last = nn.Conv3d(64, 1, kernel_size=1)

    def forward(self, x, feature_maps):
        # 逐步上采样
        dec3 = torch.cat([self.upconv3(x), feature_maps[2]], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = torch.cat([self.upconv2(dec3), feature_maps[1]], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = torch.cat([self.upconv1(dec2), feature_maps[0]], dim=1)
        dec1 = self.dec1(dec1)
        
        out = self.conv_last(dec1)
        return out
    
# 完整的分离网络架构
class SQNet(nn.Module):
    def __init__(self):
        super(SQNet, self).__init__()
        
        # 图像编码器：用于提取多尺度特征金字塔
        self.Unet_encoder = ImageEncoder3D()
        
        # R2' 编码器，用于生成引导向量
        self.QSM_encoder = QSMEncoder3D()
        
        self.CascadeInteractionModule3D = CascadeInteractionModule3D()

        self.bottleneck_pos = ConvBlock(256, 512)

        self.bottleneck_neg = ConvBlock(256, 512)

        # 图像恢复层，用于将特征转换回图像域
        self.decoder_pos = ImageDecoder3D()
        self.decoder_neg = ImageDecoder3D()

    def forward(self, QSM_image, R2_prime_image, Lfs_image):
        # 编码 QSM 图像，提取多尺度特征金字塔
        image_inputs = torch.cat([QSM_image, R2_prime_image, Lfs_image], dim=1)
        fvin, feature_maps = self.Unet_encoder(image_inputs)

        # 编码 R2_flirt_image 图像生成引导向量
        guide_vector_pos, guide_vector_neg = self.QSM_encoder(QSM_image)

        f_chi_pos, f_chi_neg = self.CascadeInteractionModule3D(fvin, guide_vector_pos, guide_vector_neg)

        pre_chi_pos = self.bottleneck_pos(f_chi_pos)
        pre_chi_neg = self.bottleneck_neg(f_chi_neg)

        # 图像恢复层，将分离特征转换回图像域
        chi_pos = self.decoder_pos(pre_chi_pos, feature_maps)
        chi_neg = self.decoder_neg(pre_chi_neg, feature_maps)
        # print(chi_pos.max())
        return chi_pos, chi_neg, f_chi_pos, f_chi_neg, guide_vector_pos, guide_vector_neg

if __name__ == "__main__":
    # 假设输入是 64x64x64 的 R2_Flirt 和 QSM 图像
    QSM_image = torch.randn(1, 1, 256, 256, 256)  # batch_size=1, channels=1, depth=64, height=64, width=64
    R2_prime_image = torch.randn(1, 1, 256, 256, 256)
    Lfs_image = torch.randn(1, 1, 256, 256, 256)
    
    # 初始化网络
    model = SQNet()
    
    # 前向传播，得到 Chi-pos 和 Chi-neg
    chi_pos, chi_neg, f_chi_pos, f_chi_neg, guide_vector_pos, guide_vector_neg = model(QSM_image, R2_prime_image, Lfs_image)
    
    # 打印输出的尺寸
    print(f"Chi-pos 输出尺寸: {chi_pos.size()}")  # 预期尺寸: (16, 1, 258, 256, 256) 
    print(f"Chi-neg 输出尺寸: {chi_neg.size()}")  # 预期尺寸: (16, 1, 258, 256, 256) 

    print(f"f_chi_pos 输出尺寸: {f_chi_pos.size()}")  # 预期尺寸: (16, 1, 258, 256, 256) 
    print(f"guide_vector_pos 输出尺寸: {guide_vector_pos.size()}")  # 预期尺寸: (16, 1, 258, 256, 256) 

    # 获取模型参数量和详细信息
    model_summary = summary(model, input_data=(QSM_image, R2_prime_image, Lfs_image))

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"总参数量: {total_params}")
    print(f"可训练参数量: {trainable_params}")
