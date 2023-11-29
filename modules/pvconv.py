import torch.nn as nn
import torch
import modules.functional as F
from modules.voxelization import Voxelization
from modules.shared_mlp import SharedMLP
from modules.se import SE3d
import math
__all__ = ['PVConv']

class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)  #
        
        # bn = nn.BatchNorm1d(in_dim // 4)

    def forward(self, x_0, x_1):     # voxel, point

        batch_0, C_0, width_0 = x_0.size()
        batch_1, C_1, width_1 = x_1.size()
        # print("x_0.shape, x_1.shape:", x_0.shape, x_1.shape)
        proj_query = self.query_conv(x_0) 
        proj_key = self.key_conv(x_1).permute(0, 2, 1) 
        # print("proj_key.shape, proj_query.shape:", proj_key.shape, proj_query.shape)
        energy = torch.matmul(proj_key, proj_query)/math.sqrt(C_0)
        print("energy.shape:", energy.shape, math.sqrt(C_0))
        attention = self.softmax(energy)  # the shape are K_number * N_number
        # proj_value = self.value_conv(x_0)
        # print("proj_value.shape:", proj_value.shape)
        out = torch.matmul(attention, x_0.permute(0, 2, 1)).permute(0, 2, 1)
        # print("torch.matmul(attention, proj_value.permute(0, 2, 1)).shape:", torch.matmul(attention, proj_value.permute(0, 2, 1)).shape)
        out = self.gamma * out + x_1     # (K_number * 512 * 8 * 8) attention + x_1 (error cases)
        return out, attention


    
    
class Attn(nn.Module):
    def __init__(self, in_dim):
        super(Attn, self).__init__()

        self.chanel_in = in_dim
        self.conv_a = nn.Conv1d(in_channels=in_dim, out_channels=in_dim//4, kernel_size=1)
        self.conv_b = nn.Conv1d(in_channels=in_dim, out_channels=in_dim//4, kernel_size=1)
        self.conv = nn.Conv1d(in_channels=in_dim//2, out_channels=1, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm1d(in_dim)
        self.relu = nn.ReLU(True)
        
        # bn = nn.BatchNorm1d(in_dim // 4)
    def forward(self, x_0, x_1):     # voxel, point
        batch_0, C_0, width_0 = x_0.size()
        batch_1, C_1, width_1 = x_1.size()

        x_a = self.conv_a(x_0)
        x_b = self.conv_b(x_1)
        x_all = torch.cat([x_a, x_b], dim=1)
        # print("x_0.shape, x_1.shape:", x_0.shape, x_1.shape, x_all.shape)
        proj_query = self.conv(x_all) 
        # print("proj_query.shape:", proj_query.shape)
        attention = 1 + self.sigmoid(proj_query)  # the shape are K_number * N_number
        out = torch.mul(x_0, attention)
        # print("out.shape:", out.shape)
        # out = self.gamma * out + x_1     # (K_number * 512 * 8 * 8) attention + x_1 (error cases)
        out = out + x_1
        # out = self.bn(out)
        # out = self.relu(out)
        # return out, attention
        return out, attention

class BasicConvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(inc,outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride,padding=1),
            nn.BatchNorm3d(outc),
            nn.ReLU(True),
        )

    def forward(self, x):
        out = self.net(x)
        # print("out:",out.shape)
        return out

class ResidualBlock(nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride,padding=1),
            nn.BatchNorm3d(outc),
            nn.ReLU(True),
            nn.Conv3d(outc, outc, kernel_size=ks, dilation=dilation,
                        stride=1,padding=1),
            nn.BatchNorm3d(outc),
        )

        if inc == outc and stride == 1:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(
                nn.Conv3d(inc, outc, kernel_size=1, dilation=1,
                            stride=stride),
                nn.BatchNorm3d(outc),
            )

        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out


class PVConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, resolution, with_se=False, normalize=True, eps=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.resolution = resolution
        self.voxelization = Voxelization(resolution, normalize=normalize, eps=eps)
        voxel_layers = [
            BasicConvolutionBlock(in_channels,in_channels,kernel_size,stride=1),
            ResidualBlock(in_channels,out_channels,kernel_size,stride=1),
            ResidualBlock(out_channels,out_channels,kernel_size,stride=1)
         ]
        if with_se:
            voxel_layers.append(SE3d(out_channels))
        self.voxel_layers = nn.Sequential(*voxel_layers)
        self.point_features = SharedMLP(in_channels, out_channels)
        self.atten_64 = Attn(64)
        self.atten_128 = Attn(128)

        self.atten_256 = Attn(256)
        self.output_64 = nn.Conv3d(64, 1, 1)
        self.output_128 = nn.Conv3d(128, 1, 1)
        self.output_256 = nn.Conv3d(256, 1, 1)
    def forward(self, inputs, targets=None):
        features, coords = inputs
        # print("features.shape:", features.shape)
        voxel_features, voxel_coords = self.voxelization(features, coords)#coords 坐标
        # print("voxel_coords.shape:", voxel_coords.shape)

        voxel_features = self.voxel_layers(voxel_features)
        # print("voxel_features.shape:", voxel_features.shape)
        

        """这里需要将voxel_features经过一个输出卷积，用来输出预测的类别，然后将预测的类别与label一起返回，返回后进行loss监督"""
        
        voxel_targets, target_coords = self.voxelization(targets.float(), coords)#coords 坐标
#         print("targets.shape, targets.max(), targets.min():", targets.shape, targets.max(), targets.min())
#         print("voxel_targets.max(), voxel_targets.min():", voxel_targets.shape, voxel_targets.max(), voxel_targets.min())

#         print("voxel_targets[voxel_targets==8].sum():", voxel_targets[voxel_targets==8].sum())
#         print("voxel_targets[voxel_targets==7].sum():", voxel_targets[voxel_targets==7].sum())
#         print("voxel_targets[voxel_targets==6].sum():", voxel_targets[voxel_targets==6].sum())
#         print("voxel_targets[voxel_targets==5].sum():", voxel_targets[voxel_targets==5].sum())
#         print("voxel_targets[voxel_targets==4].sum():", voxel_targets[voxel_targets==4].sum())
#         print("voxel_targets[voxel_targets==3].sum():", voxel_targets[voxel_targets==3].sum())
#         print("voxel_targets[voxel_targets==2].sum():", voxel_targets[voxel_targets==2].sum())
#         print("voxel_targets[voxel_targets==1].sum():", voxel_targets[voxel_targets==1].sum())
#         print("voxel_targets[voxel_targets==0].sum():", voxel_targets[voxel_targets==0].sum())
        if voxel_features.shape[1] == 64:
            voxel_outputs = self.output_64(voxel_features)
        if voxel_features.shape[1] == 128:
            voxel_outputs = self.output_128(voxel_features)
        if voxel_features.shape[1] == 256:
            voxel_outputs = self.output_256(voxel_features)
        voxel_features_p = F.trilinear_devoxelize(voxel_features, voxel_coords, self.resolution, self.training)
        if voxel_features_p.shape[1] == 64:
            fused_features, attention = self.atten_64(voxel_features_p, self.point_features(features))
        if voxel_features_p.shape[1] == 128:
            fused_features, attention = self.atten_128(voxel_features_p, self.point_features(features))
        if voxel_features_p.shape[1] == 256:
            fused_features, attention = self.atten_256(voxel_features_p, self.point_features(features))
        # fused_features = voxel_features + self.point_features(features)
        # print("---voxel_features.shape, self.point_features(features).shape, features.shape:", voxel_features.shape, self.point_features(features).shape, features.shape)
        # print("---fused_features.shape:", fused_features.shape)
        return fused_features, coords, (voxel_outputs, voxel_targets)
