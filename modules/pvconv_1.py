import torch.nn as nn

import modules.functional as F
from modules.voxelization import Voxelization
from modules.shared_mlp import SharedMLP
from modules.se import SE3d

__all__ = ['PVConv']
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

class PVConv_1(nn.Module):
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

    def forward(self, inputs):
        features, coords = inputs
        voxel_features, voxel_coords = self.voxelization(features, coords)#coords 坐标
        voxel_features = self.voxel_layers(voxel_features)
        voxel_features = F.trilinear_devoxelize(voxel_features, voxel_coords, self.resolution, self.training)
        fused_features = voxel_features + self.point_features(features)
        return fused_features, coords
