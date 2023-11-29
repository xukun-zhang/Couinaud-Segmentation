import torch
import torch.nn as nn

from models.utils import create_pointnet_components, create_mlp_components

__all__ = ['PVCNN']


class PVCNN(nn.Module):
    blocks = ((64, 1, 64), (128, 1,32),(256,1,16), (512, 1, None), (2048, 1, None))

    def __init__(self, num_classes, num_shapes, extra_feature_channels=3,
                 width_multiplier=1, voxel_resolution_multiplier=1):
        super().__init__()
        assert extra_feature_channels >= 0
        # self.in_channels = extra_feature_channels + 3
        self.in_channels = extra_feature_channels + 1
        self.num_shapes = num_shapes

        layers, channels_point, concat_channels_point = create_pointnet_components(
            blocks=self.blocks, in_channels=self.in_channels, with_se=True, normalize=False,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.point_features = nn.ModuleList(layers)

        layers, _ = create_mlp_components(in_channels=(num_shapes + channels_point + concat_channels_point),
                                          out_channels=[256, 0.2, 256, 0.2, 128, num_classes],
                                          classifier=True, dim=2, width_multiplier=width_multiplier)
        self.classifier = nn.Sequential(*layers)
        self.fea_linear_a = nn.Conv1d(5057, 1024, 1)
        self.fea_linear_b = nn.Conv1d(1024, 128, 1)
        
    def forward(self, inputs, target):
    # def forward(self, input,coords):
        # inputs : [B, in_channels + S, N]
        features = inputs[:, :self.in_channels, :]
        # print("inputs.shape:", inputs.shape)
        # print("target.max():", target.max(), target.min())
        one_hot_vectors = inputs[:, -self.num_shapes:, :]
        # print("one_hot_vectors.max(), one_hot_vectors.min():", one_hot_vectors.max(), one_hot_vectors.min())
        num_points = inputs.size(-1)
        # features = torch.cat((coords, input), dim=1)
        coords = features[:, :3, :]
        out_features_list = [one_hot_vectors]
        # out_features_list = []
        
        output_voxel_pre = []     # 用于保存下方前面3个列表，用于返回计算列表里3个元组对应的loss 
        for i in range(len(self.point_features)):
            # print("len(self.point_features):", len(self.point_features))

            if i < 3:
                features, _, output_aux = self.point_features[i]((features, coords), target)
                # print("output_aux:", type(output_aux), output_aux[0].shape, output_aux[1].shape)
            else:
                features, _ = self.point_features[i]((features, coords), target)
            # print("_.shape:", _.shape)

            out_features_list.append(features)
            output_voxel_pre.append(output_aux)

        out_features_list.append(features.max(dim=-1, keepdim=True).values.repeat([1, 1, num_points]))
        # print("self.classifier():", self.classifier)
        input_x = torch.cat(out_features_list, dim=1)
        # linear_out = self.fea_linear_a(input_x)
        # linear_out = self.fea_linear_b(linear_out)
        # print("input_x.shape:", input_x.shape, linear_out.shape)
        for i in range(len(self.classifier)):
            input_x = self.classifier[i](input_x)
            # print("i, input_x.shape:", i, input_x.shape)
            if i == 4:
                linear_out = input_x
        # print("linear_out.shape:", linear_out.shape, input_x.shape)
        return input_x, linear_out, output_voxel_pre
