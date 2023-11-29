import sys

sys.path.append('..')

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

from torchstat import stat
import math
import torch.nn.functional as F
from skimage.measure import label as la
from data_load import Dataset_Test
from torch.utils.data import DataLoader
from torchvision import transforms as tfs
from util_unet_train import train,test
# from util_test import test
from scipy.ndimage import zoom
import SimpleITK as sitk
from datetime import datetime
import os
# import segmentation_models_pytorch as smp
# import seg_transforms
# from Unet2D import UNet2D
from pvcnn import PVCNN
# from .utils.loss import DiceLoss
img_size = 64
img_weight = 128
img_height = 128
# print("**************下面是检验densenet******************")
# test_net = UNet2D((1, 256, 256))
# stat(test_net, (1, 256, 256))
# test_x = Variable(torch.zeros(1, 1, 256, 256))
# test_x = torch.Tensor(test_x)
# test_y = test_net(test_x)
# print('output: {}'.format(test_y.shape))


# 重写collate_fn函数，其输入为一个batch的sample数据
batch_points = 90000



'''
这里要进行改动，即对送进来的血管和肝脏的点，随机抽取30000和20000，但是要确保每次抽取都要抽取以前没有取到的，直到剩余的点不够30000或者20000的话，然后就从整体里随机进行抽取
'''

def collate_fn(batch):
    print("type(batch), len(batch):", type(batch), len(batch))
    xyz_sequence, features_sequence, target_sequence, name = batch[0][0], batch[0][1], batch[0][2], batch[0][3]
    xyz_sequence_v, features_sequence_v, target_sequence_v = batch[0][4], batch[0][5], batch[0][6]
    print("xyz", type(xyz_sequence))
    print(xyz_sequence.shape, xyz_sequence_v.shape, name)

    batch_num = int(xyz_sequence.shape[1] / 10000)
    batch_num_v = int(xyz_sequence_v.shape[1] / 40000)

    print("batch_num, and batch_num_v:", batch_num, batch_num_v)
    xyz_list = []
    features_list = []
    target_list = []
    
    xyz_list_v = []
    features_list_v = []
    target_list_v = []

    for i in range(batch_num):
        start = i * 10000
        end = (i + 1) * 10000
        xyz_list.append(xyz_sequence[:, start:end])
        features_list.append(features_sequence[:, start:end])
        target_list.append(target_sequence[:, start:end])
        
    tem_xyz = xyz_sequence[:, batch_num * 10000:]
    tem_xyz_a = xyz_sequence[:, :10000-(xyz_sequence.shape[1] - (batch_num * 10000))]
    points_xyz = np.concatenate((tem_xyz,tem_xyz_a),axis=1)

    tem_fea = features_sequence[:, batch_num * 10000:]
    tem_fea_a = features_sequence[:, :10000-(xyz_sequence.shape[1] - (batch_num * 10000))]
    points_fea = np.concatenate((tem_fea,tem_fea_a),axis=1)
    
    tem_tar = target_sequence[:, batch_num * 10000:]

    tem_tar_a = target_sequence[:, :10000-(xyz_sequence.shape[1] - (batch_num * 10000))]
    points_tar = np.concatenate((tem_tar,tem_tar_a),axis=1)
    
    xyz_list.append(points_xyz)
    features_list.append(points_fea)
    target_list.append(points_tar)
    
    for i in range(batch_num_v):
        start = i * 40000
        end = (i + 1) * 40000
        xyz_list_v.append(xyz_sequence_v[:, start:end])
        features_list_v.append(features_sequence_v[:, start:end])
        target_list_v.append(target_sequence_v[:, start:end])
        
    tem_xyz = xyz_sequence_v[:, batch_num_v * 40000:]
    tem_xyz_a = xyz_sequence_v[:, :40000-(xyz_sequence_v.shape[1] - (batch_num_v * 40000))]
    points_xyz = np.concatenate((tem_xyz,tem_xyz_a),axis=1)

    tem_fea = features_sequence_v[:, batch_num_v * 40000:]
    tem_fea_a = features_sequence_v[:, :40000-(xyz_sequence_v.shape[1] - (batch_num_v * 40000))]
    points_fea = np.concatenate((tem_fea,tem_fea_a),axis=1)
    
    tem_tar = target_sequence_v[:, batch_num_v * 40000:]

    tem_tar_a = target_sequence_v[:, :40000-(xyz_sequence_v.shape[1] - (batch_num_v * 40000))]
    points_tar = np.concatenate((tem_tar,tem_tar_a),axis=1)
    print("points_xyz.shape, points_fea.shape, points_tar.shape:", points_xyz.shape, points_fea.shape, points_tar.shape)
    xyz_list_v.append(points_xyz)
    features_list_v.append(points_fea)
    target_list_v.append(points_tar)
    xyz_list_all = []
    features_list_all = []
    target_list_all = []

    max_len = max(len(xyz_list), len(xyz_list_v))
    print("len(xyz_list), len(xyz_list_v):", len(xyz_list), len(xyz_list_v))
    if len(xyz_list) <= len(xyz_list_v):
        for i in range(len(xyz_list_v)):
            xyz_list_all.append(np.concatenate((xyz_list_v[i], xyz_list[i%len(xyz_list)]),axis=1))
            features_list_all.append(np.concatenate((features_list_v[i], features_list[i%len(xyz_list)]),axis=1))
            target_list_all.append(np.concatenate((target_list_v[i], target_list[i%len(xyz_list)]),axis=1))
    else:
        for i in range(len(xyz_list)):
            xyz_list_all.append(np.concatenate((xyz_list_v[i%len(xyz_list_v)], xyz_list[i]),axis=1))
            features_list_all.append(np.concatenate((features_list_v[i%len(xyz_list_v)], features_list[i]),axis=1))
            target_list_all.append(np.concatenate((target_list_v[i%len(xyz_list_v)], target_list[i]),axis=1))

    print("len(xyz_list_all), len(features_list_all), len(target_list_all):", len(xyz_list_all), len(features_list_all), len(target_list_all))
    
    name = name.split(".nii.gz")[0]
      
    # points = np.concatenate((v_p,l_p),axis=0)
    

    return xyz_list_all, features_list_all, target_list_all, name  # 这里将血管和肝脏的混到了一起，提取出50000个点，作为xyz_list的值 
from PIL import Image


# image, label, mask



def data_test_tf(img, label,liver, name, size,direction,spacing):
    factor = (img_size/label.shape[0], img_weight / label.shape[1], img_height / label.shape[2])
    label_zoom = zoom(label, factor, order=0)

    liver_zoom = zoom(liver, factor, order=0)
    # factor = (img_size/label.shape[0], 1, img_weight / img.shape[2], img_height / img.shape[3])
    img_zoom = zoom(img, factor, order=2)
    # vessel_zoom = zoom(vessel, factor, order=0)

    return img_zoom, label_zoom,liver_zoom,name, size,direction,spacing


test_dataset = Dataset_Test('./new_testpoints',transform=None)
# test_dataset = Dataset_Test('./Dataset/Point_Data/allpoints/test',transform=None)
# test_dataset = custom_dataset('./datasets/',transform=data_test_tf)  # 读入 .pkl 文件
test_data = DataLoader(test_dataset, 1, shuffle=False,collate_fn=collate_fn)  # batch size 设置为 8


net = torch.load("./model/best/fine_model_co_enhance.pth")
# net = torch.load("./model/epoch/0.7591_900_fine_model_co_enhance.pth")
# print("./model/save_best_model/fine_model_co_enhance.pth")
# criterion = DiceLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
print("---- 开始训练：")
# train(net, train_data, test_data, 1000, optimizer, criterion)

test(net,test_data)
# net1 = torch.load('./save/bNet_e_g_1.pth')
# net2 = torch.load('./save/bNet_e_g_2.pth')
# optimizer1 = torch.optim.SGD(net1.parameters(), lr=0.01)
# optimizer2 = torch.optim.SGD(net2.parameters(), lr=0.01)
# train_newlabel(net1, train_data_100_1,net2, test_data, 3000, optimizer1,optimizer2, criterion)