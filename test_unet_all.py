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
from data_load_all import Dataset_Test
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
batch_points = 50000

def collate_fn(batch):
    # print("type(batch), len(batch):", type(batch), len(batch))
    xyz_sequence, features_sequence, target_sequence, xyz_v, fea_v, tar_v, name = batch[0][0], batch[0][1], batch[0][2], batch[0][3], batch[0][4], batch[0][5], batch[0][6]
    yw_l, yw_v = batch[0][7], batch[0][8]
    print("name", type(xyz_sequence), name, xyz_sequence.shape, features_sequence.shape, xyz_v.shape, fea_v.shape)

    print("yw_l.shape, yw_v.shape:", yw_l.shape, yw_v.shape)
    # print(xyz_sequence.shape)

    batch_num = int(xyz_sequence.shape[1] / batch_points)     # 这句话要改一改,新的batch_num应该为最大的liver或者vessel,即两者较大的那个 
    batch_v, batch_liver = int(xyz_v.shape[1] / 40000), int(xyz_sequence.shape[1] / 10000)
    print("batch_v, batch_liver:", batch_v, batch_liver)
    xyz_list = []
    features_list = []
    target_list = []
    number_list = []     # 这里面应该是一个2个元素的列表组成的列表,表示每个batch内部多少点是需要纳入的 
    # name_list = []
    batch_max = max(batch_v, batch_liver)
    xyz_yw_list = []
    num_liver, num_v = 0, 0
    # xyz_liver, xyz_v = [], []
    # fea_liver, fea_v = [], []
    # tar_liver, tar_v = [], []
    for i in range(batch_max+1):
        if i<batch_v:
            start_v = i * 40000
            end_v = (i + 1) * 40000
            n_xyz_v = xyz_v[:, start_v:end_v]
            n_fea_v = fea_v[:, start_v:end_v]
            n_tar_v = tar_v[:, start_v:end_v]
            num_v = 40000
            
            yw_xyz_v = yw_v[:, start_v:end_v]
            # 传入所有,否则传入一个固定的点列表.但是注意,如果i==batch_v时传入需要纳入的点的数量,如果i>batch_v时,需要纳入的点的数量将为0 
        elif i == batch_v:
            n_xyz_v = xyz_v[:, batch_v * 40000:]
            n_fea_v = fea_v[:, batch_v * 40000:]
            n_tar_v = tar_v[:, batch_v * 40000:]
            num_v = xyz_v.shape[1]-batch_v*40000
            
            yw_xyz_v = yw_v[:, batch_v * 40000:]
        else:
            n_xyz_v = xyz_v[:, 0:40000]
            n_fea_v = fea_v[:, 0:40000]
            n_tar_v = tar_v[:, 0:40000]
            num_v = 0
            
            yw_xyz_v = yw_v[:, 0:40000]

        if i<batch_liver:
            start_l = i * 10000
            end_l = (i + 1) * 10000
            n_xyz_l = xyz_sequence[:, start_l:end_l]
            n_fea_l = features_sequence[:, start_l:end_l]
            n_tar_l = target_sequence[:, start_l:end_l]
            num_liver = 10000
            
            yw_xyz_l = yw_l[:, start_l:end_l]
        elif i == batch_liver:
            n_xyz_l = xyz_sequence[:, batch_liver * 10000:]
            n_fea_l = features_sequence[:, batch_liver * 10000:]
            n_tar_l = target_sequence[:, batch_liver * 10000:]
            num_liver = xyz_sequence.shape[1]-batch_liver*10000
            
            yw_xyz_l = yw_l[:, batch_liver * 10000:]
        else:
            n_xyz_l = xyz_sequence[:, 0:10000]
            n_fea_l = features_sequence[:, 0:10000]
            n_tar_l = target_sequence[:, 0:10000]
            num_liver = 0
            
            yw_xyz_l = yw_l[:, 0:10000]

        xyz_vl = np.concatenate((n_xyz_v, n_xyz_l),axis=1)
        fea_vl = np.concatenate((n_fea_v, n_fea_l),axis=1)
        tar_vl = np.concatenate((n_tar_v, n_tar_l),axis=1)
        
        xyz_yw = np.concatenate((yw_xyz_v, yw_xyz_l),axis=1)
        # print("i, xyz_vl.shape, fea_vl.shape, tar_vl.shape:", i, xyz_vl.shape, fea_vl.shape, tar_vl.shape, num_v, num_liver)
        xyz_list.append(xyz_vl)
        features_list.append(fea_vl)
        target_list.append(tar_vl)
        number_list.append([num_v, num_liver])
        
        xyz_yw_list.append(xyz_yw)
    # xyz_list.append(xyz_sequence[:, batch_num * batch_points:])
    # features_list.append(features_sequence[:, batch_num * batch_points:])
    # target_list.append(target_sequence[:, batch_num * batch_points:])
    name = name.split(".nii.gz")[0]
    return xyz_list, features_list, target_list, number_list, name, xyz_yw_list  # maybe map change to entorpy

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



# test_dataset = Dataset_Test('./Dataset/Point_Data/allpoints/test',transform=None)
test_dataset = Dataset_Test('./new_testpoints/AllPoints',transform=None)
test_data = DataLoader(test_dataset, 1, shuffle=False,collate_fn=collate_fn)


net = torch.load("./model/best/fine_model_co_enhance.pth")
# net = torch.load("./model/best/0706.pth")
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