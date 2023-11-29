import sys

sys.path.append('..')

import numpy as np
import os
from datetime import datetime
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import sys
import pickle
import pandas as pd
# import nibabel as nib
import SimpleITK as sitk
from tqdm import tqdm
import math

from skimage.measure import label as la
from PIL import Image
from scipy.ndimage import zoom

from PIL import ImageEnhance

img_depth, img_weight, img_height = 64, 128, 128
# def fun_Contrast(image, coefficient):
#     # 对比度，增强因子为1.0是原始图片; 对比度增强 1.5; 对比度减弱 0.8
#     enh_con = ImageEnhance.Contrast(image)
#     image_contrasted1 = enh_con.enhance(coefficient)
#     return image_contrasted1
#
#
# def fun_Sharpness(image, coefficient):
#     # 锐度，增强因子为1.0是原始图片; 锐度增强 3; 锐度减弱 0.8
#     enh_sha = ImageEnhance.Sharpness(image)
#     image_sharped1 = enh_sha.enhance(coefficient)
#     return image_sharped1
#
#
# def fun_bright(image, coefficient):
#     # 变亮 1.5; 变暗 0.8; 亮度增强,增强因子为0.0将产生黑色图像； 为1.0将保持原始图像。
#     enh_bri = ImageEnhance.Brightness(image)
#     image_brightened1 = enh_bri.enhance(coefficient)
#     return image_brightened1



num_seg = 8
points = 20000

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class CrossEntropyLoss(nn.Module):

    def forward(self, uout, target):
        # cel = nn.CrossEntropyLoss()
        # # return cel(uout.transpose(1,2).float(),target[:,0,:].long())/(points*uout.shape[0])
        # C = 0
        # for i in range(uout.shape[0]):
        #     y_p = uout[i]
        #     y_p = torch.transpose(y_p,dim0=1,dim1=0)
        #     y_true = target[i,0]
        #     print("y_pred:",y_p.shape)
        #     print("y_true:",y_true.shape)
        #     C += cel(y_p.float(),y_true.long())

        # return torch.tensor(C/uout.shape[0], requires_grad=True)
        # # for i in range(uout.shape[0]):
        # #
        # # C = 0
        # # for i in range(len(uout)):
        # #     print(i)
        # #     C += cel(uout[i],target[i])
        # #

        y_pred = uout
        organ_target = torch.zeros((target.size(0),num_seg, target.size(2)))
        # print("y_pred", y_pred.shape)
        for organ_index in range(num_seg):
            temp_target = torch.zeros(target.shape)
            temp_target[target == organ_index + 1] = 1
            temp_target = temp_target.squeeze(1)
            # print(temp_target.shape)
            organ_target[:,organ_index, :] = temp_target
        # y_true = organ_target.transpose((1,0))
        organ_target = organ_target.cuda()
        dice = 0.0
        for organ_index in range(num_seg):

            iflat = (y_pred[:,organ_index, :].contiguous().view(-1)).float()
            tflat = organ_target[:,organ_index, :].contiguous().view(-1)
            intersection = (iflat * tflat).sum()
            dice += 2. * intersection / ((iflat ** 2).sum() + (tflat ** 2).sum())

        dice_loss = 1 - dice / (num_seg)
        # loss = 0.5*dice_loss + 0.5*(C/uout.shape[0])

        return dice_loss
        # y_pred = y_pred.transpose(1,0)
        # organ_target = organ_target.transpose(1,0)
        # print(y_pred.shape)
        # C = 0



class dice_loss(nn.Module):
    # def forward(self, uout, label, label_1, label_2):

    def forward(self, uout, label):
        # def forward(self, uout, uout_1, label, label_1):
        """soft dice loss"""
        eps = 1e-7

        L2_sum = 0
        L2_mean = 0
        all_sort = []

        iflat = uout.contiguous().view(-1)

        tflat = label.view(-1)

        intersection = (iflat * tflat).sum()
        dice_loss = 1 - 2. * intersection / ((iflat ** 2).sum() + (tflat ** 2).sum() + eps)

        return dice_loss

criterion_cld = nn.CrossEntropyLoss().cuda()     # 会自动加上softmax 
# torch.manual_seed(2023)
# torch.cuda.manual_seed(2023)

class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=8, feat_dim=128, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
    def forward(self, x, labels, loss1):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        # print("x.shape, labels.shape:", x.shape, labels.shape)
        # print("self.centers.shape:", self.centers.shape)
        ce_loss_1 = loss1.view(-1)
        ind_1_sorted = np.argsort(ce_loss_1.cpu().data).cuda()  # 从小到大排列，然后输出下标列表
        ce_loss_1_sorted = ce_loss_1[ind_1_sorted]
        # print("ce_loss_1_sorted:", ce_loss_1_sorted)     # 此时的loss为从小到大的排列顺序 
        num_remember = int(0.9 * len(ce_loss_1_sorted))

        ind_1_update = ind_1_sorted[num_remember:]
        
        # print("hard_x.shape, hard_labels.shape, ce_loss_1-ind_1_update-:", hard_x.shape, hard_labels.shape, ce_loss_1[ind_1_update])

        batch_size = x.size(0)
        # print(torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes).shape)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        """distmat = 1 * distmat - 2 * (x @ self.centers.t())"""
        distmat.addmm_(1, -2, x, self.centers.t())     # 加速计算欧氏距离，因为distmat已经是两个向量的平方和了，但中心是学习出来的
        # 注意上面的中心值是学习出来的，不是计算出的特征中心（为啥不可以计算出特征中心呢？浪费时间效率麽？）
        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        # print("batch_size, self.centers.shape, x.shape, cld_lab.shape:", batch_size, self.centers.shape, x.shape, labels[:,0].shape, labels[:8,0])

        # 按理说这里应该得到血管那些点和label，先不用计算血管来看看 
        hard_x = x[ind_1_update] 
        hard_labels = labels[ind_1_update]
        affnity = torch.mm(hard_x, self.centers.t())     # 其实也可以换为中心点自身的乘积，表示自身的相似性，使得中心点feature互相具有差异 
        CLD_loss = criterion_cld(affnity.div_(1), hard_labels[:,0]-1)

        # print("affnity, classes:", affnity, classes)
        # CLD_loss = criterion_cld(affnity.div_(1), classes)
        # print("CLD_loss:", CLD_loss)
        return loss, CLD_loss
        """其实上面可以通过CEloss先得到loss较大的那些点，然后仅对那些点进行距离上的拉近以及鉴别，使得模型更好训练！！！"""

class LossFunction(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, linear_out, voxel_output_list):
        """
        :param pred: (B, 14, 128, 256, 256)
        :param target: (B, 128, 256, 256)
        :return: Dice距离
        """
        cel = nn.CrossEntropyLoss(reduction='none')
        loss1 = cel(pred, torch.squeeze(target - 1, dim=1))
        
        msl = nn.MSELoss()
        ceter_loss = CenterLoss()
#         for i in range(len(voxel_output_list)):
#             lab = voxel_output_list[i][1].view(-1)
#             mask = voxel_output_list[i][0].view(-1)
#             # print("mask.shape, lab.shape:", mask.shape, lab.shape)
#             lab_loss = lab[lab>=1]
#             mask_loss = mask[lab>=1]

#             mean_loss = msl(mask_loss, lab_loss)
#             # print("mean_loss:", mean_loss)
#             loss1 = loss1+0.01*mean_loss

        # print("target.shape, linear_out.shape:", target.shape, linear_out.shape, pred.shape)

        tar = target.squeeze(0).squeeze(0)
        lin = linear_out.squeeze(0).permute(1, 0)
        # print("loss1.shape, ")
        # print("loss1.shape, tar.shape, pred.shape, lin.shape:", loss1.shape, tar.shape, pred.shape, lin.shape)
        loss_center, cld_loss = ceter_loss(lin, tar, loss1)
        # print("loss_center:", loss_center)
        
#         lab_64 = torch.zeros(voxel_output_list[0][1].shape)
#         lab_128 = torch.zeros(voxel_output_list[1][1].shape)
#         lab_256 = torch.zeros(voxel_output_list[2][1].shape)
#         print("lab_64.shape, lab_128.shape, lab_256.shape:", lab_64.shape, lab_128.shape, lab_256.shape)
#         for i in range(1, 9):
#             lab_64[voxel_output_list[0][1]==i], lab_128[voxel_output_list[1][1]==i], lab_256[voxel_output_list[2][1]==i] = i, i, i

#         print("lab_64[lab_64>0].sum(), lab_128[lab_128>0].sum(), lab_256[lab_256>0].sum():", lab_64[lab_64>0].sum(), lab_128[lab_128>0].sum(), lab_256[lab_256>0].sum())
#         b_0, c_0 = voxel_output_list[0][0].shape[0], voxel_output_list[0][0].shape[1]
#         b_1, c_1 = voxel_output_list[1][0].shape[0], voxel_output_list[1][0].shape[1]
#         b_2, c_2 = voxel_output_list[2][0].shape[0], voxel_output_list[2][0].shape[1]
#         mask_64, mask_128, mask_256 = voxel_output_list[0][0].view(b_0, c_0, -1), voxel_output_list[1][0].view(b_1, c_1, -1), voxel_output_list[2][0].view(b_2, c_2, -1)
        
#         print("mask_64.shape, mask_128.shape, mask_256.shape:", mask_64.shape, mask_128.shape, mask_256.shape)
#         lab_64, lab_128, lab_256 = lab_64.view(b_0, 1, -1), lab_128.view(b_1, 1, -1), lab_256.view(b_2, 1, -1)
#         print("lab_64.shape, lab_128.shape, lab_256.shape:", lab_64.shape, lab_128.shape, lab_256.shape)

#         label_64, label_128, label_256 = lab_64[lab_64>0], lab_128[lab_128>0], lab_256[lab_256>0]
#         print("label_64.shape, label_128.shape, label_256.shape:", label_64.shape, label_128.shape, label_256.shape)
        # 首先将金标准拆开
        pred = torch.softmax(pred,dim=1)
        y_pred = pred
        # print(pred.shape)
        # print(target.shape)
        organ_target = torch.zeros((target.size(0), num_seg, target.size(2)))
        # print(organ_target.shape)
        # print("y_pred", y_pred.shape)
        for organ_index in range(num_seg):
            temp_target = torch.zeros(target.shape)
            temp_target[target == organ_index+1] = 1
            temp_target = temp_target.squeeze(1)
            # print(temp_target.shape)
            organ_target[:, organ_index, :] = temp_target
        # y_true = organ_target.transpose((1,0))
        organ_target = organ_target.cuda()
        dice = 0.0
        for organ_index in range(num_seg):
            iflat = (y_pred[:, organ_index, :].contiguous().view(-1)).float()
            tflat = organ_target[:, organ_index, :].contiguous().view(-1)
            intersection = (iflat * tflat).sum()
            dice += 2. * intersection / ((iflat ** 2).sum() + (tflat ** 2).sum())

        dice_loss = 1 - dice / (num_seg)
        # 返回的是dice距离
        loss1 = loss1.mean()
        # loss = loss1 + dice_loss + 0.01*(0.9*loss_center + 0.1*cld_loss)
        loss = loss1 + dice_loss + 0.01*(0.1*loss_center + 0.9*cld_loss)
        # loss = loss1 + dice_loss + 0.001*loss_center
        # loss = loss1 + dice_loss 
        # print("loss1, dice_loss, loss_center, cld_loss:", loss1.item(), dice_loss.item(), loss_center.item(), cld_loss.item())
        return loss

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        """
        :param pred: (B, 14, 128, 256, 256)
        :param target: (B, 128, 256, 256)
        :return: Dice距离
        """

        # 首先将金标准拆开
        organ_target = torch.zeros((target.size(0), num_seg + 1, target.size(2), target.size(3), target.size(4)))

        for organ_index in range(num_seg + 1):
            temp_target = torch.zeros(target.shape)
            temp_target[target == organ_index] = 1
            temp_target = temp_target.squeeze(1)
            organ_target[:, organ_index, :, :, :] = temp_target
            # organ_target: (B, 14, 128, 256, 256)

        organ_target = organ_target.cuda()

        dice = 0.0
        for organ_index in range(num_seg + 1):
            iflat = (pred[:, organ_index, :, :, :].contiguous().view(-1)).float()
            tflat = organ_target[:, organ_index, :, :, :].contiguous().view(-1)
            intersection = (iflat * tflat).sum()
            dice += 2. * intersection / ((iflat ** 2).sum() + (tflat ** 2).sum())

        dice_loss = 1 - dice / (num_seg + 1)
        # 返回的是dice距离
        return dice_loss


def get_batch_acc(uout, label):
    # def get_acc(uout, uout_1, label, label_1):
    """soft dice score"""
    eps = 1e-7
    uout = torch.Tensor(uout)
    label = torch.Tensor(label)

    # print("type(uout), uout.shape, type(label), label.shape:", type(uout), uout.shape, type(label), label.shape)
    iflat = uout.view(-1).float()
    tflat = label.view(-1).float()
    intersection = (iflat * tflat).sum()
    dice_0 = 2. * intersection / ((iflat ** 2).sum() + (tflat ** 2).sum() + eps)

    return dice_0


'''
    是否可以设置一个3、5循环，然后train函数放在循环里面，每一次循环使用的train_data不同，
    但是每一次循环都加载上一次保存最优的那个模型！
'''
import matplotlib.pyplot as plt
save_results_data = "./save_result"

def train(net, train_data, valid_data, num_epochs, optimizer, criterion):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        net = net.cuda()
        print("使用了 cuda")
    else:
        print("没使用cuda")
    prev_time = datetime.now()
    # 超算上用于保存模型的路径
    save = './model'  # /home/zxk/Code/self-sub/code/save
    save_results_data = "./save_result"
    # 定义初始化正确率为 0
    best_acc = 0
    forget_rate = 0.5
    v_name = 'xxx.nii.gz'

    for epoch in range(num_epochs):
        print(f'\n==> training epoch {epoch}/{num_epochs}')
        if epoch % 20 == 0:
            for p in optimizer.param_groups:
                # p['lr'] *= 0.9
                p['lr'] *= 1.0

        print("当前学习率为{:.6f}".format(p['lr']))

        train_loss = 0
        IoU = [0, 0, 0, 0, 0, 0, 0, 0]
        avg_IoU = 0
        number_tra, number_val = 0, 0
        train_case_n, val_case_n = 0, 0
        net = net.train()
        for xyz_origin, features, target,name in tqdm(train_data, desc='train', ncols=0):
            number_tra += 1
            # print("xyz",xyz_origin.shape)
            # print("f",features.shape)
            # print("t",target.shape)
            xyz_origin, features, target = xyz_origin.cuda(), features.cuda(), target.cuda()  #
            # print("im:",im.shape)
            inputs = torch.cat((xyz_origin,features),dim=1)
            # print("input:",inputs.shape)
            # print("xyz_origin.shape, features.shape, target.shape:", xyz_origin.shape, features.shape, target.shape)
            uout, linear_out, uout_aux_list = net(inputs, target)  # 因为术中所有的轮廓都是label，所以先使用全1的label进行训练；
            # print("len(uout_aux_list):", len(uout_aux_list))
            # print("1:", uout_aux_list[0][0].shape, uout_aux_list[0][1].shape)
            # print("2:", uout_aux_list[1][0].shape, uout_aux_list[1][1].shape)
            # print("3:", uout_aux_list[2][0].shape, uout_aux_list[2][1].shape)

            # print("uout.shape, target.shape:", uout.shape, uout.max(), uout.min(), target.shape, target.max(), target.min())



            loss = criterion(uout,target, linear_out, uout_aux_list)
            # if math.isnan(loss):
            #     print("Loss is NaN!")
            #     break

            # print("loss:", loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            uout = torch.softmax(uout, dim=1)
            new_uout = uout.cpu().detach().numpy()
            target = target.cpu().detach().numpy()
            # print("traget",target.shape)
            # label = label.cpu().numpy()
            
            for index in range(len(new_uout)):
                mask = new_uout[index]
                mask = np.argmax(mask, axis=0)
                mask += 1
                xyz = xyz_origin[index]
                xyz = xyz.cpu().detach().numpy()
                label = target[index]
                # label = label[np.newaxis,:]
                mask = mask[np.newaxis,:]
                # print(xyz.shape)
                # print(label.shape)
                # print(mask.shape)
                avg_IoU += np.sum(mask == label) / points

                preTxt = np.concatenate((xyz, label,mask), axis=0)

                preTxt = preTxt.transpose(1, 0)
                # print(preTxt.shape)
                # preTxt = np.concatenate((preTxt, new_uout), axis=1)
                df = pd.DataFrame(preTxt.tolist())
                # print("name[i]",name[index])
                path = os.path.join(save_results_data, "train", str(name[index]) + ".txt")
                df.to_csv(path, header=None, index=False)
                train_case_n = train_case_n + 1

                train_loss += loss.item()


        print('index in train-data, and the length of train-data:', train_case_n)
        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        number = 0
        # im_list, uout_list, map_list, label_list = [], [], [], []
        # name_zero = ["0"]
        if valid_data is not None:
            val_acc = 0
            # val_dice = [0,0,0,0,0,0,0,0]

            with torch.no_grad():
                net = net.eval()
                val_points = 50000
                for xyz_origin, features, target,name in valid_data:
                    number_val += 1

                    """得到tensor形式的数据"""
                    number = number + 1
                    xyz_origin, features, target = xyz_origin.cuda(), features.cuda(), target.cuda()  #
                    inputs = torch.cat((xyz_origin, features), dim=1)
                    # print("input:", inputs.shape)
                    uout, linear_out, uout_aux_list = net(inputs, target)  # 因为术中所有的轮廓都是label，所以先使用全1的label进行训练；
                    # print("im:",im.shape)
                    # uout = net(features, xyz_origin)  # 因为术中所有的轮廓都是label，所以先使用全1的label进行训练；
                    uout = torch.softmax(uout, dim=1)

                    new_uout = uout.cpu().detach().numpy()
                    target = target.cpu().detach().numpy()
                    for index in range(len(new_uout)):
                        mask = new_uout[index]
                        mask = np.argmax(mask, axis=0)
                        mask += 1
                        xyz = xyz_origin[index]
                        xyz = xyz.cpu().detach().numpy()
                        label = target[index]
                        mask = mask[np.newaxis, :]
                        val_acc += np.sum(mask == label) / val_points

                        preTxt = np.concatenate((xyz, label, mask), axis=0)

                        preTxt = preTxt.transpose(1, 0)
                        df = pd.DataFrame(preTxt.tolist())
                        path = os.path.join(save_results_data, "test", str(name[index]) + ".txt")
                        df.to_csv(path, header=None, index=False)
                        val_case_n = val_case_n + 1
            print("val_case_n", val_case_n)
            print("train_case_n", train_case_n)
            epoch_str = (
                    "Epoch %d. Train Loss: %f,train avg dice:%f,Valid avg dice:%f,len(valid_data): %d"
                    % (epoch, train_loss / train_case_n,
                       avg_IoU / (train_case_n),
                       val_acc / (val_case_n),
                       val_case_n))

            # print('dice list, and conf list:', dice_list, conf_list)
            sys.stdout.flush()
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                         (epoch, train_loss / len(train_data),
                          avg_IoU / len(train_data)))
        prev_time = cur_time
        print(epoch_str + time_str)
        '''
        保存最终的模型：
            torch.save(net.state_dict(), os.path.join(save, 'model_half.dat'))
        '''
        # Determine if model is the best
        if (val_acc / (val_case_n)) > best_acc:
            best_acc = (val_acc / (val_case_n))
            save_path = os.path.join(save, "best/")
            # if os.path.exists(save_path):
            #     os.mkdir(save_path)
            torch.save(net.state_dict(), os.path.join(save_path, 'fine_model_co_enhance.dat'))
            torch.save(net, os.path.join(save_path, 'fine_model_co_enhance.pth'))

        if epoch > 100 and epoch % 100 == 0:
            now_acc = (val_acc / (val_case_n))
            save_path_epoch = os.path.join(save, "epoch/")
            # if os.path.exists(save_path):
            #     os.mkdir(save_path)
            torch.save(net.state_dict(), os.path.join(save_path_epoch, str(now_acc.item())[0:6] + '_' + str(
                epoch) + '_fine_model_co_enhance.dat'))
            torch.save(net, os.path.join(save_path_epoch,
                                         str(now_acc.item())[0:6] + '_' + str(epoch) + '_fine_model_co_enhance.pth'))
        print("best_val_acc:", best_acc)


'''
Compute img-entropy(confidence-level)
'''


def entropy_fn(map, point_number):
    map[map < 0.5] = 0
    # index = 1
    # for i in range(len(map)):
    #     for j in range(len(map[i])):
    #         if map[i][j] != 0:
    #             index = index + 1
    # print("index, point_number:", index, point_number)
    map = torch.tensor(map)
    entropy = ((-1) * map.contiguous().view(-1) * torch.log2(map.contiguous().view(-1) + 1e-7)).sum() / point_number
    return entropy.item()


from scipy.ndimage import zoom


def test(net, test_data):
    test_points = 400000
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        net = net.cuda()
        print("使用了 cuda")
    else:
        print("没使用cuda")
    prev_time = datetime.now()

    # 定义初始化正确率为 0
    dice_all = 0
    val_sum_batch = 0
    origh_error_batch = 0
    origh_good_batch = 0
    new_good_batch = 0
    ide = 0
    origh_all = 0
    Jaccard = 0
    val_acc_1 = 0
    ASD = 0
    HD = 0
    # val_acc_2 = 0
    # val_acc_3 = 0
    with torch.no_grad():
        net = net.eval()
        name_tem = ''
        mask_list = []

        data_list = []
        ACC_list = []
        Conf_list = []
        number_val = 0
        if test_data is not None:
            val_acc = 0
            val_case_n = 0
            # val_dice = [0,0,0,0,0,0,0,0]

            with torch.no_grad():
                net = net.eval()
                for xyz_list, features_list, target_list, number_list, name, xyz_yw_list in test_data:     # 再传入一个列表来形容liver和vessel的数量，用于后续采纳这些数量的点 
                    total_points_num = 0
                    correct_points_num = 0
                    txt_list, txt_list_liver, txt_list_vessel = [], [], [] 
                    for i in range(len(xyz_list)):
                        xyz = xyz_list[i]
                        xyz_sequence = np.array(xyz)[np.newaxis,:,:]
                        xyz_tensor = torch.Tensor(xyz_sequence)
                        # print("xyz_sequence.shape:",xyz_sequence.shape)     # xyz_sequence.shape: (1, 3, 50000) 
                        features = features_list[i]
                        features_sequence = np.array(features)[np.newaxis,:,:]
                        features_tensor = torch.Tensor(features_sequence)
                        target = target_list[i]
                        target_sequence = np.array(target)[np.newaxis,:,:]
                        target_tensor = torch.Tensor(target_sequence)
                        xyz_tensor, features_tensor, target_tensor = xyz_tensor.cuda(), features_tensor.cuda(), target_tensor.cuda()

                        
                        num_list = number_list[i]
                        
                        xyz_yw = xyz_yw_list[i]

                        #网络输入
                        inputs = torch.cat((xyz_tensor, features_tensor), dim=1)


                        uout, linear_out, uout_aux_list = net(inputs, target_tensor)
                        uout = torch.softmax(uout, dim=1)
                        # print("uout.shape:", uout.shape)     # uout.shape: torch.Size([1, 8, 50000]) 
                        new_uout = uout.cpu().detach().numpy()
                        target = target_tensor.cpu().detach().numpy()
                        total_points_num += xyz_tensor.shape[2]
                        for index in range(len(new_uout)):     # 其实这里没有遍历
                            mask = new_uout[index]
                            mask = np.argmax(mask, axis=0)
                            mask += 1
                            xyz = xyz_tensor[index]
                            xyz = xyz.cpu().detach().numpy()
                            # print("xyz.shpae, xyz_yw.shape:", xyz.shape, xyz_yw.shape)
                            # print("xyz.shpae, xyz_yw.shape:", xyz[:, :4], xyz_yw[:, :4])
                            xyz = xyz_yw
                            label = target[index]
                            mask = mask[np.newaxis, :]
                            # print("mask:",mask.shape)
                            # print("label:",label.shape)
                            # print(xyz.shape)
                            # print(label.shape)
                            # print(mask.shape)
                            # print("np.sum(mask==label):", np.sum(mask==label))
                            correct_points_num += np.sum(mask == label)
                            preTxt = np.concatenate((xyz, label, mask), axis=0) 
                            preTxt = preTxt.transpose(1, 0)
                            # print("preTxt.shape, num_list:", preTxt.shape, num_list)     # 记得vessel是从列表前面开始,liver可以从列表后面开始 
                            # 需要将下面3个列表进行改动即可，即仅保存需要保存的点！
                            
                            all_vl = np.concatenate((preTxt[:num_list[0], :], preTxt[-num_list[1]:, :]), axis=0) 
                            txt_list_liver.extend(preTxt[-num_list[1]:, :])
                            txt_list_vessel.extend(preTxt[:num_list[0], :])
                            txt_list.extend(all_vl)
                    print("len(txt_list_liver), len(txt_list_vessel), len(txt_list):", len(txt_list_liver), len(txt_list_vessel), len(txt_list))
                    print("correct_points_num:", correct_points_num)
                    print("total_points_num:", total_points_num)
                    temp_acc = correct_points_num / total_points_num
                    val_acc += temp_acc

                    ACC_list.append([name[0], temp_acc])
                    list_array = np.array(txt_list)
                    list_array_l = np.array(txt_list_liver)
                    list_array_v = np.array(txt_list_vessel) 
                    
                    lab_num, mask_num = list_array[:, 3], list_array[:, 4]
                    print("list_array.shape, lab_num.shape, mask_num.shape:", list_array.shape, lab_num.shape, mask_num.shape)
                    print("当前样例的所有点的ACC为:", np.sum(lab_num == mask_num)/list_array.shape[0])
                    path = os.path.join(save_results_data, "test_all", name + ".txt")     # 输出的肝脏和血管也要分开保存，即将50000个点进行拆分，肝脏保存肝脏，血管保存血管；
                    df = pd.DataFrame(list_array.tolist())
                    print("name:", name, list_array.shape, list_array_l.shape, list_array_v.shape)
                    print("ACC:", temp_acc)

                    df.to_csv(path, header=None, index=False)     # 这里保存的是所有的点 
                    path_l = os.path.join(save_results_data, "test_all", "liver", name + ".txt") 
                    df_l = pd.DataFrame(list_array_l.tolist())
                    df_l.to_csv(path_l, header=None, index=False)
                    path_v = os.path.join(save_results_data, "test_all", "vessel", name + ".txt") 
                    df_v = pd.DataFrame(list_array_v.tolist())
                    df_v.to_csv(path_v, header=None, index=False)
                    
                    val_case_n = val_case_n + 1
                print("val_case_n", val_case_n)
                epoch_str = (
                            " Valid avg dice:%f,len(valid_data): %d"
                            % (val_acc / (val_case_n),
                               val_case_n))
                print(epoch_str)
            df = pd.DataFrame(ACC_list, columns=["name", "ACC"])
            df.to_csv(os.path.join(save_results_data, "./test_acc_all.csv"), header=None, index=False)


