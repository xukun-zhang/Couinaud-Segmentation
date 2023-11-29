from torch.utils.data import Dataset
import pickle
import os
from medpy.io import load, save
import numpy as np
import cv2
from skimage.measure import label as la
import SimpleITK as sitk
from scipy.ndimage import zoom
import random
import pandas as pd

"""
    定义一个 读取数据集.pkl文件的类：
"""



def window_level_processing(image):
    win_min = -100
    win_max = 300
    # print("win_max, win_min:", win_max, win_min)
    image = (image - win_min) / (win_max - win_min)
    image[image > 1] = 1
    image[image < 0] = 0

    return image


def nomalize(xyz_origin):
    xyz_scope = np.array([max(xyz_origin[:, 0]) - min(xyz_origin[:, 0]), max(xyz_origin[:, 1]) - min(xyz_origin[:, 1]),
                          max(xyz_origin[:, 2]) - min(xyz_origin[:, 2])])
    xyz_min = np.array([min(xyz_origin[:, 0]), min(xyz_origin[:, 1]), min(xyz_origin[:, 2])])
    # xyz_min = np.array([x_axis * Spacing_arr[0], y_axis * Spacing_arr[1], z_axis * Spacing_arr[2]])
    xyz_origin = (xyz_origin - xyz_min) / xyz_scope
    xyz_origin = xyz_origin.astype(np.float32)

    return xyz_origin


# 定义一个子类叫 custom_dataset，继承与 Dataset
save_path = "./save_result/all_yuanwei"


'''
这里要进行改动，即要加载血管和肝脏的两个txt文件，这样的话即可以一次加载血管和肝脏，然后从血管中随机取30000个点，然后其余的从肝脏中进行抽取20000个点，但要确保将血管中的点全部抽取完
'''

class Dataset_Test(Dataset):
    def __init__(self, path, transform=None):
        self.transform = transform  # 传入数据预处理
        self.image_data = {}  # length is the number of cases, and each case have a list, that have a sequence data
        self.label_data = {}
        self.feature_data = {}
        self.image_data_v = {}  # length is the number of cases, and each case have a list, that have a sequence data
        self.label_data_v = {}
        self.feature_data_v = {}
   
        self.image_name = []  # length is the number of cases
        floder_path = "./Dataset/Point_Data/test"
        # floder_path = "../Dataset/Tiaoshi"
        file_list = os.listdir(floder_path)
        numbers = 0

        for file_name in file_list:
            print(file_name)
            number = 0

            file = os.path.join(path, "liver_Points", file_name)
            data = np.loadtxt(file, dtype=float, delimiter=",", skiprows=0, usecols=None, unpack=False)
            number = number + 1

            if number == 1:
                numbers = numbers + 1
                np.random.shuffle(data)
                df = pd.DataFrame(data.tolist())
                save_file_path = os.path.join(save_path, "liver", file_name)

                print("save_file_path:", save_file_path, data.shape)
                df.to_csv(save_file_path, header=None, index=False)
                xyz_origin = data[:, :3]
                features = data[:, -2]
                target = data[:, -1]
                xyz_origin = nomalize(xyz_origin)
                xyz_origin = xyz_origin.transpose((1, 0))
                features = features[np.newaxis, :]
                target = target[np.newaxis, :]

                self.image_name.extend([file_name])
                self.image_data[file_name] = xyz_origin
                self.label_data[file_name] = target
                self.feature_data[file_name] = features

            else:
                print("the label name not in dataset:", number, file_name, file_name)
        
        for file_name in file_list:
            print("血管：", file_name)
            number = 0

            file = os.path.join(path, "vessel", file_name)
            data = np.loadtxt(file, dtype=float, delimiter=",", skiprows=0, usecols=None, unpack=False)
            number = number + 1
            if number == 1:
                numbers = numbers + 1
                np.random.shuffle(data)
                df = pd.DataFrame(data.tolist())
                save_file_path = os.path.join(save_path, "vessel", file_name)     # 这里肝脏和血管要分开保存
                print("vessel data.shape:", data.shape)
                df.to_csv(save_file_path, header=None, index=False)
                xyz_origin = data[:, :3]
                features = data[:, -2]
                target = data[:, -1]
                xyz_origin = nomalize(xyz_origin)
                xyz_origin = xyz_origin.transpose((1, 0))
                features = features[np.newaxis, :]
                target = target[np.newaxis, :]
                self.image_data_v[file_name] = xyz_origin
                self.label_data_v[file_name] = target
                self.feature_data_v[file_name] = features

            else:
                print("the label name not in dataset:", number, file_name, file_name)

        print("the data numbers are:", numbers)

    def __getitem__(self, idx):  # 根据 idx 取出其中一个name
        name = self.image_name[idx]
        xyz_origin = self.image_data[name]
        target = self.label_data[name]
        features = self.feature_data[name]
        xyz_origin_v = self.image_data_v[name]
        target_v = self.label_data_v[name]
        features_v = self.feature_data_v[name]

        return xyz_origin.astype(np.float32), features.astype(np.float32), target.astype(np.int64), str(name.split(".txt")[0]), xyz_origin_v.astype(np.float32), features_v.astype(np.float32), target_v.astype(np.int64)

    def __len__(self):  # 总数据的多少
        return len(self.label_data)
class Dataset_Val(Dataset):
    def __init__(self, path, transform=None):

        self.transform = transform  # 传入数据预处理
        self.image_data = {}  # length is the number of cases, and each case have a list, that have a sequence data
        self.label_data = {}
        self.feature_data = {}
        self.image_name = []  # length is the number of cases

        file_list = os.listdir(path)
        numbers = 0

        for file_name in file_list:
            number = 0
            file = os.path.join(path, file_name)
            print(file_name)
            data = np.loadtxt(file, dtype=float, delimiter=",", skiprows=0, usecols=None, unpack=False)
            number = number + 1

            if number == 1:
                numbers = numbers + 1
                xyz_origin = data[:, :3]
                features = data[:, -2]
                target = data[:, -1]
                xyz_origin = nomalize(xyz_origin)     # 应该是 直接 世界坐标 进行正则化？
                xyz_origin = xyz_origin.transpose((1, 0))     # 将这个坐标变换的再后面再用两次 
                features = features[np.newaxis, :]
                target = target[np.newaxis, :]

                self.image_name.extend([file_name])
                self.image_data[file_name] = xyz_origin
                self.label_data[file_name] = target
                self.feature_data[file_name] = features

            else:
                print("the label name not in dataset:", number, file_name, file_name)

        print("the data numbers are:", numbers)

    def __getitem__(self, idx):  # 根据 idx 取出其中一个name
        name = self.image_name[idx]
        xyz_origin = self.image_data[name]
        target_all = self.label_data[name]
        features_all = self.feature_data[name]

        # 这里应该使用个增强函数，既随机采样50000个点（先打乱顺序，再随机采样） 
        xyz,features,target = self.transform(xyz_origin, features_all, target_all)

        return xyz.astype(np.float32), features.astype(np.float32), target.astype(np.int64), str(
            name.split(".txt")[0])

    def __len__(self):  # 总数据的多少
        return len(self.label_data)


class Dataset_Train(Dataset):
    def __init__(self, path, transform=None):
        self.transform = transform  # 传入数据预处理
        self.Image_datas = {}
        self.vessel_points_datas = {}
        self.liver_points_datas = {}
        self.spacing_datas = {}
        self.dir_datas = {}
        self.origin_datas = {}
        self.image_name = []  # length is the number of cases

        numbers = 0
        liver_points_path = "liver_Points"
        vessel_around_points_path = "vessel"
        yw_image_path = "/home/dataset1/zxk_data/Train_data/MSD/image"
        file_list = os.listdir(os.path.join(path, vessel_around_points_path))
        print("file:",file_list)
        for file_name in file_list:
            number = 0
            vessel_around_points_file = os.path.join(path, vessel_around_points_path, file_name)
            vessel_around_points_data = np.loadtxt(vessel_around_points_file, dtype=float, delimiter=",", skiprows=0,
                                                   usecols=None, unpack=False)
            # print(vessel_around_points_data.shape)
            liver_points_file = os.path.join(path, liver_points_path, file_name)
            liver_points_data = np.loadtxt(liver_points_file, dtype=float, delimiter=",", skiprows=0, usecols=None,
                                           unpack=False)
            # origin_file = os.path.join(path, origin_path, file_name)
            # origin_data = np.loadtxt(origin_file, dtype=np.float, delimiter=",", skiprows=0, usecols=None,
            #                                unpack=False)
            # Spacing_Dir_file = os.path.join(path, Spacing_Dir_path, file_name)
            # Spacing_Dir_data = np.loadtxt(Spacing_Dir_file, dtype=np.float, delimiter=",", skiprows=0, usecols=None,
            #                                unpack=False)
            name = file_name.split(".txt")[0]
            print(name)
            Image = sitk.ReadImage(os.path.join(yw_image_path, name + ".nii.gz"))
            Image_arr = sitk.GetArrayFromImage(Image)
            Image_arr = window_level_processing(Image_arr)
            numbers = numbers + 1

            self.image_name.extend([name])
            self.Image_datas[name] = Image_arr
            self.vessel_points_datas[name] = vessel_around_points_data
            self.liver_points_datas[name] = liver_points_data
            self.spacing_datas[name] = np.array(Image.GetSpacing())
            self.dir_datas[name] = np.array(Image.GetDirection()).reshape((3, 3))
            self.origin_datas[name] = np.array(Image.GetOrigin())
        print("the data numbers are:", numbers)

    def __getitem__(self, idx):  # 根据 idx 取出其中一个name
        name = self.image_name[idx]
        vessel_around_points_data = self.vessel_points_datas[name]
        liver_points_data = self.liver_points_datas[name]
        spacing_data = self.spacing_datas[name]
        dir_data = self.dir_datas[name]
        origin_data = self.origin_datas[name]
        image_arr = self.Image_datas[name]


        # 下面的将图像坐标 基于扰动 转换为了 世界坐标 
        xyz,features,target = self.transform(vessel_around_points_data, liver_points_data, spacing_data,
                                                      origin_data, image_arr, dir_data)
        xyz = nomalize(xyz)     # 对 世界坐标 正则化 
        xyz = xyz.transpose((1, 0))
        features = features[np.newaxis, :]
        target = target[np.newaxis, :]

        return xyz.astype(np.float32), features.astype(np.float32), target.astype(np.int64), name

    def __len__(self):  # 总数据的多少
        return len(self.vessel_points_datas)