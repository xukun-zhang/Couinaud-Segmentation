import numpy as np
import pandas as pd
import SimpleITK as sitk
import os
import random
from skimage.measure import label as la

vessel_file = "vessel"
image_file = "image"
label_file = "label"
liver_file = "liver"
DataSet_file = "./Dataset/Nii_Data"
save_path = "./new_testpoints/"
save_normal_path = "./PointDataChange/normal"
save_20w_path = "./PointDataChange/test_20w"
def window_level_processing(image):
    win_min = -100
    win_max = 300
    # print("win_max, win_min:", win_max, win_min)
    image = (image - win_min) / (win_max - win_min)
    image[image > 1] = 1
    image[image < 0] = 0

    return image
# list = ["train","test"]
list = ["test"]
# list = ["train"]
points = 5000
test_20w_points = 200000
pixel = 10
def convert2point(image,label,vessel,liver,spacing,origin,direction,flag):
    loc_img, num = la(liver, background=0, return_num=True, connectivity=2)
    max_label = 0
    max_num = 0
    for i in range(1, num + 1):
        if np.sum(loc_img == i) > max_num:
            max_num = np.sum(loc_img == i)
            max_label = i
    mcr = (loc_img == max_label)
    mcr = mcr + 0
    z_true, y_true, x_true = np.where(mcr)

    # 下面这行代码得到了 肝脏所在 3D立方体 空间 的网格 
    box = np.array([[np.min(z_true), np.max(z_true)], [np.min(y_true), np.max(y_true)], [np.min(x_true), np.max(x_true)]])
    z_min, z_max = box[0]
    y_min, y_max = box[1]
    x_min, x_max = box[2]

    Spacing_arr = np.array(spacing)
    Origin_arr = np.array(origin)
    Direction_arr = np.array(direction).reshape((3, 3))

    vessel[label == 0] = 0     # Couinaud segments 外面的血管 Mask 设为0 

    print("在处理的CT图像的大小为:", image.shape)
    """下面保存原本的图像坐标:"""
    vessel_list_yw = [[x,y,z,image[z][y][x],label[z][y][x]]  for x in range(x_min,x_max) for y in range(y_min,y_max) for z in range(z_min,z_max) if (vessel[z][y][x] != 0)]
    
    liver_list_yw = [[x,y,z,image[z][y][x],label[z][y][x]]  for x in range(x_min,x_max) for y in range(y_min,y_max) for z in range(z_min,z_max) if (label[z][y][x] != 0 and vessel[z][y][x]==0)]
    
    """下面转换为世界坐标:"""
    vessel_around_list = [[x * Spacing_arr[0]*Direction_arr[0,0] + Origin_arr[0], y * Spacing_arr[1]*Direction_arr[1,1] + Origin_arr[1], z * Spacing_arr[2]*Direction_arr[2,2] + Origin_arr[2], image[z][y][x],label[z][y][x]]  for x in range(x_min,x_max) for y in range(y_min,y_max) for z in range(z_min,z_max) if
        (vessel[z][y][x] != 0)]
    liver_list = [[x * Spacing_arr[0]*Direction_arr[0,0] + Origin_arr[0], y * Spacing_arr[1]*Direction_arr[1,1] + Origin_arr[1], z * Spacing_arr[2]*Direction_arr[2,2] + Origin_arr[2], image[z][y][x],label[z][y][x]]  for x in range(x_min,x_max) for y in range(y_min,y_max) for z in range(z_min,z_max) if
        (label[z][y][x] != 0 and vessel[z][y][x]==0)]
    print("vessel:",len(vessel_around_list))
    print("liver:",len(liver_list))
    
    """下面这行得到了完整肝脏空间中所有点- 肝实质与血管 -的坐标列表:"""
    all_list = vessel_around_list + liver_list

    return vessel_around_list,liver_list,all_list, vessel_list_yw, liver_list_yw
    # all_arr = np.array(random.sample(all_list,points*2))
    # v_a_arr = np.array(random.sample(vessel_around_list,points))
    # # point_data = random.sample(vessel_around_list,points)
    # l_arr = np.array(random.sample(liver_list,points))
    # # point_data.append(random.sample(liver_list,points))
    # point_data = np.concatenate((v_a_arr,l_arr),axis=0)

    # print(point_data.shape)

    # return point_data,all_arr


if __name__ == '__main__':
    for i in range(1):
        Data_floder_path = os.path.join(DataSet_file,list[i])     # 选择 训练集 或 测试集进行处理,得到点数据 
        Data_floder = os.listdir(os.path.join(Data_floder_path,"image"))
        
        for data_name in Data_floder:
        # for z in range(1):
            # data_name = "volume-71.nii.gz"

            if data_name == ".DS_Store":
                continue
            print("打印当前处理的图像数据名称:", data_name)
            # if os.path.exists(os.path.join(save_path,list[i],"vessel",data_name.split(".nii")[0] + ".txt")):
            #     continue
            print("训练 或 测试的路径:", Data_floder_path)
            image_path = os.path.join(Data_floder_path,image_file,data_name)
            # image_path = "./Dataset/train/image/volume-71.nii.gz"
            print("得到但去处理的图像样例的路径-image_path:",image_path)
            
            Image = sitk.ReadImage(image_path)
            image_arr = sitk.GetArrayFromImage(Image)     # 得到 CT 图像 的 HU值 
            spacing = Image.GetSpacing()
            origin = Image.GetOrigin()
            direction = Image.GetDirection()
            
            tem = data_name.split("-")[-1]     # 图像样例的名称 
            print("tem:", tem)

            label_path = os.path.join(Data_floder_path, label_file,tem)
            Label = sitk.ReadImage(label_path)
            label_arr = sitk.GetArrayFromImage(Label)     # 得到肝脏 Couinaud segments 的 Mask 

            liver_path = os.path.join(Data_floder_path, liver_file, tem)
            liver = sitk.ReadImage(liver_path)
            liver_arr = sitk.GetArrayFromImage(liver)     # 得到 肝脏 Liver 的 Mask 

            vessel_path = os.path.join(Data_floder_path, vessel_file,data_name)
            vessel = sitk.ReadImage(vessel_path)
            vessel = sitk.DilateObjectMorphology(vessel, kernelRadius=(10, 10, 3))
            vessel_arr = sitk.GetArrayFromImage(vessel)     # 对血管进行膨胀, 得到 血管 Vessel 的 Mask 
            
            image_arr = window_level_processing(image_arr)     # 对 CT 图像 进行 窗宽 窗位 的设置 

            vessel_list,liver_list,all_points_list, vessel_yw, liver_yw = convert2point(image_arr, label_arr, vessel_arr, liver_arr, spacing,origin, direction, i)

            df_v = pd.DataFrame(vessel_list,columns=None)
            df_v.to_csv(os.path.join(save_path,"vessel",data_name.split(".nii")[0] + ".txt"), header=None, index=False)

            df_l = pd.DataFrame(liver_list, columns=None)
            df_l.to_csv(os.path.join(save_path,"liver_Points", data_name.split(".nii")[0] + ".txt"), header=None, index=False)

            df_A = pd.DataFrame(all_points_list, columns=None)
            df_A.to_csv(os.path.join(save_path, "AllPoints", data_name.split(".nii")[0] + ".txt"), header=None,
                        index=False)
            
            df_v_yw = pd.DataFrame(vessel_yw,columns=None)
            df_v_yw.to_csv(os.path.join("./new_testpoints_yw/vessel",data_name.split(".nii")[0] + ".txt"), header=None, index=False)

            df_l_yw = pd.DataFrame(liver_yw, columns=None)
            df_l_yw.to_csv(os.path.join("./new_testpoints_yw/liver_Points", data_name.split(".nii")[0] + ".txt"), header=None, index=False)

            # vessel_list = random.sample(vessel_list, min(30000,len(vessel_list)))
            # liver_list = random.sample(liver_list, 30000)
            # df_v = pd.DataFrame(vessel_list, columns=None)
            # df_v.to_csv(os.path.join(save_path, list[i]+"_3w", "vessel", data_name.split(".nii")[0] + ".txt"), header=None,
            #             index=False)
            # df_l = pd.DataFrame(liver_list, columns=None)
            # df_l.to_csv(os.path.join(save_path, list[i]+"_3w", "liver_Points", data_name.split(".nii")[0] + ".txt"),
            #             header=None, index=False)

            # df_S = pd.DataFrame(spacing_dir.tolist(), columns=None)
            # df_S.to_csv(os.path.join(save_path, "SpacingDir", data_name.split(".nii")[0] + ".txt"), header=None,
            #             index=False)
            #
            # pd.DataFrame(origin_list.tolist(), columns=None).to_csv(os.path.join(save_path, "origin", data_name.split(".nii")[0] + ".txt"), header=None, index=False)