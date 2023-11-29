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
DataSet_file = "/home/zxk/code/3Diradb/PVRCNN-vessel-raodong/Dataset/Nii_Data"
save_path = "./new_trainpoints/"
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
# list = ["test"]
list = ["train"]
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
    box = np.array([[np.min(z_true), np.max(z_true)], [np.min(y_true), np.max(y_true)],
                    [np.min(x_true), np.max(x_true)]])

    z_min, z_max = box[0]
    y_min, y_max = box[1]
    x_min, x_max = box[2]

    Spacing_arr = np.array(spacing)
    Origin_arr = np.array(origin)
    # vessel[label == 0] = 0
    # z_axis, x_axis, y_axis = image.shape
    Direction_arr = np.array(direction).reshape((3, 3))

    vessel[label == 0] = 0
    # z_axis, x_axis, y_axis = image.shape
    print(image.shape)

    # vessel_around_list = [[x,y,z,image[z][y][x],label[z][y][x]]  for x in range(x_min,x_max) for y in range(y_min,y_max) for z in range(z_min,z_max) if
    #     (vessel[z][y][x] != 0)]
    # liver_list = [[x,y,z,image[z][y][x],label[z][y][x]]  for x in range(x_min,x_max) for y in range(y_min,y_max) for z in range(z_min,z_max) if
    #     (label[z][y][x] != 0 and vessel[z][y][x]==0)]
    vessel_around_list = [[x * Spacing_arr[0]*Direction_arr[0,0] + Origin_arr[0], y * Spacing_arr[1]*Direction_arr[1,1] + Origin_arr[1], z * Spacing_arr[2]*Direction_arr[2,2] + Origin_arr[2], image[z][y][x],label[z][y][x]]  for x in range(x_min,x_max) for y in range(y_min,y_max) for z in range(z_min,z_max) if
        (vessel[z][y][x] != 0)]
    liver_list = [[x * Spacing_arr[0]*Direction_arr[0,0] + Origin_arr[0], y * Spacing_arr[1]*Direction_arr[1,1] + Origin_arr[1], z * Spacing_arr[2]*Direction_arr[2,2] + Origin_arr[2], image[z][y][x],label[z][y][x]]  for x in range(x_min,x_max) for y in range(y_min,y_max) for z in range(z_min,z_max) if
        (label[z][y][x] != 0 and vessel[z][y][x]==0)]
    print("vessel:",len(vessel_around_list))
    print("liver:",len(liver_list))
    all_list = vessel_around_list + liver_list

    return vessel_around_list,liver_list,all_list
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
        Data_floder_path = os.path.join(DataSet_file,list[i])

        Data_floder = os.listdir(os.path.join(Data_floder_path,"image"))
        
        for data_name in Data_floder:
        # for z in range(1):
            # data_name = "volume-71.nii.gz"

            if data_name == ".DS_Store":
                continue
            print(data_name)
            # if os.path.exists(os.path.join(save_path,list[i],"vessel",data_name.split(".nii")[0] + ".txt")):
            #     continue
            print(Data_floder_path)

            image_path = os.path.join(Data_floder_path,image_file,data_name)
            # image_path = "./Dataset/train/image/volume-71.nii.gz"
            print("image_path:",image_path)
            Image = sitk.ReadImage(image_path)
            image_arr = sitk.GetArrayFromImage(Image)
            spacing = Image.GetSpacing()
            origin = Image.GetOrigin()
            direction = Image.GetDirection()
            tem = data_name.split("-")[-1]

            label_path = os.path.join(Data_floder_path, label_file,tem)
            Label = sitk.ReadImage(label_path)

            label_arr = sitk.GetArrayFromImage(Label)

            liver_path = os.path.join(Data_floder_path, liver_file, tem)
            liver = sitk.ReadImage(liver_path)
            liver_arr = sitk.GetArrayFromImage(liver)

            vessel_path = os.path.join(Data_floder_path, vessel_file,data_name)
            vessel = sitk.ReadImage(vessel_path)
            vessel = sitk.DilateObjectMorphology(vessel, kernelRadius=(10, 10, 3))
            vessel_arr = sitk.GetArrayFromImage(vessel)
            image_arr = window_level_processing(image_arr)
            print("save_path:",os.path.join(save_path, list[i],data_name.split(".nii")[0] + ".txt"))
            vessel_list,liver_list,all_points_list = convert2point(image_arr, label_arr, vessel_arr, liver_arr, spacing,origin, direction, i)

            df_v = pd.DataFrame(vessel_list,columns=None)
            df_v.to_csv(os.path.join(save_path,"vessel",data_name.split(".nii")[0] + ".txt"), header=None, index=False)

            df_l = pd.DataFrame(liver_list, columns=None)
            df_l.to_csv(os.path.join(save_path,"liver_Points", data_name.split(".nii")[0] + ".txt"), header=None, index=False)

            df_A = pd.DataFrame(all_points_list, columns=None)
            df_A.to_csv(os.path.join(save_path, "AllPoints", data_name.split(".nii")[0] + ".txt"), header=None,
                        index=False)

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