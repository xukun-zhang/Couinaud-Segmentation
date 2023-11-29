import os
from os.path import join, exists, dirname, abspath
import numpy as np
import SimpleITK as sitk
from helper_ply import write_ply
import random
import pandas as pd
# from helper_tool import DataProcessing as DP
typeimg = ['x','y', 'z', 'value', 'class']
out_format = ".ply"
sub_grid_size = 0.01
save_path = "./point_data"
def load_volume(ID):

    def window_level_processing(image):
        win_min = -100
        win_max = 300
        # print("win_max, win_min:", win_max, win_min)
        image = (image - win_min) / (win_max - win_min)
        image[image > 1] = 1
        image[image < 0] = 0

        return image

    name_id = ID.split("-")[1]
    # label_path = os.path.join("./Couinaud","couinaud-"+name_id)
    label_path = os.path.join("./C","couinaud-"+name_id)
    # image_path = os.path.join("./image",ID)
    image_path = os.path.join("./I",ID)

    label_nii = sitk.ReadImage(label_path)
    image_nii = sitk.ReadImage(image_path)

    Spacing_arr = np.array(image_nii.GetSpacing())
    Origin_arr = np.array(image_nii.GetOrigin())

    label = sitk.GetArrayFromImage(label_nii)
    image = sitk.GetArrayFromImage(image_nii)
    z_axis,x_axis, y_axis = image.shape
    print(image.shape)
    data_list = [
        [x*Spacing_arr[0]+Origin_arr[0], y*Spacing_arr[1]+Origin_arr[1], z*Spacing_arr[2]+Origin_arr[2],image[z][x][y],label[z][x][y]]
        for x in range(x_axis) for y in range(y_axis) for z in range(z_axis) if
        (label[z][x][y] != 0)]
    pc_data = np.array(data_list)
    # print(pc_data.shape)
    # print(pc_data[:,:3])
    xyz_origin = pc_data[:, :3]
    np.save(os.path.join(sub_pc_folder, ID + "_xyz_origin.npy"), xyz_origin)

    xyz_min = np.array([x_axis*Spacing_arr[0], y_axis*Spacing_arr[1], z_axis*Spacing_arr[2]])
    pc_data[:, 0:3] /= xyz_min
    xyz = pc_data[:, :3].astype(np.float32)
    value = pc_data[:, -2].astype(np.float32)
    labels = pc_data[:, -1].astype(np.uint8)
    points = len(labels)
    (unique, counts) = np.unique(labels, return_counts=True)
    print(ID, " n point ", len(labels), unique, counts)
    pc_list = pc_data.tolist()
    sub_points = (int)(points*sub_grid_size)
    sub_pc_list = random.sample(pc_list,sub_points)
    name = ID.split(".nii")[0]
    save_name = os.path.join(save_path,name+".txt")
    sub_pc_list_df = pd.DataFrame(sub_pc_list)
    sub_pc_list_df.to_csv(save_name,header=typeimg,index=False)



    # write_ply(os.path.join(original_pc_folder, ID+out_format ), (xyz,value, labels),
    #           ['x', 'y', 'z', 'value','class'])
    # sub_xyz, sub_value, sub_labels = DP.grid_sub_sampling(xyz, value, labels, sub_grid_size)
    # write_ply(os.path.join(sub_pc_folder, ID + out_format), [sub_xyz, sub_value, sub_labels],
    #           ['x', 'y', 'z', 'value', 'class'])


def convert_pc2ply(param, ID):
    pass


def process_data_and_save(ID):
    load_volume(ID)


if __name__ == '__main__':
    outPC_path = "./point_data"
    dataset_path = "./I"
    original_pc_folder = os.path.join(outPC_path,"original_ply") ##保存路径
    sub_pc_folder =  os.path.join(outPC_path,"input0.01")   ##
    if not exists(original_pc_folder):
        os.makedirs(original_pc_folder)
    if not exists(sub_pc_folder):
        os.makedirs(sub_pc_folder)
    list_ID = os.listdir(dataset_path)
    for i, ID in enumerate(list_ID):
        if ID == ".DS_Store":
            continue
        # print(ID)
        process_data_and_save(ID)