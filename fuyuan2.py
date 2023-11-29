import numpy as np
import SimpleITK as sitk
import os
import pandas as pd
from reprocessing import reprocessing
from skimage.measure import label as la
import torch
# Txt_file = "../PVRCNN-Vessel/save_result/test"
Txt_file = "./save_result/test_all"
# Txt_file = "../PVRCNN/save_result/test"
save_file = "./save_result/pvrcnn_final"
# save_file = "./save_result/pvrcnn"
mask_file = "./save_result/pvrcnn_final/all"
# mask_file = "./save_result/pvrcnn/point"
label_file = "./Dataset/Nii_Data/test/label"
image_nii_file = "./Dataset/Nii_Data/test/image"
# image_file = "../Dataset/Point_Data/allpoints/all"
image_file = "./save_result/all_yuanwei"
# nii_file ="./nii_image"
Txt_floder = os.listdir(Txt_file)
print("Txt_floder:",Txt_floder)
liver_path = "./Dataset/Nii_Data/test/liver"
dices = [0, 0, 0, 0, 0, 0, 0, 0]
l = 0
list = []
dice_total = 0
def get_batch_acc(uout, label):

    """soft dice score"""
    eps = 1e-7
    uout = torch.Tensor(uout)
    label = torch.Tensor(label)


    #print("type(uout), uout.shape, type(label), label.shape:", type(uout), uout.shape, type(label), label.shape)
    iflat = uout.view(-1) .float()
    tflat = label.view(-1).float()
    intersection = (iflat * tflat).sum()
    dice_0 = 2. * intersection / ((iflat ** 2).sum() + (tflat ** 2).sum() + eps)

    return dice_0

# for i in range(1):
#     file_name = "volume-41.txt"
for file_name in Txt_floder:
    dice = 0


    print(file_name)
    if ".txt" not in file_name:
        continue


    # fiel_index = file_name.split(".txt")[0].split("b1.")[-1]
    name = file_name.split(".txt")[0] + ".nii.gz"
    tem = name.split("-")[-1]
    print("tem:", tem)
    label_name = tem
    mask_name = tem

    image_name = file_name
    file_path = os.path.join(Txt_file,file_name)
    file_path_l = os.path.join(Txt_file, "liver", file_name)
    file_path_v = os.path.join(Txt_file, "vessel", file_name)
    
    label_path = os.path.join(label_file,label_name)
    save_path = os.path.join(save_file,mask_name)
    save_point_path = os.path.join(mask_file, mask_name)
    if os.path.exists(save_point_path):
        continue

    image_path = os.path.join(image_file,image_name)
    image_path_l = os.path.join(image_file, "liver", image_name)
    image_path_v = os.path.join(image_file, "vessel", image_name)
    
    nii_path = os.path.join(image_nii_file,name)
    NiiImage = sitk.ReadImage(nii_path)

    # liver_name = name
    liver_name = tem
    liver_p = os.path.join(liver_path, liver_name)
    Liver = sitk.ReadImage(liver_p)
    Liver.SetDirection(NiiImage.GetDirection())
    Liver.SetOrigin(NiiImage.GetOrigin())
    Liver.SetSpacing(NiiImage.GetSpacing())
    liver_array = sitk.GetArrayFromImage(Liver)
    # point_arr = np.array(pd.read_csv(file_path,index_col=0,encoding='ISO-8859-1',delimiter=",",skiprows=0))
    # image_arr = np.array(pd.read_csv(image_path,index_col=0,encoding='ISO-8859-1',delimiter=",",skiprows=0))
    point_arr = np.loadtxt(file_path,delimiter=",")
    image_arr = np.loadtxt(image_path,delimiter=",")
    
    point_arr_l = np.loadtxt(file_path_l,delimiter=",")
    image_arr_l = np.loadtxt(image_path_l,delimiter=",")
    point_arr_v = np.loadtxt(file_path_v,delimiter=",")
    image_arr_v = np.loadtxt(image_path_v,delimiter=",")

    
    

    print("-----point_arr_l.shape, point_arr_v.shape, image_arr_l.shape, image_arr_v.shape:", point_arr_l.shape, point_arr_v.shape, image_arr_l.shape, image_arr_v.shape)
    # image_arr = np.loadtxt(file_path,delimiter=",").astype(int)
    # print(point_arr.shape)
    label_Image = sitk.ReadImage(label_path)
    label_arr = sitk.GetArrayFromImage(label_Image)
    spacing_arr = np.array(NiiImage.GetSpacing())
    Direction_arr = np.array(NiiImage.GetDirection()).reshape((3,3))
    origin_arr = np.array(NiiImage.GetOrigin())
    # print(((image_arr[:,0]-origin_arr[0])/(Direction_arr[0,0]*spacing_arr[0])).shape)
    # print("-----image_arr.shape, point_arr.shape:", image_arr.shape, point_arr.shape)
    # image_arr[:, 0] = (image_arr[:, 0] - origin_arr[0]) / (Direction_arr[0, 0] * spacing_arr[0]).astype(int)
    # image_arr[:, 1] = (image_arr[:, 1] - origin_arr[1]) / (Direction_arr[1, 1] * spacing_arr[1]).astype(int)
    # image_arr[:, 2] = (image_arr[:, 2] - origin_arr[2]) / (Direction_arr[2, 2] * spacing_arr[2]).astype(int)
    # print("image_arr.shape:", image_arr.shape, point_arr[:image_arr.shape[0], 4:5].shape)


    print("origin_arr, Direction_arr, spacing_arr:", origin_arr, Direction_arr, spacing_arr)
    image_arr_l[:, 0] = (image_arr_l[:, 0] - origin_arr[0]) / (Direction_arr[0, 0] * spacing_arr[0])
    image_arr_l[:, 1] = (image_arr_l[:, 1] - origin_arr[1]) / (Direction_arr[1, 1] * spacing_arr[1])
    image_arr_l[:, 2] = (image_arr_l[:, 2] - origin_arr[2]) / (Direction_arr[2, 2] * spacing_arr[2])

    image_arr_v[:, 0] = (image_arr_v[:, 0] - origin_arr[0]) / (Direction_arr[0, 0] * spacing_arr[0])
    image_arr_v[:, 1] = (image_arr_v[:, 1] - origin_arr[1]) / (Direction_arr[1, 1] * spacing_arr[1])
    image_arr_v[:, 2] = (image_arr_v[:, 2] - origin_arr[2]) / (Direction_arr[2, 2] * spacing_arr[2])
    
    # mask_arr = np.concatenate((image_arr[:, :3], point_arr[:image_arr.shape[0], 4:5]), axis=1).astype(int)
    # print("mask_arr:", mask_arr.shape)
    arr = np.zeros(label_arr.shape)

    # for i in range(len(mask_arr)):
    #     arr[mask_arr[i, 2], mask_arr[i, 1], mask_arr[i, 0]] = mask_arr[i, 4]

    image_arr_l, image_arr_v = image_arr_l.astype(int), image_arr_v.astype(int)
    point_arr_l, point_arr_v = point_arr_l.astype(int), point_arr_v.astype(int)
    print("image_arr_l[:, :3].max(), image_arr_v[:, :3].max():", image_arr_l[:, :3].max(), image_arr_v[:, :3].max())

    print("image_arr_l[:, 0].max(), image_arr_l[:, 1].max(), image_arr_l[:, 2].max():", image_arr_l[:, 0].max(), image_arr_l[:, 1].max(), image_arr_l[:, 2].max())
    for i in range(len(image_arr_l)):
        arr[image_arr_l[i, 2], image_arr_l[i, 1], image_arr_l[i, 0]] = point_arr_l[i, 4]

    for i in range(len(image_arr_v)):
        arr[image_arr_v[i, 2], image_arr_v[i, 1], image_arr_v[i, 0]] = point_arr_v[i, 4]


        


        
        
    
    save_Image = sitk.GetImageFromArray(arr)
    save_Image.SetDirection(label_Image.GetDirection())
    save_Image.SetOrigin(label_Image.GetOrigin())
    save_Image.SetSpacing(label_Image.GetSpacing())
    sitk.WriteImage(save_Image, save_point_path)

    # arr = reprocessing(arr)

    

    l += 1
    # mask_array = mask_array[z_min:z_max+1,y_min:y_max+1,x_min:x_max+1]
    # liver_array = label_arr[z_min:z_max+1,y_min:y_max+1,x_min:x_max+1]

    # z_l,y_l,x_l = mask_array.shape




    # save_arr = np.zeros((z,y,x))
    arr[label_arr == 0] = 0


    for i in range(1, 9):
        # new_uout[uout[:, i:i + 1, :, :] > 0.5] = i
        new_uout_tmp = np.zeros(arr.shape)
        new_uout_tmp[arr == i] = i
        label_tmp = np.zeros(label_arr.shape)
        label_tmp[label_arr == i] = i
        di = get_batch_acc(new_uout_tmp, label_tmp).numpy()
        dices[i - 1] += di
        dice += di
        # print(dices[i-1])



    list.append([name, dice/8])

    print(
        "Train dice 1: %f, Train dice 2: %f,Train dice 3: %f, Train dice 4: %f,Train dice 5: %f,Train dice 6: %f,Train dice 7: %f,Train dice 8: %f,train avg dice:%f"
        % (dices[0] / l,
           dices[1] / l,
           dices[2] / l,
           dices[3] / l,
           dices[4] / l,
           dices[5] / l,
           dices[6] / l,
           dices[7] / l,
           dice / 8))

for i in range(8):
    dice_total += dices[i]
print(dice_total/(l*8))
df = pd.DataFrame(list,columns=["name","Dice"])
# p = "./save_result/pvrcnn/test_dice.csv"
p = "./save_result/pvrcnn_final/test_dice_all.csv"
df.to_csv(p,header=None,index=False)


