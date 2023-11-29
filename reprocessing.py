import cv2
import numpy as np
import SimpleITK as sitk
import numpy as np
from hole_filling import hole_filling
def reprocessing(label):


    flag_5_8 = True
    flag_2_3 = True
    for i in range (label.shape[0]):

        l = label[i,:,:]
        num_2 = np.sum(l == 2)
        num_3 = np.sum(l == 3)
        num_5 = np.sum(l == 5)
        num_8 = np.sum(l == 8)
        num_6 = np.sum(l == 6)
        num_7 = np.sum(l == 7)

        if  flag_2_3:
            if  num_3 >= num_2:
                l[l == 2] = 3
            else:
                flag_2_3 = False
                l[l == 3] = 2
        else:
            l[l == 3] = 2

        if flag_5_8:
            if  num_5 >= num_8:
                l[l == 8] = 5
            else:
                l[l == 5] = 8
                l[l == 6] = 7
                flag_5_8 = False
            if  num_6 >= num_7:
                l[l == 7] = 6
            else:
                l[l == 6] = 7
                l[l == 5] = 8
                flag_5_8 = False
        else:
            l[l == 5] = 8
            l[l == 6] = 7





    # num = 10000
    #
    # new_label_1 = np.zeros(label.shape)
    # new_label_1[label == 1] =1
    # new_label_1 = hole_filling(new_label_1,0,num,False)
    # #
    # # #
    # new_label_2 = np.zeros(label.shape)
    # new_label_2[label == 2] =1
    # new_label_2 = hole_filling(new_label_2,0,num,False)
    # #
    # # #
    # new_label_3 = np.zeros(label.shape)
    # new_label_3[label == 3] =1
    # new_label_3 = hole_filling(new_label_3,0,num,False)
    # #
    # # # new_label_3_nii = sitk.GetImageFromArray(new_label_3)
    # # # new_label_3_nii = sitk.BinaryFillhole(new_label_3_nii)
    # # # new_label_3 = sitk.GetArrayFromImage(new_label_3)
    # # #
    # new_label_4 = np.zeros(label.shape)
    # new_label_4[label == 4] =1
    # new_label_4 = hole_filling(new_label_4,0,num,False)
    # #
    # # #
    # new_label_5 = np.zeros(label.shape)
    # new_label_5[label == 5] =1
    # new_label_5 = hole_filling(new_label_5,0,num,False)
    # #
    # new_label_6 = np.zeros(label.shape)
    # new_label_6[label == 6] =1
    # new_label_6 = hole_filling(new_label_6,0,num,False)
    # #
    #
    # # #
    # new_label_7 = np.zeros(label.shape)
    # new_label_7[label == 7] =1
    # new_label_7 = hole_filling(new_label_7,0,num,False)
    # #
    #
    # # #
    # new_label_8 = np.zeros(label.shape)
    # new_label_8[label == 8] =1
    # new_label_8 = hole_filling(new_label_8,0,num,False)
    # #
    #
    # # #
    # label[new_label_1 == 1] = 1
    # label[new_label_2 == 1] = 2
    # label[new_label_3 == 1] = 3
    # label[new_label_4 == 1] = 4
    # label[new_label_5 == 1] = 5
    # label[new_label_6 == 1] = 6
    # label[new_label_7 == 1] = 7
    # label[new_label_8 == 1] = 8
    #
    # #

    return label








