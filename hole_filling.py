from scipy.ndimage.morphology import binary_fill_holes
import numpy as np
from scipy import ndimage
import nibabel as nib
from skimage.measure import label
import matplotlib.pyplot as plt


def hole_filling(bw, hole_min, hole_max, fill_2d=True):
    bw = bw > 0
    if len(bw.shape) == 2:
        background_lab = label(~bw, connectivity=1)
        fill_out = np.copy(background_lab)
        component_sizes = np.bincount(background_lab.ravel())
        too_big = component_sizes > hole_max
        too_big_mask = too_big[background_lab]
        fill_out[too_big_mask] = 0
        too_small = component_sizes < hole_min
        too_small_mask = too_small[background_lab]
        fill_out[too_small_mask] = 0
    elif len(bw.shape) == 3:
        if fill_2d:
            fill_out = np.zeros_like(bw)
            for zz in range(bw.shape[1]):
                background_lab = label(~bw[:, zz, :], connectivity=1)   # 1表示4连通， ~bw[zz, :, :]1变为0， 0变为1
                # 标记背景和孔洞， target区域标记为0
                out = np.copy(background_lab)
                # plt.imshow(bw[:, :, 87])
                # plt.show()
                component_sizes = np.bincount(background_lab.ravel())
                # 求各个类别的个数
                too_big = component_sizes > hole_max
                too_big_mask = too_big[background_lab]

                out[too_big_mask] = 0
                too_small = component_sizes < hole_min
                too_small_mask = too_small[background_lab]
                out[too_small_mask] = 0
                # 大于最大孔洞和小于最小孔洞的都标记为0， 所以背景部分被标记为0了。只剩下符合规则的孔洞
                fill_out[:, zz, :] = out
                # 只有符合规则的孔洞区域是1， 背景及target都是0
        else:
            background_lab = label(~bw, connectivity=1)
            fill_out = np.copy(background_lab)
            component_sizes = np.bincount(background_lab.ravel())
            too_big = component_sizes > hole_max
            too_big_mask = too_big[background_lab]
            fill_out[too_big_mask] = 0
            too_small = component_sizes < hole_min
            too_small_mask = too_small[background_lab]
            fill_out[too_small_mask] = 0
    else:
        print('error')
        return

    return np.logical_or(bw, fill_out)  # 或运算，孔洞的地方是1，原来target的地方也是1
