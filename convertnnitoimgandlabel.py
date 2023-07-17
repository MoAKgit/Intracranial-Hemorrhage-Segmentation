# -*- coding: utf-8 -*-
"""
Created on Sat May 20 07:23:04 2023

@author: Mohammad
"""

import os
import nibabel as nib
import glob
import numpy as np
from matplotlib import pyplot as plt



def window(ct_scan, w_level, w_width):
    """extract bone information or subdural information
    Args:
        ct_scan (npy): 3-D array of CT-scan image
        w_level (int, optional): find it in dataset instruction.
        w_width (int, optional): find it in dataset instruction.

    Returns:
        numpy array: extracted information from original CT-scan image
    """
    
    
    
    w_min = w_level - w_width / 2
    w_max = w_level + w_width / 2
    
    # w_min = np.min(ct_scan[:])
    # w_max = np.max(ct_scan[:])
    
    num_slices = ct_scan.shape[2]
    for s in range(num_slices):
        slice_s = ct_scan[:, :, s]
        slice_s = (slice_s - w_min) * (255 / (w_max - w_min))
        slice_s[slice_s < 0] = 0
        slice_s[slice_s > 255] = 255
        
        ct_scan[:, :, s] = slice_s
    return ct_scan




if __name__ == "__main__":
    
    w_level = 40
    w_width = 100
    path_imgs = "../dataSet/data/image/*.nii"
    path_imgs = glob.glob(path_imgs)
    stride = 80
    length = 160
    
    # path_cts_neg = "../converted_dataset/train_cts_neg/"
    # path_masks_neg = "../converted_dataset/train_masks_neg/"

    # path_cts_pos = "../converted_dataset/train_cts_pos/"
    # path_masks_pos = "../converted_dataset/train_masks_pos/"
    
    # for path in path_imgs:
    #     print(path)
    #     img_path = path
    #     mask_path = path.replace("/image", "/label")
    #     imgs = nib.load(img_path)
    #     masks = nib.load(mask_path)
        
    #     imgs = imgs.get_fdata()
    #     masks = masks.get_fdata()
        
    #     imgs = window(imgs, w_level, w_width)
        
    #     for i in range(imgs.shape[2]):
    #         img = imgs[:,:,i]
    #         mask = masks[:,:,i]
    #         crop_num = 0
    #         for w in range(0, imgs.shape[0], stride):
    #             for h in range(0, imgs.shape[0], stride):
    #                 crop_img = img[w:w+length,h:h+length]
    #                 crop_mask = mask[w:w+length,h:h+length]
    #                 crop_num += 1
    #                 if crop_mask.shape[0] == length and crop_mask.shape[1] == length:

    #                     if crop_mask.sum() > 1 and crop_img.sum() > 0:
    #                         name = path.split("\\")[1]
    #                         name = name.split(".")
    #                         name = name[0] + '_{:02d}_{:02d}'.format(i,crop_num)
    #                         crop_img_path = path_cts_neg +  name
    #                         crop_mask_path = path_masks_neg +  name
    #                         # print(crop_mask_path)
    #                         np.save(crop_img_path, crop_img)
    #                         np.save(crop_mask_path, crop_mask)
    #                         # print(crop_img_path)
    #                     else:
    #                         name = path.split("\\")[1]
    #                         name = name.split(".")
    #                         name = name[0] + '_{:02d}_{:02d}'.format(i,crop_num)
    #                         crop_img_path = path_cts_pos +  name
    #                         crop_mask_path = path_masks_pos +  name
    #                         np.save(crop_img_path, crop_img)
    #                         np.save(crop_mask_path, crop_mask)
                    
      
    path_imgs_val = "../converted_dataset/val_cts/"
    path_masks_val = "../converted_dataset/val_masks/"      
    
    path_imgs = "../dataSet/val_cts/*.nii"
    path_imgs = glob.glob(path_imgs)
    for path in path_imgs:
        img_path = path
        mask_path = path.replace("/val_cts", "/val_masks")
        imgs = nib.load(img_path)
        masks = nib.load(mask_path)
        
        imgs = imgs.get_fdata()
        masks = masks.get_fdata()
        imgs = window(imgs, w_level, w_width) 
        for i in range(imgs.shape[2]):
            img = imgs[:,:,i]
            mask = masks[:,:,i]
            name = path.split("\\")[1]
            name = name.split(".")
            name = name[0] + '_{:02d}'.format(i)
            
            slice_img_path = path_imgs_val + name
            slice_mask_path = path_masks_val + name
            np.save(slice_img_path, img)
            np.save(slice_mask_path, mask)

            
            
    
    
    
    