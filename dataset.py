# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 13:52:34 2023

@author: Mohammad
"""

import os
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import nibabel as nib
import numpy as np
import torchvision
import torchvision.transforms as TF
import albumentations as albu




class CTDataset(Dataset):
    """a class to import CT scans into the framework"""

    def __init__(
        self,
        image_dir,
        mask_dir,
        augm = False,
        resize_size = (128, 128),
    ) -> None:
        """_summary_

        Args:
            image_dir (str): pass image directory.
            mask_dir (str): pass mask directory.
            Augment train images whenever augm is True (Including 
                                                        Hrizental, 
                                                        Vertical flip and 
                                                        rotation)
            
        """
        
        
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)# Choose number of training samples
        self.maskes = os.listdir(mask_dir)
        self.resize_size = resize_size
        self.augm = augm
        
        
            
        # self.preprocess = albu.Compose([albu.HorizontalFlip(p=0.5),
        #                                 albu.VerticalFlip(p=0.5),
        #                                 albu.Rotate(p=0.5),
        #                                 albu.ShiftScaleRotate(
        #                                     shift_limit=0.1,
        #                                     scale_limit=0.1,
        #                                     rotate_limit=15,
        #                                     p=0.5)
        #                                 ])
        self.preprocess = albu.Compose([albu.HorizontalFlip(p=0.5),
                                        albu.VerticalFlip(p=0.5),
                                        albu.Rotate(p=0.5)
                                        ])
            
                
        
    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> torch.Tensor:
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.maskes[index])

        image = np.load(img_path)
        mask = np.load(mask_path)
        """augmnetation"""
        if self.augm:
            augmented  = self.preprocess(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        """convert images from numpy to tensor"""
        image = transforms.Compose([transforms.ToTensor()])(image)
        mask = transforms.Compose([transforms.ToTensor()])(mask)

        # image = torch.unsqueeze(image, dim=1).float()
        # mask = torch.unsqueeze(mask, dim=1).float()

        """Normalize values between 0 and 1"""
        image = image / 255.0
        mask = torch.round(mask / 255.0)
        # NOTE: change below code proportionally to your dataset and your goal
        # mask = torch.cat((mask, 1 - mask), dim=0)
        
        return image.float(), mask
