# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 13:42:03 2023

@author: Mohammad
"""

import torch
import numpy as np
import os
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import *
import argparse
from dataset import *
from torch.utils.data import DataLoader
from itertools import cycle
from losses import *
from utils import *
# from loss_funcs import DiceLoss, IoULoss, Combined_Loss 



def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    parser = argparse.ArgumentParser(description='Segmentation')
    
    parser.add_argument("--max_epochs",dest= 'max_epochs', default= 1000) 
    
    parser.add_argument("--batch_size",dest= 'batch_size', default= 2) 
    
    parser.add_argument("--learning_rate",dest= 'learning_rate', default= 1e-4) 
    
    return parser.parse_args()


drive_path = 'D:\Dataset/medical data/Hemorrhage/'

TRAIN_IMG_DIR_POS = drive_path + "converted_dataset/train_cts_pos"
TRAIN_IMG_DIR_NEG = drive_path + "converted_dataset/train_cts_neg"
TRAIN_MASK_DIR_POS = drive_path + "converted_dataset/train_masks_pos"
TRAIN_MASK_DIR_NEG = drive_path + "converted_dataset/train_masks_neg"

# VAL_IMG_DIR = TRAIN_IMG_DIR_NEG
# VAL_MASK_DIR = TRAIN_MASK_DIR_NEG

VAL_IMG_DIR = drive_path + "converted_dataset/val_cts"
VAL_MASK_DIR = drive_path + "converted_dataset/val_masks"

# VAL_IMG_DIR = drive_path + "converted_dataset/train_cts_neg"
# VAL_MASK_DIR = drive_path + "converted_dataset/train_masks_neg"




if __name__ == '__main__':
    args = arg_parse()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    dataset_neg = CTDataset(TRAIN_IMG_DIR_NEG, TRAIN_MASK_DIR_NEG, augm = True)
    dataset_pos = CTDataset(TRAIN_IMG_DIR_POS, TRAIN_MASK_DIR_POS, augm = True)
    
    dataset_val = CTDataset(VAL_IMG_DIR, VAL_MASK_DIR)
    
    train_loader_neg = DataLoader(dataset_neg,
                                  batch_size= args.batch_size, 
                                  pin_memory= False,
                                  shuffle = False ,
                                  num_workers= 2)
    
    train_loader_pos = DataLoader(dataset_pos,
                                  batch_size= args.batch_size, 
                                  pin_memory= False,
                                  shuffle = False ,
                                  num_workers= 2)
    
    val_loader = DataLoader(dataset_val,
                                  batch_size= 1, 
                                  pin_memory= False,
                                  shuffle = False ,
                                  num_workers= 2)
    
    model = UNET2D(in_channels=1, out_channels=1).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterian = DiceLoss()
    for epoch in range(args.max_epochs):
        losses_trn = []
        loop = tqdm(zip(train_loader_pos, cycle(train_loader_neg)) )
        for idx, (item1, item2) in enumerate(loop):
            
            data_pos,targets_pos =  item1
            data_neg,targets_neg =  item2
            
            data_pos = data_pos.to(device=device)
            targets_pos = targets_pos.to(device=device)
            
            data_neg = data_neg.to(device=device)
            targets_neg = targets_neg.to(device=device)
            
            data = torch.cat( (data_pos,data_neg ), 0)
            targets = torch.cat( (targets_pos, targets_neg), 0)
            
            pred = model(data)
            loss = criterian(pred, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(data.shape)
            losses_trn.append(loss.detach().cpu().numpy())
        losses_trn = np.asarray(losses_trn).mean()
        print("Epoch: {}, loss train: {:05f}".format(epoch, losses_trn))
        
        if epoch%1 == 0:
            loss_val = save_predictions_as_imgs(val_loader, 
                                     model, 
                                     criterian, 
                                     folder="saved_images", 
                                     device= device )
            print("loss val: {:05f}".format(loss_val))
            
        










