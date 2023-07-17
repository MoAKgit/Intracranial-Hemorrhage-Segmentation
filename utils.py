# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 14:26:10 2023

@author: Mohammad
"""

import os
import torch
import torch.nn as nn
import numpy as np
import torchvision
from dataset import CTDataset
from torch.utils.data import DataLoader




def save_predictions_as_imgs(
    loader, 
    model, 
    criterian, 
    folder="saved_images", 
    device="cuda"
    ):
    
    losses = []
    # model.eval()
    for idx, (x, y) in enumerate(loader):
        # y = y.to(device=device)
        # x = x.to(device=device)

        x = x.to(device=device)
        with torch.no_grad():
            
            preds = model(x)
            # print(preds.shape)
            y_ = y.long()
            loss = criterian(preds, y_.to(device=device))

        losses.append(loss.detach().cpu().numpy())
        
        preds = preds>0.9
        torchvision.utils.save_image(preds.float(), f"{folder}/{idx}_pred.png")
        torchvision.utils.save_image(x, f"{folder}/{idx}_img.png")
        torchvision.utils.save_image(
            y[:, 0, :, :].unsqueeze(1), f"{folder}/{idx}_save.png"
        )
    losses = np.asarray(losses)
    loss_val = losses.mean()
    return loss_val
    # np.save(os.path.join(os.path.dirname(__file__), "..\\losses\\val_losses"), val_losses)
    
