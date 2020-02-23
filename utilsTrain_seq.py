from datetime import datetime
from pathlib import Path
import random
import numpy as np
import time
import torch
import copy      

from collections import defaultdict
import torch.nn.functional as F
import os
import torch.nn as nn
from loss import dice_loss,metric_jaccard
from metrics_prediction_2 import calc_loss_seq,print_metrics_seq 



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
if (device=='cpu'):
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

def train_model(out_file,name_file_VHR,model_HR, model_VHR, optimizer, scheduler,dataloaders_VHR_lab,
                dataloaders_VHR_unlab,fold_out,fold_in,name_model_VHR='UNet11',n_steps=15,num_epochs=25):

    best_model_wts = copy.deepcopy(model_VHR.state_dict())
    best_loss = 1e10
 
    f = open("history_{}/history_model{}_{}_foldout{}_foldin{}_{}epochs.txt".format(out_file,name_file_VHR,name_model_VHR,fold_out,fold_in,num_epochs), "w+")  
   #--------------------------------------------------------
    upsample = nn.Upsample(scale_factor= 8, mode='bicubic')
    #------------------------------------------------------------
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        f.write('Epoch {}/{}'.format(str(epoch), str(num_epochs - 1)) + "\n")
        print('-' * 10)
        f.write(str('-' * 10) + "\n") 
        since = time.time()

        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
                    f.write("LR" +  str(param_group['lr']) + "\n") 

                model_VHR.train()  # Set model to training mode
            else:
                model_VHR.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples_l2 = epoch_samples_l3 = epoch_samples_loss = 0

            
            for i in range(n_steps): 
                input_VHR_lab, labels_VHR_lab= next(iter (dataloaders_VHR_lab[phase]))           
                input_VHR_unlab, labels = next(iter(dataloaders_VHR_unlab[phase]))  
                    
                input_VHR_lab = input_VHR_lab.to(device)
                labels_VHR_lab = labels_VHR_lab.to(device)
                input_VHR_unlab = input_VHR_unlab.to(device)                
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    pred_VHR_lab = model_VHR(input_VHR_lab)
                    inputs_VHR_unlab_ds = nn.functional.interpolate(input_VHR_unlab, scale_factor= 0.125, mode='bicubic')
                    
                    pred_VHR_unlab_ds = model_HR(inputs_VHR_unlab_ds)
                    target_VHR_unlab_us = upsample(pred_VHR_unlab_ds)
                    target_VHR_unlab_us = torch.sigmoid(target_VHR_unlab_us)    
                    pred_VHR_unlab = model_VHR(input_VHR_unlab)

                    loss = calc_loss_seq(pred_VHR_lab, labels_VHR_lab,
                                     pred_VHR_unlab, target_VHR_unlab_us, metrics)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                epoch_samples_loss += input_VHR_lab.size(0) + input_VHR_unlab.size(0) 
                epoch_samples_l2 += input_VHR_lab.size(0) 
                epoch_samples_l3 += input_VHR_unlab.size(0)
    
            print_metrics_seq(metrics, epoch_samples_l2, epoch_samples_l3, epoch_samples_loss, phase, f)
            epoch_loss = metrics['loss'] / epoch_samples_loss

            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                f.write("saving best model" + "\n")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model_VHR.state_dict())

        
        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        f.write('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60) + "\n")
    print('Best val loss: {:4f}'.format(best_loss))
    f.write('Best val loss: {:4f}'.format(best_loss)  + "\n")
    f.close()

   
    model_VHR.load_state_dict(best_model_wts)
    return model_VHR