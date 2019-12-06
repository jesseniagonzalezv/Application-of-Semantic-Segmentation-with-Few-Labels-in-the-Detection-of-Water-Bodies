from datetime import datetime
from pathlib import Path

import random #
import numpy as np#
import time 
import torch 
import copy      

from collections import defaultdict
import torch.nn.functional as F
from loss import dice_loss,metric_jaccard
import os
import torch.nn as nn

import cv2
from skimage.io import imread, show




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device',device)
if (device=='cpu'):
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')




def upsampling_HR(imagen_LR,scale,keepdims=False):

    lr_img = imagen_LR.transpose((1, 2, 0))  
    
    #print(np.shape(lr_img))
    #plt.imshow(lr_img[:,:,0]/lr_img.max())# ·HR
    #plt.show()
    lr_img_dims = (lr_img.shape[1], lr_img.shape[0])
    #print(hr_img_dims)

        # Downsample image 2.8/10 x   ----
    hr_img_x = cv2.resize(lr_img, (0, 0), fx=scale, fy=scale,
                               interpolation=cv2.INTER_CUBIC)

    if keepdims==True:
        hr_img_x = cv2.resize(hr_img_x, hr_img_dims,
                                   interpolation=cv2.INTER_CUBIC)
    else:
        hr_img_x = cv2.resize(hr_img_x, (512,512),
                                   interpolation=cv2.INTER_CUBIC)
    #cv2.imwrite(os.path.join(lr_image_dir + "/x", filename), lr_img_x)
    #plt.imshow(hr_img_x/hr_img_x.max()) #HR
    #plt.show()
    #print(hr_img_x.shape)
  
    return hr_img_x




def downsampling_LR(imagen_HR,scale,keepdims=False):

    # Read HR image
    #hr_image_dir='/home/jgonzalez/Test_2019/Test_network/'
    #lr_image_dir='/home/jgonzalez/Test_2019/Test_network/imagetiffLR'

    # Create LR image dirs
    #os.makedirs(lr_image_dir + "/x", exist_ok=True)
    
    #########################################################
    hr_img = imagen_HR.transpose((1, 2, 0))
    #########################################################
    #hr_img = np.load(imagen_HR).transpose((1, 2, 0))  #
    #print('hr',np.shape(hr_img))
    #plt.imshow(hr_img/hr_img.max())
    #plt.show()
    hr_img_dims = (hr_img.shape[1], hr_img.shape[0])
    #print(hr_img_dims)

        # Downsample image 2.8/10 x   ----
    lr_img_x = cv2.resize(hr_img, (0, 0), fx=scale, fy=scale,
                               interpolation=cv2.INTER_CUBIC)

    if keepdims==True:
        lr_img_x = cv2.resize(lr_img_x, hr_img_dims,
                                   interpolation=cv2.INTER_CUBIC)
    else:
        lr_img_x = cv2.resize(lr_img_x, (64,64),
                                   interpolation=cv2.INTER_CUBIC)
    #cv2.imwrite(os.path.join(lr_image_dir + "/x", filename), lr_img_x)
    #plt.imshow(lr_img_x/lr_img_x.max())
    #plt.show()
    #print('lr',lr_img_x.shape)
  
    return lr_img_x


def calc_loss(pred_lr, target_lr, 
              pred_hr_lab, target_hr_lab,
              pred_hr_unlab, target_hr_unlab_up, 
              metrics, weight_loss = 0.1, bce_weight=0.5):
    
    #loss for LR model
    bce_l1 = F.binary_cross_entropy_with_logits(pred_lr, target_lr)
    pred_lr = torch.sigmoid(pred_lr) 
    dice_l1 = dice_loss(pred_lr, target_lr)
    loss_l1 = bce_l1 * bce_weight + dice_l1 * (1 - bce_weight)
    
    #loss for HR model label
    bce_l2 = F.binary_cross_entropy_with_logits(pred_hr_lab, target_hr_lab)
    pred_hr_lab = torch.sigmoid((pred_hr_lab))
    dice_l2 = dice_loss(pred_hr_lab, target_hr_lab)
    loss_l2 = bce_l2 * bce_weight + dice_l2 * (1 - bce_weight)
    
    #loss for HR model unlabel                        
    bce_l3 = F.binary_cross_entropy_with_logits(pred_hr_unlab, target_hr_unlab_up)
    pred_hr_unlab = torch.sigmoid((pred_hr_unlab)) 
    dice_l3 = dice_loss(pred_hr_unlab, target_hr_unlab_up)
    loss_l3 = bce_l3 * bce_weight + dice_l3 * (1 - bce_weight)
    
    #loss for full-network
    loss =  (loss_l1 + loss_l2 + loss_l3 * weight_loss )
    
    #pred_hr_lab=(pred_hr_lab >0.50).float()  #with 0.55 is a little better
    #pred_hr_unlab=(pred_hr_unlab >0.50).float() 
    
    #jaccard_HR_lb = metric_jaccard(pred_hr_lab, target_hr_lab)
    #jaccard_HR_unlab = metric_jaccard(pred_hr_unlab, target_hr_unlab_up )
    
    metrics['loss_LR'] += loss_l1.data.cpu().numpy() * target_lr.size(0)
    metrics['loss_HRlab'] += loss_l2.data.cpu().numpy() * target_hr_lab.size(0)
    metrics['loss_HRunlab'] += loss_l3.data.cpu().numpy() * target_hr_unlab_up.size(0)
    metrics['loss'] += loss.data.cpu().numpy() *(target_hr_lab.size(0)+target_hr_unlab_up.size(0))#* target.size(0)
    metrics['loss_dice_lb'] += dice_l2.data.cpu().numpy() * target_hr_lab.size(0)  
    #metrics['jaccard_lb'] += jaccard_HR_lb.data.cpu().numpy() * target_hr_lab.size(0) 
    
    metrics['loss_dice_unlab'] += dice_l3.data.cpu().numpy() * target_hr_unlab_up.size(0)  #cambiar por hr_label
   # metrics['jaccard_unlab'] += jaccard_HR_unlab.data.cpu().numpy() * target_hr_unlab_up.size(0) #cambiar por hr_label
    
    
    return loss

def print_metrics(metrics, epoch_samples_l1, epoch_samples_l2, epoch_samples_l3, epoch_samples_loss, phase,f): # print by epoch
    outputs = []
    epoch_samples = [epoch_samples_l1, epoch_samples_l2, epoch_samples_l3, 
                     epoch_samples_loss,epoch_samples_l2, epoch_samples_l3] # l3 is unlab dice and jaccard now is from unlabel
    i = 0
    for k in metrics.keys():       #metricas(frist-mean-input.size)samples
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples[i]))
        i += 1
    print("{}: {}".format(phase, ", ".join(outputs)))
    f.write("{}: {}".format(phase, ", ".join(outputs))+ "\n")
##_______________________________________________________________________________________________

def train_model(name_file_HR,model_LR, model_HR, optimizer_ft, scheduler,dataloaders_HR_lab,
                dataloaders_HR_unlab,dataloaders_LR, fold_out,name_model_HR='UNet11',n_steps=15, num_epochs=25):
    #finally_path = Path('logs')
    #f = open('logs/history_model1.txt',"w+")
    #f = open("history_model.txt", "w+")
    #f = open("history_model_100.txt", "w+")
    #f = open("history_model_400.txt", "w+")
    #f = open("history_model_disti_fake.txt", "w+")
    


    best_loss = 1e10
    
    f = open("history_paral/history_model{}_{}_fold{}.txt".format(name_file_HR,name_model_HR,fold_out), "w+")  
     #--------------------------------------------------------
    #upsample = nn.Upsample(scale_factor= 8,  mode='bicubic') 
    #upsample = nn.Upsample(size=64,scale_factor= 3.57,  mode='bicubic') 

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
                for param_group in optimizer_ft.param_groups:
                    f.write("LR" +  str(param_group['lr']) + "\n") 

                model_LR.train()  
                model_HR.train()  # Set model to training mode
            else: 
                model_LR.eval()
                model_HR.eval()
                
            metrics = defaultdict(float)
            epoch_samples_l1 = epoch_samples_l2 = epoch_samples_l3 = epoch_samples_loss = 0
            
            print("dataloader_lb:",len(dataloaders_HR_lab[phase]) )
            f.write("dataloader:" + str(len(dataloaders_HR_lab[phase])) + "\n")     
    
            for i in range(n_steps):  
               
                #print("step_number:", i)

                
                #Load input data------------------------------------------
                input_LR, labels_LR= next(iter(dataloaders_LR[phase]))
                


                #print(input_LR.type())
                #print(labels_LR.type())
                input_HR_lab, labels_HR_lab = next(iter(dataloaders_HR_lab[phase]))
                #print(phase)
                #if phase == "train":
                input_HR_unlab, labels = next(iter(dataloaders_HR_unlab[phase]))  #propocional batches ?
                #else:
                    #input_HR_unlab, labels = next(iter(dataloaders_HR_unlab["unlb_val"])) 
                    
                
                
                

                input_LR = input_LR.to(device)
                labels_LR = labels_LR.to(device)
                input_HR_lab = input_HR_lab.to(device)
                labels_HR_lab = labels_HR_lab.to(device)
                ###############################
                #input_HR_unlab = input_HR_unlab.to(device)
                input_HR_unlab = input_HR_unlab.numpy()
                ###############################


                optimizer_ft.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    #outputs-----------------------------------
                    ############################### fake LR
                    #input_LR = nn.functional.interpolate(input_LR, scale_factor= 0.125, mode='bicubic')                 
                   # labels_LR = nn.functional.interpolate(labels_LR, scale_factor= 0.125, mode='bicubic')      ############################### end fake LR
                    pred_LR = model_LR(input_LR)
                    pred_HR_lab = model_HR(input_HR_lab)
                    ###############################
                    #inputs_HR_unlab_ds = nn.functional.interpolate(input_HR_unlab, scale_factor= 0.125, mode='bicubic')
                    input_HR_unlab_ds=[]
                    for idx in enumerate((input_HR_unlab)):
                        print(idx)
                        lr_img_x=downsampling_LR(input_HR_unlab[idx],scale=2.8/10)
                        lr_img_x=torch.from_numpy(np.moveaxis(lr_img_x, -1, 0)).float().to(device)

                        #input_HR_unlab_ds.append(torch.from_numpy(np.moveaxis(lr_img_x, -1, 0)))
                    ###############################
                    pred_HR_unlab_ds=model_LR(((lr_img_x)).float().to(device))
                    #pred_HR_unlab_ds = model_LR(inputs_HR_unlab_ds)
                    
                    #target_HR_unlab_us = upsample(pred_HR_unlab_ds)
                    target_HR_unlab_us=[]
                    for idx in range(len(pred_HR_unlab_ds)):
                        pred_LR=pred_HR_unlab_ds[idx].to('cpu').detach().numpy()                    
                        target_HR_unlab_us.append((upsampling_HR(pred_LR,scale=10/2.8)))

                    #target_HR_unlab_us=(torch.from_numpy(np.asarray(target_HR_unlab_us))).float().to(device)
                    target_HR_unlab_us=(torch.from_numpy((target_HR_unlab_us))).float().to(device)
                    
                    #target_HR_unlab_us[target_HR_unlab_us > 1] = 1
                    #target_HR_unlab_us[target_HR_unlab_us < 0] = 0
                    #print('target_out',target_HR_unlab_us)
                    #target_HR_unlab_us = torch.div(target_HR_unlab_us,2)  ###############    check??
                    #print('target_out_div',target_HR_unlab_us)

                    target_HR_unlab_us = torch.sigmoid(target_HR_unlab_us)    ###############    check??
                    #print('target_out_sig',target_HR_unlab_us)                
                    
                    #print('target_HR',labels_HR_lab)
                    

                    pred_HR_unlab = model_HR(input_HR_unlab)
                    
                    
                    loss = calc_loss(pred_LR, labels_LR, 
                                     pred_HR_lab, labels_HR_lab,
                                     pred_HR_unlab, target_HR_unlab_us, metrics)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer_ft.step()  
                #________________________________________________________________________
                # statistics
                epoch_samples_loss += input_LR.size(0) + input_HR_lab.size(0) + input_HR_unlab.size(0)  # ctd of samples in a batch
                epoch_samples_l1 += input_LR.size(0) 
                epoch_samples_l2 += input_HR_lab.size(0) 
                epoch_samples_l3 += input_HR_unlab.size(0)
            print_metrics(metrics, epoch_samples_l1, epoch_samples_l2, epoch_samples_l3, epoch_samples_loss, phase,f)
            
            epoch_loss = metrics['loss'] / epoch_samples_loss
            
            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                f.write("saving best model" + "\n")

                best_loss = epoch_loss
                best_model_wts_LR = copy.deepcopy(model_LR.state_dict())
                best_model_wts_HR =  copy.deepcopy(model_HR.state_dict())
                


                
        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        f.write('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60) + "\n")

    print('Best val loss: {:4f}'.format(best_loss))
    f.write('Best val loss: {:4f}'.format(best_loss)  + "\n")
    f.close()

    # load best model weights
    model_LR.load_state_dict(best_model_wts_LR)
    model_HR.load_state_dict(best_model_wts_HR)
    
    return model_LR, model_HR