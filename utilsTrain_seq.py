from datetime import datetime
from pathlib import Path
import random
import numpy as np
import time
import torch
import copy      

from collections import defaultdict
import torch.nn.functional as F
from loss import dice_loss,metric_jaccard
import os
import torch.nn as nn


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
if (device=='cpu'):
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')


#def cuda(x):
#    return x.cuda(async=True) if torch.cuda.is_available() else x


def calc_loss(pred_hr_lab, target_hr_lab,
              pred_hr_unlab, target_hr_unlab_up, 
              metrics, weight_loss = 0.1, bce_weight=0.5):

    
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
    loss =  (loss_l2 + loss_l3 * weight_loss)

    pred_hr_lab=(pred_hr_lab >0.50).float()  #with 0.55 is a little better
    pred_hr_lab=(pred_hr_unlab >0.50).float() 
    
    jaccard_HR_lb = metric_jaccard(pred_hr_lab, target_hr_lab)
    jaccard_HR_unlab = metric_jaccard(pred_hr_unlab, target_hr_unlab_up )

    metrics['loss_lab'] += loss_l2.data.cpu().numpy() * target_hr_lab.size(0)
    metrics['loss_unlab'] += loss_l3.data.cpu().numpy() * target_hr_unlab_up.size(0)
    metrics['loss'] += loss.data.cpu().numpy()*(target_hr_lab.size(0)+target_hr_unlab_up.size(0))     #check
    
    metrics['loss_dice_lb'] += dice_l2.data.cpu().numpy() * target_hr_lab.size(0)  
    metrics['jaccard_lb'] += jaccard_HR_lb.data.cpu().numpy() * target_hr_lab.size(0) 
    
    metrics['loss_dice_unlab'] += dice_l3.data.cpu().numpy() * target_hr_unlab_up.size(0)      
    metrics['jaccard_unlab'] += jaccard_HR_unlab.data.cpu().numpy() * target_hr_unlab_up.size(0) 
   
    return loss

def print_metrics(metrics,  epoch_samples_l2, epoch_samples_l3, epoch_samples_loss, phase, f):    
    outputs = []
    epoch_samples = [epoch_samples_l2, epoch_samples_l3, 
                     epoch_samples_loss,epoch_samples_l2,epoch_samples_l2, epoch_samples_l3, epoch_samples_l3]
    
    i = 0
    for k in metrics.keys():       #metricas(frist-mean-input.size)samples
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples[i]))
        i += 1
    print("{}: {}".format(phase, ", ".join(outputs)))
    f.write("{}: {}".format(phase, ", ".join(outputs)))
##______________________________________________________

def train_model(name_file,model_LR, model_HR, optimizer, scheduler,dataloaders_HR_lab,
                dataloaders_HR_unlab,name_model_HR='UNet11',n_steps=15,num_epochs=25):

    #finally_path = Path('logs')
    #final_layer = finally_path/'mapping'/'final_layer'
    #final_layer = Path('logs/mapping/final_layer')
    #final_layer_npy_outpath = 'final_layer_{}.npy'



    #hist_lst = []
    best_model_wts = copy.deepcopy(model_HR.state_dict())
    best_loss = 1e10

        #f = open("history_model1.txt", "w+")       #--------------------------------------------------------
    #f = open("history_model1_100.txt", "w+")  
    f = open("history_seq/history_model{}_{}.txt".format(name_file,name_model), "w+")  
         #--------------------------------------------------------
    upsample = nn.Upsample(scale_factor= 8, mode='bicubic')
    #------------------------------------------------------------
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        f.write('Epoch {}/{}'.format(str(epoch), str(num_epochs - 1)) + "\n")
        print('-' * 10)
        f.write(str('-' * 10) + "\n") 
        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
                    f.write("LR" +  str(param_group['lr']) + "\n") 

                model_HR.train()  # Set model to training mode
            else:
                    model_HR.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples_l2 = epoch_samples_l3 = epoch_samples_loss = 0
            
            #itr = 0
            size_dataloader_HR_unlab=len(dataloaders_HR_unlab[phase])
            print("dataloader_unlb:",size_dataloader_HR_unlab) 
            f.write("dataloader_unlb:" + str(size_dataloader_HR_unlab) + "\n")  
            
            for i in range(n_steps): #input_HR_lab, labels_HR_lab in dataloaders_HR_lab[phase]:
                input_HR_lab, labels_HR_lab= next(iter ( dataloaders_HR_lab[phase]))           
                ##if phase == "train": #si pongo train pensaria q tiene label
                input_HR_unlab, labels = next(iter(dataloaders_HR_unlab[phase]))  #propocional batches ?
                ##else:
                ##   input_HR_unlab, labels = next(iter(dataloaders_HR_unlab["unlb_val"])) 
                    
                input_HR_lab = input_HR_lab.to(device)
                labels_HR_lab = labels_HR_lab.to(device)
                input_HR_unlab = input_HR_unlab.to(device)
                
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    #outputs, conv1 = model(inputs)
                    pred_HR_lab = model_HR(input_HR_lab)
                    inputs_HR_unlab_ds = nn.functional.interpolate(input_HR_unlab, scale_factor= 0.125, mode='bicubic')
                    
                    pred_HR_unlab_ds = model_LR(inputs_HR_unlab_ds)
                    target_HR_unlab_us = upsample(pred_HR_unlab_ds)
                    ##########################################################
                    #target_HR_unlab_us = torch.div(target_HR_unlab_us,4) ###############    check
                    target_HR_unlab_us = torch.sigmoid(target_HR_unlab_us)    ###############    check??
                    pred_HR_unlab = model_HR(input_HR_unlab)
    ######################## saving batch prediction _final_layer 
                    #if itr == 0:
    
    #final_layer_data = outputs.data.cpu().numpy()
                        #outpath_final_layer=str(os.path.join(final_layer,final_layer_npy_outpath.format(int(epoch))))

                        #np.save(final_layer/"final_layer_" + str(epoch) + ".npy" , final_layer_data)
                        #np.save(outpath_final_layer, final_layer_data)
                        
                    #saving conv1_data
                    #conv1_data = conv1.data.cpu().numpy()
                    #np.save(str(conv_path) + "conv1_" + str(epoch) + "_" + str(itr) + ".npy", conv1_data )
                   #itr = itr + 1
                    #print(itr)
                    
    ########################  
                    loss = calc_loss(pred_HR_lab, labels_HR_lab,
                                     pred_HR_unlab, target_HR_unlab_us, metrics)
                    #print("I am here 2")
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples_loss += input_HR_lab.size(0) + input_HR_unlab.size(0)  # ctd of samples in a batch
                epoch_samples_l2 += input_HR_lab.size(0) 
                epoch_samples_l3 += input_HR_unlab.size(0)
                #print(epoch_samples)
            #print("I am here 3")
            print_metrics(metrics, epoch_samples_l2, epoch_samples_l3, epoch_samples_loss, phase, f)
            epoch_loss = metrics['loss'] / epoch_samples_loss

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                f.write("saving best model" + "\n")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model_HR.state_dict())
        #hist_lst.append(metrics)
        #with open(final_layer/'loss.txt', "w") as file:
         #   file.write(str(hist_lst))
        
        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        f.write('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60) + "\n")
    print('Best val loss: {:4f}'.format(best_loss))
    f.write('Best val loss: {:4f}'.format(best_loss)  + "\n")
    f.close()

    # load best model weights
    model_HR.load_state_dict(best_model_wts)
    return model_HR