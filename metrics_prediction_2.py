import helper
from collections import defaultdict
from helper import reverse_transform
from torch.utils.data import DataLoader
from loss import dice_loss,metric_jaccard  #this is loss
from dataset import WaterDataset
import torch.nn.functional as F
from models import UNet11, UNet, AlbuNet34,SegNet
import numpy as np
import torch
import glob
import os
import numpy as np
from pathlib import Path
from scalarmeanstd import meanstd

from transformsdata import (DualCompose,
                        ImageOnly,
                        Normalize,
                        Normalize2,                            
                        HorizontalFlip,
                        Rotate,
                        CenterCrop,
                        VerticalFlip)






def calc_loss(pred, target, metrics,phase='train', bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    pred = torch.sigmoid(pred)

    # convering tensor to numpy to remove from the computationl graph 
    if  phase=='test':
        pred=(pred >0.50).float()  #with 0.55 is a little better
        dice = dice_loss(pred, target)
        jaccard_loss = metric_jaccard(pred, target)    
        loss = bce * bce_weight + dice * (1 - bce_weight)
    
        metrics['bce'] = bce.data.cpu().numpy() * target.size(0)
        metrics['loss'] = loss.data.cpu().numpy() * target.size(0)
        metrics['dice'] =1- dice.data.cpu().numpy() * target.size(0)
        metrics['jaccard'] = 1-jaccard_loss.data.cpu().numpy() * target.size(0)
    else:
        dice = dice_loss(pred, target)
        jaccard_loss = metric_jaccard(pred, target)    
        loss = bce * bce_weight + dice * (1 - bce_weight)
        metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
        metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
        metrics['dice_loss'] += dice.data.cpu().numpy() * target.size(0)
        metrics['jaccard_loss'] += jaccard_loss.data.cpu().numpy() * target.size(0)

    return loss



def print_metrics(metrics, file, phase='train', epoch_samples=1 ):    
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples ))
    if phase=='test':
        file.write("{}".format(",".join(outputs)))
    else:          
        print("{}: {}".format(phase, ", ".join(outputs)))
        file.write("{}: {}".format(phase, ", ".join(outputs)))    ### f

 
def calc_loss_paral(pred_hr, target_hr, 
              pred_vhr_lab, target_vhr_lab,
              pred_vhr_unlab, target_vhr_unlab_up, 
              metrics, weight_loss = 0.1, bce_weight=0.5):
    
    #loss for HR model
    bce_l1 = F.binary_cross_entropy_with_logits(pred_hr, target_hr)
    pred_hr = torch.sigmoid(pred_hr) 
    
    dice_l1 = dice_loss(pred_hr, target_hr)
    loss_l1 = bce_l1 * bce_weight + dice_l1 * (1 - bce_weight)
    
    #loss for HR model label
    bce_l2 = F.binary_cross_entropy_with_logits(pred_vhr_lab, target_vhr_lab)   
    pred_vhr_lab = torch.sigmoid((pred_vhr_lab))

    dice_l2 = dice_loss(pred_vhr_lab, target_vhr_lab)
    loss_l2 = bce_l2 * bce_weight + dice_l2 * (1 - bce_weight)
    
    #loss for VHR model unlabel                        
    bce_l3 = F.binary_cross_entropy_with_logits(pred_vhr_unlab, target_vhr_unlab_up)
    pred_vhr_unlab = torch.sigmoid((pred_vhr_unlab)) 
    dice_l3 = dice_loss(pred_vhr_unlab, target_vhr_unlab_up)
    loss_l3 = bce_l3 * bce_weight + dice_l3 * (1 - bce_weight)
    
    #loss for full-network
    loss =  (loss_l1 + loss_l2 + loss_l3 * weight_loss )    
    
    metrics['loss_HR'] += loss_l1.data.cpu().numpy() * target_hr.size(0)
    metrics['loss_VHR_lab'] += loss_l2.data.cpu().numpy() * target_vhr_lab.size(0)
    metrics['loss_VHR_unlab'] += loss_l3.data.cpu().numpy() * target_vhr_unlab_up.size(0)
    metrics['loss'] += loss.data.cpu().numpy() *(target_vhr_lab.size(0)+target_vhr_unlab_up.size(0))#* target.size(0)
    metrics['loss_dice_lb'] += dice_l2.data.cpu().numpy() * target_vhr_lab.size(0)  
    metrics['loss_dice_unlab'] += dice_l3.data.cpu().numpy() * target_vhr_unlab_up.size(0)  #cambiar por hr_label
    
    return loss

def calc_loss_seq(pred_vhr_lab, target_vhr_lab,
              pred_vhr_unlab, target_vhr_unlab_up, 
              metrics, weight_loss = 0.1, bce_weight=0.5):

    
    #loss for VHR model label
    bce_l2 = F.binary_cross_entropy_with_logits(pred_vhr_lab, target_vhr_lab)
    pred_vhr_lab = torch.sigmoid((pred_vhr_lab))
    dice_l2 = dice_loss(pred_vhr_lab, target_vhr_lab)
    loss_l2 = bce_l2 * bce_weight + dice_l2 * (1 - bce_weight)
    
    #loss for VHR model unlabel                        
    bce_l3 = F.binary_cross_entropy_with_logits(pred_vhr_unlab, target_vhr_unlab_up)
    pred_vhr_unlab = torch.sigmoid((pred_vhr_unlab)) 
    dice_l3 = dice_loss(pred_vhr_unlab, target_vhr_unlab_up)
    loss_l3 = bce_l3 * bce_weight + dice_l3 * (1 - bce_weight)
    
    #loss for full-network
    loss =  (loss_l2 + loss_l3 * weight_loss)

    metrics['loss_VHR_lab'] += loss_l2.data.cpu().numpy() * target_vhr_lab.size(0)
    metrics['loss_VHR_unlab'] += loss_l3.data.cpu().numpy() * target_vhr_unlab_up.size(0)
    metrics['loss'] += loss.data.cpu().numpy()*(target_vhr_lab.size(0)+target_vhr_unlab_up.size(0))     #check   
    metrics['loss_dice_lb'] += dice_l2.data.cpu().numpy() * target_vhr_lab.size(0)   
    metrics['loss_dice_unlab'] += dice_l3.data.cpu().numpy() * target_vhr_unlab_up.size(0)      
   
    return loss

def print_metrics_paral(metrics, epoch_samples_l1, epoch_samples_l2, epoch_samples_l3, epoch_samples_loss, phase,f): # print by epoch
    outputs = []
    epoch_samples = [epoch_samples_l1, epoch_samples_l2, epoch_samples_l3, 
                     epoch_samples_loss,epoch_samples_l2, epoch_samples_l3] # l3 is unlab dice and jaccard now is from unlabel
    i = 0
    for k in metrics.keys():       #metricas(frist-mean-input.size)samples
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples[i]))
        i += 1
    print("{}: {}".format(phase, ", ".join(outputs)))
    f.write("{}: {}".format(phase, ", ".join(outputs))+ "\n")
 
def print_metrics_seq(metrics,  epoch_samples_l2, epoch_samples_l3, epoch_samples_loss, phase, f):    
    outputs = []
    epoch_samples = [epoch_samples_l2, epoch_samples_l3, 
                     epoch_samples_loss,epoch_samples_l2,#epoch_samples_l2, epoch_samples_l3, 
epoch_samples_l3]
    
    i = 0
    for k in metrics.keys():       #metricas(frist-mean-input.size)samples
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples[i]))
        i += 1
    print("{}: {}".format(phase, ", ".join(outputs)))
    f.write("{}: {}".format(phase, ", ".join(outputs))+"\n")
    
    
    
def make_loader(file_names, shuffle=False, transform=None,mode='train',batch_size=1, limit=None):
        return DataLoader(
            dataset=WaterDataset(file_names, transform=transform,mode=mode, limit=limit),
            shuffle=shuffle,            
            batch_size=batch_size,
            pin_memory=torch.cuda.is_available() 
        )

def find_metrics(train_file_names,val_file_names, test_file_names, max_values, mean_values, std_values,model,fold_out='0', fold_in='0',  name_model='UNet11', epochs='40',out_file='VHR',dataset_file='VHR' ,name_file='_VHR_60_fake' ):
                            
    outfile_path = ('predictions_{}').format(out_file)
        
    f = open(("predictions_{}/metric{}_{}_foldout{}_foldin{}_{}epochs.txt").format(out_file,name_file,name_model,fold_out, fold_in,epochs), "w+")
    f2 = open(("predictions_{}/pred_loss_test{}_{}_foldout{}_foldin{}_{}epochs.txt").format(out_file,name_file,name_model, fold_out, fold_in,epochs), "w+")
    f.write("Training mean_values:[{}], std_values:[{}] \n".format(mean_values, std_values))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #####Dilenames ###############################################

   
    print(len(test_file_names))
    #####Dataloder ###############################################

    if(dataset_file == 'VHR'):
        all_transform = DualCompose([
                CenterCrop(512),
                ImageOnly(Normalize(mean=mean_values, std=std_values))
            ])

    if(dataset_file =='HR'):
        all_transform = DualCompose([
                CenterCrop(64),
                ImageOnly(Normalize2(mean=mean_values, std=std_values))
            ])

    train_loader = make_loader(train_file_names,shuffle=True, transform=all_transform)
    val_loader = make_loader(val_file_names, transform=all_transform)
    test_loader = make_loader(test_file_names, transform=all_transform)

    dataloaders = {
    'train': train_loader, 'val': val_loader, 'test':test_loader    }

    for phase in ['train', 'val','test']:
        model.eval()
        metrics = defaultdict(float)
    ###############################  train images ###############################

        count_img=0
        input_vec= []
        labels_vec = []
        pred_vec = []
        result_dice = []
        result_jaccard = []

        
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)              
            with torch.set_grad_enabled(False):
                input_vec.append(inputs.data.cpu().numpy())
                labels_vec.append(labels.data.cpu().numpy())
                pred = model(inputs)

                loss = calc_loss(pred, labels, metrics,'test')
                
                if phase=='test':
                    print_metrics(metrics,f2, 'test')

                pred=torch.sigmoid(pred)    
                pred_vec.append(pred.data.cpu().numpy())    

                result_dice += [metrics['dice']]

                result_jaccard += [metrics['jaccard'] ]

                count_img += 1

        print(("{}_{}").format(phase,out_file))
        print('Dice = ', np.mean(result_dice), np.std(result_dice))
        print('Jaccard = ',np.mean(result_jaccard), np.std(result_jaccard),'\n')

        f.write(("{}_{}\n").format(phase,out_file))
        f.write("dice_metric: {:4f}, std: {:4f} \n".format(np.mean(result_dice),np.std(result_dice)))
        f.write("jaccard_metric: {:4f}, std: {:4f}  \n".format(np.mean(result_jaccard), np.std(result_jaccard)))    

    
        if phase=='test':      
            np.save(str(os.path.join(outfile_path,"inputs_test{}_{}_foldout{}_foldin{}_{}epochs_{}.npy".format(name_file,name_model,fold_out,fold_in,epochs,int(count_img)))), np.array(input_vec))
            np.save(str(os.path.join(outfile_path,"labels_test{}_{}_foldout{}_foldin{}_{}epochs_{}.npy".format(name_file,name_model, fold_out,fold_in,epochs,int(count_img)))), np.array(labels_vec))
            np.save(str(os.path.join(outfile_path,"pred_test{}_{}_foldout{}_foldin{}_{}epochs_{}.npy".format(name_file,name_model, fold_out,fold_in,epochs,int(count_img)))), np.array(pred_vec))



