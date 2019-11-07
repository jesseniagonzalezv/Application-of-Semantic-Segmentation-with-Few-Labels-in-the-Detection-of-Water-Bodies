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
#from plotting import plotting_figures

from transformsdata import (DualCompose,
                        ImageOnly,
                        Normalize,
                        Normalize2,                            
                        HorizontalFlip,
                        Rotate,
                        CenterCrop,
                        VerticalFlip)


#PATH = 'logs/mapping/HR_model_temp2.pth'
#PATH = 'logs/mapping/HR_model_temp2_100.pth'
#PATH = 'logs/mapping/HR_model_temp2_400.pth'
#PATH = 'logs/mapping/HR_model_fake.pth'
######################### setting all data paths#######

#train_path= "data_HR/train/images" ##
#val_path= "data_HR/val/images" ##
#test_path="data_HR/test/images" #

#train_path= "data_HR/train_100/images" ##
#val_path= "data_HR/val_100/images" ##
#test_path="data_HR/test/images" #

#train_path= "data_HR/train_400/images" ##
#val_path= "data_HR/val_400/images" ##
#test_path="data_HR/test/images" #

#train_path= "data_HR/dist_per/train_HR/images" ##
#val_path= "data_HR/dist_per/val_HR/images" ##
#test_path="data_HR/test/images" #

###################################

#f = open("predictions/metric_HR_paral_916.txt", "w+")
#f = open("predictions/metric_HR_paral_100.txt", "w+")
#f = open("predictions/metric_HR_paral_400.txt", "w+")
#f = open("predictions/metric_HR_paral_fake_116.txt", "w+")


#f2 = open("predictions/pred_loss_dist_paral_916.txt", "w+")
#f2 = open("predictions/pred_loss_dist_paral_100.txt", "w+")
#f2 = open("predictions/pred_loss_dist_paral_400.txt", "w+")
#f2 = open("predictions/pred_loss_dist_paral_fake.txt", "w+")




def make_loader(file_names, shuffle=False, transform=None,mode='train', limit=None):  #mode ='train' with labels
    return DataLoader(
        dataset=WaterDataset(file_names, transform=transform, limit=limit),
        shuffle=shuffle,            
        batch_size=1,
        pin_memory=torch.cuda.is_available() #### in process arguments
    )


def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    
    #pred=(pred >0).float()   #the same result with the next 2 lines
    pred = torch.sigmoid(pred)
    pred=(pred >0.50).float()  #with 0.55 is a little better

    dice = dice_loss(pred, target)
    
    jaccard_loss = metric_jaccard(pred, target)
    
    loss = bce * bce_weight + dice * (1 - bce_weight)
    # convering tensor to numpy to remove from the computationl graph 
    metrics['bce'] = bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] =1- dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] = loss.data.cpu().numpy() * target.size(0)
    metrics['jaccard'] = 1-jaccard_loss.data.cpu().numpy() * target.size(0)
    return loss

def print_metrics(metrics, epoch_samples, file):    
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] ))#/ epoch_samples))

    #print("{}".format(", ".join(outputs)))
    file.write("{}".format(",".join(outputs)))

    
def find_metrics(train_file_names,val_file_names,max_values, mean_values, std_values, name_model='UNet11', out_file='HR',dataset_file='HR' ,name_file='_HR_60_fake' ):
                            #LR , dist_paral, dist_sec   #LR
        
    #max_values, mean_values, std_values=meanstd(name_file,("data_{}").format(dataset_file)) #_60, data_HR/LR 

    PATH = ('logs_{}/mapping/model_40epoch{}_{}.pth').format(out_file,name_file,name_model) 
    #PATH = ('logs_{}/mapping/model_40epoch{}_UNet.pth').format(out_file,name_file)

    outfile_path = ('predictions_{}').format(out_file)
        
    f = open(("predictions_{}/metric{}_{}.txt").format(out_file,name_file,name_model), "w+")
    f2 = open(("predictions_{}/pred_loss_test{}_{}.txt").format(out_file,name_file,name_model), "w+")
    f3 = open(("predictions_{}/pred_loss_val{}_{}.txt").format(out_file,name_file,name_model), "w+")
    
    #####Initialise the model ###############################################MAKE CODE MODELSSSS
    num_classes = 1 
        
    if name_model== 'UNet11':
        model = UNet11(num_classes=num_classes)
    elif name_model == 'UNet':
        model = UNet(num_classes=num_classes)
    elif name_model == 'AlbuNet34':
        model = AlbuNet34(num_classes=num_classes, num_input_channels=4, pretrained=False)
    elif name_model== 'SegNet':
        model = SegNet(num_classes=num_classes, num_input_channels=4, pretrained=False)
    else:
        model = UNet11(num_classes=num_classes, input_channels=4)
        
###########################MAKE CODE MODELSSSS

        
    model.cuda()
    model.load_state_dict(torch.load(PATH))
    model.eval()   # Set model to evaluate mode
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #####Dilenames ###############################################

    
    #train_path= ("data_{}/train{}/images").format(dataset_file,name_file) ##
    #val_path= ("data_{}/val{}/images").format(dataset_file,name_file)##
 #************************************************************ 
    test_path=("data_{}/test{}/images").format(dataset_file,name_file) 
 #************************************************************ 
#.format(dataset_file) whrn the test is 29 images and all the data 
    #test_path=("data_{}/test_dist/images").format(dataset_file) #.when i test with the test random and 60 samples
    #test_path=("data_{}/test_HR_916/images").format(dataset_file) #.when i test with the test random and 60 samples

 #************************************************************ 

    #get_files_path1 = train_path + "/*.npy"
    #get_files_path2 = val_path + "/*.npy"
    get_files_path3 = test_path + "/*.npy"


    #train_file_names = np.array(sorted(glob.glob(get_files_path1)))
    #val_file_names = np.array(sorted(glob.glob(get_files_path2)))
    test_file_names = np.array(sorted(glob.glob(get_files_path3)))

    print(len(test_file_names))
    #####Dataloder ###############################################

    if(dataset_file == 'HR'):
        val_transform = DualCompose([
                CenterCrop(512),
                ImageOnly(Normalize(mean_values, std_values))
            ])

    if(dataset_file =='LR'):
        val_transform = DualCompose([
                CenterCrop(64),
                ImageOnly(Normalize2(mean_values, std_values))
            ])

    train_loader = make_loader(train_file_names,shuffle=True, transform=val_transform)
    val_loader = make_loader(val_file_names, transform=val_transform)
    test_loader = make_loader(test_file_names, transform=val_transform)

    metrics = defaultdict(float)
    ###############################  train images ###############################

    count=0
    input_vec= []
    labels_vec = []
    pred_vec = []
    epoch_samples = 0  #########

    result_dice1 = []
    result_jaccard1 = []
    result_dice2 = []
    result_jaccard2 = []
    result_dice3 = []
    result_jaccard3 = []

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)              
        with torch.set_grad_enabled(False):
            input_vec.append(inputs.data.cpu().numpy())
            labels_vec.append(labels.data.cpu().numpy())
            pred = model(inputs)

            epoch_samples += inputs.size(0) #### 

            loss = calc_loss(pred, labels, metrics)
            #print_metrics(metrics,epoch_samples, f)

            pred=torch.sigmoid(pred) #####   
            pred_vec.append(pred.data.cpu().numpy())    

            result_dice1 += [metrics['dice']]
            
            #if((metrics['jaccard'] == 0 )and (metrics['dice'] > 0.8)): 
            #    result_jaccard1 += [metrics['dice']]  #because  is all water IOU ==1
            #elif((metrics['jaccard'] == 0 )and (metrics['dice'] < 0.8)): 
            #    result_jaccard1 += [metrics['jaccard'] ]
            #else:           
            result_jaccard1 += [metrics['jaccard'] ]

            count += 1
            #print(count)

    print(("Training_{}").format(out_file))
    print('Dice = ', np.mean(result_dice1), np.std(result_dice1))
    print('Jaccard = ',np.mean(result_jaccard1), np.std(result_jaccard1),'\n')

    f.write(("Training_{}\n").format(out_file))
    f.write("dice_metric: {:4f}, std: {:4f} \n".format(np.mean(result_dice1),np.std(result_dice1)))
    f.write("jaccard_metric: {:4f}, std: {:4f}  \n".format(np.mean(result_jaccard1), np.std(result_jaccard1)))  
    ############################### val images ###############################

    count=0
    epoch_samples = 0  #########

    input_vec= []
    labels_vec = []
    pred_vec = []

    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)              
        with torch.set_grad_enabled(False):
            input_vec.append(inputs.data.cpu().numpy())
            labels_vec.append(labels.data.cpu().numpy())
            pred = model(inputs)

            epoch_samples += inputs.size(0) #### 

            loss = calc_loss(pred, labels, metrics)
            print_metrics(metrics,epoch_samples, f3)

            pred=torch.sigmoid(pred) #####   
            pred_vec.append(pred.data.cpu().numpy())    

            result_dice2 += [metrics['dice']]

            #if((metrics['jaccard'] == 0 )and (metrics['dice'] > 0.8)): 
             #   result_jaccard2 += [metrics['dice']]  #because  is all water IOU ==1
            #elif((metrics['jaccard'] == 0 )and (metrics['dice'] < 0.8)): 
            #    result_jaccard2 += [metrics['jaccard'] ]
            #else:           
            result_jaccard2 += [metrics['jaccard'] ]

                              

            count += 1
            #print(count)

    print(("Validation_{}").format(out_file))
    print('Dice = ', np.mean(result_dice2), np.std(result_dice2))
    print('Jaccard = ',np.mean(result_jaccard2), np.std(result_jaccard2),'\n')


    f.write(("Validation_{}\n").format(out_file))
    f.write("dice_metric: {:4f}, std: {:4f} \n".format(np.mean(result_dice2),np.std(result_dice2)))
    f.write("jaccard_metric: {:4f}, std: {:4f}  \n".format(np.mean(result_jaccard2), np.std(result_jaccard2)))

    ###############################  Plotting val  ###############################
    np.save(str(os.path.join(outfile_path,"inputs_val{}_{}_{}.npy".format(name_file,int(count),name_model))), np.array(input_vec))
    np.save(str(os.path.join(outfile_path,"labels_val{}_{}_{}.npy".format(name_file,int(count),name_model))), np.array(labels_vec))
    np.save(str(os.path.join(outfile_path,"pred_val{}_{}_{}.npy".format(name_file,int(count),name_model))), np.array(pred_vec))

    ###############################  test images ###############################

    count=0
    epoch_samples = 0  #########
    input_vec= []
    labels_vec = []
    pred_vec = []

    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)              
        with torch.set_grad_enabled(False):
            input_vec.append(inputs.data.cpu().numpy())
            labels_vec.append(labels.data.cpu().numpy())
            pred = model(inputs)

            epoch_samples += inputs.size(0) #### 

            loss = calc_loss(pred, labels, metrics)
            print_metrics(metrics,epoch_samples, f2)

            pred=torch.sigmoid(pred) #####   
            pred_vec.append(pred.data.cpu().numpy())    


            result_dice3 += [metrics['dice']]

            #if((metrics['jaccard'] == 0 )and (metrics['dice'] > 0.8)): 
            #    result_jaccard3 += [metrics['dice']]  #because  is all water IOU ==1
            #elif((metrics['jaccard'] == 0 )and (metrics['dice'] < 0.8)): 
            #       result_jaccard3 += [metrics['jaccard'] ]
            #else:           
            result_jaccard3 += [metrics['jaccard'] ]

                
                
               
            count += 1
            #print(count)

            #final_layer_npy_outpath.format(int(epoch)
    #np.save(str(os.path.join(outfile_path,"inputs_HR{}_paral_916.npy".format(int(count)))), np.array(input_vec))
    #np.save(str(os.path.join(outfile_path,"labels_HR{}_paral_916.npy".format(int(count)))), np.array(labels_vec))
    #np.save(str(os.path.join(outfile_path,"pred_HR{}_paral_916.npy".format(int(count)))), np.array(pred_vec))


    #np.save(str(os.path.join(outfile_path,"inputs_HR{}_paral_100.npy".format(int(count)))), np.array(input_vec))
    #np.save(str(os.path.join(outfile_path,"labels_HR{}_paral_100.npy".format(int(count)))), np.array(labels_vec))
    #np.save(str(os.path.join(outfile_path,"pred_HR{}_paral_100.npy".format(int(count)))), np.array(pred_vec))

    #np.save(str(os.path.join(outfile_path,"inputs_HR{}_paral_400.npy".format(int(count)))), np.array(input_vec))
    #np.save(str(os.path.join(outfile_path,"labels_HR{}_paral_400.npy".format(int(count)))), np.array(labels_vec))
    #np.save(str(os.path.join(outfile_path,"pred_HR{}_paral_400.npy".format(int(count)))), np.array(pred_vec))

    #np.save(str(os.path.join(outfile_path,"inputs_HR{}_paral_fake.npy".format(int(count)))), np.array(input_vec))
    #np.save(str(os.path.join(outfile_path,"labels_HR{}_paral_fake.npy".format(int(count)))), np.array(labels_vec))
    #np.save(str(os.path.join(outfile_path,"pred_HR{}_paral_fake.npy".format(int(count)))), np.array(pred_vec))

    print(("Test_{}").format(out_file))
    print('Dice = ', np.mean(result_dice3), np.std(result_dice3))
    print('Jaccard = ',np.mean(result_jaccard3), np.std(result_jaccard3),'\n')


    f.write(("Test_{}\n").format(out_file))
    f.write("dice_metric: {:4f}, std: {:4f} \n".format(np.mean(result_dice3),np.std(result_dice3)))
    f.write("jaccard_metric: {:4f}, std: {:4f}  \n".format(np.mean(result_jaccard3), np.std(result_jaccard3)))
    
    np.save(str(os.path.join(outfile_path,"inputs_test{}_{}_{}.npy".format(name_file,int(count),name_model))), np.array(input_vec))
    np.save(str(os.path.join(outfile_path,"labels_test{}_{}_{}.npy".format(name_file,int(count),name_model))), np.array(labels_vec))
    np.save(str(os.path.join(outfile_path,"pred_test{}_{}_{}.npy".format(name_file,int(count),name_model))), np.array(pred_vec))


    

    #stage='test'
    #plotting_figures(stage,name_file,out_file )

#####################################################
######################call funciton###############################


#train_file_names = np.load("logs_HR/mapping/train_files_dist_60.npy")
#val_file_names = np.load("logs_HR/mapping/val_files_dist_60.npy")
#max_values= 3521
#mean_values=(0.10994662, 0.10066561, 0.1125644, 0.13298954)

#std_values=(0.09256749, 0.06976779, 0.05923646, 0.11411727)
#find_metrics(train_file_names,val_file_names,max_values, mean_values, std_values, name_model='UNet11', out_file='HR',dataset_file='HR' ,name_file='_dist_60' )
#########################end############################
