'''
This is the main code 
Ask the argument
Make the loaders
Make the train
'''
import argparse
from pathlib import Path
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import torch.backends.cudnn as cudnn
import torch.backends.cudnn
import json
from models import UNet11,UNet, AlbuNet34, SegNet
from dataset import WaterDataset
from torch.optim import lr_scheduler   ####
import utilsTrain_seq
import torch.optim as optim 
import numpy as np 
import cv2
import glob  ###
import os

from split_train_val import get_files_names
from scalarmeanstd import meanstd
from metrics_prediction import find_metrics

from transformsdata import (DualCompose,
                        ImageOnly,
                        Normalize,
                        HorizontalFlip,
                        Rotate,
                        CenterCrop,
                        VerticalFlip)



def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--device-ids', type=str, default='0', help='For example 0,1 to run on two GPUs')
    arg('--root', default='runs/debug', help='checkpoint root')
    arg('--batch-size', type=int, default=4)
    arg('--limit', type=int, default=10000, help='number of images in epoch')
    arg('--n-epochs', type=int, default=40)
    arg('--lr', type=float, default=1e-3)
    arg('--model', type=str, default='UNet11', choices=['UNet11','UNet','AlbuNet34','SegNet'])

    args = parser.parse_args()
    
    root = Path(args.root)
    root.mkdir(exist_ok=True, parents=True)

    num_classes = 1 
    if args.model == 'UNet11':
        model_HR = UNet11(num_classes=num_classes)
    elif args.model == 'UNet':
        model_HR = UNet(num_classes=num_classes)
    elif args.model == 'AlbuNet34':
        model_HR = AlbuNet34(num_classes=num_classes, num_input_channels=4, pretrained=False)
    elif args.model == 'SegNet':
        model_HR = SegNet(num_classes=num_classes, num_input_channels=4, pretrained=False)
    else:
        model_HR = UNet11(num_classes=num_classes, input_channels=4)

    if torch.cuda.is_available():
        if args.device_ids:#
            device_ids = list(map(int, args.device_ids.split(',')))
        else:
            device_ids = None
        model_HR = nn.DataParallel(model_HR, device_ids=device_ids).cuda()# to run the code in multiple gpus
    #loss = utilsTrain.calc_loss(pred, target, metrics, bce_weight=0.5) #check it is utilstrain

    cudnn.benchmark = True


    out_path = Path('logs_seq/mapping/')
    
    #Data-paths:--------------------------Hr-------------------------------------

    #train_file_names, val_file_names = get_split(args.fold)
    data_path_HR = Path('data_HR') # change the data path here 
    
    print("data_path:",data_path_HR)
    #name_file='_HR_dist'
    name_file='_HR_60_fake'

    ##--------------------------------------
    train_file_HR_lab, val_file_HR_lab = get_files_names(data_path_HR,name_file)
    np.save(str(os.path.join(out_path,"train_files{}_{}.npy".format(name_file,args.model))), train_file_HR_lab)
    np.save(str(os.path.join(out_path,"val_files{}_{}.npy".format(name_file,args.model))), val_file_HR_lab)
    #train_path_HR_lab= data_path_HR/'train'/'images'
    #val_path_HR_lab= data_path_HR/'val'/'images'
    
    #train_path_HR_lab= data_path_HR/'train_100'/'images'
    #val_path_HR_lab= data_path_HR/'val_100'/'images'
    
    #train_path_HR_lab= data_path_HR/'train_400'/'images'
    #val_path_HR_lab= data_path_HR/'val_400'/'images'
    
    #train_file_HR_lab = np.array(sorted(list(train_path_HR_lab.glob('*.npy'))))
    #val_file_HR_lab = np.array(sorted(list(val_path_HR_lab.glob('*.npy'))))
      #Data-paths:--------------------------unlabeled HR-------------------------------------    
    
    train_path_HR_unlab= data_path_HR/'unlabel'/'train'/'images'
    val_path_HR_unlab = data_path_HR/'unlabel'/'val'/'images'
    
    

    train_file_HR_unlab = np.array(sorted(list(train_path_HR_unlab.glob('*.npy'))))
    val_file_HR_unlab = np.array(sorted(list(val_path_HR_unlab.glob('*.npy'))))
   
    #--------------------------------------------------------------------------------------------------------


    print('num train_lab = {}, num_val_lab = {}'.format(len(train_file_HR_lab), len(val_file_HR_lab)))
    print('num train_unlab = {}, num_val_unlab = {}'.format(len(train_file_HR_unlab), len(val_file_HR_unlab)))
    
    max_values, mean_values, std_values=meanstd(train_file_HR_lab, val_file_HR_lab,name_file,str(data_path_HR)) #_60 --data_HR, data_LR

    def make_loader(file_names, shuffle=False, transform=None,mode='train',batch_size=4, limit=None):
        return DataLoader(
            dataset=WaterDataset(file_names, transform=transform,mode=mode, limit=limit),
            shuffle=shuffle,            
            batch_size=batch_size #args.batch_size,
            
        )
     
        
    train_transform_HR = DualCompose([
            CenterCrop(512),
            HorizontalFlip(),
            VerticalFlip(),
            Rotate(),
            ImageOnly(Normalize(mean_values, std_values))
        ])
    
    val_transform_HR = DualCompose([
            CenterCrop(512),
            ImageOnly(Normalize(mean_values, std_values))
        ])

######################## DATA-LOADERS ###########################################################49
    train_loader_HR_lab = make_loader(train_file_HR_lab, shuffle=True, transform=train_transform_HR , batch_size = 4, mode = "train")
    valid_loader_HR_lab = make_loader(val_file_HR_lab, transform=val_transform_HR, batch_size = 4, mode = "train")
    
    train_loader_HR_unlab = make_loader(train_file_HR_unlab, shuffle=True, transform=train_transform_HR, batch_size = 2, mode = "unlb_train")
    valid_loader_HR_unlab = make_loader(val_file_HR_unlab, transform=val_transform_HR, batch_size = 2, mode = "unlb_val")

    
    dataloaders_HR_lab= {
        'train': train_loader_HR_lab, 'val': valid_loader_HR_lab
    }
    
    dataloaders_HR_unlab= {
        'train': train_loader_HR_unlab, 'val': valid_loader_HR_unlab
    }

#----------------------------------------------    
    root.joinpath('params_seq.json').write_text(
        json.dumps(vars(args), indent=True, sort_keys=True))
    
    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(model_HR.parameters(), lr= args.lr)  #poner como parametro check later
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1) 

#--------------------------model LR-------------------------------------
    PATH_LR= 'logs_LR/mapping/model_40epoch_LR_UNet11.pth'

    #Initialise the model
    model_LR = UNet11(num_classes=num_classes)
    model_LR.cuda()
    model_LR.load_state_dict(torch.load(PATH_LR))

#---------------------------------------------------------------
    model_HR= utilsTrain_seq.train_model(
        name_file,
        model_LR, model_HR,
        optimizer_ft,
        exp_lr_scheduler,
        dataloaders_HR_lab,
        dataloaders_HR_unlab,
        args.model,
        60,
        args.n_epochs 
        
        )

    #torch.save(model_HR.module.state_dict(), out_path/'model_dis_HR_40epoch_916.pth')
    #torch.save(model_HR.module.state_dict(), out_path/'model_dis_HR_40epoch_100.pth')
    torch.save(model_HR.module.state_dict(), (str(out_path)+'/model_40epoch{}_{}.pth').format(name_file,args.model))

    print(args.model)
    find_metrics(train_file_HR_lab, val_file_HR_lab, max_values, mean_values, std_values,args.model, out_file='seq', dataset_file='HR',name_file=name_file)
    
if __name__ == '__main__':
    main()
