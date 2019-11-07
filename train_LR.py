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
import utilsTrain_LR 

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
                        Normalize2,
                        HorizontalFlip,
                        Rotate,
                        #RandomBrightness,
                        #RandomRotate90,
                        CenterCrop,
                        VerticalFlip)


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--device-ids', type=str, default='0', help='For example 0,1 to run on two GPUs')
    arg('--root', default='runs/debug', help='checkpoint root')
    arg('--batch-size', type=int, default=8)
    arg('--limit', type=int, default=10000, help='number of images in epoch')
    arg('--n-epochs', type=int, default=40)
    arg('--lr', type=float, default=1e-3)
    arg('--model', type=str, default='UNet11', choices=['UNet11','UNet','AlbuNet34','SegNet'])

    args = parser.parse_args()
    
    root = Path(args.root)
    root.mkdir(exist_ok=True, parents=True)

    num_classes = 1 
    if args.model == 'UNet11':
        model = UNet11(num_classes=num_classes)
    elif args.model == 'UNet':
        model = UNet(num_classes=num_classes)
    elif args.model == 'AlbuNet34':
        model = AlbuNet34(num_classes=num_classes, num_input_channels=4, pretrained=False)
    elif args.model == 'SegNet':
        model = SegNet(num_classes=num_classes, num_input_channels=4, pretrained=False)
    else:
        model = UNet11(num_classes=num_classes, input_channels=4)

    if torch.cuda.is_available():
        if args.device_ids:#
            device_ids = list(map(int, args.device_ids.split(',')))
        else:
            device_ids = None
        model = nn.DataParallel(model, device_ids=device_ids).cuda()# to run the code in multiple gpus
    #loss = utilsTrain.calc_loss(pred, target, metrics, bce_weight=0.5) #check it is utilstrain

    cudnn.benchmark = True

    def make_loader(file_names, shuffle=False, transform=None,mode='train',batch_size=4, limit=None):
        return DataLoader(
            dataset=WaterDataset(file_names, transform=transform,mode=mode, limit=limit),
            shuffle=shuffle,            
            batch_size=batch_size,  #args.batch_size
            pin_memory=torch.cuda.is_available() #### in process arguments
        )
    out_path = Path('logs_LR/mapping/')

    ####################Change the files_names ######################################
    data_path = Path('data_LR') # change the data path here 

    name_file='_LR'
     ####################End Change the files_names ######################################

    print("data_path:",data_path)
    train_path= data_path/'train'/'images'
    val_path= data_path/'val'/'images'

    train_file_names, val_file_names = get_files_names(data_path,name_file)

    np.save(str(os.path.join(out_path,"train_files{}_{}.npy".format(name_file,args.model))), train_file_names)
    np.save(str(os.path.join(out_path,"val_files{}_{}.npy".format(name_file,args.model))), val_file_names)
    
    
    print('num train = {}, num_val = {}'.format(len(train_file_names), len(val_file_names)))
    
    max_values, mean_values, std_values=meanstd(train_file_names, val_file_names,name_file,str(data_path)) #_LR


    train_transform = DualCompose([
        CenterCrop(64),
        HorizontalFlip(),
        VerticalFlip(),
        Rotate(), 
        #RandomRotate90(),
        #RandomBrightness(),
        ImageOnly(Normalize2(mean_values, std_values))
    ])

    val_transform = DualCompose([
        CenterCrop(64),
       # VerticalFlip(),
       # HorizontalFlip(),
        ImageOnly(Normalize2(mean_values, std_values))
    ])

    train_loader = make_loader(train_file_names, shuffle=True, transform=train_transform, mode='train', batch_size = 16)
    valid_loader = make_loader(val_file_names, transform=val_transform, mode = "train", batch_size = 8)


    
    dataloaders = {
    'train': train_loader, 'val': valid_loader
    }

    dataloaders_sizes = {
    x: len(dataloaders[x]) for x in dataloaders.keys()
    }

    
    root.joinpath('params_LR.json').write_text(
        json.dumps(vars(args), indent=True, sort_keys=True))
    
    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(model.parameters(), lr= args.lr)  #poner como parametro check later
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1) 
    
    
    utilsTrain_LR.train_model( 
        name_file,
        model,
        optimizer_ft,
        exp_lr_scheduler,
        dataloaders,
        args.model,

        args.n_epochs 
        
        )

    torch.save(model.module.state_dict(), (str(out_path)+'/model_{}epoch{}_{}.pth').format(args.n_epochs , name_file,args.model))
    
    print(args.model)
    find_metrics(train_file_names, val_file_names, max_values, mean_values, std_values,args.model, out_file='LR', dataset_file='LR',name_file=name_file)   
                 
    
if __name__ == '__main__':
    main()
