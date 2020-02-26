'''
This is the main code to train

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
from torch.optim import lr_scheduler 
import utilsTrain_paral
import torch.optim as optim
import numpy as np
import glob
import os

from get_train_test_kfold import get_split_out, percent_split, get_split_in

from split_train_val import get_files_names
from scalarmeanstd import meanstd
from metrics_prediction_2 import find_metrics

from transformsdata import (DualCompose,
                        ImageOnly,
                        Normalize,
                        Normalize2,
                        HorizontalFlip,
                        Rotate,
                        CenterCrop,
                        VerticalFlip)


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--device-ids', type=str, default='0', help='For example 0,1 to run on two GPUs')
    arg('--fold-out', type=int, help='fold-train-val test', default=0)
    arg('--fold-in', type=int, help='fold- train val', default=0)
    arg('--percent', type=float, help='percent of data', default=1)
    arg('--root', default='runs/debug', help='checkpoint root')
    arg('--batch-size', type=int, default=4)
    arg('--limit', type=int, default=10000, help='number of images in epoch')
    arg('--n-epochs', type=int, default=40)
    arg('--n-steps', type=int, default=200)
    arg('--lr', type=float, default=1e-3)
    arg('--modelVHR', type=str, default='UNet11', choices=['UNet11','UNet','AlbuNet34','SegNet'])
    arg('--dataset-path-HR', type=str, default='data_HR', help='ain path  of the HR dataset')
    arg('--model-path-HR', type=str, default='logs_HR/mapping/model_40epoch_HR_UNet11.pth', help='path of the model of HR')
    arg('--dataset-path-VHR', type=str, default='data_VHR', help='main path  of the VHR dataset')
    arg('--name-file-HR', type=str, default='_HR', help='name file of HR dataset')
    arg('--dataset-file', type=str, default='VHR', help='main dataset resolution,depend of this correspond a specific crop' )
    arg('--out-file', type=str, default='paral', help='the file in which save the outputs')
    arg('--train-val-file-HR', type=str, default='train_val_HR', help='name of the train-val file' )
    arg('--test-file-HR', type=str, default='test_HR', help='name of the test file' )
    arg('--train-val-file-VHR', type=str, default='train_val_850', help='name of the train-val file' )
    arg('--test-file-VHR', type=str, default='test_850', help='name of the test file' )

    args = parser.parse_args()
    
    root = Path(args.root)
    root.mkdir(exist_ok=True, parents=True)

    num_classes = 1
    input_channels=4

    if args.modelVHR == 'UNet11':
        model_HR = UNet11(num_classes=num_classes, input_channels=input_channels)
        model_VHR = UNet11(num_classes=num_classes, input_channels=input_channels)
    elif args.modelVHR == 'UNet':
        model_HR = UNet(num_classes=num_classes, input_channels=input_channels)
        model_VHR = UNet(num_classes=num_classes, input_channels=input_channels)

    elif args.modelVHR == 'AlbuNet34':
        model_HR = AlbuNet34(num_classes=num_classes, num_input_channels=input_channels, pretrained=False)
        model_VHR = AlbuNet34(num_classes=num_classes, num_input_channels=input_channels, pretrained=False)
        
    elif args.modelVHR == 'SegNet':
        model_HR = SegNet(num_classes=num_classes, num_input_channels=input_channels, pretrained=False)
        model_VHR = SegNet(num_classes=num_classes, num_input_channels=input_channels, pretrained=False)

    else:
        model_HR = UNet11(num_classes=num_classes, input_channels=input_channels)
        
        model_VHR = UNet11(num_classes=num_classes, input_channels=input_channels)


        
    if torch.cuda.is_available():
        if args.device_ids:#
            device_ids = list(map(int, args.device_ids.split(',')))
        else:
            device_ids = None
        model_HR = nn.DataParallel(model_HR, device_ids=device_ids).cuda()
        model_VHR = nn.DataParallel(model_VHR, device_ids=device_ids).cuda()# to run the code in multiple gpus

    cudnn.benchmark = True



    out_path = Path(('logs_{}/mapping/').format(args.out_file))

#Data-paths:--------------------------HR-------------------------------------
    data_path_HR = Path(args.dataset_path_HR) 
    print("dataset_path:",data_path_HR) 
    
    train_file_HR, val_file_HR = get_files_names(data_path_HR,args.train_val_file_HR)
    test_path = (data_path_HR/args.test_file_HR/'images' )   
    test_file_HR = np.array(sorted(list(test_path.glob('*.npy'))))

    max_values_HR, mean_values_HR, std_values_HR=meanstd(train_file_HR, val_file_HR,test_file_HR,str(data_path_HR),input_channels) #_HR

    #Data-paths:--------------------------VHR-------------------------------------
    data_path_VHR = Path(args.dataset_path_VHR) 

    print("dataset_path:",data_path_VHR)
    name_file_VHR = '_'+ str(int(args.percent*100))+'_percent_'+args.out_file
    data_all='data'


      #Data-paths:--------------------------HR-------------------------------------
    #############################   nested cross validation K-fold train test
    #train_val_file_names, test_file_names_HR = get_split_out(data_path_HR,data_all,args.fold_out)
    ############################  

    train_val_file_names=np.array(sorted(glob.glob(str((data_path_VHR/args.train_val_file_VHR/'images'))+ "/*.npy")))
    test_file_names_VHR =  np.array(sorted(glob.glob(str((data_path_VHR/args.test_file_VHR/'images')) + "/*.npy")))
    
    if args.percent !=1:
        extra, train_val_file_names= percent_split(train_val_file_names, args.percent) 

    train_file_VHR_lab,val_file_VHR_lab = get_split_in(train_val_file_names,args.fold_in)    
    
    np.save(str(os.path.join(out_path,"train_files{}_{}_fold{}_{}.npy".format(name_file_VHR, args.modelVHR,args.fold_out,args.fold_in))), train_file_VHR_lab)
    np.save(str(os.path.join(out_path,"val_files{}_{}_fold{}_{}.npy".format(name_file_VHR, args.modelVHR,args.fold_out,args.fold_in))), val_file_VHR_lab)
    
      

        
    train_path_VHR_unlab= data_path_VHR/'unlabel'/'train'/'images'
    val_path_VHR_unlab = data_path_VHR/'unlabel'/'val'/'images'
    
    train_file_VHR_unlab = np.array(sorted(list(train_path_VHR_unlab.glob('*.npy'))))
   
    val_file_VHR_unlab = np.array(sorted(list(val_path_VHR_unlab.glob('*.npy'))))
    

        
    print('num train_lab = {}, num_val_lab = {}'.format(len(train_file_VHR_lab), len(val_file_VHR_lab)))
    print('num train_unlab = {}, num_val_unlab = {}'.format(len(train_file_VHR_unlab), len(val_file_VHR_unlab)))   
    print('num train_unlab = {}, num_val_unlab = {}'.format(len(train_file_HR), len(val_file_HR)))   #----------------------------------------------------------------------------
    
    max_values_VHR, mean_values_VHR, std_values_VHR=meanstd(train_file_VHR_lab, val_file_VHR_lab,test_file_names_VHR,str(data_path_VHR),input_channels) 
    
    
    def make_loader(file_names, shuffle=False, transform=None, limit=None,  mode = "train",batch_size=4,limite=None) :
             return DataLoader(
                dataset=WaterDataset(file_names, transform=transform, mode = mode,limit=limit),
                shuffle=shuffle,
                batch_size= batch_size,
                pin_memory=torch.cuda.is_available() 
        ) 
    
    #transformations ---------------------------------------------------------------------------                      
    train_transform_VHR = DualCompose([
            CenterCrop(512),
            HorizontalFlip(),
            VerticalFlip(),
            Rotate(),
            ImageOnly(Normalize(mean=mean_values_VHR, std=std_values_VHR))
        ])
    
    val_transform_VHR = DualCompose([
            CenterCrop(512),
            ImageOnly(Normalize(mean=mean_values_VHR, std=std_values_VHR))
        ])
    
                     
    train_transform_VHR_unlab = DualCompose([
            CenterCrop(512),
            HorizontalFlip(),
            VerticalFlip(),
            Rotate(),
            ImageOnly(Normalize(mean=mean_values_HR, std=std_values_HR))
        ])
    
    val_transform_VHR_unlab = DualCompose([
            CenterCrop(512),
            ImageOnly(Normalize(mean_values_HR, std=std_values_HR))
        ])
    #mean_values_HR=(0.11952524, 0.1264638 , 0.13479991, 0.15017026)
    #std_values_HR=(0.08844988, 0.07304429, 0.06740904, 0.11003125)
    
    train_transform_HR = DualCompose([
            CenterCrop(64),
            HorizontalFlip(),
            VerticalFlip(),
            Rotate(),
            ImageOnly(Normalize2(mean=mean_values_HR,std= std_values_HR))  
        ])
    
    val_transform_HR = DualCompose([
            CenterCrop(64),
            ImageOnly(Normalize2(mean=mean_values_HR, std=std_values_HR))    
        ])
    
    
    ######################## DATA-LOADERS ###########################################################
      
    train_loader_VHR_lab = make_loader(train_file_VHR_lab, shuffle=True, transform=train_transform_VHR , batch_size = args.batch_size , mode = "train")
    valid_loader_VHR_lab = make_loader(val_file_VHR_lab, transform=val_transform_VHR, batch_size =args.batch_size, mode = "train")
    
    train_loader_VHR_unlab = make_loader(train_file_VHR_unlab, shuffle=True, transform=train_transform_VHR, batch_size = args.batch_size//2, mode = "unlb_train")
    valid_loader_VHR_unlab = make_loader(val_file_VHR_unlab, transform=val_transform_VHR, batch_size = args.batch_size//2, mode = "unlb_val")
 
    train_loader_HR = make_loader(train_file_HR, shuffle=True, transform=train_transform_HR, batch_size = args.batch_size, mode = "train" )
    valid_loader_HR = make_loader(val_file_HR, transform=val_transform_HR, batch_size = args.batch_size, mode = "train")
        
        
    dataloaders_VHR_lab= {
        'train': train_loader_VHR_lab, 'val': valid_loader_VHR_lab
    }
    
    dataloaders_VHR_unlab= {
        'train': train_loader_VHR_unlab, 'val': valid_loader_VHR_unlab
    }
    
    dataloaders_HR= {
        'train': train_loader_HR, 'val': valid_loader_HR
    }
    
    #PRINT THE SIZES----------------------------------------------
    dataloaders_sizes = {
        x: len(dataloaders_VHR_lab[x]) for x in dataloaders_VHR_lab.keys()
        
    }
    print('VHR',dataloaders_sizes)
    
    dataloaders_sizes = {
        x: len(dataloaders_HR[x]) for x in dataloaders_HR.keys()
    }
    print('HR',dataloaders_sizes)

    #----------------------------------------------------------------------------------------------
    root.joinpath(('params_{}.json').format(args.out_file)).write_text(
        json.dumps(vars(args), indent=True, sort_keys=True))
    
    optimizer_ft = optim.Adam(list(model_HR.parameters()) + list(model_VHR.parameters()), lr=args.lr)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1) 

#--------------------------model HR-------------------------------------
#    PATH_HR= args.model_path_HR

    #Initialise the model
#    model_HR = UNet11(num_classes=num_classes)
#    model_HR.cuda()
#    model_HR.load_state_dict(torch.load(PATH_HR))
#---------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------
    
    
    utilsTrain_paral.train_model(out_file=args.out_file,
                                 name_file_VHR=name_file_VHR,
                                 model_HR=model_HR, 
                                 model_VHR= model_VHR, 
                                 optimizer_ft=optimizer_ft,                                                    
                                 scheduler=exp_lr_scheduler,
                                 dataloaders_VHR_lab=dataloaders_VHR_lab,
                                 dataloaders_VHR_unlab=dataloaders_VHR_unlab,
                                 dataloaders_HR=dataloaders_HR,
                                 fold_out=args.fold_out,
                                 fold_in=args.fold_in,
                                 name_model_VHR=args.modelVHR,                       
                                 n_steps=args.n_steps, 
                                 num_epochs=args.n_epochs) 

    
    torch.save(model_HR.module.state_dict(), (str(out_path)+'/model{}_{}_foldout{}_foldin{}_{}epochs.pth').format(args.name_file_HR,args.modelVHR,args.fold_out,args.fold_in,args.n_epochs)) 
    torch.save(model_VHR.module.state_dict(),(str(out_path)+'/model{}_{}_foldout{}_foldin{}_{}epochs.pth'). format(name_file_VHR,args.modelVHR,args.fold_out,args.fold_in,args.n_epochs))
               
    print(args.modelVHR)
    max_values_all_VHR=3521

    find_metrics(train_file_names=train_file_VHR_lab, 
                 val_file_names=val_file_VHR_lab,
                 test_file_names=test_file_names_VHR, 
                 max_values=max_values_all_VHR, 
                 mean_values=mean_values_VHR, 
                 std_values=std_values_VHR,
                 model= model_VHR,
                 fold_out=args.fold_out, 
                 fold_in=args.fold_in, 
                 name_model=args.modelVHR,
                 epochs=args.n_epochs, 
                 out_file=args.out_file, 
                 dataset_file=args.dataset_file,
                 name_file=name_file_VHR)
    
if __name__ == '__main__':
    main()
