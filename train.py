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
import utilsTrain 
import torch.optim as optim 
import numpy as np 
import glob  ###
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
    arg('--fold-out', type=int, default='0', help='fold train-val test')
    arg('--fold-in', type=int, default='0', help='fold train val')
    arg('--percent', type=float, default=1, help='percent of data')
    arg('--root', default='runs/debug', help='checkpoint root')
    arg('--batch-size', type=int, default=4,help='HR:4,VHR:8')
    arg('--limit', type=int, default=10000, help='number of images in epoch')
    arg('--n-epochs', type=int, default=40)
    arg('--lr', type=float, default=1e-3)
    arg('--model', type=str, default='UNet11', choices=['UNet11','UNet','AlbuNet34','SegNet'])
    arg('--dataset-path', type=str, default='data_VHR', help='main file,in which the dataset is:  data_VHR or data_HR')
    arg('--dataset-file', type=str, default='VHR', help='resolution of the dataset VHR,HR' )
    #arg('--out-file', type=str, default='VHR', help='the file in which save the outputs')
    arg('--train-val-file', type=str, default='train_val_850', help='name of the train-val file VHR:train_val_850 or train_val_HR' )
    arg('--test-file', type=str, default='test_850', help='name of the test file VHR:test_850 or HR:test_HR' )



    args = parser.parse_args()
    
    root = Path(args.root)
    root.mkdir(exist_ok=True, parents=True)

    num_classes = 1
    input_channels=4

    if args.model == 'UNet11':
        model = UNet11(num_classes=num_classes, input_channels=input_channels)
    elif args.model == 'UNet':
        model = UNet(num_classes=num_classes, input_channels=input_channels)
    elif args.model == 'AlbuNet34':
        model = AlbuNet34(num_classes=num_classes, num_input_channels=input_channels, pretrained=False)
    elif args.model == 'SegNet':
        model = SegNet(num_classes=num_classes, num_input_channels=input_channels, pretrained=False)
    else:
        model = UNet11(num_classes=num_classes, input_channels=input_channels)

    
    if torch.cuda.is_available():
        if args.device_ids:#
            device_ids = list(map(int, args.device_ids.split(',')))
        else:
            device_ids = None
        model = nn.DataParallel(model, device_ids=device_ids).cuda()

    cudnn.benchmark = True



    ####################Change the files_names ######################################
    out_path = Path(('logs_{}/mapping/').format(args.dataset_file))
    name_file = '_'+ str(int(args.percent*100))+'_percent_'+ args.dataset_file
    data_all='data' ##file with all the data 

    data_path = Path(args.dataset_path) 
    print("data_path:",data_path)
    #################################################################################  
    # Nested cross validation K-fold train test
    #train_val_file_names, test_file_names = get_split_out(data_path,data_all,args.fold_out)
    #################################################################################  
    #eWe are consider the same test in all the cases
    train_val_file_names=np.array(sorted(glob.glob(str(data_path/args.train_val_file/'images')+ "/*.npy")))
    test_file_names =  np.array(sorted(glob.glob(str(data_path/args.test_file/'images') + "/*.npy")))
    
    if args.percent !=1:
        extra, train_val_file_names= percent_split(train_val_file_names, args.percent) 

    #################################################################################  
    
 
    
    train_file_names,val_file_names = get_split_in(train_val_file_names,args.fold_in)   

    np.save(str(os.path.join(out_path,"train_files{}_{}_fold{}_{}.npy".format(name_file,args.model,args.fold_out,args.fold_in))), train_file_names)
    np.save(str(os.path.join(out_path,"val_files{}_{}_fold{}_{}.npy".format(name_file,args. model,args.fold_out, args.fold_in))), val_file_names)
    

    print('num train = {}, num_val = {}'.format(len(train_file_names), len(val_file_names)))

    
    def make_loader(file_names, shuffle=False, transform=None,mode='train',batch_size=4, limit=None):
        return DataLoader(
            dataset=WaterDataset(file_names, transform=transform,mode=mode, limit=limit),
            shuffle=shuffle,            
            batch_size=batch_size, 
            pin_memory=torch.cuda.is_available() 
        )
    max_values, mean_values, std_values=meanstd(train_file_names, val_file_names,test_file_names,str(data_path),input_channels) #_60 
    print(max_values,mean_values, std_values)
    if(args.dataset_file == 'VHR'):
        train_transform = DualCompose([
            CenterCrop(512),
            HorizontalFlip(),
            VerticalFlip(),
            Rotate(),    
            ImageOnly(Normalize(mean=mean_values, std=std_values))
        ])
  
        val_transform = DualCompose([
            CenterCrop(512),
            ImageOnly(Normalize(mean=mean_values, std=std_values))
        ])
        max_values=3521
        train_loader = make_loader(train_file_names, shuffle=True, transform=train_transform, mode='train', batch_size = args.batch_size)  #4 batch_size
        valid_loader = make_loader(val_file_names, transform=val_transform, batch_size = args.batch_size, mode = "train")
        
    if(args.dataset_file == 'HR'):
        train_transform = DualCompose([ 
            CenterCrop(64),
            HorizontalFlip(),
            VerticalFlip(),
            Rotate(),    
            ImageOnly(Normalize2(mean=mean_values, std=std_values))
        ])        


        val_transform = DualCompose([
            CenterCrop(64),
            ImageOnly(Normalize2(mean=mean_values,std= std_values))
        ])
        train_loader = make_loader(train_file_names, shuffle=True, transform=train_transform, mode='train', batch_size = args.batch_size)  #8 batch_size
        valid_loader = make_loader(val_file_names, transform=val_transform, mode = "train", batch_size = args.batch_size//2)
        

#albunet 34 with only 3 batch_size
 


    dataloaders = {
    'train': train_loader, 'val': valid_loader
    }

    dataloaders_sizes = {
    x: len(dataloaders[x]) for x in dataloaders.keys()
    }

    
    root.joinpath(('params_{}.json').format(args.dataset_file)).write_text(
        json.dumps(vars(args), indent=True, sort_keys=True))
    
    optimizer_ft = optim.Adam(model.parameters(), lr= args.lr)  #
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1) 
    
    
    utilsTrain.train_model(
        dataset_file=args.dataset_file,
        name_file=name_file,
        model=model,
        optimizer=optimizer_ft,
        scheduler=exp_lr_scheduler,
        dataloaders=dataloaders,
        fold_out=args.fold_out,
        fold_in=args.fold_in,
        name_model=args.model,
        num_epochs=args.n_epochs 
        )


    torch.save(model.module.state_dict(),(str(out_path)+'/model{}_{}_foldout{}_foldin{}_{}epochs').format(name_file,args.model,args.fold_out,args.fold_in,args.n_epochs)) 
    
    print(args.model)
    

    find_metrics(train_file_names=train_file_names, 
                 val_file_names=val_file_names, 
                 test_file_names=test_file_names, 
                 max_values=max_values, 
                 mean_values=mean_values, 
                 std_values=std_values,
                 model=model, 
                 fold_out=args.fold_out, 
                 fold_in=args.fold_in,
                 name_model=args.model, 
                 epochs=args.n_epochs,
                 out_file=args.dataset_file, 
                 dataset_file=args.dataset_file,
                 name_file=name_file)   
 
        
if __name__ == '__main__':
    main()
