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
from models import UNet11
from dataset import WaterDataset
from torch.optim import lr_scheduler   ####
import utilsTrain 
#import prediction_mask #debjani
import torch.optim as optim #debjani
import numpy as np 
import cv2
import glob  ###


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
    arg('--batch-size', type=int, default=1)
    arg('--limit', type=int, default=100, help='number of images in epoch')
    arg('--n-epochs', type=int, default=100)
    arg('--lr', type=float, default=0.001)
    arg('--model', type=str, default='UNet11', choices=['UNet11'])

    args = parser.parse_args()
    
    root = Path(args.root)
    root.mkdir(exist_ok=True, parents=True)

    num_classes = 1 
    if args.model == 'UNet11':
        model = UNet11(num_classes=num_classes)
        """elif args.model == 'other':
            model = other(num_classes=num_classes, pretrained=True)
        elif args.model == 'other':
            model = other(num_classes=num_classes, pretrained=True)
        """
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
            batch_size=args.batch_size,
            pin_memory=torch.cuda.is_available() #### in process arguments
        )

    #train_file_names, val_file_names = get_split(args.fold)
    data_path = Path('data_HR') # change the data path here 
    print("data_path:",data_path)
    train_path= data_path/'train'/'images'
    val_path= data_path/'val'/'images'
    out_path = Path('logs/mapping/')

    train_file_names = np.array(sorted(list(train_path.glob('*.npy'))))
    print(len(train_file_names))
    val_file_names = np.array(sorted(list(val_path.glob('*.npy'))))


    print('num train = {}, num_val = {}'.format(len(train_file_names), len(val_file_names)))

    train_transform = DualCompose([
        CenterCrop(512),
        HorizontalFlip(),
        VerticalFlip(),
        Rotate(),    
        ImageOnly(Normalize())
    ])

    val_transform = DualCompose([
        CenterCrop(512),
        ImageOnly(Normalize())
    ])

    train_loader = make_loader(train_file_names, shuffle=True, transform=train_transform, mode='train', batch_size = 4)
    valid_loader = make_loader(val_file_names, transform=val_transform, batch_size = 4, mode = "train")


    
    dataloaders = {
    'train': train_loader, 'val': valid_loader
    }

    dataloaders_sizes = {
    x: len(dataloaders[x]) for x in dataloaders.keys()
    }

    
    root.joinpath('params.json').write_text(
        json.dumps(vars(args), indent=True, sort_keys=True))
    
    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(model.parameters(), lr= 1e-3)  #poner como parametro check later
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1) 
    
    
    utilsTrain.train_model( #debjani
        model,
        optimizer_ft,
        exp_lr_scheduler,
        dataloaders,
        num_epochs = 40
        
        )

    torch.save(model.module.state_dict(), out_path/'modelHR_40epoch.pth')
         
if __name__ == '__main__':
    main()
