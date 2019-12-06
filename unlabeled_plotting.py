import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from PIL import Image
import glob
import unlabeled_helper



from collections import defaultdict

from torch.utils.data import DataLoader
from dataset import WaterDataset
import torch.nn.functional as F
from models import UNet11
import numpy as np
import torch
import glob

from transformsdata import (DualCompose,
                        ImageOnly,
                        Normalize,
                        HorizontalFlip,
                        Rotate,
                        CenterCrop,
                        VerticalFlip)


def unlabel_prediction(PATH):
    #PATH = 'logs/mapping/HR_model_temp2.pth'

    #Initialise the model
    num_classes = 1 
    model = UNet11(num_classes=num_classes)
    model.cuda()
    model.load_state_dict(torch.load(PATH))
    model.eval()   # Set model to evaluate mode

    ######################### setting all data paths#######
    outfile_path = 'predictions_HR/unlabel_test/'
    data_path = 'data_HR'
    #test_path= "data_HR/unlabel/images_jungle"  #crear otro test
    #test_path= "data_HR/unlabel/images"  #crear otro test
    test_path= "data_HR/test_850/images" 

    get_files_path = test_path + "/*.npy"
    test_file_names = np.array(sorted(glob.glob(get_files_path)))
    ###################################
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_transform = DualCompose([
            CenterCrop(512),
            ImageOnly(Normalize())
        ])

    def make_loader(file_names, shuffle=False, transform=None, limit=None):
        return DataLoader(
            dataset=WaterDataset(file_names, transform=transform, limit=limit, mode='test'),
            shuffle=shuffle,            
            batch_size=1,
            pin_memory=torch.cuda.is_available() #### in process arguments
        )
    test_loader = make_loader(test_file_names, shuffle=False, transform=test_transform)
    metrics = defaultdict(float)

    count=0
    input_vec= []
    pred_vec = []
    for inputs , name in test_loader:
        inputs = inputs.to(device)            
        with torch.set_grad_enabled(False):
            input_vec.append(inputs.data.cpu().numpy())
            pred = model(inputs)
            pred=torch.sigmoid(pred) #####   

            pred_vec.append(pred.data.cpu().numpy())
            count += 1
            print(count)
    np.save(outfile_path + "inputs_unlab" + str(count) + ".npy" , np.array(input_vec))
    np.save(outfile_path + "pred_unlab" +  str(count) + ".npy" , np.array(pred_vec))



def plot_prediction(path): # #LR •dist
#PATH = 'logs/mapping/HR_model_temp2.pth'
    unlabel_prediction(path)
    
    val_file = "predictions_HR/unlabel_test/inputs_unlab94.npy"
    pred_file = "predictions_HR/unlabel_test/pred_unlab94.npy"

    val_images = np.load(val_file)
    pred_images = np.load(pred_file)
    pred_images[0,0,:,:,:].shape

    input_images_rgb = [unlabeled_helper.reverse_transform(x) for x in val_images[:,0,:3,:,:]]   #new metrics
    pred_rgb = [unlabeled_helper.masks_to_colorimg(x) for x in pred_images[:,0,:,:,:]]
    unlabeled_helper.plot_side_by_side([input_images_rgb, pred_rgb],save=1)
    

plot_prediction('logs_HR/mapping/model_40epoch_8_percent_UNet11_fold0.pth')
#plot_prediction('logs_paral/mapping/model_40epoch_8_percent_UNet11_fold0.pth')
#plot_prediction('logs_HR/mapping/model_40epoch_80_percent_UNet11_fold0.pth')
#plot_prediction('logs_paral/mapping/model_40epoch_80_percent_UNet11_fold0.pth')