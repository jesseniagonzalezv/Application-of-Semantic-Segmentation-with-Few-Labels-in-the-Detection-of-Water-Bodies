import unlabeled_helper

from collections import defaultdict
from unlabeled_helper import reverse_transform2

from torch.utils.data import DataLoader
from loss import dice_loss,metric_jaccard
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


PATH = 'logs/mapping/modelHR_40epoch.pth'

#Initialise the model
num_classes = 1 
model = UNet11(num_classes=num_classes)
model.cuda()
model.load_state_dict(torch.load(PATH))
model.eval()   # Set model to evaluate mode

######################### setting all data paths#######
outfile_path = 'predictions/unlabel_test/'
data_path = 'data_HR'
test_path= "data_HR/unlabel/images_jungle"  #crear otro test
get_files_path = test_path + "/*.npy"
test_file_names = np.array(sorted(glob.glob(get_files_path)))
###################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_transform = DualCompose([
        CenterCrop(512),
        ImageOnly(Normalize())
    ])

def make_loader(file_names, shuffle=False, transform=None,mode='train',batch_size=1, limit=None):
    return DataLoader(
        dataset=WaterDataset(file_names, transform=transform, limit=limit),
        shuffle=shuffle,            
        batch_size=batch_size,
        pin_memory=torch.cuda.is_available() #### in process arguments
    )
test_loader = make_loader(test_file_names, shuffle=False, transform=test_transform, mode='test', batch_size = 1)
metrics = defaultdict(float)

count=0
input_vec= []
pred_vec = []
for inputs , name in test_loader:
    inputs = inputs.to(device)            
    with torch.set_grad_enabled(False):
        input_vec.append(inputs.data.cpu().numpy())
        pred = model(inputs)
        #pred=torch.sigmoid(pred) #####   

        pred_vec.append(pred.data.cpu().numpy())
        count += 1
        print(count)
np.save(outfile_path + "inputs_unlab_" + str(count) + ".npy" , np.array(input_vec))
np.save(outfile_path + "pred_unlab_" +  str(count) + ".npy" , np.array(pred_vec))
