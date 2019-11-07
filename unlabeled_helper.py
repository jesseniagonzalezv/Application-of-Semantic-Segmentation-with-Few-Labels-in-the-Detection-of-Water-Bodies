import matplotlib.pyplot as plt
import numpy as np
from functools import reduce
import itertools
import torchvision.utils


def reverse_transform2(inp): 
    inp = inp.to('cpu').numpy().transpose((1, 2, 0))
         #mean, std Data_HR
    mean = np.array([0.11239524, 0.101936, 0.11311523])
    std = np.array([0.08964322, 0.06702993, 0.05725554]) 
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)*3415
    inp = (inp/inp.max()).astype(np.float32)

    return inp


def plot_img_array(img_array,save, ncol=2):
    nrow = len(img_array)  // ncol
    f, plots = plt.subplots(nrow, ncol, sharex='all', sharey='all', figsize=(ncol * 4, nrow * 4))    
    for i in range(len(img_array)):
        plots[i // ncol, i % ncol]
        plots[i // ncol, i % ncol].imshow(img_array[i])
    if(save==1):
        f.savefig("predictions/unlabeled_test/unlabeled_prediction_dist.pdf", bbox_inches='tight')
    

def plot_side_by_side(img_arrays,save=0):
    flatten_list = reduce(lambda x,y: x+y, zip(*img_arrays))

    plot_img_array(np.array(flatten_list),save, ncol=len(img_arrays))


def masks_to_colorimg(masks):

    colors = np.asarray([(0, 140, 150)])
    colorimg = np.zeros((masks.shape[1], masks.shape[2], 3), dtype=np.uint8) * 255
    channels, height, width = masks.shape

    for y in range(height):
        for x in range(width):
            selected_colors = colors[masks[:,y,x] > 0.5]

            if len(selected_colors) > 0:
                colorimg[y,x,:] = np.mean(selected_colors, axis=0)

    return colorimg.astype(np.uint8)

def reverse_transform(inp):     
    inp = inp.transpose(1,2,0)
    mean = np.array([0.11239524, 0.101936, 0.11311523])
    std = np.array([0.08964322, 0.06702993, 0.05725554]) 
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)*3415
    #inp = (inp/inp.max()).astype(np.float32)
    inp = (inp/inp.max())#.astype(np.float32)
    inp = (inp*255).astype(np.uint8)
    return inp