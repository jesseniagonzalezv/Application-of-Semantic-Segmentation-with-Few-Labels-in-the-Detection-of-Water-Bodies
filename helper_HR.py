import matplotlib.pyplot as plt
import numpy as np
from functools import reduce
import itertools
import torchvision.utils


def reverse_transform2(inp): 
    inp = inp.to('cpu').numpy().transpose((1, 2, 0))
    mean = np.array([0.11239524, 0.101936, 0.11311523])
    std = np.array([0.08964322, 0.06702993, 0.05725554]) 
 
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)*3415  #check value maximun
    inp = (inp/inp.max()).astype(np.float32)  

    return inp


def plot_img_array(img_array, loss, ncol=3):
    nrow = len(img_array)  // ncol
    f, plots = plt.subplots(nrow, ncol, sharex='all', sharey='all', figsize=(ncol * 4, nrow * 4))    
    count = 0
    for i in range(len(img_array)):
        while count < 29:
            #print(count)
            temp = str(loss[count])
            count += 1
        #print(len(img_array))
        plots[i // ncol, i % ncol]
        if (i % ncol == 1):
                #print(temp,loss,count)
            plots[i // ncol, i % ncol].set_title("predictions with loss: " + temp)
        plots[i // ncol, i % ncol].imshow(img_array[i])
    
    f.savefig("predictions/prediction.pdf", bbox_inches='tight')
    



def plot_side_by_side(img_arrays, loss):
    flatten_list = reduce(lambda x,y: x+y, zip(*img_arrays))

    plot_img_array(np.array(flatten_list),loss, ncol=len(img_arrays))


def masks_to_colorimg(masks):

    colors = np.asarray([(0, 2, 255)])
    colorimg = np.zeros((masks.shape[1], masks.shape[2], 3), dtype=np.float32) * 255
    channels, height, width = masks.shape

    for y in range(height):
        for x in range(width):
            selected_colors = colors[masks[:,y,x] > 0.5]

            if len(selected_colors) > 0:
                colorimg[y,x,:] = np.mean(selected_colors, axis=0)

    return colorimg.astype(np.uint8)