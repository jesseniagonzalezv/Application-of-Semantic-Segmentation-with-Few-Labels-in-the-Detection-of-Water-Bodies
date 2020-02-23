
'''
Histogram of all the dataset
'''
import matplotlib.pyplot as plt
import numpy as np
import torch
import glob
import os
from pathlib import Path

##########################################################################

def hist_data(files_root,pixel_max,name,proce):
 
    get_files_path = str(files_root) + "/*.npy"
    file_names = np.array(sorted(glob.glob(get_files_path)))

    minimo_pixel=[]
    maximo_pixel=[]
    maximo_pixel_nor=[]

    print(len(file_names))
    fig, axes = plt.subplots(nrows=2, ncols=2)
    ax0, ax1, ax2, ax3 = axes.flatten()
    for i in file_names:
        img = np.load(str(i))
        #print(np.max(img))
        img=img.transpose((1, 2, 0))
        img_nor=img/pixel_max

        minimo_pixel.append(np.min(img))
        maximo_pixel.append(np.max(img))
        maximo_pixel_nor.append(np.max(img_nor))


        ax0.hist((img[:,:,0]/pixel_max).ravel(), bins=256, range=(0.0, 1.0), fc='none', ec='r', histtype='step') 
        ax1.hist((img[:,:,0]/pixel_max).ravel(), bins=256, range=(0.0, 1.0), fc='none', ec='g', histtype='step')
        ax2.hist((img[:,:,0]/pixel_max).ravel(), bins=256, range=(0.0, 1.0), fc='none', ec='b', histtype='step') 
        ax3.hist((img[:,:,0]/pixel_max).ravel(), bins=256, range=(0.0, 1.0), fc='none', ec='k', histtype='step') 
    ax0.set_xlabel('Pixels')
    ax1.set_xlabel('Pixels')
    ax2.set_xlabel('Pixels')
    ax3.set_xlabel('Pixels')

    ax0.set_ylabel('Probabilty')
    ax1.set_ylabel('Probabilty')
    ax2.set_ylabel('Probabilty')
    ax3.set_ylabel('Probabilty')

    ax0.set_title('Histogram of {} - Red'.format(name))
    ax1.set_title('Histogram of {} - Green'.format(name))
    ax2.set_title('Histogram of {} - Blue'.format(name))
    ax3.set_title('Histogram of {} - NIR'.format(name))


    fig.tight_layout()
    plt.xlabel('Pixels')
    plt.ylabel('Probabilty')
    plt.show()
    print('{} - min: {} max: {}'.format(name,np.min(minimo_pixel),np.max(maximo_pixel)))# 0-3521
    fig.savefig("histogram_{}_{}_normalized.pdf".format(name,proce), bbox_inches='tight')
    return fig
    
    
######################################Perusat histogram###############################3
def histogram():
    data_path_VHR = Path('data_VHR')
    train_root_VHR= data_path_VHR/'train'/'images'
    val_root_VHR= data_path_VHR/'val'/'images'
    test_root_VHR= data_path_VHR/'test'/'images'
    data_all_root_VHR= '/home/jgonzalez/Test_2019/Test_PreProcessing/data' #all the dataset
    #######################################plot histogram###############################3

    pixel_max_VHR=3521
    fig_train_VHR=hist_data(train_root_VHR, pixel_max_VHR,'PeruSat','train')
    fig_train_VHR=hist_data(val_root_VHR,pixel_max_VHR, 'PeruSat','val')
    fig_train_VHR=hist_data(test_root_VHR,pixel_max_VHR, 'PeruSat','test')
    fig_train_VHR=hist_data(data_all_root_VHR,pixel_max_VHR, 'PeruSat','all')

    ####################################################################################3
    #################################Sentinel histogram###############################3

    data_path_HR = Path('data_HR')
    train_root_HR= data_path_HR/'train'/'images'
    val_root_HR= data_path_HR/'val'/'images'
    test_root_HR= data_path_HR/'test'/'images'
    data_all_root_HR= '/home/jgonzalez/Test_2019/Test_network/model_HR/data_HR/data/images' #all the dataset

    #######################################plot histogram###############################3
    pixel_max_HR=1
    fig_train_HR=hist_data(train_root_HR, pixel_max_HR,'Sentinel','train')
    fig_train_HR=hist_data(val_root_HR,pixel_max_HR, 'Sentinel','val')
    fig_train_HR=hist_data(test_root_HR,pixel_max_HR, 'Sentinel','test')
    fig_train_HR=hist_data(data_all_root_HR,pixel_max_HR, 'Sentinel','all')

    #######################################plot histogram###############################3
