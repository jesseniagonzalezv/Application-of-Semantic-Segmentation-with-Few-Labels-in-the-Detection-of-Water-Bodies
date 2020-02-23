"""
Create the Sentinel-2 dataset 
Images with RGBNIR bands  

Input:
Original dataset: 7670 images of 13 bands 
3 red, 2 green, 1 blue, 8 nir
https://eo-learn.readthedocs.io/en/latest/examples/visualization/EOPatchVisualization.html

Output:
Dataset: 7670 images of 4 bands (RGBNIR) 
Image output: C X H X W  
Image output: C X H X W  
C: 0 red, 1 green, 2 blue, 3 nir
Label:C X H X W  
"""
from pathlib import Path
import numpy as np
import cv2
import gzip
import rasterio
import matplotlib.pyplot as plt
import os

data_path = Path('eopatches_png')
image_pth= data_path/'jpg'/'eopatch-{}.jpg'

#Images
image_name_path='/home/jgonzalez/Test_2019/Test_network/eopatches/eopatch-{}/data/BANDS-S2-L1C.npy.gz'
#Labels
RGBNIR1_path='/home/jgonzalez/Test_2019/Test_network/eopatches/eopatch-{}/mask_timeless/water_label.npy.gz'

out_path_images = '/home/jgonzalez/Test_2019/Test_network/data_LR/train/images'
out_path_masks = '/home/jgonzalez/Test_2019/Test_network/data_LR/train/masks'

img_filename_npy_outpath = 'rgbnir{}.npy'
mask_filename_npy_outpath = 'rgbnir{}_a.npy'


for patches in range(0,7670): #0372 this images is not 
    img_path=str(image_name_path.format(str(patches).zfill(4)))
    f = gzip.GzipFile(img_path, "r")
    image_name=np.load(f)
    #print(image_name.shape)
    outpath_img_npy = str(os.path.join(out_path_images,img_filename_npy_outpath.format(int(patches))))
    outpath_mask_npy = str(os.path.join(out_path_masks,mask_filename_npy_outpath.format(int(patches))))

    arrayRGBNIR=np.zeros((64,64,4))
    arrayRGBNIR[:,:,0]=(image_name[0,:,:,3])
    arrayRGBNIR[:,:,1]=(image_name[0,:,:,2])
    arrayRGBNIR[:,:,2]=(image_name[0,:,:,1])
    arrayRGBNIR[:,:,3]=(image_name[0,:,:,8])
    arrayRGBNIR=arrayRGBNIR.transpose((2, 0, 1))   #image: C X H X W  other dataset
    #print(arrayRGBNIR.shape,arrayRGBNIR.dtype)
    np.save(outpath_img_npy,arrayRGBNIR)

    f_a = gzip.GzipFile(str(RGBNIR1_path.format(str(patches).zfill(4))), "r")
    mask=np.load(f_a)
    #print(RGBNIR1.shape)
    arraymask1=(mask).transpose((2, 0, 1))
    np.save(outpath_mask_npy,arraymask1)
print("completed dataset LR")

