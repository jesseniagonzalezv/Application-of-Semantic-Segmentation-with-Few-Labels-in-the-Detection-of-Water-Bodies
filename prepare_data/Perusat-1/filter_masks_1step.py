'''
Code to get more precise label of the complete image (approximately 6000x6000 size)
output: C X H X W filter labels

'''
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import rasterio as rio
import cv2



def cal_area(arraymask1,min_size):
    #find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(arraymask1, connectivity=8)
    #connectedComponentswithStats yields every seperated component with information on each of them, such as size
    #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    #min_size = 100 #114 imagen1   

    maskout = np.zeros((output.shape))
    #for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            maskout[output == i + 1] = 1
    return maskout


def filter_small_area(masks_path,min_size_water,min_size_nowater):
    
    #read the masks  water
    mask = rio.open(masks_path)    
    #mask1.indexes
    arraymask=(mask.read()).transpose(1,2,0)    
    arraymask=(arraymask[:,:,0]).astype(np.uint8)  #importante uint8
    N=arraymask.shape[0]
    M=arraymask.shape[1]
    ones=np.ones((N,M))    

    #print('mask', arraymask1.shape) # Ch,Hy,Wx tensor
    #plt.imshow(arraymask1)  
    maskout=cal_area(arraymask, min_size_water) #first filter
    #masks  water
    arraymasknowater=(ones-maskout).astype(np.uint8) #no water
    #print('mask', arraymask1nowater.shape) # Ch,Hy,Wx tensor
    maskout2=cal_area(arraymasknowater, min_size_nowater) #second filter no water
    mask_outf2=(ones-maskout2).astype(np.uint8)
    return mask_outf2,N,M

def save_raster(inpath_mask,maskout,outpath_mask,N,M):
    #save the new mask1 water only filter

    array_out=np.zeros((1,N,M))
    #print(maskout.shape)
    array_out[0,:,:]=maskout
    #print(array_out.shape)

    with rio.open(inpath_mask) as inds:
          meta = inds.meta.copy()

    with rio.open(outpath_mask, 'w', **meta) as outds:          
          outds.write(array_out)  


            


mask_path = Path('imagenes')
masks_0 = os.path.join(mask_path, 'imagen0/maskout0.tif') 
masks_1 = os.path.join(mask_path, 'imagen1/maskout1.tif') 
masks_2 = os.path.join(mask_path, 'imagen2/maskout2.tif') 
masks_3 = os.path.join(mask_path, 'imagen3/maskout3.tif') 
masks_4 = os.path.join(mask_path, 'imagen4/maskout4.tif') 
masks_5 = os.path.join(mask_path, 'imagen5/maskout5.tif') 
masks_6 = os.path.join(mask_path, 'imagen6_conida/maskout6.tif') 

masksout_0 = os.path.join(mask_path,'imagen0/maskoutf0.tif')
masksout_1 = os.path.join(mask_path,'imagen1/maskoutf1.tif')
masksout_2 = os.path.join(mask_path,'imagen2/maskoutf2.tif')
masksout_3 = os.path.join(mask_path,'imagen3/maskoutf3.tif')
masksout_4 = os.path.join(mask_path,'imagen4/maskoutf4.tif')
masksout_5 = os.path.join(mask_path,'imagen5/maskoutf5.tif')
masksout_6 = os.path.join(mask_path,'imagen6_conida/maskoutf6.tif')

maskout0,N0,M0=filter_small_area(masks_0, min_size_water=50,min_size_nowater=50) 
maskout1,N1,M1=filter_small_area(masks_1, min_size_water=100,min_size_nowater=100) 
maskout2,N2,M2=filter_small_area(masks_2, min_size_water=100,min_size_nowater=25) #297 ,78
maskout3,N3,M3=filter_small_area(masks_3, min_size_water=53,min_size_nowater=44) #  150,125
maskout4,N4,M4=filter_small_area(masks_4, min_size_water=50,min_size_nowater=25) #  180 ,156
maskout5,N5,M5=filter_small_area(masks_5, min_size_water=14,min_size_nowater=17) # 40 ,50
maskout6,N6,M6=filter_small_area(masks_6, min_size_water=700,min_size_nowater=2) #1500 no watermm


save_raster(masks_0,maskout0,masksout_0,N0,M0)
save_raster(masks_1,maskout1,masksout_1,N1,M1)
save_raster(masks_2,maskout2,masksout_2,N2,M2)
save_raster(masks_3,maskout3,masksout_3,N3,M3)
save_raster(masks_4,maskout4,masksout_4,N4,M4)
save_raster(masks_5,maskout5,masksout_5,N5,M5)
save_raster(masks_6,maskout6,masksout_6,N6,M6)
