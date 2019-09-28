"""
Create the dataset MASK
Crop masks.TIF in split of 512x512x1 
"""
from pathlib import Path
import timeit
import csv
from cropMasks import splits_masks


mask_path = Path('imagenes')
out_path_mask= '/home/jgonzalez/Test_2019/Test_network/data/train/masks'
myData = [["input_id", "source_id", "coordinates(rows,col)"]] 

myFile = open('splits_masks.csv', 'w')
with myFile:
    writer = csv.writer(myFile)
    writer.writerows(myData)  
    
    
input_filename0 = 'imagen0/maskout0.tif'
output_filename0 = 'masktif/rgbnir0{}-{}_a.tif'
output_filename_npy0 = 'rgbnir0{}-{}_a.npy'
splits_masks(mask_path,out_path_mask,input_filename0,output_filename0,output_filename_npy0)


input_filename1 = 'imagen1/maskout1.tif'
output_filename1 = 'masktif/rgbnir1{}-{}_a.tif'
output_filename_npy1 = 'rgbnir1{}-{}_a.npy'
splits_masks(mask_path,out_path_mask,input_filename1,output_filename1,output_filename_npy1)

input_filename2 = 'imagen2/maskout2.tif'
output_filename2 = 'masktif/rgbnir2{}-{}_a.tif'
output_filename_npy2 = 'rgbnir2{}-{}_a.npy'
splits_masks(mask_path,out_path_mask,input_filename2,output_filename2,output_filename_npy2)

input_filename3 = 'imagen3/maskout3.tif'
output_filename3 = 'masktif/rgbnir3{}-{}_a.tif'
output_filename_npy3 = 'rgbnir3{}-{}_a.npy'
splits_masks(mask_path,out_path_mask,input_filename3,output_filename3,output_filename_npy3)

input_filename4 = 'imagen4/maskout4.tif'
output_filename4 = 'masktif/rgbnir4{}-{}_a.tif'
output_filename_npy4 = 'rgbnir4{}-{}_a.npy'
splits_masks(mask_path,out_path_mask,input_filename4,output_filename4,output_filename_npy4)

input_filename5 = 'imagen5/maskout5.tif'
output_filename5 = 'masktif/rgbnir5{}-{}_a.tif'
output_filename_npy5 = 'rgbnir5{}-{}_a.npy'
splits_masks(mask_path,out_path_mask,input_filename5,output_filename5,output_filename_npy5)
#######################################################################################################################