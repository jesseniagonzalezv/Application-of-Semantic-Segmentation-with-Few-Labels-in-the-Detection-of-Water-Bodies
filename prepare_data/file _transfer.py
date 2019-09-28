#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 18:17:43 2019
Script: This is for making the validation mask data
"""

# ----------------Importing required packages---------------------------
import shutil
import glob
import os

#----------------making mask folder for validation data------------------------
original_dataset_dir = "../../data/dataset/val/images/"
data_dir  = "../../data/dataset/val/mask/"
label_dir  = "../../data/dataset/train/masks_old/"
# copying files to their correspondig folder-------------------------
## Getting names of all files in the folder------------------------------------
get_files_path = original_dataset_dir + "/*.*"
fpath_list = sorted(glob.glob(get_files_path))
for file in fpath_list:
    fname = str(file[30:-4] + "_a" + ".npy")
    print(fname)
    src = os.path.join(label_dir, fname)
    dst = os.path.join(data_dir, fname)
    shutil.copyfile(src, dst) 
#rename the masks_____________________________________________________

    
    

