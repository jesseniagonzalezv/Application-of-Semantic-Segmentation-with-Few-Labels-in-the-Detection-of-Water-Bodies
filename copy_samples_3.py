#####################
'''
#916 total
#     80%  20%  
#    train val  test
#916   733   183   29

#      90%  10%  
#560   504   56     60% de 916
#400   360   40     43.66%
#240   216   24     26.20%
#100    90   10     9.8%

'''
#####################
import numpy as np
import shutil
import glob
import os
from pathlib import Path

########################################################################

def n_samples(original_dataset_dir= "data_HR/data/images",base_dir = "data_HR" 
, train_n='train_400',split = 0.437): #400):
    get_files_path = original_dataset_dir + "/*.npy"
    fpath_list = sorted(glob.glob(get_files_path))


    # make the directories


    #train_dir = os.path.join(base_dir, 'train_100','images')
    train_dir0 = os.path.join(base_dir, train_n)
    if not os.path.exists(train_dir0):
            os.mkdir(train_dir0)
            
    train_dir = os.path.join(base_dir, train_n,'images')
    if not os.path.exists(train_dir):
            os.mkdir(train_dir)

    ###############################################################################
    #split = 0.11 #100
    #split = 0.437 #400

    dataset_size = len(fpath_list)
    indices = list(range(dataset_size))
    split = int(np.floor(split * dataset_size))
    if 1 :
        np.random.seed(13)
        np.random.shuffle(indices)
    extra_indices, train_indices = indices[split:], indices[:split]
    print(dataset_size,len(train_indices), len(extra_indices))
    ############################################################


    for i in train_indices:
        fname = fpath_list[i]
        fname = fname.split("/")
        fname = (fname[-1])
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_dir, fname)
        shutil.copyfile(src, dst)    

              