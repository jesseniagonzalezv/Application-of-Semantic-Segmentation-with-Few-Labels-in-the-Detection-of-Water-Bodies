#945 HR Perusat Images
# 116 HR_dataset +29 test_HR// 800 fake_LR_dataset 
#
# 116 (12%)       80%  20%  
#        train val  test
#145_HR   93   23    29

# 116 (12%)       80%  20%  
#        train val  test
#145_HR   50   12    29

#800      70%  20%   10%
#        train val  test
#800_LR   560  160   80
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from get_train_val_1 import split_train_val
from get_test_2 import split_train_test
from transfer_maks import obtained_mask
from copy_samples_3 import n_samples

########################################################################
split_train_val(original_dataset_dir= '/home/jgonzalez/Test_2019/Test_network/model_LR_HR/data_HR/data/images',base_dir = "data_HR/dist_per",validation_split = 0.12664, train_file='train_LR',val_file='train_HR')


split_train_test(original_dataset_dir_total='/home/jgonzalez/Test_2019/Test_network/model_LR_HR_paralel/data_HR/dist_per/train_HR/images', base_dir = "data_HR/dist_per",test_split = 0.2, train_file='train_HR',test_file='val_HR') 

split_train_test(original_dataset_dir_total='/home/jgonzalez/Test_2019/Test_network/model_LR_HR_paralel/data_HR/dist_per/train_LR/images', base_dir = "data_HR/dist_per",test_split = 0.2, train_file='train_LR',test_file='val_LR') 

split_train_test(original_dataset_dir_total='/home/jgonzalez/Test_2019/Test_network/model_LR_HR_paralel/data_HR/dist_per/train_LR/images', base_dir = "data_HR/dist_per",test_split = 0.125, train_file='train_LR',test_file='test_LR') 


obtained_mask(original_dataset_dir_train="data_HR/dist_per/train_LR/images/",label_dir_train='/home/jgonzalez/Test_2019/Test_network/model_HR/data_HR/data/masks/',data_dir_train="data_HR/dist_per/train_LR/masks/" ,original_dataset_dir_val="data_HR/dist_per/val_LR/images",label_dir_val='/home/jgonzalez/Test_2019/Test_network/model_HR/data_HR/data/masks/',data_dir_val="data_HR/dist_per/val_LR/masks/" ,original_dataset_dir_test="data_HR/dist_per/test_LR/images",label_dir_test='/home/jgonzalez/Test_2019/Test_network/model_HR/data_HR/data/masks/',data_dir_test="data_HR/dist_per/test_LR/masks/")

obtained_mask(original_dataset_dir_train="data_HR/dist_per/train_HR/images/",label_dir_train='/home/jgonzalez/Test_2019/Test_network/model_HR/data_HR/data/masks/',data_dir_train="data_HR/dist_per/train_HR/masks/" ,original_dataset_dir_val="data_HR/dist_per/val_HR/images",label_dir_val='/home/jgonzalez/Test_2019/Test_network/model_HR/data_HR/data/masks/',data_dir_val="data_HR/dist_per/val_HR/masks/" ,original_dataset_dir_test="data_HR/dist_per/test_HR/images",label_dir_test='/home/jgonzalez/Test_2019/Test_network/model_HR/data_HR/data/masks/',data_dir_test="data_HR/dist_per/test_HR/masks/")



#########split a small data
#get samples train +val

n_samples(original_dataset_dir= "/home/jgonzalez/Test_2019/Test_network/model_LR_HR_paralel/data_HR/dist_per/train_HR/images",base_dir = "data_HR/dist_per", train_n ='train_HR_60',split = 0.64) #60):
    
split_train_test(original_dataset_dir_total='/home/jgonzalez/Test_2019/Test_network/model_LR_HR_paralel/data_HR/dist_per/train_HR_60/images'
,base_dir = "data_HR/dist_per",test_split = 0.166, train_file='train_HR_60',test_file='val_HR_60') 


obtained_mask(mode="notest",original_dataset_dir_train="data_HR/dist_per/train_HR_60/images/",label_dir_train='/home/jgonzalez/Test_2019/Test_network/model_HR/data_HR/data/masks/',data_dir_train="data_HR/dist_per/train_HR_60/masks" ,original_dataset_dir_val="data_HR/dist_per/val_HR_60/images",label_dir_val='/home/jgonzalez/Test_2019/Test_network/model_HR/data_HR/data/masks/',data_dir_val="data_HR/dist_per/val_HR_60/masks/")
########################################################################

# create a test file with 10 % of all the dataset 945 


from get_train_val_1 import split_train_val
from get_test_2 import split_train_test
from transfer_maks_4 import obtained_mask
from copy_samples_3 import n_samples

split_train_val(original_dataset_dir= '/home/jgonzalez/Test_2019/Test_network/model_LR_HR/data_HR/data/images',base_dir = "data_HR",validation_split = 0.10, train_file='train_val_HR_dist',val_file='test_HR_dist')
## run then in separate 
obtained_mask(mode="val",original_dataset_dir_train="data_HR/train_val_HR_dist/images/",label_dir_train='/home/jgonzalez/Test_2019/Test_network/model_LR_HR/data_HR/data/masks/',data_dir_train="data_HR/train_val_HR_dist/masks" ,original_dataset_dir_val="data_HR/test_HR_dist/images",label_dir_val='/home/jgonzalez/Test_2019/Test_network/model_LR_HR/data_HR/data/masks/',data_dir_val="data_HR/test_HR_dist/masks/")


#####################
# create a sample of 60 images of the data before 851 sample also transfer the mask dist_60
from get_train_val_1 import split_train_val
from get_test_2 import split_train_test
from transfer_maks_4 import obtained_mask
from copy_samples_3 import n_samples
#de train_val_dist use 70 images with the same test dist_hr
n_samples(original_dataset_dir= "/home/jgonzalez/Test_2019/Test_network/model_LR_HR/data_HR/train_val_HR_dist/images",base_dir = "data_HR", train_n ='train_val_dist_60',split = 0.07) #60):


obtained_mask(mode="noall",original_dataset_dir_train="data_HR/train_val_dist_60/images/",label_dir_train='/home/jgonzalez/Test_2019/Test_network/model_LR_HR/data_HR/data/masks/',data_dir_train="data_HR/train_val_dist_60/masks")
#####################
