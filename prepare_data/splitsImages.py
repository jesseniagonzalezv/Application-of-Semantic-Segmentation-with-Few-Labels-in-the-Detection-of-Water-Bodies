from pathlib import Path
import timeit
import csv
from cropImages import splits_images



data_path = Path('imagenes')
out_path_images = '/home/jgonzalez/Test_2019/Test_network/data/train/images'

myData = [["input_id", "source_id", "coordinates(rows,col)", "porcentaje"]]              
myFile = open('splits_images.csv', 'w')
with myFile:
    writer = csv.writer(myFile)
    writer.writerows(myData)  #list
####### Images ########################################################################################################
start = timeit.default_timer()
input_filename0  = 'imagen0/IMG_PER1_20170422154946_ORT_MS_003749.TIF'
output_filename0 = 'imagentif/rgbnir0{}-{}.tif'
output_filename_npy0 = 'rgbnir0{}-{}.npy'
output_filename_npyblack0 = 'rgbnir0{}-{}_b.npy'
splits_images(data_path,out_path_images,input_filename0,output_filename0,output_filename_npy0,output_filename_npyblack0)

end = timeit.default_timer()
print("elapsed time: {}".format(end-start))


input_filename1  = 'imagen1/IMG_PER1_20170422154946_ORT_MS_003131.TIF'
output_filename1 = 'imagentif/rgbnir1{}-{}.tif'
output_filename_npy1 = 'rgbnir1{}-{}.npy'
output_filename_npyblack1 = 'rgbnir1{}-{}_b.npy'
splits_images(data_path,out_path_images,input_filename1,output_filename1,output_filename_npy1,output_filename_npyblack1)

input_filename2  = 'imagen2/IMG_PER1_20170422154946_ORT_MS_002513.TIF'
output_filename2 = 'imagentif/rgbnir2{}-{}.tif'
output_filename_npy2 = 'rgbnir2{}-{}.npy'
output_filename_npyblack2 = 'rgbnir2{}-{}_b.npy'
splits_images(data_path,out_path_images,input_filename2,output_filename2,output_filename_npy2,output_filename_npyblack2)

input_filename3  = 'imagen3/IMG_PER1_20190703144250_ORT_MS_000672.TIF'
output_filename3 = 'imagentif/rgbnir3{}-{}.tif'
output_filename_npy3 = 'rgbnir3{}-{}.npy'
output_filename_npyblack3 = 'rgbnir3{}-{}_b.npy'
splits_images(data_path,out_path_images,input_filename3,output_filename3,output_filename_npy3,output_filename_npyblack3)

input_filename4  = 'imagen4/IMG_PER1_20190703144250_ORT_MS_001290.TIF'
output_filename4 = 'imagentif/rgbnir4{}-{}.tif'
output_filename_npy4 = 'rgbnir4{}-{}.npy'
output_filename_npyblack4 = 'rgbnir4{}-{}_b.npy'
splits_images(data_path,out_path_images,input_filename4,output_filename4,output_filename_npy4,output_filename_npyblack4)

input_filename5 = 'imagen5/IMG_PER1_20190703144250_ORT_MS_002526.TIF'
output_filename5= 'imagentif/rgbnir5{}-{}.tif'
output_filename_npy5= 'rgbnir5{}-{}.npy'
output_filename_npyblack5 = 'rgbnir5{}-{}_b.npy'
splits_images(data_path,out_path_images,input_filename5,output_filename5,output_filename_npy5,output_filename_npyblack5)
