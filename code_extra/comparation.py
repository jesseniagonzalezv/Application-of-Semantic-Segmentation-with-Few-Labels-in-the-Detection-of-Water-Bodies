'''
Compare the metrics in the 8% 20% 40% 80% of all data (850)
Comparation of model with only VHR,  parallel and sequential distillation
'''

import matplotlib.pyplot as plt
import numpy as np
import argparse
import re
import glob  

#############################################################################
def read_data(path, name):
    loss_file = open(path)
    filedata = loss_file.read()
    filedata = filedata.replace("bce",",bce")
    filedata = filedata.split(",")
    metric=[]
    for i in filedata:
        i = i.strip(" ")
        if str(i).startswith(name):
            i = i.split(" ")
            metric.append(float(i[1]))
    return metric
#############################################################################

def get_metrics(out_file,stage,name_file, name,name_model,fold_out,fold_in):
    path = ("predictions_{}/metric{}_{}_foldout{}_foldin{}.txt").format(out_file,name_file,name_model,fold_out,fold_in)
    loss_file = open(path)
    filedata = loss_file.readlines()
    value=[]
    for i in filedata:
        i = i.strip(" ")
        if str(i).startswith(name):
            i=re.split(' |,',i)
            value.append(float(i[1]))
    value_train = value[0]
    value_val = value[1]
    value_test = value[2]
    return value_train,value_val,value_test
    
#############################################################################
    
def metrics_values(name):
    f = open("comparative_metrics/compare_metrics_models_jaccard.txt", "w+")
    name_model='UNet11'
    #name= 'jaccard_metric' #'dice_metric'
    stage='test'
    fold_out=0

    for name_file in ['_8_percent','_20_percent','_40_percent','_80_percent']:
        for out_file in ['VHR','paral']:
            value_train_all=[]
            value_val_all=[]
            value_test_all=[]
            for  fold_in in range(5):
                value_train,value_val,value_test=get_metrics(out_file,stage,name_file, name,name_model,fold_out,fold_in)
                value_train_all.append(value_train)
                value_val_all.append(value_val)
                value_test_all.append(value_test)
                   
            print(("Train_{}{}{}_foldin0_to_foldin4,{},{:4f} \n").format(out_file,name_file,name_model,value_train_all,np.mean(value_train_all)))
            print(("Val_{}{}{}_foldin0_to_foldin4,{},{:4f} \n").format(out_file,name_file,name_model,value_val_all,np.mean(value_val_all)))
            print(("Test_{}{}{}_foldin0_to_foldin4,{},{:4f} \n").format(out_file,name_file,name_model,value_test_all,np.mean(value_test_all)))
            f.write(("Train_{}{}{}_foldin0_to_foldin4,{},{:4f} \n").format(out_file,name_file,name_model,value_train_all,np.mean(value_train_all)))
            f.write(("Val_{}{}{}_foldin0_to_foldin4,{},{:4f} \n").format(out_file,name_file,name_model,value_val_all,np.mean(value_val_all)))
            f.write(("Test_{}{}{}_foldin0_to_foldin4,{},{:4f}  \n").format(out_file,name_file,name_model,value_test_all,np.mean(value_test_all)))
            f.write("\n")
        

            
#############################################################################
#metrics_values(name='dice_metric')
#metrics_values(name='jaccard_metric')
#############################################################################
def comparative_dice(percent,name_model,fold_out,fold_in):
    dice_VHR=read_data(("/home/jgonzalez/Test_2019/Test_network/model_LR_VHR/predictions_VHR/pred_loss_test_{}_percent_{}_foldout{}_foldin{}.txt").format((percent),name_model,(fold_out), (fold_in)), name="jaccard") #name="dice")
    dice_dist_paral=read_data(("/home/jgonzalez/Test_2019/Test_network/model_LR_VHR/predictions_paral/pred_loss_test_{}_percent_{}_foldout{}_foldin{}.txt").format((percent),name_model,(fold_out), (fold_in)), name="jaccard") #name="dice")
    dice_dist_seq=read_data(("/home/jgonzalez/Test_2019/Test_network/model_LR_VHR/predictions_seq/pred_loss_test_{}_percent_{}_foldout{}_foldin{}.txt").format((percent),name_model,(fold_out), (fold_in)), name="jaccard") #name="dice")


    y_VHR = dice_VHR #np.asarray(dice_VHR)
    y_dist_paral = dice_dist_paral #np.asarray(dice_dist_paral)
    y_dist_seq = dice_dist_seq #np.asarray(dice_dist_seq)

    return y_VHR,y_dist_paral,y_dist_seq 

#############################################################################
def comparative_plot(percent,y_VHR,y_dist_paral,y_dist_seq,fold_out,fold_in,average='average'):
        plt.close('all')

        f, (ax1, ax2,ax3,ax4)=  plt.subplots(4,sharey=True, figsize=(25,10)) #plt.figure()
        x =np.asarray(list(range(0,len(y_VHR))))
        ax1.set_xlabel("Test images")
        ax2.set_xlabel("Test images")
        ax3.set_xlabel("Test images")
        ax4.set_xlabel("Test images")

        ax1.set_ylabel("Dice")
        ax2.set_ylabel("Dice")
        ax3.set_ylabel("Dice")
        ax4.set_ylabel("Dice")

        if average!='average':           
            name_title="predictions_with_dice_{}percent_{}_{}".format(percent,fold_out,fold_in)
            ax1.set_title(name_title) #f.suptitle(name_title)
        else:
            name_title="predictions_with_dice_{}percent_{}".format(percent,average)   
            ax1.set_title(name_title) #f.suptitle(name_title)
            
        ax1.plot(x,y_dist_paral ,'k:', label = 'y_dist_paral')
        ax1.plot(x,y_dist_seq ,'g:', label = 'y_dist_seq')
        ax1.plot(x,y_VHR ,'r:', label = 'y_VHR')
        ax1.set_xticks(x)
        ax1.legend()  # plot items
        ax1.grid(True)


        ax2.plot(x,y_VHR ,'r*', label = 'y_VHR')
        ax2.set_xticks(x)
        ax2.legend()  # plot items
        ax2.grid(True)

        #ax2.plot(3,1,2)
        ax3.plot(x,y_dist_paral ,'k+', label = 'y_dist_paral')
        ax3.set_xticks(x)
        ax3.legend()  # plot items
        ax3.grid(True)

        #ax2.plot(3,1,3)
        ax4.plot(x,y_dist_seq ,'g.', label = 'y_dist_seq')
        ax4.set_xticks(x)
        ax4.legend()  # plot items
        ax4.grid(True)
        
        
        plt.tight_layout()    
        plt.show()
        f.savefig(("comparative_metrics/comparative_{}.pdf").format(name_title), bbox_inches='tight')
        plt.close()

###################################################################################
def comparative_perce_plot(out_file='VHR',stage='test',name='dice'):
    name_model='UNet11'
    
    name_file_1='_8_percent'
    name_file_2='_20_percent'
    name_file_3='_40_percent'
    name_file_4='_80_percent'
    name_file_5='_100_percent'

    dice_in_1=[]
    dice_out_1=[]
    dice_in_2=[]
    dice_out_2=[]
    dice_in_3=[]
    dice_out_3=[]
    dice_in_4=[]
    dice_out_4=[]
    dice_in_5=[]
    dice_out_5=[]
    y_1_all=[]
    y_2_all=[]
    y_3_all=[]
    y_4_all=[]
    y_5_all=[]

    plt.close('all')

    f, (ax1, ax2,ax3,ax4,ax5,ax6)=  plt.subplots(6,sharey=True, figsize=(25,15)) #plt.figure()

    ax1.set_xlabel("Test images")
    ax2.set_xlabel("Test images")
    ax3.set_xlabel("Test images")
    ax4.set_xlabel("Test images")
    ax5.set_xlabel("Test images")
    ax6.set_xlabel("Test images")

    ax1.set_ylabel("Dice")
    ax2.set_ylabel("Dice")
    ax3.set_ylabel("Dice")
    ax4.set_ylabel("Dice")
    ax5.set_ylabel("Dice")
    ax6.set_ylabel("Dice")

        
    if  out_file == 'VHR':         
        ax1.set_title("VHR predictions{}".format(name_file_1)) 
        ax2.set_title("VHR predictions{}".format(name_file_2)) 
        ax3.set_title("VHR predictions{}".format(name_file_3)) 
        ax4.set_title("VHR predictions{}".format(name_file_4))
        ax5.set_title("VHR predictions{}".format(name_file_5))

        ax6.set_title("VHR all predictions/ Fold average") 

    elif out_file == 'paral': 
        ax1.set_title("Model Combined_predictions{}".format(name_file_1)) 
        ax2.set_title("Model Combined_predictions{}".format(name_file_2)) 
        ax3.set_title("Model Combined_predictions{}".format(name_file_3)) 
        ax4.set_title("Model Combined_predictions{}".format(name_file_4))
        ax5.set_title("Model Combined_predictions{}".format(name_file_5)) 
        ax6.set_title("Model Combined/ Fold Average") 
            
    fold_out=0
    for fold_in in range(5):

            path1=(("predictions_{}/pred_loss_{}{}_{}_foldout{}_foldin{}.txt").format(out_file,stage,name_file_1,name_model,fold_out, fold_in))
            path2=(("predictions_{}/pred_loss_{}{}_{}_foldout{}_foldin{}.txt").format(out_file,stage,name_file_2,name_model,fold_out, fold_in))
            path3=(("predictions_{}/pred_loss_{}{}_{}_foldout{}_foldin{}.txt").format(out_file,stage,name_file_3,name_model,fold_out, fold_in))
            path4=(("predictions_{}/pred_loss_{}{}_{}_foldout{}_foldin{}.txt").format(out_file,stage,name_file_4,name_model,fold_out, fold_in))
            path5=(("predictions_{}/pred_loss_{}{}_{}_foldout{}_foldin{}.txt").format(out_file,stage,name_file_5,name_model,fold_out, fold_in))


            dice_VHR_1=read_data(path1, name)
            dice_in_1.append(np.mean(dice_VHR_1))
            print('8 values')
            print('24',dice_VHR_1[23])
            print('89',dice_VHR_1[88])
            print('91',dice_VHR_1[90])


            '''
            print('dice_VHR_8',np.mean(dice_VHR_1),dice_in_1)
            print('bad_sample',dice_VHR_1[89])'''
            
            dice_VHR_2=read_data(path2, name)
            dice_in_2.append(np.mean(dice_VHR_2))
            #print('dice_VHR_20',np.mean(dice_VHR_2),dice_in_2)

            dice_VHR_3=read_data(path3, name)
            dice_in_3.append(np.mean(dice_VHR_3))
            #print('dice_VHR_40',np.mean(dice_VHR_3),dice_in_3)

            dice_VHR_4=read_data(path4, name)
            dice_in_4.append(np.mean(dice_VHR_4))
            #print('dice_VHR_80',np.mean(dice_VHR_4),dice_in_4)
            
            dice_VHR_5=read_data(path5, name)
            dice_in_5.append(np.mean(dice_VHR_5))
            print('80 values')
            print('24',dice_VHR_4[23])
            print('89',dice_VHR_4[88])
            print('91',dice_VHR_4[90])
            
            print('\n mean 80--')
            print('14',dice_VHR_4[13])
            print('12',dice_VHR_4[11])
            print('42',dice_VHR_4[41])
            print('74',dice_VHR_4[73])
            print('93',dice_VHR_4[92])   
            print('90',dice_VHR_4[89])
            y_VHR_1 = np.asarray(dice_VHR_1)
            y_VHR_2 = np.asarray(dice_VHR_2)
            y_VHR_3 = np.asarray(dice_VHR_3)
            y_VHR_4 = np.asarray(dice_VHR_4)
            y_VHR_5 = np.asarray(dice_VHR_5)


            x =np.asarray(list(range(0,len(y_VHR_1))))
            #print('fout{}_fin{}'.format(fold_out, fold_in))
            ax1.plot(x,y_VHR_1,label = (('fout{}_fin{}').format(fold_out, fold_in)))
            ax1.set_xticks(x)
            ax1.legend()  # plot items
            ax1.grid(True)
            ax2.plot(x,y_VHR_2,label = (('fout{}_fin{}').format(fold_out, fold_in)))
            ax2.set_xticks(x)
            ax2.legend()  # plot items
            ax2.grid(True)
            ax3.plot(x,y_VHR_3,label = (('fout{}_fin{}').format(fold_out, fold_in)))
            ax3.set_xticks(x)
            ax3.legend()  # plot items
            ax3.grid(True)
            ax4.plot(x,y_VHR_4,label = (('fout{}_fin{}').format(fold_out, fold_in)))
            ax4.set_xticks(x)
            ax4.legend()  # plot items
            ax4.grid(True)
            ax5.plot(x,y_VHR_5,label = (('fout{}_fin{}').format(fold_out, fold_in)))
            ax5.set_xticks(x)
            ax5.legend()  # plot items
            ax5.grid(True)

            y_1_all.append(y_VHR_1)
            y_2_all.append(y_VHR_2)
            y_3_all.append(y_VHR_3)
            y_4_all.append(y_VHR_4)
            y_5_all.append(y_VHR_5)

 
    dice_out_1.append(np.mean(dice_in_1))
    print('dice_out_mean_8',np.mean(dice_in_1),dice_out_1)      
    dice_out_2.append(np.mean(dice_in_2))
    print('dice_out_mean_20',np.mean(dice_in_2),dice_out_2)  
    dice_out_3.append(np.mean(dice_in_3))
    print('dice_out_mean_40',np.mean(dice_in_3),dice_out_3)
    dice_out_4.append(np.mean(dice_in_4))
    print('dice_out_mean_80',np.mean(dice_in_4),dice_out_4)
    dice_out_5.append(np.mean(dice_in_5))
    print('dice_out_mean_100',np.mean(dice_in_5),dice_out_5,'\n')

    y_1_all=np.mean(y_1_all,axis=0) #in this case there is only 1 fold_out it is not needed
    y_2_all=np.mean(y_2_all,axis=0)
    y_3_all=np.mean(y_3_all,axis=0)
    y_4_all=np.mean(y_4_all,axis=0)
    y_5_all=np.mean(y_5_all,axis=0)
    print('mean 8')
    print('24',y_1_all[23])
    print('89',y_1_all[88])
    print('91',y_1_all[90])
    print('mean 80')
    print('24',y_4_all[23])
    print('89',y_4_all[88])
    print('91',y_4_all[90])
    
    print('mean 80')
    print('14',y_4_all[13])
    print('12',y_4_all[11])
    print('42',y_4_all[41])
    print('74',y_4_all[73])
    print('93',y_4_all[92])   
    print('90',y_4_all[89])

    ax6.plot(x,y_1_all ,'.', label = '8_percent')
    ax6.plot(x,y_2_all ,'*', label = '20_percent')
    ax6.plot(x,y_3_all ,'+', label = '40_percent')
    ax6.plot(x,y_4_all ,'^', label = '80_percent')
    ax6.plot(x,y_5_all ,'^', label = '100_percent')

    ax6.legend()  # plot items

    ax6.set_xticks(x)
    ax6.grid(True)

    plt.tight_layout()    
    plt.show()
    f.savefig("comparative_metrics/metric_by_percent_{}_all.pdf".format(out_file), bbox_inches='tight')
    plt.close()
    
    return y_1_all,y_2_all,y_3_all,y_4_all,y_5_all  # average 8,20,40,80
#############################################################################
print(' -dice')
comparative_perce_plot(out_file='VHR',stage='test',name='dice')
print('jaccard')
comparative_perce_plot(out_file='VHR',stage='test',name='jaccard')
print(' ---------------------------------------------------------------------')

print('parala')
print('parala -dice')

comparative_perce_plot(out_file='paral',stage='test',name='dice')
print('parala -jacara')

comparative_perce_plot(out_file='paral',stage='test',name='jaccard')

#############################################################################


#############################################################################
def water_images_v1(path):
    test_file_names =  np.array(sorted(glob.glob(str(path) + "/*.npy")))
    area_percent =[]
    name_file_1='_8_percent'
    name_file_2='_20_percent'
    name_file_3='_40_percent'
    name_file_4='_80_percent'
    name_file_5='_100_percent'

    for path_mask in test_file_names:
        mask = np.load(str(path_mask))
        mask=mask.transpose(1, 2, 0).reshape(mask.shape[1],-1)
        mask=(mask > 0).astype(np.uint8) #512,512
        plt.imshow(mask)
        
        equals0=(mask==1).astype(np.uint8)        
        sum_percent=np.sum(equals0)/(512*512)*100+50 #*100
        area_percent.append(sum_percent)
        #print(sum_percent)
    pvalue_1,pvalue_2,pvalue_3,pvalue_4,pvalue_5=comparative_perce_plot(out_file='VHR',stage='test') #considering shuffle=False
    pvalue_a,pvalue_b,pvalue_c,pvalue_d,pvalue_e=comparative_perce_plot(out_file='paral',stage='test') #considering shuffle=False

    plt.close('all')

    f, (ax1, ax2,ax3,ax4,ax5)=  plt.subplots(5,sharey=True, figsize=(25,15)) #plt.figure()

    ax1.set_xlabel("Test images")  #("Water % in the image")
    ax2.set_xlabel("Test images") #("Water % in the image")
    ax3.set_xlabel("Test images") #("Water % in the image")
    ax4.set_xlabel("Test images") #("Water % in the image")
    ax5.set_xlabel("Test images") #("Water % in the image")

    ax1.set_ylabel("Dice")
    ax2.set_ylabel("Dice")
    ax3.set_ylabel("Dice")
    ax4.set_ylabel("Dice")
    ax5.set_ylabel("Dice")

    ax1.set_title("Water % in the image vs Dice {}".format(name_file_1)) 
    ax2.set_title("Water % in the image vs Dice {}".format(name_file_2)) 
    ax3.set_title("Water % in the image vs Dice {}".format(name_file_3)) 
    ax4.set_title("Water % in the image vs Dice {}".format(name_file_4))
    ax5.set_title("Water % in the image vs Dice {}".format(name_file_5))
  
        
    x =np.asarray(list(range(0,len(area_percent))))
    c = np.sqrt(area_percent)
    print(np.shape(x),np.shape(pvalue_1),np.shape(area_percent))
    #ax1.scatter(x,pvalue_1 ,s=area_percent,c=c, label = 'VHR_8_percent')
    ax1.scatter(x,pvalue_1 ,s=area_percent, label = 'VHR_8_percent')
    ax1.scatter(x,pvalue_a ,s=area_percent, label = 'Combined_8_percent')
    ax1.set_xticks(x)
    ax1.legend()  # plot items
    ax1.grid(True)
        
    ax2.scatter(x,pvalue_2 ,s=area_percent, label = 'VHR_20_percent')
    ax2.scatter(x,pvalue_b ,s=area_percent, label = 'Combined_20_percent')
    ax2.set_xticks(x)
    ax2.legend()  # plot items
    ax2.grid(True)
    
    ax3.scatter(x,pvalue_3,s=area_percent, label = 'VHR_40_percent')
    ax3.scatter(x,pvalue_c,s=area_percent, label = 'Combined_40_percent')
    ax3.set_xticks(x)
    ax3.legend()  # plot items
    ax3.grid(True)
    
    ax4.scatter(x,pvalue_4 ,s=area_percent, label = 'VHR_80_percent')
    ax4.scatter(x,pvalue_d ,s=area_percent, label = 'Combined_80_percent')
    ax4.set_xticks(x)
    ax4.legend()  # plot items
    ax4.grid(True)
    
    ax5.scatter(x,pvalue_5 ,s=area_percent, label = 'VHR_100_percent')
    ax5.scatter(x,pvalue_e ,s=area_percent, label = 'Combined_100_percent')
    ax5.set_xticks(x)
    ax5.legend()  # plot items
    ax5.grid(True)

    plt.tight_layout()    
    plt.show()
    f.savefig("comparative_metrics/sample_vs_metric_all_water.pdf", bbox_inches='tight')
    plt.close()
###########    
#water_images_v1("data_VHR/test_850/masks")


#############################################################################
def water_images_v2(path):
    test_file_names =  np.array(sorted(glob.glob(str(path) + "/*.npy")))
    area_percent =[]
    name_file_1='_8_percent'
    name_file_2='_20_percent'
    name_file_3='_40_percent'
    name_file_4='_80_percent'
    name_file_5='_100_percent'

    for path_mask in test_file_names:
        mask = np.load(str(path_mask))
        mask=mask.transpose(1, 2, 0).reshape(mask.shape[1],-1)
        mask=(mask > 0).astype(np.uint8) #512,512
        plt.imshow(mask)
        
        equals0=(mask==1).astype(np.uint8)        
        sum_percent=np.sum(equals0)/(512*512)*100 #*100
        area_percent.append(sum_percent)
        #print(sum_percent)
    pvalue_1,pvalue_2,pvalue_3,pvalue_4,pvalue_5=comparative_perce_plot(out_file='VHR',stage='test') #considering shuffle=False
    pvalue_a,pvalue_b,pvalue_c,pvalue_d,pvalue_e=comparative_perce_plot(out_file='paral',stage='test') #considering shuffle=False

    plt.close('all')

    f, (ax1, ax2,ax3,ax4,ax5)=  plt.subplots(5,sharey=True, figsize=(18,15)) #plt.figure()

    ax1.set_xlabel("Water % in the image")
    ax2.set_xlabel("Water % in the image")
    ax3.set_xlabel("Water % in the image")
    ax4.set_xlabel ("Water % in the image")
    ax5.set_xlabel ("Water % in the image")

    ax1.set_ylabel("Dice")
    ax2.set_ylabel("Dice")
    ax3.set_ylabel("Dice")
    ax4.set_ylabel("Dice")
    ax5.set_ylabel("Dice")

    ax1.set_title("Water % in the image vs Dice {}".format(name_file_1)) 
    ax2.set_title("Water % in the image vs Dice {}".format(name_file_2)) 
    ax3.set_title("Water % in the image vs Dice {}".format(name_file_3)) 
    ax4.set_title("Water % in the image vs Dice {}".format(name_file_4))
    ax5.set_title("Water % in the image vs Dice {}".format(name_file_5))

    x =np.asarray(list(range(0,len(area_percent))))
    c = np.sqrt(area_percent)
    print(np.shape(x),np.shape(pvalue_1),np.shape(area_percent))
    #ax1.scatter(x,pvalue_1 ,s=area_percent,c=c, label = 'VHR_8_percent')
    ax1.scatter(area_percent,pvalue_1 ,s=40,color='red', label = 'VHR_8_percent')
    ax1.scatter(area_percent, pvalue_a ,s=30,marker='^', label = 'Combined_8_percent')
    #ax1.set_xticks(x)
    ax1.legend()  # plot items
    ax1.grid(True)
        
    ax2.scatter(area_percent,pvalue_2 ,s=40,color='red',  label = 'VHR_20_percent')
    ax2.scatter(area_percent,pvalue_b ,s=30,marker='^',  label = 'Combined_20_percent')
    #ax2.set_xticks(x)
    ax2.legend()  # plot items
    ax2.grid(True)
    
    ax3.scatter(area_percent,pvalue_3,s=40,color='red',  label = 'VHR_40_percent')
    ax3.scatter(area_percent,pvalue_c,s=30,marker='^',  label = 'Combined_40_percent')
    #ax3.set_xticks(x)
    ax3.legend()  # plot items
    ax3.grid(True)
    
    ax4.scatter(area_percent,pvalue_4 ,s=40,color='red',  label = 'VHR_80_percent')
    ax4.scatter(area_percent,pvalue_d ,s=30,marker='^',  label = 'Combined_80_percent')
    #ax4.set_xticks(x)
    ax4.legend()  # plot items
    ax4.grid(True)
    
    ax5.scatter(area_percent,pvalue_5 ,s=40,color='red',  label = 'VHR_100_percent')
    ax5.scatter(area_percent,pvalue_e ,s=30,marker='^',  label = 'Combined_100_percent')
    #ax4.set_xticks(x)
    ax5.legend()  # plot items
    ax5.grid(True)

    plt.tight_layout()    
    plt.show()
    f.savefig("comparative_metrics/water_vs_metric_all.pdf", bbox_inches='tight')
    plt.close()
#############################################################################
#water_images_v2("data_VHR/test_850/masks")

#############################################################################
#############################################################################

def main():   
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--percent', type=str, default='8,20,40,80', help='For example VHR, paral, seq')
    arg('--name-model', type=str, default='UNet11', choices=['UNet11','UNet','AlbuNet34','SegNet'])
    arg('--fold-out', type=str, default='0', help='fold train test')
    arg('--fold-in', type=str, default='0,1,2,3,4', help='fold train val')
    

    args = parser.parse_args()    
        
    if args.percent:
            percent = list(map(int, args.percent.split(',')))
        
    if args.fold_out:
            fold_out = list(map(int, args.fold_out.split(',')))
       
    if args.fold_in:
            fold_in = list(map(int, args.fold_in.split(',')))
                
        
    #print(percent, fold_out, fold_in)    
    #### all of the same percent
    for perc in percent:
        #print(perc)    
        y_VHR_all=[]
        y_dist_paral_all=[]
        y_dist_seq_all=[]
        #print('0-vhr',y_VHR_all,'paral',y_dist_paral_all,'seq',y_dist_seq_all)   

        for i in (fold_out):
            #print(perc, i)    
            for j in (fold_in):
                #print(perc, i, j)    

                y_VHR,y_dist_paral,y_dist_seq= comparative_dice(perc,name_model=args.name_model,fold_out=int(i),fold_in=int(j)) 
                #print('2-0-vhr',y_VHR,'paral',y_dist_paral,'seq',y_dist_seq)   


                comparative_plot(perc, y_VHR, y_dist_paral, y_dist_seq,fold_out=int(i),fold_in=int(j),average='noaverage')
                
                y_VHR_all.append(y_VHR)
                #print('2-1-vhr',y_VHR_all,'paral',y_dist_paral_all,'seq',y_dist_seq_all)   

                y_dist_paral_all.append(y_dist_paral)
                #print('2-2-vhr',y_VHR_all,'paral',y_dist_paral_all,'seq',y_dist_seq_all)   

                y_dist_seq_all.append(y_dist_seq)
                #print('1-vhr',y_VHR,'paral',y_dist_paral,'seq',y_dist_seq)   #(y_VHR[89],y_dist_paral[89])
        #print('2-hr',y_HR_all,'paral',y_dist_paral_all,'seq',y_dist_seq_all)   

        y_VHR_all=np.mean(y_HR_all,axis=0)
        y_dist_paral_all=np.mean(y_dist_paral_all,axis=0)
        y_dist_seq_all=np.mean(y_dist_seq_all,axis=0)       
        print('3-vhr',y_VHR_all,'paral',y_dist_paral_all,'seq',y_dist_seq_all)   
        comparative_plot(perc,y_VHR_all,y_dist_paral_all,y_dist_seq_all,fold_out=int(i),fold_in=int(j),average='average')

    return y_VHR_all,y_dist_paral_all,y_dist_seq

#all_comparative_dice(percent=[6],name_model='UNet11',fold_out=[0],fold_in=[0,1,2,3,4])
#all_comparative_dice(percent=[70],name_model='UNet11',fold_out=[0],fold_in=[0,2,4])

#if __name__ == '__main__':
#    main()
