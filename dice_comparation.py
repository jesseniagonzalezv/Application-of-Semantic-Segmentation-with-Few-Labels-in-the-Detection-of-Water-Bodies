import matplotlib.pyplot as plt
import numpy as np


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


def comparative_dice(percent,name_model,fold_out,fold_in):
    dice_HR=read_data(("/home/jgonzalez/Test_2019/Test_network/model_LR_HR/predictions_HR/pred_loss_test_{}_percent_{}_foldout{}_foldin{}.txt").format((percent),name_model,(fold_out), (fold_in)), name="dice")
    dice_dist_paral=read_data(("/home/jgonzalez/Test_2019/Test_network/model_LR_HR/predictions_paral/pred_loss_test_{}_percent_{}_foldout{}_foldin{}.txt").format((percent),name_model,(fold_out), (fold_in)), name="dice")
    dice_dist_seq=read_data(("/home/jgonzalez/Test_2019/Test_network/model_LR_HR/predictions_seq/pred_loss_test_{}_percent_{}_foldout{}_foldin{}.txt").format((percent),name_model,(fold_out), (fold_in)), name="dice")
    #dice_dist_100=read_data("/home/jgonzalez/Test_2019/Test_network/model_LR_HR_paralel/predictions/pred_loss_dist_paral_100.txt",name="dice")

    y_HR = np.asarray(dice_HR)
    y_dist_paral = np.asarray(dice_dist_paral)
    y_dist_seq = np.asarray(dice_dist_seq)


    return y_HR,y_dist_paral,y_dist_seq  

def comparative_plot(percent,y_HR,y_dist_paral,y_dist_seq,fold_out,fold_in,average='average'):
        plt.close('all')

        f = plt.figure()

        x =np.asarray(list(range(0,len(y_HR))))
        plt.xlabel("Number of test images")
        plt.ylabel("Dice")
        if average!='average':
            plt.title("predictions_with_dice_{}percent_{}_{}".format(percent,fold_out,fold_in)) 
        else:
            plt.title("predictions_with_dice_{}percent_{}".format(percent,average)) 

        plt.plot(x,y_HR ,'r-', label = 'y_HR')
        plt.plot(x,y_dist_paral ,'k-', label = 'y_dist_paral')
        plt.plot(x,y_dist_seq ,'g-', label = 'y_dist_seq')

        plt.legend()
        plt.show()

def samecomparative_plot(percent,y_z,fold_out,fold_in,average='average'):

        f = plt.figure()

        x =np.asarray(list(range(0,len(y_z))))
        plt.xlabel("Number of test images")
        plt.ylabel("Dice")
        if average!='average':
            plt.title("predictions_with_dice_{}percent_{}_{}".format(percent,fold_out,fold_in)) 
        else:
            plt.title("predictions_with_dice_{}percent_{}".format(percent,average)) 

        plt.plot(x,y_HR ,'r-', label = 'y_HR')
        plt.plot(x,y_dist_paral ,'k-', label = 'y_dist_paral')
        plt.plot(x,y_dist_seq ,'b-', label = 'y_dist_seq')

        plt.legend()
        plt.show()

    
    #f.savefig("predictions/comparation_dice.pdf", bbox_inches='tight')
def all_comparative_dice(percent=[6,70],fold_out=[0,1],fold_in=[0,1,2,3,4],name_model='UNet11'):
    y_HR_all=y_dist_paral_all=y_dist_seq_all=[]
    #### all of the same percent
    for perc in percent:
        for i in (fold_out):
            for j in (fold_in):
                
                y_HR,y_dist_paral,y_dist_seq= comparative_dice(perc,name_model='UNet11',fold_out=int(i),fold_in=int(j)) 
                comparative_plot(perc, y_HR, y_dist_paral, y_dist_seq,fold_out=int(i),fold_in=int(j),average='noaverage')
                
                y_HR_all.append(y_HR)
                y_dist_paral_all.append(y_dist_paral)
                y_dist_seq_all.append(y_dist_seq)
    
        y_HR_all=np.mean(y_HR_all,axis=0)
        y_dist_paral_all=np.mean(y_dist_paral_all,axis=0)
        y_dist_seq_all=np.mean(y_dist_seq_all,axis=0)
        comparative_plot(perc,y_HR_all,y_dist_paral_all,y_dist_seq_all,fold_out=int(i),fold_in=int(j),average='average')

    return y_HR_all,y_dist_paral_all,y_dist_seq

all_comparative_dice(percent=[6],name_model='UNet11',fold_out=[0],fold_in=[0,1,2,3,4])
#all_comparative_dice(percent=[70],name_model='UNet11',fold_out=[0],fold_in=[0,2,4])