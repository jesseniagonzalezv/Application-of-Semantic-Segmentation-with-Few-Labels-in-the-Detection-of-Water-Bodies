import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from PIL import Image
import glob
import helper


def read_metric(out_file,stage,name_file, name,name_model):
    loss_file = open(("predictions_{}/pred_loss_{}{}_{}.txt").format(out_file,stage,name_file,name_model))
    filedata = loss_file.read()
    filedata = filedata.replace("bce",",bce")
    filedata = filedata.split(",")
    metric=[]
    for i in filedata:
        i = i.strip(" ")
        if str(i).startswith(name):
            i = i.split(" ")
            metric.append(float(i[1]))
            
    plt.close('all')
    f = plt.figure()
    y_axe = np.asarray(metric)
    
    x =np.asarray(list(range(0,len(metric))))
    plt.xlabel(("Number of {} images").format(stage))
    plt.ylabel(name)
    plt.title(("predictions_with_{}").format(name) )
    plt.plot(x,y_axe,label = name_file)
    plt.show()
    f.savefig(("predictions_{}/metric_{}_{}{}_{}.pdf").format(out_file,name,stage,name_file,name_model), bbox_inches='tight')
    print('Ctd:',len(x),name,np.mean(metric))

    #return metric


def plot_history_train(out_file,name_file,name_model):
    file = open(("history_{}/history_model{}_{}.txt").format(out_file,name_file,name_model), "r")

    filedata = file.read()
    filedata = filedata.split(",")
    loss = []
    for i in filedata:
        i = i.strip(" ")
        #print(i)
        if str(i).startswith("loss"):
            i = i.split(":")
            loss.append(float(i[1]))
            #print(i[1])
            
    plt.close('all')
    f= plt.figure()  
    y_train = np.asarray(loss[0::2])
    y_val = np.asarray(loss[1::2])

    x =np.asarray(list(range(0,len(y_val))))
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("loss graph")
    #plt.yticks([0,0.2,0.4,0.6,0.8,1])
    plt.plot(x[1:120:],y_train[1:120:],label = 'train_loss') ##0 has a bad value too high
    plt.plot(x[1:120:],y_val[1:120:],'k', label = 'val_loss')
    plt.legend()
    plt.show()
    f.savefig(("history_{}/loss_convergence_{}_{}.pdf").format(out_file,name_file,name_model), bbox_inches='tight')
    


def plot_prediction(stage='test',name_file='_HR_60_fake',out_file='HR',name_model='UNet11', count=29): # #LR •dist

    loss_file = open(("predictions_{}/pred_loss_{}{}_{}.txt").format(out_file,stage,name_file,name_model))
    #loss_file = open("predictions/pred_loss_HR_fake_60.txt")
  
    filedata = loss_file.read()
    filedata = filedata.replace("bce",",bce")
    filedata = filedata.split(",")

    val_file = (("predictions_{}/inputs_{}{}_{}_{}.npy").format(out_file, stage, name_file, count,name_model))
    pred_file =(("predictions_{}/pred_{}{}_{}_{}.npy").format(out_file, stage, name_file, count,name_model))
    label_file = (("predictions_{}/labels_{}{}_{}_{}.npy").format(out_file, stage, name_file, count,name_model))

    #val_file = "predictions/inputs_testHR_60fake29.npy"
    #pred_file = "predictions/pred_testHR_60fake29.npy"
    #label_file = "predictions/labels_testHR_60fake29.npy"

    val_images = np.load(val_file)
    pred_images = np.load(pred_file)
    val_label = np.load(label_file)
    pred_images[0,0,:,:,:].shape
    print(val_images.shape)
    input_images_rgb = [helper.reverse_transform(x) for x in val_images[:94,0,:3,:,:]]   #new metrics
    # Map each channel (i.e. class) to each color
    target_masks_rgb = [helper.masks_to_colorimg(x) for x in val_label[:94,0,:3,:,:]]
    pred_rgb = [helper.masks_to_colorimg(x) for x in pred_images[:94,0,:,:,:]]
    #print(np.shape(input_images_rgb))
    #print(len([input_images_rgb, target_masks_rgb, pred_rgb]))
    name_output=stage + name_file
    #print(name_output,filedata)
    helper.plot_side_by_side([input_images_rgb, target_masks_rgb, pred_rgb],filedata, out_file, name_output, save=1)
    
###############################call the functions

plot_history_train(out_file='LR', name_file='_LR',name_model='UNet11')   

def plotting_figures(stage,name_file,out_file,name_model='UNet11',count=29):
    #stage='test',name_file='_HR_60_fake',out_file='HR',count=29) # #LR •dist

    plot_history_train(out_file, name_file,name_model)   
    plot_prediction(stage, name_file, out_file, name_model,count) # #LR •dist
    #plot_prediction(stage='val',name_file=,out_file,count=11) # #LR •dist

    dice=read_metric(out_file, stage, name_file, name='dice',name_model=name_model)
    jaccard=read_metric(out_file, stage, name_file, name='jaccard',name_model=name_model)


#plotting_figures(stage='test',name_file='_HR_60_fake',out_file='HR',name_model='UNet11',count=29)
#plotting_figures(stage='test',name_file='_HR_916',out_file='HR',name_model='UNet11',count=29)
#plotting_figures(stage='test',name_file='_HR_dist',out_file='HR',name_model='UNet11',count=94)
#plotting_figures(stage='test',name_file='_dist_60',out_file='HR',name_model='UNet11',count=94)
#plotting_figures(stage='test',name_file='_dist_60_2',out_file='HR',name_model='UNet11',count=94)

############################### LR
plotting_figures(stage='test',name_file='_LR',out_file='LR',name_model='UNet11', count=613)

############################### LR plotting history
#plot_history_train(out_file='LR',name_file='_LR',name_model='UNet11') #change the name output