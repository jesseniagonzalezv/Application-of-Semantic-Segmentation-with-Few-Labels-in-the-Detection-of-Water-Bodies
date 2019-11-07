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

dice_HR=read_data("/home/jgonzalez/Test_2019/Test_network/model_HR/predictions/pred_loss_HR.txt", name="dice")
dice_dist=read_data("/home/jgonzalez/Test_2019/Test_network/model_LR_HR_paralel/predictions/pred_loss_dist_paral.txt",name="dice")
dice_dist_400=read_data("/home/jgonzalez/Test_2019/Test_network/model_LR_HR_paralel/predictions/pred_loss_dist_paral_400.txt",name="dice")
dice_dist_100=read_data("/home/jgonzalez/Test_2019/Test_network/model_LR_HR_paralel/predictions/pred_loss_dist_paral_100.txt",name="dice")

print(len(dice_HR), len(dice_dist))

f = plt.figure()
y_HR = np.asarray(dice_HR)
y_dist = np.asarray(dice_dist)
y_dist_400 = np.asarray(dice_dist_400)
y_dist_100 = np.asarray(dice_dist_100)


x =np.asarray(list(range(0,len(dice_HR))))
plt.xlabel("Number of test images")
plt.ylabel("Dice")
plt.title("predictions_with_dice") 

plt.plot(x,y_HR,label = 'model_HR')
plt.plot(x,y_dist ,'k', label = 'dist_916')
plt.plot(x,y_dist_400 ,'r', label = 'dist_400')
plt.plot(x,y_dist_100 ,'g', label = 'dist_100')

plt.legend()
plt.show()
#f.savefig("predictions/comparation_dice.pdf", bbox_inches='tight')
