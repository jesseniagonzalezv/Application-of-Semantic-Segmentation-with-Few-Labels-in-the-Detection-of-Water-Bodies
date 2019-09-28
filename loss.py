import torch
import torch.nn as nn

def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    cdice = (2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)         #1-Dice
    loss = 1 - cdice
    return loss.mean() #mean of the batch

#The Jaccard coefficient measures similarity between finite sample sets.
def metric_jaccard(pred, target):
    pred = pred.contiguous()
    target = target.contiguous()  
    epsilon= 1e-15  #epsilon! para evitar el indeterminado
    intersection = (pred*target).sum(dim=2).sum(dim=2)
    #print("union_shape:", union.shape)
    union = target.sum(dim=2).sum(dim=2) + pred.sum(dim=2).sum(dim=2) - intersection
    cjaccard = (intersection + epsilon)/ (union + epsilon)
    loss = 1 - cjaccard
    #print(cjaccard.shape)
    return loss.mean()#mean of the batch


####################metric ???
def value_jaccard(pred, target):
    pred = pred.contiguous()
    target = target.contiguous()  
    epsilon= 1e-15  #epsilon! para evitar el indeterminado
    intersection = (pred*target).sum(dim=-2).sum(dim=-1)
    union = target.sum(dim=-2).sum(dim=-1) + pred.sum(dim=-2).sum(dim=-1) - intersection
    djaccard=(intersection +epsilon ) / (union + epsilon)
    return djaccard.mean()


def value_dice(pred, target):
    pred = pred.contiguous()
    target = target.contiguous()  
    epsilon= 1e-15   
    intersection = (pred*target).sum(dim=-2).sum(dim=-1)
    ddice=2* intersection / (target.sum(dim=-2).sum(dim=-1)+ pred.sum(dim=-2).sum(dim=-1) + 1e-15)
    return ddice.mean()