"""
Based on https://github.com/asanakoy/kaggle_carvana_segmentation/blob/master/albu/src/transforms.py
"""

import random
import cv2
import numpy as np
import math


class DualCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, mask=None):
        for t in self.transforms:
            x, mask = t(x, mask)
            #print('en DC')

        return x, mask

class CenterCrop:
    def __init__(self, size):
        if isinstance(size, int):
            size = (size, size)

        self.height = size[0]
        self.width = size[1]

    def __call__(self, img, mask=None):
        h, w, c = img.shape
        dy = (h - self.height) // 2
        dx = (w - self.width) // 2

        y1 = dy
        y2 = y1 + self.height
        x1 = dx
        x2 = x1 + self.width
        img = img[y1:y2, x1:x2]

        if mask is not None:
            mask = mask[y1:y2, x1:x2]
        #print('en CC')

        return img, mask


class HorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            img = cv2.flip(img, 1)
            if mask is not None:
                mask = cv2.flip(mask, 1)
            #print('en HC')

        return img, mask
    
    
class VerticalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            img = cv2.flip(img, 0)
            if mask is not None:
                mask = cv2.flip(mask, 0)
            #print('en VF')

        return img, mask
    
    
class Rotate:
    def __init__(self, limit=90, prob=0.5):
        self.prob = prob
        self.limit = limit

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            angle = random.uniform(-self.limit, self.limit)

            height, width = img.shape[0:2]
            mat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
            img = cv2.warpAffine(img, mat, (height, width),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)
            if mask is not None:
                mask = cv2.warpAffine(mask, mat, (height, width),
                                      flags=cv2.INTER_NEAREST,
                                      borderMode=cv2.BORDER_REFLECT_101)
        #print('en Rot')

        return img, mask

class ImageOnly:
    def __init__(self, trans):
        self.trans = trans

    def __call__(self, x, mask=None):
        #print('en IO')

        return self.trans(x), mask
    
    
class Normalize:
    def __init__(self, mean=(0.11239524, 0.101936, 0.11311523, 0.13813571), std=(0.08964322, 0.06702993, 0.05725554, 0.11082901)):
#    def __init__(self, mean=(0.12121853, 0.10802471, 0.11964872, 0.137894), std=(0.09692554, 0.07299206, 0.06229836, 0.11779822)): db
        
        
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img= img.astype(np.float32)
        max_pixel_value=3413
        img = img/max_pixel_value 
        img -= np.ones(img.shape) * self.mean
        img /= np.ones(img.shape) * self.std
        #print('en Nor')

        return img


class Normalize2:
    def __init__(self, mean=(0.11946253, 0.12642327, 0.13482856, 0.15008255), std=(0.08853241, 0.07311754, 0.06746538, 0.10958234)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img= img.astype(np.float32)
        max_pixel_value=1.0176
        img = img/max_pixel_value 
        img -= np.ones(img.shape) * self.mean
        img /= np.ones(img.shape) * self.std
        #print('en Nor2')

        return img

class RandomRotate90:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            factor = random.randint(0, 4)
            img = np.rot90(img, factor)
            if mask is not None:
                mask = np.rot90(mask, factor)
            #print('en Random')

        return img.copy(), mask.copy()



