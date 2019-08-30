import tensorflow as tf
import tensorlayer as tl
from tensorlayer.prepro import *
# from config import config, log_config
#
# img_path = config.TRAIN.img_path

import scipy
import scipy.misc
import cv2
import numpy as np

def get_imgs_fn(file_name, path):
    """ Input an image path and name, return an image array """
    # return scipy.misc.imread(path + file_name).astype(np.float)
    return scipy.misc.imread(path + file_name, mode='RGB')
#    return cv2.imread(path + file_name)
    return x

def scale_imgs_fn(x):
    x = x-127.5
    x = x/(255./2.)
    return x


def crop_sub_imgs_fn(x, is_random=True, wrg=384, hrg=384):
    x = crop(x, wrg=wrg, hrg=hrg, is_random=is_random)
    return x

def crop_sub_frames_fn(xs, is_random=True, wrg=384, hrg=384):    
    xs = crop_multi(xs, wrg=wrg, hrg=hrg, is_random=is_random)
    return xs              
    

def downsample_fn(x, is_random=True, fx=0.25, fy=0.25, inter=cv2.INTER_CUBIC):
    x = cv2.resize(x,None,fx=fx,fy=fy,interpolation=inter)
    return x

def upsample_fn(x, is_random=True, fx=4, fy=4, inter=cv2.INTER_CUBIC):    
    x = cv2.resize(x,None,fx=fx,fy=fy,interpolation=inter)
    return x


