#! /usr/bin/python
# -*- coding: utf8 -*-
import matplotlib.pyplot as plt 
import os, time, pickle, random
import numpy as np
import cv2

import tensorflow as tf
import tensorlayer as tl
from model import *
from utils import *
from config import config 

def  modcrop(imgs, modulo):

    tmpsz = imgs.shape
    sz = tmpsz[0:2]

    h = sz[0] - sz[0]%modulo
    w = sz[1] - sz[1]%modulo
    imgs = imgs[0:h, 0:w,:]
    return imgs

def read_all_imgs(img_list, path='', n_threads=32):
    """ Returns all images in array by given path and name of each image file. """
    imgs = []
    for idx in range(0, len(img_list), n_threads):
        b_imgs_list = img_list[idx : idx + n_threads]
        b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=path)
        # print(b_imgs.shape)
        imgs.extend(b_imgs)
        print('read %d from %s' % (len(imgs), path))
    return imgs

def scale_image(image,do_scale):
    if(do_scale):
        return np.maximum(0,np.minimum((image+1)*(255./2.),255))
#        return np.maximum(0,np.minimum((image+128.),255))
    else:
        return np.maximum(0,np.minimum(image,255))

def evaluate():
    ## create folders to save result images
    # save_dir = 'Evaluation7_up_new1_2'
    save_dir = 'result/'
    tl.files.exists_or_mkdir(save_dir)
    # checkpoint_dir = 'checkpoint7_up_new1_2'
    checkpoint_dir = 'checkpoint'
    im_path_lr = 'input/'
    ###====================== PRE-LOAD DATA ===========================###
    valid_img_list = sorted(tl.files.load_file_list(path=im_path_lr, regx='.*.png', printable=False))
    
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    
    t_image = tf.placeholder('float32', [1, 128, 128, 3], name='input_image') # set an arbitrary size of tensor for initialization
    
    net_g_init= EnhanceNet(t_image, is_train=False, reuse=False, hrg=128, wrg=128)

    ###========================== RESTORE G =============================###
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/model.npz', network=net_g_init)
    
    for imid in range(len(valid_img_list)):
        valid_img = modcrop(get_imgs_fn(valid_img_list[imid],im_path_lr),4)
        print(valid_img.shape)
        t_image = tf.placeholder('float32', [1, valid_img.shape[0], valid_img.shape[1], 3], name='input_image')    
        net_g = EnhanceNet(t_image, is_train=False, reuse=True, hrg=valid_img.shape[0], wrg=valid_img.shape[1])

        in_ori = valid_img
        valid_img = (valid_img / 127.5) - 1   # rescale to ［－1, 1]
        ###======================= EVALUATION =============================###
        start_time = time.time()
        out= sess.run(net_g.outputs, {t_image: [valid_img]})
        out_lab = cv2.cvtColor(scale_image(out[0],True).astype(np.float32)/255.,cv2.COLOR_RGB2LAB)
        in_lab = cv2.cvtColor(in_ori.astype(np.float32)/255.,cv2.COLOR_RGB2LAB)
        out_l, out_a, out_b = cv2.split(out_lab)
        in_l, in_a, in_b = cv2.split(in_lab)
        out_ = cv2.merge((out_l, in_a, in_b))
        out_ = cv2.cvtColor(out_,cv2.COLOR_Lab2RGB)*255.

            
        print("took: %4.4fs" % (time.time() - start_time))    
        
        print("[*] save images")
        scipy.misc.toimage(out_,cmin=0,cmax=255).save(save_dir+'/' + valid_img_list[imid][0:-4] +'.png')
        # tl.vis.save_image(out[0], save_dir+'/' + valid_lr_img_list[imid][0:-4] +'.png')    


if __name__ == '__main__':
    evaluate()
 