#! /usr/bin/python
# -*- coding: utf8 -*-

import tensorflow as tf
import tensorlayer as tl
import time
from tensorlayer.layers import *

def EnhanceNet(t_image, is_train=False, reuse=False, hrg=128, wrg=128):
    w_init = tf.random_normal_initializer(stddev=0.01)
    b_init = None # tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x : tf.nn.leaky_relu(x, 0.2)
    with tf.variable_scope("enhance_net", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)        
        n = InputLayer(t_image, name='in2')
        n0 = n     
        n = Conv2d(n, 32, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='f0/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='f0/b')
        f0 = n
        n = Conv2d(n, 64, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init, name='d1/c1')
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='d1/b1')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='d1/c2')
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='d1/b2')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='d1/c3')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='d1/b3')

        f1_2 = n
        n = Conv2d(n, 128, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init, name='d2/c1')
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='d2/b1')
        n = Conv2d(n, 128, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n64s1/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='d2/b2')
        temp = n

        ## B residual blocks
        for i in range(16):
            nn = Conv2d(n, 128, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init,b_init=b_init, name='n64s1/c1/%s' % i)
            nn = BatchNormLayer(nn, act=lrelu, is_train=is_train, gamma_init=g_init, name='n64s1/b1/%s' % i)
            nn = Conv2d(nn, 128, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init,b_init=b_init, name='n64s1/c2/%s' % i)
            nn = BatchNormLayer(nn, is_train=is_train, gamma_init=g_init, name='n64s1/b2/%s' % i)
            nn = ElementwiseLayer([n, nn], tf.add, name = 'b_residual_add/%s' % i)
            n = nn       

        n = Conv2d(n, 128, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init,b_init=b_init, name='n64s1/c/m')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n64s1/b1')
        n = ElementwiseLayer([temp, n], tf.add, name = 'add3')
        
        # n = DeConv2d(n, 64, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init, name='u2/d')
        n = UpSampling2dLayer(n, (2,2), method=1, name='u2/u')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='u2/c0')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='u2/b0')
        n = ElementwiseLayer([n, f1_2], tf.add, act=lrelu, name='s3')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='u2/c1')
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='u2/b1')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='u2/c2')
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='u2/b2')
        
        # n = DeConv2d(n, 32, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init, name='u1/d')
        n = UpSampling2dLayer(n, (2,2), method=1, name='u1/u')
        n = Conv2d(n, 32, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='u1/c0')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='u1/b0')
        n = ElementwiseLayer([n, f0], tf.add, act=lrelu, name='s2')
        n = Conv2d(n, 32, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='u1/c1')
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='u1/b1')
        n = Conv2d(n, 3, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='u1/c2')
        n.outputs = n.outputs*10. + n0.outputs
        return n
