
import tensorflow as tf
import numpy as np
import os
from convolutional import Conv3D
from numpy import genfromtxt
from keras import backend as K
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate, ZeroPadding3D#, Conv3D 
from keras.layers import AtrousConvolution2D, AtrousCo
from keras.layers import LeakyReLU, Deconvolution2D, Deconvolution3D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D, MaxPooling3D, AveragePooling3D
from keras.layers.pooling import Upsampling2D, Upsampling3D
from keras.models import Model
import utils
from keras.layers.core import Lambda, Flatten, Dense
# https://arxiv.org/pdf/1806.08482.pdf
# Video Inpainting by Jointly Learning Temporal Structure and Spatial Details
# INIT 2019.05.13.

def CN3D(video_size, ,sampling_frame= 32,  vid_net_mid_depth = 3):
    Activ = lambda x: LeakyReLU(alpha=0.2)(x)
    #Activ = LeakyReLU(alpha=0.2)
    W = video_size[0]
    H = video_size[1]
    W = W/2
    H = H/2
    
    input_video = Input( np.array((,sampling_frame, W, H, 3)) )

    e0 = Conv3D(filter=16,padding='valid', kernel_size=5)(input_video)
    e0 = Activ(e0)
    e0_C = Conv3D(filter=32, padding='valid', kernel_size=3, strides =2)(e0)
    e0_C = Activ(e0_C)
    #e0_D = AveragePooling3D()(e0_C)
    # AFTER DOWNSAMPLE
    e1 = Conv3D(filter=64,padding='valid', kernel_size=3)(e0_D)
    e1 = Activ(e1)
    e1_C = Conv3D(filter=128, padding='valid', kernel_size=3, strides = 2)(e1)
    e1_C = Activ(e1_C)
    #e1_D = AveragePooling3D()(e1_C)
    '''
    e2 = Conv3D(filter=128,padding='valid', kernel_size=3)(input_video)
    e2 = Activ(e2)
    #e2_C = Conv3D(filter=128, padding='valid', kernel_size=3, strdies=2 )(input_videod)
    #e2_C = Activ(e2_C)
    #e2_D = e1_D = AveragePooling3D()(e2_C)
    #fc_mid = e2_D
    '''

    fc_mid = e1_C
    for i in range(vid_net_mid_depth):
        fc_mid = Conv3D(filters= 256, dilation_rate=(2,2,1), kernel_size = 3)(fc_mid)
        fc_mid = Activ(fc_mid)

    fc_mid = Concatenate()([fc_mid, e2_D])
    fc_mid = Conv3D(filters= 256, dilation_rate=(2,2,1), kernel_size = 3)(fc_mid)
    fc_mid = Activ(fc_mid)
    fc_mid = Deconvolution3D(strides=2, filter=256, kernel_size= 4, padding='valid')(fc_mid)
    fc_mid = Activ(fc_mid)
    '''
    #d0_U = Upsampling3D()(fc_mid)
    #d0_U = Concatenate()([d0_U, e2_C])
    d0_U = Concatenate()([fc_mid, e2_C])
    d0_C = Conv3D(filter=128,padding='valid', kernel_size=3)(d0_U)
    d0_C = Activ(d0_C)
    d0_CC = Conv3D(filter=128,padding='valid', kernel_size=3)(d0_C)
    d0_CC = Activ(d0_CC)
    d0_CC = Deconvolution3D(strides=2, filter=128, kernel_size= 4, padding='valid')(d0_CC)
    d0_CC = Activ(d0_CC)
    #d1_U = Upsampling3D()(d0_CC)
    #d1_U = Concatenate()([d1_U, e1_C])
    #d1_U = Concatenate()([d0_CC, e1_C])
    '''
    d1_U =  Concatenate()([fc_mid, e1_C])
    d1_C = Conv3D(filter=128,padding='valid', kernel_size=3)(d1_U)
    d1_C = Activ(d1_C)
    d1_CC = Conv3D(filter=64,padding='valid', kernel_size=3)(d1_C)
    d1_CC = Activ(d1_CC)
    d1_CC = Deconvolution3D(strides=2, filter=32, kernel_size= 4, padding='valid')(d1_CC)
    d1_CC = Activ(d1_CC)

    #d2_U = Upsampling3D()(d1_CC)
    #d2_U = Concatenate()([d2_U, e0_C])
    d2_U = Concatenate()([d1_CC, e0_C])
    d2_C = Conv3D(filter=16,padding='valid', kernel_size=3)(d2_U)
    d2_C = Activ(d2_C)
    d2_CC = Conv3D(filter=3,padding='valid', kernel_size=3)(d2_C)
    d2_CC = Activ(d2_CC)
    
    video_3DCN = Model (input_video, d2_CC)

    return video_3DCN


def CombCN(video_size, depth, sampling_frame,  frame_net_mid_depth, frame_3DCN):
    Activ = lambda x: LeakyReLU(alpha=0.2)(x)
    W = video_size[0]
    H = video_size[1]
    input_image = Input(np.array((,W, H, 3))

    e0 = Conv2D(filter=(),padding='valid', kernel_size=32)(input_image)
    e0 = Activ(e0)
    e0_C = Conv2D(filter=(), padding='valid', kernel_size=32)(e0)
    e0_C = Activ(e0_C)
    e0_D = AveragePooling2D()(e0_C)

    e1 = Concatenate()([frame_3DCN, e0_D])
    e1 = Conv2D(filter=(),padding='valid', kernel_size=128)(e0_D)
    e1 = Activ(e1)
    e1_C = Conv2D(filter=(), padding='valid', kernel_size=128)(e1)
    e1_C = Activ(e1_C)
    e1_D = AveragePooling2D()(e1_C)

    e2 = Conv2D(filter=(),padding='valid', kernel_size=512)(input_video)
    e2 = Activ(e2)
    e2_C = Conv2D(filter=(), padding='valid', kernel_size=512)(input_videod)
    e2_C = Activ(e2_C)
    e2_D = e1_D = AveragePooling2D()(e2_C)

    fc_mid = e2_D
    for i in range(vid_net_mid_depth):
        fc_mid = Conv2D(strides=1 , filters= (), dilation_rate=(2,2,1), kernel_size = 512)(fc_mid)
        fc_mid = Activ(fc_mid)
    fc_mid = Concatenate()([fc_mid, e2_D])
    fc_mid = Conv2D(strides=1 , filters= (), dilation_rate=(2,2,1), kernel_size = 2048)(fc_mid)
    fc_mid = Activ(fc_mid)

    d0_U = Upsampling2D()(fc_mid)
    d0_U = Concatenate()([d0_U, e2_C])
    d0_C = Conv2D(filter=(),padding='valid', kernel_size=512)(d0_U)
    d0_C = Activ(d0_C)
    d0_CC = Conv2D(filter=(),padding='valid', kernel_size=512)(d0_C)
    d0_CC = Activ(d0_CC)

    d1_U = Upsampling2D()(d0_CC)
    d1_U = Concatenate()([d1_U, e1_C])
    d1_C = Conv2D(filter=(),padding='valid', kernel_size=128)(d1_U)
    d1_C = Activ(d1_C)
    d1_C = Concatenate()([d1_C, frame_3DCN])
    d1_CC = Conv2D(filter=(),padding='valid', kernel_size=128)(d1_C)
    d1_CC = Activ(d1_CC)

    d2_U = Upsampling2D()(d1_CC)
    d2_U = Concatenate()([d2_U, e0_C])
    d2_C = Conv2D(filter=(),padding='valid', kernel_size=32)(d2_U)
    d2_C = Activ(d2_C)
    d2_CC = Conv2D(filter=(),padding='valid', kernel_size=32)(d2_C)
    d2_CC = Activ(d2_CC)

    image_Como03DCN = Model (input_image, d2_CC)
    
    full_model = Model(, )


    return full_model


def network_generate(vid_size, depth, sampling_frame,  vid_net_mid_depth, frame_net_mid_depth):

    


    return final_model