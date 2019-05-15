
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
import math

def CN3D(video_size, ,sampling_frame= 32,  vid_net_mid_depth = 3):
    Activ = lambda x: LeakyReLU(alpha=0.2)(x)
    Bat = lambda x: BatchNormalization()(x)
    #Activ = LeakyReLU(alpha=0.2)
    #Bat = BatchNormalization()
    W = video_size[0]
    H = video_size[1]
    W = W/2
    H = H/2
    
    input_video = Input( np.array((,sampling_frame, W, H, 3)) )
    
    e0 = Conv3D(filter=16,padding='valid', kernel_size=5)(input_video)
    e0 = Bat(e0)
    e0 = Activ(e0)
    #WITH NO CONCATENATE FOR ENCODING IN 3DCN BUT WITH CONCAT FOR ENCODING IN combination part
    e0_C = Conv3D(filter=32, padding='valid', kernel_size=3, strides =2)(e0)
    e0_C = Bat(e0_C)
    e0_C = Activ(e0_C)
    
    e1 = Conv3D(filter=64,padding='valid', kernel_size=3)(e0_D)
    e1 = Bat(e1)
    e1 = Activ(e1)

    e1_C = Conv3D(filter=128, padding='valid', kernel_size=3, strides = 2)(e1)
    e1_C = Bat(e1_C)
    e1_C = Activ(e1_C)
    
    fc_mid = e1_C
    p_num = 2

    for i in range(vid_net_mid_depth):
        fc_mid = Conv3D(filters= 256, dilation_rate=(p_num, p_num, 1), kernel_size = 3)(fc_mid)
        fc_mid = Bat(fc_mid)
        fc_mid = Activ(fc_mid)
        p_num = p_num * 2
    fc_mid = Concatenate()([fc_mid, e1_C])

    fc_mid = Conv3D(strides=1, filter=128, kernel_size= 4, padding='valid')(fc_mid)
    fc_mid = Bat(fc_mid)
    fc_mid = Activ(fc_mid)
    fc_mid = Concatenate()([fc_mid, e1])

    fc_mid = Deconvolution3D(strides=2, filter=64, kernel_size= 4, padding='valid')(fc_mid)
    fc_mid = Bat(fc_mid)
    fc_mid = Activ(fc_mid)
    d1_U = Concatenate()([fc_mid, e0_C])    

    d1_C = Conv3D(filter=32,padding='valid', kernel_size=3)(d1_U)
    d1_C = Bat(d1_C)
    d1_C = Activ(d1_C)
    d1_C = Concatenate()([d1_C, e0])

    d1_CC = Deconvolution3D(strides=2, filter=16, kernel_size= 4, padding='valid')(d1_C)
    d1_CC = Bat(d1_CC)
    d1_CC = Activ(d1_CC)

    d2_CC = Conv3D(filter=3, padding='valid', kernel_size=3)(d1_CC)
    d2_CC = Bat(d2_CC)
    d2_CC = Activation('tanh')(d2_CC)
    
    video_3DCN = Model(input_video, d2_CC)

    return video_3DCN


def CombCN(video_size, sampling_frame, frame_3DCN, frame_net_mid_depth = 4):
    #frame_3DCN => total frames or jsut frame of Vk in Vin 

    Activ = lambda x: LeakyReLU(alpha=0.2)(x)
    Bat = lambda x: BatchNormalization()(x)
    W = video_size[0]
    H = video_size[1]
    input_image = Input(np.array((,W, H, 3))

    e0 = Conv3D(filter=16,padding='valid', kernel_size=5)(input_image)
    e0 = Bat(e0)
    e0 = Activ(e0)
    
    e0_C = Conv3D(filter=32, padding='valid', kernel_size=3, strides =2)(e0)
    e0_C = Bat(e0_C)
    e0_C = Activ(e0_C)
    e0_C = Concatenate()([frame_3DCN, e0_C])

    e1 = Conv3D(filter=64,padding='valid', kernel_size=3)(e0_D)
    e1 = Bat(e1)
    e1 = Activ(e1)

    e1_C = Conv3D(filter=128, padding='valid', kernel_size=3, strides = 2)(e1)
    e1_C = Bat(e1_C)
    e1_C = Activ(e1_C)

    e2 = Conv3D(filter=256, padding='valid', kernel_size=3)(e1_c)
    e2 = Bat(e2)
    e2 = Activ(e2)
    
    # on original paper, this part is strides=1, channels=256
    e2_C = Conv3D(filter=512, padding='valid', kernel_size=3, strides = 2)(e2)
    e2_C = Bat(e2_C)
    e2_C = Activ(e2_C)

    fc_mid = e2_C
    p_num = 2

    # on original paper, this part is strides=1, channels=256,  dilation 2**n
    for i in range(vid_net_mid_depth):
        fc_mid = Conv3D(filters= 512, dilation_rate=(p_num, p_num, 1), kernel_size = 3)(fc_mid)
        fc_mid = Bat(fc_mid)
        fc_mid = Activ(fc_mid)
        p_num = p_num * 2

    fc_mid = Concatenate()([e2_C, fc_mid])

    fc_mid = Deconvolution3D(strides=2, filter=256, kernel_size= 4, padding='valid')(fc_mid)
    fc_mid = Bat(fc_mid)
    fc_mid = Activ(fc_mid)
    fc_mid = Concatenate()([e2, fc_mid])

    fc_mid = Conv3D(strides=1, filter=128, kernel_size= 3, padding='valid')(fc_mid)
    fc_mid = Bat(fc_mid)
    fc_mid = Activ(fc_mid)
    fc_mid = Concatenate()([e1_C, fc_mid])

    fc_mid = Deconvolution3D(strides=2, filter=128, kernel_size= 4, padding='valid')(fc_mid)
    fc_mid = Bat(fc_mid)
    fc_mid = Activ(fc_mid)
    fc_mid = Concatenate()([e1, fc_mid ])    

    d1_C = Conv3D(filter= 64,padding='valid', kernel_size=3)(fc_mid)
    d1_C = Bat(d1_C)
    d1_C = Activ(d1_C)
    d1_C = Concatenate()([frame_3DCN, d1_C])

    d1_CC = Deconvolution3D(strides=2, filter=32, kernel_size= 4, padding='valid')(d1_C)
    d1_CC = Bat(d1_CC)
    d1_CC = Activ(d1_CC)

    d2_CC = Conv3D(filter=3, padding='valid', kernel_size=3)(d1_CC)
    d2_CC = Bat(d2_CC)
    d2_CC = Activation('tanh')(d2_CC)

    image_Comb3DCN = Model(input_image, d2_CC)
    
    return image_Comb3DCN


def network_generate(vid_size, depth, sampling_frame,  vid_net_mid_depth=3, frame_net_mid_depth=4):
    # loss_f = loss of 3DNN + loss of CombCN
    # loss_CombCN = Sig( M * G(V,M,I) - V )....
    loss = None
    optimazer = None




    final_model = Model( , )
    final_model.summary()
    
    return final_model