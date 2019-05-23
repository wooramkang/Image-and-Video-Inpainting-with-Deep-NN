
import tensorflow as tf
import numpy as np
import os
#from convolutional import Conv3D
from numpy import genfromtxt
from keras import backend as K
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate, ZeroPadding3D#, Conv3D 
#from keras.layers import AtrousConvolution2D, AtrousCo
from keras.layers import Conv3D, Concatenate
from keras.layers import LeakyReLU, Deconvolution2D, Deconvolution3D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D, MaxPooling3D, AveragePooling3D
from keras.layers import UpSampling2D, UpSampling3D
from keras.models import Model
from keras.layers.core import Lambda, Flatten, Dense
import math
from keras.layers import Conv2DTranspose, Conv3DTranspose, Reshape
from keras.optimizers import adam, Adam
from pconv_layer_2D import PConv2D
#NEED TO IMPORT PConv3D
from DataWeight_load import *


def CN3D(input_video = None, sampling_frame= 8,  vid_net_mid_depth = 3):
    Activ = lambda x: LeakyReLU(alpha=0.2)(x)
    Bat = lambda x: BatchNormalization()(x)
    
    video_size = None
    
    if not video_size:
        W = 320
        H = 240
    else:
        W = video_size[0]
        H = video_size[1]
    
    W = int(W/2)
    H = int(H/2)
    
    if input_video is None:
        input_video = Input( shape=(sampling_frame, W, H, 3) )
    
    #print(input_video.get_shape())
    e0 = Conv3D(filters=32,padding='same', kernel_size=(5,5,5))(input_video)
    e0 = Bat(e0)
    e0 = Activ(e0)
    #print(e0.get_shape())
    #WITH NO CONCATENATE Init_dataloader()DING IN 3DCN BUT WITH CONCAT FOR ENCODING IN combination part

    e0_C = Conv3D(filters= 64,  padding='same', kernel_size=4, strides =2)(e0)
    e0_C = Bat(e0_C)
    e0_C = Activ(e0_C)
    #print(e0_C.get_shape())

    e1 = Conv3D(filters=128,padding='same', kernel_size=(3,3,3))(e0_C)
    e1 = Bat(e1)
    e1 = Activ(e1)
    #print(e1.get_shape())

    e1_C = Conv3D(filters=256, padding='same', kernel_size=4, strides = 2)(e1)
    e1_C = Bat(e1_C)
    e1_C = Activ(e1_C)
    #print(e1_C.get_shape())

    e2 = Conv3D(filters=512,padding='same', kernel_size=(3,3,3))(e1_C)
    e2 = Bat(e2)
    e2 = Activ(e2)
    #print(e2.get_shape())

    fc_mid = e2
    p_num = 2
    
    for i in range(vid_net_mid_depth):
        fc_mid = Conv3D(filters= 512, dilation_rate= p_num, kernel_size = 3, padding='same')(fc_mid)
        fc_mid = Bat(fc_mid)
        fc_mid = Activ(fc_mid)
        #print(fc_mid.get_shape())
        p_num = p_num * 2

    d0_CC = Concatenate()([fc_mid, e2])    
    d0_CC = Conv3D(filters=256,padding='same', kernel_size=3)(d0_CC)
    d0_CC = Bat(d0_CC)
    d0_CC = Activ(d0_CC)
    #print(d0_CC.get_shape())

    d1 = Concatenate()([d0_CC, e1_C])
    d1 = UpSampling3D()(d1)
    d1 = Conv3D(strides=1, filters=128, kernel_size= 4, padding='same')(d1)
    d1 = Bat(d1)
    d1 = Activ(d1)
    #print(d1.get_shape())

    d1_C = Concatenate()([d1, e1])    
    d1_C = Conv3D(filters=64,padding='same', kernel_size=3)(d1_C)
    d1_C = Bat(d1_C)
    d1_C = Activ(d1_C)
    #print(d1_C.get_shape())

    d1_CC = Concatenate()([d1_C, e0_C])
    #d1_CC = Deconvolution3D(strides=2, filters=16, kernel_size= 4, padding='same')(d1_CC)
    d1_CC = UpSampling3D()(d1_CC)
    d1_CC = Conv3D(strides=1, filters=32, kernel_size= 4, padding='same')(d1_CC)
    d1_CC = Bat(d1_CC)
    d1_CC = Activ(d1_CC)
    #print(d1_CC.get_shape())

    d2_CC = Concatenate()([d1_CC, e0])
    d2_CC = Conv3D(filters=3, padding='same', kernel_size= 5)(d2_CC)
    d2_CC = Bat(d2_CC)
    d2_CC = Activation('tanh')(d2_CC)
    #print(d2_CC.get_shape())
    video_3DCN = d2_CC
    #video_3DCN = Model(input_video, d2_CC)
    #t_Model = Model(input_video, d2_CC)
    #t_Model.summary()
    return video_3DCN


def CombCN(input_frame, input_video, video_size=None, sampling_frame=8, frame_net_mid_depth = 4):
    #frame_3DCN => total frames or jsut frame of Vk in Vin 
    Activ = lambda x: LeakyReLU(alpha=0.2)(x)
    Bat = lambda x: BatchNormalization()(x)
    
    video_size = None
    if not video_size:
        W = 320
        H = 240
    else:
        W = video_size[0]
        H = video_size[1]
    
    if input_frame is None:
        input_frame = Input( shape=(W, H, 3) )
    if input_video is None:
        input_video = Input( shape=(sampling_frame, W/2, H/2, 3) )

    print(input_frame.get_shape())

    e0 = Conv2D(filters=32,padding='same', kernel_size=(5,5))(input_frame)
    e0 = Bat(e0)
    e0 = Activ(e0)
    print(e0.get_shape())
    #WITH NO CONCATENATE Init_dataloader()DING IN 3DCN BUT WITH CONCAT FOR ENCODING IN combination part

    e0_C = Conv2D(filters= 64,  padding='same', kernel_size=(4,4), strides = 2)(e0)
    e0_C = Bat(e0_C)
    e0_C = Activ(e0_C)
    print(e0_C.get_shape())

    #skip_subnet = CN3D(video_info=None , input_video = input_video, sampling_frame= 8,  vid_net_mid_depth = 3)
    skip_subnet = input_video
    size_subnet = skip_subnet.get_shape()

    # IS IT WHAT THE PAPER SAYS? NOT SURE
    skip_subnet = Reshape( (int(W/2), int(H/2), int(size_subnet[1]* size_subnet[4]) ) )(skip_subnet)
    skip_subnet = Conv2D(filters= 64,  padding='same', kernel_size=(4,4), strides = 1)(skip_subnet)
    skip_subnet = Bat(skip_subnet)
    skip_subnet = Activ(skip_subnet)
    
    e0_C = Concatenate()([e0_C, skip_subnet])
    print(e0_C.get_shape())

    e1 = Conv2D(filters=128, padding='same', kernel_size=(3,3))(e0_C)
    e1 = Bat(e1)
    e1 = Activ(e1)
    print(e1.get_shape())

    e1_C = Conv2D(filters=256, padding='same', kernel_size=4, strides = 2)(e1)
    e1_C = Bat(e1_C)
    e1_C = Activ(e1_C)
    print(e1_C.get_shape())

    e2 = Conv2D(filters=512,padding='same', kernel_size=(3,3))(e1_C)
    e2 = Bat(e2)
    e2 = Activ(e2)
    print(e2.get_shape())
    
    e2_C = Conv2D(filters=512, padding='same', kernel_size=4, strides = 2)(e2)
    e2_C = Bat(e2_C)
    e2_C = Activ(e2_C)
    print(e2_C.get_shape())

    fc_mid = e2_C
    p_num = 2
    
    for i in range(frame_net_mid_depth):
        fc_mid = Conv2D(filters= 512, dilation_rate= p_num, kernel_size = 3, padding='same')(fc_mid)
        fc_mid = Bat(fc_mid)
        fc_mid = Activ(fc_mid)
        print(fc_mid.get_shape())
        p_num = p_num * 2

    #fc_mid = Deconvolution3D(strides=2, filters=64, kernel_size= 4, padding='same')(fc_mid)
    d0 = Concatenate()([fc_mid, e2_C])    
    print(d0.get_shape())

    d0_C = UpSampling2D()(d0)
    d0_C = Conv2D(strides=1, filters=512, kernel_size= 4, padding='same')(d0_C)
    d0_C = Bat(d0_C)
    d0_C = Activ(d0_C)
    d0_C = Concatenate()([d0_C, e2])    
    print(d0_C.get_shape())

    d0_CC = Conv2D(filters=512,padding='same', kernel_size=3)(d0_C)
    d0_CC = Bat(d0_CC)
    d0_CC = Activ(d0_CC)
    d0_CC = Concatenate()([d0_CC, e1_C])
    print(d0_CC.get_shape())

    d1 = UpSampling2D()(d0_CC)
    d1 = Conv2D(strides=1, filters=256, kernel_size= 4, padding='same')(d1)
    d1 = Bat(d1)
    d1 = Activ(d1)
    d1 = Concatenate()([d1, e1])    
    print(d1.get_shape())

    d1_C = Conv2D(filters= 128, padding='same', kernel_size=3)(d1)
    d1_C = Bat(d1_C)
    d1_C = Activ(d1_C)
    d1_C = Concatenate()([d1_C, e0_C])
    print(d1_C.get_shape())
    
    d1_CC = UpSampling2D()(d1_C)
    d1_CC = Conv2D(strides=1, filters=64, kernel_size= 4, padding='same')(d1_CC)
    d1_CC = Bat(d1_CC)
    d1_CC = Activ(d1_CC)
    d1_CC = Concatenate()([d1_CC, e0])
    print(d1_CC.get_shape())

    d2_CC = Conv2D(filters=32, padding='same', kernel_size=3)(d1_CC)
    d2_CC = Bat(d2_CC)
    d2_CC = Activ(d2_CC)
    print(d2_CC.get_shape())

    d_out = Conv2D(filters=3, padding='same', kernel_size=4)(d2_CC)
    d_out = Bat(d_out)
    d_out = Activation('tanh')(d_out)
    print(d_out.get_shape())

    return d_out

# loss_total = loss of 3DNN + loss of CombCN
# loss_CombCN = Sig( M * G(V,M,I) - V )....
# final_model => sig ( CN3D  => combCN)
from keras.losses import mse
'''
def loss(Y_true, Y_pred):
   loss = K.sum(....)
   return loss
@staticmethod
def l1(y_true, y_pred):
    
    if K.ndim(y_true) == 4:
        return K.mean(K.abs(y_pred - y_true), axis=[1,2,3])

    elif K.ndim(y_true) == 3:
        return K.mean(K.abs(y_pred - y_true), axis=[1,2])

    else:
        raise NotImplementedError("not possible")

'''

def network_generate(data_shape= (320, 240, 3), sampling_frame=8, vid_net_mid_depth=3, frame_net_mid_depth=4):
    Init_dataloader()
    optimizer_subnet = Adam(lr=0.005)
    optimizer_mainnet = Adam(lr=0.005)
    optimizer_final = Adam(lr=0.0001)

    input_frame = Input( shape=data_shape )
    input_video = Input( shape=(sampling_frame, int(data_shape[0]/2), int(data_shape[1]/2), 3) )

    cn3d = CN3D(input_video=input_video)
    CN3D_model = Model(input_video, cn3d)
    CN3D_model.summary()
    '''
    def loss_3DCN():
        cn3d_loss = mse( CN3D_model(input_video), input_video )
        return cn3d_loss
        
        CN3D_model.add_loss(loss_3DCN())
    '''
    CN3D_model.compile(optimizer=optimizer_subnet, loss={'activation_1' : 'mse'} )

    combCN= CombCN(input_frame= input_frame, input_video = CN3D_model(input_video) )
    CombCN_model = Model([input_frame, input_video], combCN)
    CombCN_model.summary()
    CombCN_model.compile(optimizer=optimizer_mainnet, loss={'activation_2' : 'mse'})

    final_model = Model( inputs=[input_frame, input_video], outputs=[ CN3D_model(input_video), CombCN_model( [input_frame, input_video] ) ])
    final_model.summary()
    alpha = 1.0#0.7
    beta = 1.0
    
    '''
    def loss_total():
        
        l1 = K.sum(mse( CombCN_model( [input_frame, input_video] ), input_frame)) * alpha
        l2 = K.sum(mse( CN3D_model(input_video), input_video ))* beta
        t_loss = l1 + l2
        return t_loss
    final_model.add_loss(loss_total())
    '''
    final_model.compile(optimizer=optimizer_final, loss={'model_1' : 'mse', 'model_2' : 'mse'},
                                                    loss_weights={'model_1' :alpha , 'model_2':beta} )
    return CN3D_model, CombCN_model, final_model
#ref of LATEX for diagrams
#https://www.overleaf.com/learn/latex/Code_listing
#https://en.wikibooks.org/wiki/LaTeX/Source_Code_Listings
#ref of plotting NN with LATEX
#https://github.com/HarisIqbal88/PlotNeuralNet

## to check shapes of models to train
if __name__ == "__main__":
    network_generate()
    
    