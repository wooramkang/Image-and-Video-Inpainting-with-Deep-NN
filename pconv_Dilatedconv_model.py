
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
from keras.models import Model
from keras.layers.core import Lambda, Flatten, Dense
import math
from keras.layers import UpSampling2D, UpSampling3D
from keras.layers import Conv2DTranspose, Conv3DTranspose, Reshape
from keras.optimizers import adam, Adam
from pconv_layer_2D import PConv2D
from DataWeight_load import *

def P_D_model(input_frame, input_mask, frame_size=None, sampling_frame=8, frame_net_mid_depth = 4):
    
    Activ = lambda x: LeakyReLU(alpha=0.2)(x)
    Bat = lambda x: BatchNormalization()(x)
    Activ_re = lambda x: Activation('relu')(x)
    
    if not frame_size:
        W = 512
        H = 512

    if input_frame is None:
        input_frame = Input( shape=(W, H, 3) )
    if input_mask is None:
        input_mask = Input( shape=(W, H, 3) )
    
    inputs_img = input_frame
    inputs_mask = input_mask
    
    def encoder_layer(img_in, mask_in, filters, kernel_size, bn=True):
        conv, mask = PConv2D(filters, kernel_size, strides=2, padding='same')([img_in, mask_in])
        if bn:
            conv = BatchNormalization(name='EncBN'+str(encoder_layer.counter))(conv, training=bn)
        conv = Activation('relu')(conv)
        encoder_layer.counter += 1
        #print(conv.get_shape())
        return conv, mask

    encoder_layer.counter = 0

    #PConv_ENCODE
    e_conv1, e_mask1 = encoder_layer(inputs_img, inputs_mask, 64, 7, bn=False)
    e_conv2, e_mask2 = encoder_layer(e_conv1, e_mask1, 128, 5)
    e_conv3, e_mask3 = encoder_layer(e_conv2, e_mask2, 256, 5)
    e_conv4, e_mask4 = encoder_layer(e_conv3, e_mask3, 512, 3)
    e_conv5, e_mask5 = encoder_layer(e_conv4, e_mask4, 512, 3)
    e_conv6, e_mask6 = encoder_layer(e_conv5, e_mask5, 512, 3)
    e_conv7, e_mask7 = encoder_layer(e_conv6, e_mask6, 512, 3)
    e_conv8, e_mask8 = encoder_layer(e_conv7, e_mask7, 512, 3)
    
    #Dliated NN
    fc_mid = e_conv8
    fc_mid_mask = e_mask8
    fc_prev = e_conv8
    fc_prev_mask = e_mask8
    p_num = 2
    p_num_checker = True
    frame_net_mid_depth = frame_net_mid_depth -1

    for i in range(frame_net_mid_depth):
        
        fc_mid = Conv2D(512, 3, dilation_rate= (p_num,p_num), strides=1, padding='same')(fc_mid)
        fc_mid = Bat(fc_mid)
        fc_mid = Activ_re(fc_mid)
        f_temp = fc_mid
        fc_mid = Concatenate()([fc_prev, fc_mid])
        #print(fc_mid.get_shape())

        fc_prev = f_temp
        
        fc_mid_mask = Conv2D(512, 3, dilation_rate= (p_num,p_num), strides=1, padding='same')(fc_mid_mask)
        fc_mid_mask = Bat(fc_mid_mask)
        fc_mid_mask = Activ_re(fc_mid_mask)
        f_temp_mask = fc_mid_mask
        fc_mid_mask = Concatenate()([fc_prev_mask, fc_mid_mask])
        
        fc_prev_mask = f_temp_mask

        if p_num_checker:
            p_num= p_num * 2
            p_num_checker = False
        else:
            p_num = p_num / 2 

    fc_mid = Conv2D(512, 3, dilation_rate = (p_num,p_num), strides=1, padding='same')(fc_mid)
    fc_mid = Bat(fc_mid)
    fc_mid_prev = Activ_re(fc_mid)
    #print(fc_mid.get_shape())

    fc_mid_mask = Conv2D(512, 3,  dilation_rate = (p_num,p_num), strides=1, padding='same')(fc_mid_mask)
    fc_mid_mask = Bat(fc_mid_mask)
    fc_mid_mask_prev = Activ_re(fc_mid_mask)
    
    fc_mid = Concatenate()([e_conv8, fc_mid_prev])
    fc_mid = Conv2D(1024, 3, strides=2, padding='same')(fc_mid)#, fc_mid_mask])
    fc_mid = Bat(fc_mid)
    fc_mid = Activ_re(fc_mid)
    #print(fc_mid.get_shape())

    fc_mid_mask = Concatenate()([e_mask8, fc_mid_mask_prev])
    fc_mid_mask = Conv2D(1024, 3, strides=2, padding='same')(fc_mid_mask)#, fc_mid_mask])
    fc_mid_mask = Bat(fc_mid_mask)
    fc_mid_mask = Activ_re(fc_mid_mask)
    
    def decoder_layer(img_in, mask_in, e_conv, e_mask, filters, kernel_size, bn=True):
        up_img = UpSampling2D(size=(2,2))(img_in)
        up_mask = UpSampling2D(size=(2,2))(mask_in)
        concat_img = Concatenate(axis=3)([e_conv, up_img])
        concat_mask = Concatenate(axis=3)([e_mask, up_mask])
        conv, mask = PConv2D(filters, kernel_size, padding='same')([concat_img, concat_mask])

        if bn:
            conv = BatchNormalization()(conv)

        conv = LeakyReLU(alpha=0.2)(conv)
        #print(conv.get_shape())
        return conv, mask

    #PConv_DECODE
    d_conv_mid, d_mask_mid = decoder_layer(fc_mid, fc_mid_mask, e_conv8, e_mask8, 512, 3)
    d_conv9, d_mask9 = decoder_layer(d_conv_mid, d_mask_mid, e_conv7, e_mask7, 512, 3)
    d_conv10, d_mask10 = decoder_layer(d_conv9, d_mask9, e_conv6, e_mask6, 512, 3)
    d_conv11, d_mask11 = decoder_layer(d_conv10, d_mask10, e_conv5, e_mask5, 512, 3)
    d_conv12, d_mask12 = decoder_layer(d_conv11, d_mask11, e_conv4, e_mask4, 512, 3)
    d_conv13, d_mask13 = decoder_layer(d_conv12, d_mask12, e_conv3, e_mask3, 256, 3)
    d_conv14, d_mask14 = decoder_layer(d_conv13, d_mask13, e_conv2, e_mask2, 128, 3)
    d_conv15, d_mask15 = decoder_layer(d_conv14, d_mask14, e_conv1, e_mask1, 64, 3)
    d_conv16, d_mask16 = decoder_layer(d_conv15, d_mask15, inputs_img, inputs_mask, 3, 3, bn=False)

    outputs = Conv2D(3, 1, name='outputs_img')(d_conv16)
    outputs = Activation('tanh', name= 'final_output')(outputs)
    
    return outputs

from keras.losses import mse

###### loss function from https://github.com/MathiasGruber/PConv-Keras/blob/master/libs/pconv_model.py
from keras.applications import VGG16
vgg_model = None

def build_vgg(weights="imagenet"):
    """
    Load pre-trained VGG16 from keras applications
    Extract features to be used in loss function from last conv layer, see architecture at:
    https://github.com/keras-team/keras/blob/master/keras/applications/vgg16.py
    """        
        
    # Input image to extract features from
    img = Input(shape=(512, 512, 3))

    # Mean center and rescale by variance as in PyTorch
    processed = Lambda(lambda x: (x-self.mean) / self.std)(img)
    
    # If inference only, just return empty model        
            
    # Get the vgg network from Keras applications
    if weights in ['imagenet', None]:
        vgg = VGG16(weights=weights, include_top=False)
    else:
        vgg = VGG16(weights=None, include_top=False)
        vgg.load_weights(weights, by_name=True)

    # Output the first three pooling layers
    vgg.outputs = [vgg.layers[i].output for i in self.vgg_layers]        
    
    # Create model and compile
    model = Model(inputs=img, outputs=vgg(processed))
    model.trainable = False
    model.compile(loss='mse', optimizer='adam')

    return model
def loss_total(mask):
    """
    Creates a loss function which sums all the loss components 
    and multiplies by their weights. See paper eq. 7.
    """
    def loss(y_true, y_pred):

        # Compute predicted image with non-hole pixels set to ground truth
        y_comp = mask * y_true + (1-mask) * y_pred

        # Compute the vgg features. 
        if self.vgg_device:
            with tf.device(self.vgg_device):
                vgg_out = self.vgg(y_pred)
                vgg_gt = self.vgg(y_true)
                vgg_comp = self.vgg(y_comp)
        else:
            vgg_out = self.vgg(y_pred)
            vgg_gt = self.vgg(y_true)
            vgg_comp = self.vgg(y_comp)
        
        # Compute loss components
        l1 = self.loss_valid(mask, y_true, y_pred)
        l2 = self.loss_hole(mask, y_true, y_pred)
        l3 = self.loss_perceptual(vgg_out, vgg_gt, vgg_comp)
        l4 = self.loss_style(vgg_out, vgg_gt)
        l5 = self.loss_style(vgg_comp, vgg_gt)
        l6 = self.loss_tv(mask, y_comp)

        # Return loss function
        return l1 + 6*l2 + 0.05*l3 + 120*(l4+l5) + 0.1*l6

    return loss

def loss_hole(self, mask, y_true, y_pred):
    """Pixel L1 loss within the hole / mask"""
    return self.l1((1-mask) * y_true, (1-mask) * y_pred)

def loss_valid(self, mask, y_true, y_pred):
    """Pixel L1 loss outside the hole / mask"""
    return self.l1(mask * y_true, mask * y_pred)

def loss_perceptual(self, vgg_out, vgg_gt, vgg_comp): 
    """Perceptual loss based on VGG16, see. eq. 3 in paper"""       
    loss = 0
    for o, c, g in zip(vgg_out, vgg_comp, vgg_gt):
        loss += self.l1(o, g) + self.l1(c, g)
    return loss
    
def loss_style(self, output, vgg_gt):
    """Style loss based on output/computation, used for both eq. 4 & 5 in paper"""
    loss = 0
    for o, g in zip(output, vgg_gt):
        loss += self.l1(self.gram_matrix(o), self.gram_matrix(g))
    return loss

def loss_tv(self, mask, y_comp):
    """Total variation loss, used for smoothing the hole region, see. eq. 6"""

    # Create dilated hole region using a 3x3 kernel of all 1s.
    kernel = K.ones(shape=(3, 3, mask.shape[3], mask.shape[3]))
    dilated_mask = K.conv2d(1-mask, kernel, data_format='channels_last', padding='same')

    # Cast values to be [0., 1.], and compute dilated hole region of y_comp
    dilated_mask = K.cast(K.greater(dilated_mask, 0), 'float32')
    P = dilated_mask * y_comp

    # Calculate total variation loss
    a = self.l1(P[:,1:,:,:], P[:,:-1,:,:])
    b = self.l1(P[:,:,1:,:], P[:,:,:-1,:])        
    return a+b


def pdCN_network_generate(data_shape= (512, 512, 3), sampling_frame=8, frame_net_mid_depth=4, learn_rate = 0.01):
    Init_dataloader()
    
    optimizer_pdCN = Adam(lr=learn_rate)

    input_frame = Input( shape=data_shape )
    input_mask = Input( shape=data_shape )

    pdCN = P_D_model(input_frame= input_frame, input_mask = input_mask, frame_size=data_shape, sampling_frame=sampling_frame, frame_net_mid_depth =frame_net_mid_depth)
    pdCN_model = Model([input_frame, input_mask], pdCN)
    pdCN_model.summary()
    
    # custom loss to learn better
    #pdCN_model.compile(optimizer=optimizer_pdCN, loss=loss_func(input_frame, input_mask) )

    pdCN_model.compile(optimizer=optimizer_pdCN, loss={'final_output' : 'mae'})
    

    return pdCN_model


if __name__ == "__main__":
    pdCN_network_generate()
    
    