#from __future__ import absolute_import, division, print_function
import os
import sys
import numpy as np
sys.path.append(os.path.dirname(__file__) )
sys.path.append(os.path.dirname(__file__)+ '/opticalflow/' )

from copy import deepcopy
import cv2

from opticalflow.model_pwcnet import ModelPWCNet, _DEFAULT_PWCNET_TEST_OPTIONS
from opticalflow.visualize import display_img_pairs_w_flows
from opticalflow.optflow import flow_to_img
from DataWeight_load import *

# Build a list of image pairs to process

def init_optflow_nn(model_dir = 'opticalflow/pwcnet-lg-6-2-multisteps-chairsthingsmix/pwcnet.ckpt-595000'):
    
    gpu_devices = ['/device:GPU:0']  
    controller = '/device:GPU:0'

    ckpt_path = model_dir

    nn_opts = deepcopy(_DEFAULT_PWCNET_TEST_OPTIONS)
    nn_opts['verbose'] = True
    nn_opts['ckpt_path'] = ckpt_path
    nn_opts['batch_size'] = 1
    nn_opts['gpu_devices'] = gpu_devices
    nn_opts['controller'] = controller
    nn_opts['use_dense_cx'] = True
    nn_opts['use_res_cx'] = True
    nn_opts['pyr_lvls'] = 6
    nn_opts['flow_pred_lvl'] = 2
    nn_opts['adapt_info'] = (1, 436,1024, 2)

    nn = ModelPWCNet(mode='test', options=nn_opts)
    #nn.print_config()
    
    return nn


def img_to_optflow(frame_stream, batchsize, nn, target_hei =512, target_wid = 512, direction=True, with_resizing = True):
    
    img_pairs = []
    t_frames = []

    for i in range(batchsize):
        t_frames.append( cv2.resize(frame_stream[i], (1024, 436) ) )
    frame_stream = np.array(t_frames)

    if direction:
        for i in range(batchsize-1):
            img_pairs.append( (frame_stream[i], frame_stream[i+1]) )
    else:
        for i in range(batchsize-1):
            img_pairs.append( (frame_stream[ batchsize - (i+1) ], frame_stream[ batchsize - (i+2) ]) )

    pred_labels = nn.predict_from_img_pairs(img_pairs, batch_size=1, verbose=False)
    
    width = target_wid
    height = target_hei
    
    if with_resizing:
        resize_ori_images = []    

        for p in frame_stream:
            resize_ori_images.append(cv2.resize(p, (width, height) ))
        
        resize_optflow = []
        
        # need to check diff between auto_resize & manual's
        '''
        opt_h = p.shape[0]
        opt_w = p.shape[1]
        
        #print( (height, width) )
        for p in pred_labels:
            resize_image = np.zeros(shape=(height, width, 2))

            for H in range(height):
                n_hei = int( H * opt_h/height )
                for W in range(width):
                    n_wid = int( W * opt_w/width )
            
                    resize_image[H][W] = p[n_hei][n_wid]

            resize_optflow.append(resize_image)
        '''
        resize_optflow = flow_resize(pred_labels, (height, width)  )
        resize_ori_images = np.array(resize_ori_images)
      
        #display_img_pairs_w_flows(img_pairs, pred_labels)

    else:
        resize_ori_images = frame_stream
        resize_optflow = pred_labels

    return resize_ori_images, resize_optflow

