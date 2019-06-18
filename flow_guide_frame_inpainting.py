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

import img_to_flow
from DataWeight_load import *
from Pconv_Dilatedconv_model import *

#img_to_optflow(frame_stream, batchsize, target_hei =400, target_wid = 400, direction=True, with_origin_img=True)

#masks T F boolean np-array

def flow_guide_inpaint( flows_forward, flows_backward, frames, masks, batchsize):

    image_shape = frames[0].shape
    
    height = image_shape[0]
    width = image_shape[1]
    #completion_frames = np.zeros(shape=(batchsize, height, width, 2))
    new_masks = np.zeros(shape=(batchsize, height, width))

    for _ in range(2):
        for i in range(len(frames)-1 ):
            for h in range(height):        
                for w in range(width):
                    if masks[i][h][w]:
                        frames[i+1][h][w] = frames[i][ int(h + flows_forward[i][h][w][0]) ][ int(w + flows_forward[i][h][w][1] )]
                        new_masks[i][h][w] = 1

        for i in range(len(frames) -1):
            for h in range(height):        
                for w in range(width):
                    if masks[i][h][w]:
                        frames[i][h][w] = frames[i+1][ int(h + flows_backward[i][h][w][0]) ][ int(w + flows_backward[i][h][w][1] )]

    return frames, new_masks

def nn_guide_inpaint(frames, masks, frame_nn_model, batch_size):
    
    img_train_batch = image_normalization(frames)
    
    mask_batch = masks
    img_masked_batch = np.array(image_masking(img_train_batch, mask_batch) )
    
    model_result = frame_nn_model.predict_on_batch (  [img_masked_batch, mask_batch ] )

    img_train_batch = image_to_origin(img_train_batch)
    img_masked_batch = image_to_origin(img_masked_batch)
    model_result = image_to_origin(model_result)
    
    return img_train_batch, img_masked_batch, model_result


def inpainting_process( flows_forward, flows_backward, frames, masks, batchsize, nn_model):
    frames , masks = flow_guide_inpaint( flows_forward, flows_backward, frames, masks, batchsize)
    flow_guide_frames, masked_frames, final_framses = nn_guide_inpaint(frames, masks, nn_model, batch_size)

    return flow_guide_frames, masked_frames, final_framses
    