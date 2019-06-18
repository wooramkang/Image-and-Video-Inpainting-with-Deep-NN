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
from pconv_Dilatedconv_model import *

#img_to_optflow(frame_stream, batchsize, target_hei =400, target_wid = 400, direction=True, with_origin_img=True)

#masks T F boolean np-array

def flow_guide_inpaint( flows_forward, flows_backward, frames, masks, batchsize):

    image_shape = frames[0].shape
    
    height = image_shape[0]
    width = image_shape[1]
    #completion_frames = np.zeros(shape=(batchsize, height, width, 2))
    new_masks = np.ones(shape=(batchsize, height, width, 3))
    #new_masks = masks
    #print(masks[0])
    for _ in range(batchsize):

        for i in range(1, len(frames)-1):
            for h in range(height):        
                for w in range(width):
                    if masks[i][h][w][0] == 1:
                        new_h = int(h + flows_forward[i][h][w][1])
                        new_w = int(w + flows_forward[i][h][w][0])

                        if new_h >= height:
                            continue
                        if new_w >= width:
                            continue
                        if new_h < 0:
                            continue
                        if new_w < 0:
                            continue
                        if frames[i][ new_h ][ new_w ][0] == 255 and frames[i][ new_h ][ new_w ][1] == 255 and frames[i][ new_h ][ new_w ][2] == 255:
                            continue
                        if frames[i][ new_h ][ new_w ][0] == 0 and frames[i][ new_h ][ new_w ][2] == 0 and frames[i][ new_h ][ new_w ][2] == 0:
                            continue

                        frames[i-1][h][w] = frames[i][ new_h ][ new_w ]
                        
                        for t in range(3):
                            new_masks[i][h][w][t] = 0

        for i in range(len(frames) -1):
            for h in range(height):        
                for w in range(width):
                    if masks[i][h][w][0] == 1:
                        new_h = int(h + flows_backward[i][h][w][1])
                        new_w = int(w + flows_backward[i][h][w][0])

                        if new_h >= height:
                            continue
                        if new_w >= width:
                            continue
                        if new_h < 0:
                            continue
                        if new_w < 0:
                            continue
                        if frames[i][ new_h ][ new_w ][0] == 255 and frames[i][ new_h ][ new_w ][1] == 255 and frames[i][ new_h ][ new_w ][2] == 255:
                            continue
                        if frames[i][ new_h ][ new_w ][0] == 0 and frames[i][ new_h ][ new_w ][2] == 0 and frames[i][ new_h ][ new_w ][2] == 0:
                            continue

                        frames[i+1][h][w] = frames[i][ new_h ][ new_w ]
    
    return frames, new_masks
    '''
    refine_masks = []

    for i in range(len(masks)):
        new_mask = deepcopy(masks[i])
        
        refine_masks.append(new_mask)
    
    return frames, np.array(refine_masks)
    '''
def nn_guide_inpaint(frames, masks, frame_nn_model, batch_size):
    
    img_train_batch = image_normalization(frames)
    
    mask_batch = masks
    img_masked_batch = np.array(image_masking(img_train_batch, mask_batch) )
    
    #model_result = frame_nn_model.predict_on_batch (  [img_masked_batch, mask_batch ] )

    img_train_batch = image_to_origin(img_train_batch)
    img_masked_batch = image_to_origin(img_masked_batch)
    #model_result = image_to_origin(model_result)
    model_result = None

    return img_train_batch, img_masked_batch, model_result


def inpainting_process( flows_forward, flows_backward, frames, masks, batchsize, nn_model):
    
    frame_origin = frames
    frames , masks = flow_guide_inpaint( flows_forward, flows_backward, frames, masks, batchsize)
    flow_guide_frames, masked_frames, final_frames = nn_guide_inpaint(frames, masks, nn_model, batchsize)

    #return flow_guide_frames, masked_frames, final_frames
    return frame_origin, frames, masks
    