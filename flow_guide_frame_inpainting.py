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

def flow_guide_inpaint( flows_forward, flows_backward, frames, masks, batchsize):

    image_shape = frames[0].shape
    
    height = image_shape[0]
    width = image_shape[1]
    count = 0
    for i in range(1, len(frames)-2):
        for h in range(height):        
            for w in range(width):
                if masks[i-1][h][w][0] == 0:

                    new_h = int(h - flows_forward[i][h][w][0])
                    new_w = int(w - flows_forward[i][h][w][1])
                    
                    if (h - flows_forward[i][h][w][0]) - new_h > 0.5:
                        new_h = new_h +1
                    if (w - flows_forward[i][h][w][1]) - new_w > 0.5:
                        new_w = new_w +1

                    if new_h >= height:
                        continue
                    if new_w >= width:
                        continue
                    if new_h < 0:
                        continue
                    if new_w < 0:
                        continue

                    next_h = int(new_h - flows_forward[i+1][h][w][0])
                    next_w = int(new_w - flows_forward[i+1][h][w][1])

                    if (new_h - flows_forward[i+1][h][w][0]) - next_h > 0.5:
                        next_h = next_h + 1
                    if (new_w - flows_forward[i+1][h][w][1]) - next_w > 0.5:
                        next_w = next_w + 1

                    if next_h >= height:
                        continue
                    if next_w >= width:
                        continue
                    if next_h < 0:
                        continue
                    if next_w < 0:
                        continue

                    # frame k-1   <===>   frame k     <====>      frame k+1
                    if frames[i][new_h][new_w][0] != frames[i+1][next_h][next_w][0]:
                        continue

                    if frames[i][ new_h ][ new_w ][0] == 255 and frames[i][ new_h ][ new_w ][1] == 255 and frames[i][ new_h ][ new_w ][2] == 255:
                        continue
                    if frames[i][ new_h ][ new_w ][0] == 0 and frames[i][ new_h ][ new_w ][2] == 0 and frames[i][ new_h ][ new_w ][2] == 0:
                        continue

                    frames[i-1][h][w] = frames[i][ new_h ][ new_w ] 
                    
                    count = count + 1

                    for t in range(3):
                        masks[i-1][h][w][t] = 1
    print("times of flow guide inpainting _ forward")
    print(count)
    count = 0

    for i in range(len(frames) -2):
        for h in range(height):        
            for w in range(width):
                if masks[i+1][h][w][0] == 0:

                    new_h = int(h - flows_backward[i][h][w][0])
                    new_w = int(w - flows_backward[i][h][w][1])

                    if (h - flows_backward[i][h][w][0]) - new_h > 0.5:
                        new_h = new_h +1
                    if (w - flows_backward[i][h][w][1]) - new_w > 0.5:
                        new_w = new_w +1

                    if new_h >= height:
                        continue
                    if new_w >= width:
                        continue
                    if new_h < 0:
                        continue
                    if new_w < 0:
                        continue

                    next_h = int(new_h - flows_backward[i+1][h][w][0])
                    next_w = int(new_w - flows_backward[i+1][h][w][1])
                    
                    if (new_h - flows_backward[i+1][h][w][0]) - next_h > 0.5:
                        next_h = next_h + 1
                    if (new_w - flows_backward[i+1][h][w][1]) - next_w > 0.5:
                        next_w = next_w + 1

                    if next_h >= height:
                        continue
                    if next_w >= width:
                        continue
                    if next_h < 0:
                        continue
                    if next_w < 0:
                        continue

                    if frames[i][new_h][new_w][0] != frames[i-1][next_h][next_w][0]:
                        continue
                        
                    if frames[i][ new_h ][ new_w ][0] == 255 and frames[i][ new_h ][ new_w ][1] == 255 and frames[i][ new_h ][ new_w ][2] == 255:
                        continue
                    if frames[i][ new_h ][ new_w ][0] == 0 and frames[i][ new_h ][ new_w ][2] == 0 and frames[i][ new_h ][ new_w ][2] == 0:
                        continue

                    frames[i+1][h][w] = frames[i][ new_h ][ new_w ]
                    count = count + 1
                    for t in range(3):
                        masks[i+1][h][w][t] = 1

    print("times of flow guide inpainting _ backward")
    print(count)

    return frames, masks

def nn_guide_inpaint(frames, masks, batch_size):

    LEARN_RATE = 0.001
    img_shape = ( frames[0].shape[0], frames[0].shape[1], 3)
    mask_batch = mask_to_origin(masks)
    
    t_frames = []
    t_masks = []
    
    for i in range(len(frames)):
        t_frames.append( cv2.resize( frames[i], (512, 512)   ))
        t_masks.append( cv2.resize( mask_batch[i], (512, 512)   ))
    
    t_frames = np.array(t_frames)
    t_masks = np.array(t_masks)

    mask_batch = mask_normalization(t_masks)    
    img_train_batch = image_normalization(t_frames)

    with keras.backend.get_session().graph.as_default():
        try:
            nn_model = pdCN_network_generate(data_shape= (512, 512, 3), sampling_frame=8, frame_net_mid_depth=4, learn_rate = LEARN_RATE) 
            MODEL_DIR = "img_model_log/"
            nn_model.load_weights(MODEL_DIR + "pdCN.h5")
            #nn_model = Weight_load(nn_model, )
            print("load saved model done")
            model_result = nn_model.predict_on_batch (  [img_train_batch, mask_batch ] )
            print("completion done")
        except:
            model_result = img_train_batch
            print("No saved model")

        img_train_batch = image_to_origin(img_train_batch)
        mask_batch = mask_to_origin(mask_batch)
        model_result = image_to_origin(model_result)

    t_frames = []
    t_masks = []
    t_results = []

    for i in range(len(frames)):
        t_frames.append( cv2.resize( img_train_batch[i], (img_shape[1], img_shape[0])   ))
        t_masks.append( cv2.resize( mask_batch[i], (img_shape[1], img_shape[0])   ))
        t_results.append( cv2.resize( model_result[i], (img_shape[1], img_shape[0])   ))
    
    t_frames = np.array(t_frames)
    t_masks = np.array(t_masks)
    t_results = np.array(t_results)

    return t_frames, t_results, t_masks


def inpainting_process( flows_forward, flows_backward, frames, masks, batchsize):
    
    #for i in range(3):
    print(flows_forward.shape)
    print(flows_backward.shape)
    print(frames.shape)
    print(masks.shape)
    print("")

    print("flow guide inpainting start")
    frames , masks = flow_guide_inpaint( flows_forward, flows_backward, frames, masks, batchsize)
    print("flow guide inpainting done")
    print("")
    
    print("nn base inpainting start")
    flow_guide_frames, final_frames,  masked_frames = nn_guide_inpaint(frames, masks, batchsize)
    print("nn base inpainting done")
    print("")
    
    
    masks = mask_normalization(masked_frames)
    
    for i in range(len(masks)):
        for h in range(masks[0].shape[0]):
            for w in range(masks[0].shape[1]):
                for c in range(3):
                    if masks[i][h][w][c] < 1:
                        masks[i][h][w][c] = 0

    masked_frames = masks
    
    
    return flow_guide_frames, final_frames, masked_frames
    