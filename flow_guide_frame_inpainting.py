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
    count =0
    end_of_frame_arr = len(frames)-1
    
    for i in range(0, end_of_frame_arr):
        #print(i)
        for h in range(height):        
            for w in range(width):

                if masks[i][h][w][0] == 0:

                    next_crit_flow = flows_forward[i][h][w]

                    next_h = int(h - next_crit_flow[0])
                    next_w = int(w - next_crit_flow[1])

                    if ( h - next_crit_flow[0]) - next_h > 0.5:
                        next_h = next_h + 1

                    if ( w - next_crit_flow[1]) - next_w > 0.5:
                        next_w = next_w + 1
                        
                    if next_h >= height:
                        continue

                    if next_w >= width:
                        continue

                    if next_h < 0:
                        continue

                    if next_w < 0:
                        continue
                    
                    if masks[i + 1][ next_h ][ next_w ][0] == 0:
                        continue

                    frames[i][h][w] = frames[i + 1][next_h][next_w]

                    for t in range(3):
                        masks[i][h][w][t] = 1

                    count = count +1

    print(count)

    for i in range(0, end_of_frame_arr-1):
        #print(i)
        for h in range(height):        
            for w in range(width):
                    if masks[i][h][w][0] == 0:
                        
                        crit_flow = flows_forward[i][h][w]

                        prev_h = int(h - crit_flow[0])
                        prev_w = int(w - crit_flow[1])
                        
                        if (h - crit_flow[0]) - prev_h > 0.5:
                            prev_h = prev_h + 1
                        if (w - crit_flow[1]) - prev_w > 0.5:
                            prev_w = prev_w + 1

                        if prev_h >= height:
                            continue
                        if prev_w >= width:
                            continue
                        if prev_h < 0:
                            continue
                        if prev_w < 0:
                            continue
                        
                        next_crit_flow = flows_forward[i+1][prev_h][prev_w]

                        next_h = int(prev_h - next_crit_flow[0])
                        next_w = int(prev_w - next_crit_flow[1])

                        if (prev_h - next_crit_flow[0]) - next_h > 0.5:
                            next_h = next_h + 1                            
                        if (prev_h - next_crit_flow[0]) - next_w > 0.5:
                            next_w = next_w + 1

                        if next_h >= height:
                            continue
                        if next_w >= width:
                            continue
                        if next_h < 0:
                            continue
                        if next_w < 0:
                            continue

                        if masks[i + 2][ next_h ][ next_w ][0] == 0:
                            continue
                        if masks[i + 1][ prev_h ][ prev_w ][0] == 0:
                                                                                
                            frames[i][h][w] = frames[i + 2][next_h][next_w]
                            #for t in range(3):
                                #masks[i][h][w][t] = 1

                            frames[i + 1][prev_h][prev_w] = frames[i + 2][next_h][next_w]
                            #for t in range(3):
                                #masks[i + 1][prev_h][prev_w][t] = 1
                        else:

                            frames[i][h][w] = frames[i + 2][next_h][next_w]
                            
                            #for t in range(3):
                                #masks[i][h][w][t] = 1
                        
                        count = count +1

    print(count)
    '''
    ### backward ###
    
    for i in range(end_of_frame_arr, 0, -1):
        #print(i)
        for h in range(height):        
            for w in range(width):

                if masks[i][h][w][0] == 0:
                    
                    next_crit_flow = flows_backward[ batchsize - (i+1) ] [h][w]
                    next_h = int(h - next_crit_flow[0])
                    next_w = int(w - next_crit_flow[1])

                    if ( h - next_crit_flow[0]) - next_h > 0.5:
                        next_h = next_h + 1
                    if ( w - next_crit_flow[1]) - next_w > 0.5:
                        next_w = next_w + 1

                    if next_h >= height:
                        continue
                    if next_w >= width:
                        continue
                    if next_h < 0:
                        continue
                    if next_w < 0:
                        continue

                    if masks[ i - 1 ][ next_h ][ next_w ][0] == 0:
                        continue
                    
                    frames[i][h][w] = frames[i - 1][next_h][next_w]

                    for t in range(3):
                        masks[i][h][w][t] = 1

                    count = count +1
    
    #print(count)

    for i in range(end_of_frame_arr, 1, -1):
        for h in range(height):        
            for w in range(width):

                if masks[i][h][w][0] == 0:

                    crit_flow = flows_backward[ batchsize - (i+1) ][h][w]

                    prev_h = int(h - crit_flow[0])
                    prev_w = int(w - crit_flow[1])
                    
                    if (h - crit_flow[0]) - prev_h > 0.5:
                        prev_h = prev_h + 1
                    if (w - crit_flow[1]) - prev_w > 0.5:
                        prev_w = prev_w + 1

                    if prev_h >= height:
                        continue
                    if prev_w >= width:
                        continue
                    if prev_h < 0:
                        continue
                    if prev_w < 0:
                        continue

                    next_crit_flow = flows_backward[batchsize - (i+1) - 1 ][h][w]

                    next_h = int(prev_h - next_crit_flow[0])
                    next_w = int(prev_w - next_crit_flow[1])

                    if ( prev_h - next_crit_flow[0]) - next_h > 0.5:
                        next_h = next_h + 1

                    if ( prev_w - next_crit_flow[1]) - next_w > 0.5:
                        next_w = next_w + 1

                    if next_h >= height:
                        continue
                    if next_w >= width:
                        continue
                    if next_h < 0:
                        continue
                    if next_w < 0:
                        continue

                    if masks[i - 2][ next_h ][ next_w ][0] == 0:                    
                        continue

                    if masks[i - 1][ prev_h ][ prev_w ][0] == 0:
                        frames[i][h][w] = frames[i - 2][next_h][next_w]
                        #for t in range(3):
                            #masks[i][h][w][t] = 1
                        
                        frames[i - 1][prev_h][prev_w] = frames[i - 2][next_h][next_w]
                        #for t in range(3):
                            #masks[i - 1][prev_h][prev_w][t] = 1
                    else:

                        frames[i][h][w] = frames[i - 2][next_h][next_w]

                        #for t in range(3):
                            #masks[i][h][w][t] = 1
                        
                    count = count +1

    #print(count)
    '''
    return frames, masks, count

def nn_guide_inpaint(frames, masks, batch_size):

    LEARN_RATE = 0.001

    img_shape = ( frames[0].shape[0], frames[0].shape[1], 3)
    
    t_frames = []
    for i in range(len(frames)):
        t_frames.append( cv2.resize( frames[i], (512, 512) ))
    t_frames = np.array(t_frames)
    img_train_batch = image_normalization(t_frames)

    mask_batch = mask_resize(masks, (512, 512) )
    with keras.backend.get_session().graph.as_default():
        nn_model = pdCN_network_generate(data_shape= (512, 512, 3), sampling_frame=8, frame_net_mid_depth=4, learn_rate = LEARN_RATE) 
        #try:
        MODEL_DIR = "img_model_log/"
        nn_model.load_weights(MODEL_DIR + "pdCN.h5")
        #nn_model = Weight_load(
        #print("load saved model done")
        model_result = nn_model.predict_on_batch (  [img_train_batch, mask_batch ] )
        #print("completion done")
        #except:
            #print("No saved model")
            #model_result = img_train_batch

    img_train_batch = image_to_origin(img_train_batch)
    model_result = image_to_origin(model_result)

    t_frames = []
    t_results = []

    for i in range(len(model_result)):
        #cv2.cvtColor(masked_in[j, :]), )
        #t_frames.append( cv2.resize( img_train_batch[i], (img_shape[1], img_shape[0])   ))
        t_frames.append( np.uint8(cv2.resize(  img_train_batch[i], (img_shape[1], img_shape[0])   ) ) )  ##cv2.cvtColor( ## ,  cv2.COLOR_BGR2RGB ) ) 
        t_results.append( np.uint8(cv2.resize(  model_result[i], (img_shape[1], img_shape[0])   ) ) )
    
    t_frames = np.array(t_frames)
    t_results = np.array(t_results)
        
    #print(mask_batch)
    mask_batch = mask_resize(mask_batch, (img_shape[1], img_shape[0]))
    #print(mask_batch)

    # after masking, image distortion !
    return t_frames, mask_batch, t_results
    

def inpainting_process( flows_forward, flows_backward, frames, masks, batchsize, inference_nn_model = True):
    
    print("flow guide inpainting start")
    frames, masks, paint_count = flow_guide_inpaint( flows_forward, flows_backward, frames, masks, batchsize)
    print("flow guide inpainting done")
    print("")

    if inference_nn_model:
        print("nn base inpainting start")
        flow_guide_frames, masks_batch, final_frames = nn_guide_inpaint(frames, masks, batchsize)
        print("nn base inpainting done")
        print("")
    else:
        return frames, frames, masks, paint_count
    
    return flow_guide_frames, final_frames, masks_batch, paint_count
