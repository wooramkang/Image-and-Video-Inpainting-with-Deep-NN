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

# Build a list of image pairs to process

def img_to_optflow(frame_stream, batchsize, direction=True):
    
    img_pairs = []

    if direction:
        for i in range(batchsize-1):
            frame_stream[i] = cv2.resize(frame_stream[i], (1024, 436)  )
            frame_stream[i+1] = cv2.resize(frame_stream[i+1], (1024, 436)  )
            img_pairs.append( (frame_stream[i], frame_stream[i+1]) )
    else:
        for i in range(batchsize-1):
            frame_stream[batchsize - (i+1)] = cv2.resize(frame_stream[batchsize - (i+1)], (1024, 436)  )
            frame_stream[batchsize - (i+2)] = cv2.resize(frame_stream[batchsize - (i+2)], (1024, 436)  )
            img_pairs.append( (frame_stream[i], frame_stream[i+1]) )

    gpu_devices = ['/device:GPU:0']  
    controller = '/device:GPU:0'

    ckpt_path = 'opticalflow/pwcnet-lg-6-2-multisteps-chairsthingsmix/pwcnet.ckpt-595000'

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
    nn.print_config()

    pred_labels = nn.predict_from_img_pairs(img_pairs, batch_size=1, verbose=False)

    return pred_labels

if __name__ == "__main__":
    frames = []
    frames.append(cv2.imread("mpisintel_test_clean_ambush_1_frame_0001.png"))
    frames.append(cv2.imread("mpisintel_test_clean_ambush_1_frame_0002.png"))
    frames = np.array(frames)
    print(img_to_optflow(frames, 2))
    print("")