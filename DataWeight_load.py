from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import numpy as np
from numpy import genfromtxt
import cv2
import PIL
import keras
from keras.models import Model
global default_dir
global image_dir
global video_dir

#2019. 05 .16 video to images & only image stream loader // wooramkang

def Set_video_dir(t_str):
    global video_dir
    video_dir= t_str
def Set_image_dir(t_str):
    global image_dir
    image_dir= t_str

def Init_dataloader():
    global default_dir
    global image_dir
    global video_dir
    global shape

    default_dir = "/home/rd/recognition_research/3D_model/DATASET/UCF-101/"
    Set_video_dir(default_dir)
    Set_image_dir(default_dir)
    shape = None

def Img_loader():
    ###UCF-101
    ###use image stream not video type
    '''
        2019. 05 .16 video to images & only image stream loader // wooramkang
        dataset UCF-101 - >x_data  hash structure tree i made

        x_data[1]          ["path"][1]  [2]  [3]
              [start from 0]
        x_data[scene_order]
              [random name order of scene]
        x_data[scene_order]["name"]
        x_data[scene_order]["path"]
                                   [1]
                                   [g01 = longcut01]
                                   [start from 1]
                                        [1]
                                        [c01 = cut01]
                                        [start from 1]
                                             [0] = "/ho...."
                                             [random order of image]
                                             [start from 0]
    '''
    x_data = []
    global image_dir

    folders_root = os.listdir(image_dir)

    for name in folders_root:
        #ex) ApplyEyeMakeup
        count=0
        folders_son = os.listdir(image_dir+name)

        images = dict()
        images["name"] = name    
        images["path"] = dict()

        past_g = ""
        past_c = ""

        check_change= False

        for name_son in folders_son:
            
            folder_property = name_son.split('_')
            temp_len = len(folder_property) - 1
            now_g = folder_property[temp_len-1]
            now_c = folder_property[temp_len]
            
            if past_g != now_g :
                check_change = True
            
            if past_c != now_c :
                check_change = True
            
            if check_change:
                
                if not int( now_g[1:3] ) in images["path"]:
                    images["path"][ int( now_g[1:3] ) ] = dict()

                past_g = now_g
                past_c = now_c
                check_change= False
            
            images["path"][ int( now_g[1:3] ) ][ int(now_c[1:3]) ] = []
            
            for file in glob.glob(image_dir + name + "/" + name_son + "/*"):
                #ex) ApplyEyeMakeup_G01_C01
                count= count+1

                identity = str(file).split('.')
                if identity[len(identity)-1] != 'jpg':
                    continue

                images["path"][ int( now_g[1:3] ) ][ int(now_c[1:3]) ].append(file)
                
        x_data.append(images)

    print(len(x_data))    
    print(x_data[0]["name"])
    print(len(x_data[1]["path"]))
    print(len(x_data[1]["path"][1] ))
    print(len(x_data[1]["path"][1][1] ))
    print(x_data[1]["path"][1][2][0] )
    
    return np.array(x_data)


def Image_read(file_path):
    
    img = cv2.imread(file_path)
    '''
    img = cv2.resize(img,(128,128))
    #IN A LOT OF PAPER,
    #RESIZE 128 * 128
    '''
    img = np.transpose(img, (2, 0, 1))
    return img

def Get_image_shape(image_data):
    global shape
    if shape != None:
        return shape

    shape = image_data.get_shape()
    return shape

def Data_split(x_data, train_test_ratio = 0.7):
    #params
    #to split data to train and validate OR to train and test
    
    x_train = []
    x_test = []
    
    data_len = len(x_data)
    train_len = int(data_len * train_test_ratio)

    train_list = range(train_len)
    test_list = range(train_len, data_len)

    for i in train_list:
        x_train.append(x_data[i])
        
    for i in test_list:
        x_test.append(x_data[i])
        
    return np.array(x_train), np.array(x_test)

def Weight_load(model, weights_path):
    model.load_weights(weights_path)
    return model

'''
def get_video_shape(image_data):
    
    global shape
    shape =None

    return shape
def Video_loader(sampling_size = 30):
    video_list = None
    video_streams = None
    video_shape = None
    global video_dir

    return video_list, video_streams, video_shape
'''

#FOR DATALOADER TEST
if __name__ == "__main__":
    Init_dataloader()
    Img_loader()