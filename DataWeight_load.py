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

def __init_dataloader__ ():
    
    global default_dir
    global image_dir
    global video_dir
    default_dir = "/home/rd/recognition_research/3D_model/DATASET/UCF-101/"
    video_dir = default_dir
    image_dir = default_dir

def set_image_dir(t_str):
    global image_dir
    image_dir= t_str

def set_video_dir(t_str):
    global video_dir
    video_dir= t_str

def Video_loader(sampling_size = 30):
    video_list = None
    video_streams = None
    video_shape = None
    global video_dir


    return video_list, video_streams, video_shape

def get_video_shape():
    video_shape =None
    global video_dir

    return video_shape

def Img_loader(image_dir, img_szie):
    x_data = []
    y_data = []
    global image_dir

    # glob.glob("images/*"):
    folders = os.listdir(image_path)
    for name in folders:
        count=0
        #print(name)
        for file in glob.glob(image_path+name+"/*"):
            count= count+1
            identity = str(file).split('.')
            if identity[len(identity)-1] != 'jpg':
                break
            img_ = cv2.imread(file)
            img_ = cv2.resize(img_, (img_szie, img_szie))
            #print(img_.shape)
            img_ = np.transpose(img_, (2, 0, 1))
            x_data.append(img_)

            #y_data.append(identity)
            y_data.append(name)
            if count == 20:
                break

    print(len(x_data))
    print(len(y_data))
    print(len(folders))
    print("==============")

    return np.array(x_data), np.array(y_data)

def get_image_shape():
    img_shape =None
    global image_dir

    return img_shape

'''
def Img_load(image_path, img_size ):
    x_data = []
    y_data = []
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # glob.glob("images/*"):
    folders = os.listdir(image_path)
    print(folders)
    count = 0
    for name in folders:
        for file in glob.glob(image_path+name+"/*"):
            if count % 1000 == 0:
                print(count)

            identity = str(file).split('.')

            if identity[len(identity)-1] != 'jpg':
                continue
            
            #written by wooramkang 2018.09. 14
            #for broken images, you should check the images if it's okay or not
            
            with open(file, 'rb') as f:
                check_chars = f.read()[-2:]
                if check_chars != b'\xff\xd9':
                    print(file)
                    print('Not complete image')
                    continue
                else:
                    img_ = cv2.imread(file)

            count =count + 1
            _, img_ = make_transformed_faceset(img_)
            gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                sub_img = img_[y:y + h, x:x + w]
                sub_img = cv2.resize(sub_img, (img_size, img_size))

                #sub_imgs, _ = Removing_light([sub_img])
                #sub_img = sub_imgs[0]

                #sub_img = remove_shadow(sub_img)
                #sub_img = cv2.cvtColor(sub_img, cv2.COLOR_BGR2GRAY)
                sub_img = np.reshape(sub_img, (sub_img.shape[0], sub_img.shape[1], -1))

                sub_img = np.transpose(sub_img, (2, 0, 1))
                x_data.append(sub_img)
                y_data.append(name)

    print(len(x_data))
    print(len(y_data))
    print(len(folders))
    print("==============")

    return np.array(x_data), np.array(y_data)
'''

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
