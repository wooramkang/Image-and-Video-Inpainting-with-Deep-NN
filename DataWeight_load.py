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
default_dir = "/home/rd/recognition_research/3D_model/DATASET/UCF-101/"

#2019. 05 .16 video to images & only image stream loader // wooramkang

def video_loader(video_dir=None, sampling_size = None):

    video_list = None
    video_streams = None
    video_shape = None
    
    if video_dir is None:
        video_dir = default_dir

    if sampling_size is None:
        sampling_size = 30

    return video_list, video_streams, video_shape

def get_video_shape(video_dir=None):
    global default_dir
    video_shape =None

    if video_dir is None:
        video_dir = default_dir

    return video_shape

def Img_load(image_path, img_szie ):
    x_data = []
    y_data = []
    default_dir = "/home/rd/recognition_research/3D_model/DATASET/UCF-101/"
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
            '''
            #written by wooramkang 2018.09. 14
            #for broken images, you should check the images if it's okay or not
            
            '''
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


def Weight_load(model, weights_path):
    model.load_weights(weights_path)
    return model
