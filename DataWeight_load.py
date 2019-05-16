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

def init_dataloader():
    
    global default_dir
    global image_dir
    global video_dir
    global shape

    default_dir = "/home/rd/recognition_research/3D_model/DATASET/UCF-101/"
    video_dir = default_dir
    image_dir = default_dir
    shape = None

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
    global video_dir
    global shape
    shape =None

    return shape

def Img_loader():
    ###UCF-101
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

        print(name)
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
            
            leaf_image_dir = len(os.listdir(image_dir + name + "/" + name_son))
            
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
    print(x_data[1]["name"])
    print(len(x_data[1]["path"]))
    print(len(x_data[1]["path"][1] ))
    print(len(x_data[1]["path"][1][2] ))
    print(x_data[1]["path"][1][2][3] )
    '''
        dataset UCF-101 - >x_data  hash structure tree
        x_data[1]          ["path"][1]  [2]  [3]
        x_data[scene_order]
              [random name order of scene]
        x_data[scene_order]["name"]
        x_data[scene_order]["path"]
                                   [1]
                                   [g01 = longcut01]
                                        [1]
                                        [c01 = cut01]
                                             [0] = "/ho...."
                                             [random order of image]
    '''
    return np.array(x_data)

def image_read(file_path):
    img_ = cv2.imread(file_path)
    img_ = np.transpose(img_, (2, 0, 1))
    return img_

def get_image_shape():
    global image_dir
    global shape
    shape = None

    return shape

'''##faces apart from the rest
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

if __name__ == "__main__":
    init_dataloader()
    Img_loader()