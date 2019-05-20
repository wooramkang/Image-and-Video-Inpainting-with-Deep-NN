from model import *
from DataWeight_load import *
from result_plot import *

def train_one_epoch(model, train_data, val_data, learning_rate, ):
    
    return loss

def train():
    # TRAIN EPOCH
    epoch = 40000

    if img_shape is None:
        img_shape = Get_image_shape()

    raw_data = Img_loader()
    train_data, val_data = Data_split(raw_data, train_test_ratio = 0.7)

    train_model = network_generate(sampling_frame=8, data_shape=img_shape, 
                                    vid_net_mid_depth=3, frame_net_mid_depth=4)
    
    for i in range(epoch):
        loss = train_one_epoch(train_model, train_data, val_data, learning_rate = 0.01)    

    # FUTURE WORK
    
    Init_plot()
    sample_data = None
    sample_result=train_model(sample_data)
    result_plot(sample_data, sample_result)

if __name__ == "__main__":
    Init_dataloader()
    train()


