from model import *
from DataWeight_load import *
from result_plot import *

def test():
    # TRAIN EPOCH
    epoch = 40000
    weights_path = ""

    raw_data = Img_loader()

    if img_shape is None:
        img_shape = get_image_shape(raw_data)


    test_model = network_generate(sampling_frame=8, data_shape=img_shape, 
                                    vid_net_mid_depth=3, frame_net_mid_depth=4)
    
    test_model = Weight_load(test_model, weights_path)
    

    # FUTURE WORK
    Init_plot()
    sample_data = None
    sample_result=test_model(sample_data)
    result_plot(sample_data, sample_result)

if __name__ == "__main__":
    Init_dataloader()
    test()