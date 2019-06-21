from DataWeight_load import *
import matplotlib.pyplot as plt
from pconv_Dilatedconv_model import *
from frame_resnet_encode_upsample_decode import * 
from img_to_flow import img_to_optflow
from keras.losses import mse
from keras.optimizers import Adam

def rescale(x,max_range,min_range):
    max_val = np.max(x)
    min_val = np.min(x)
    return (max_range-min_range)/(max_val-min_val)*(x-max_val)+max_range

def train_model(mask_loader, train_dataloader, target_size, epoch):
    #SAVE_TERM_PER_EPOCH = 500
    batch_size = 4
    BATCH_SIZE = batch_size
    LEARN_RATE = 0.001
    MODEL_DIR = "flow_model_log/"

    for i in range(epoch):

        img_train_batch = np.array( iter_to_one_batch(train_dataloader, BATCH_SIZE, False) )
        img_masked_batch = None
        mask_batch = None
        
        #IMAGE BATCH
        mask_batch = np.array(mask_to_one_batch(mask_loader, batch_size))
        img_masked_batch = np.array(image_masking(img_train_batch, mask_batch) )
        
        #with keras.backend.get_session().graph.as_default():
        _, flows_masked = img_to_optflow(img_masked_batch, batch_size, target_size[0], target_size[1], direction=True, with_resizing = True)
        _, flows_origin = img_to_optflow(img_train_batch, batch_size, target_size[0], target_size[1], direction=False, with_resizing = True)
        
        max_masked = np.max(flows_masked) # raw => norm
        min_masked = np.min(flows_masked)
        norm_flows_masked = deepcopy(flows_masked)
        norm_flows_masked =  (norm_flows_masked - min_masked) / (max_masked - min_masked)

        norm_flows_origin = deepcopy(flows_origin)
        norm_flows_origin =  (norm_flows_origin - min_masked) / (max_masked - min_masked)
        
        with keras.backend.get_session().graph.as_default():

            raw_input_shape = (512, 512, 2)
            target_output_shape = raw_input_shape

            en_model = Encoder( raw_input_shape )
            #en_model.summary()

            output_encoder = en_model.output
            output_size = int(output_encoder.get_shape()[1])
            decoder_input_shape = tuple((output_size, ))

            de_model = Decoder( decoder_input_shape , target_output_shape )
            #de_model.summary()

            raw_inputs = Input( (raw_input_shape[0], raw_input_shape[1], raw_input_shape[2], ) )
            optimizer_flow_inpaint = Adam(lr=LEARN_RATE)
            full_model = Model(inputs = raw_inputs, outputs = de_model(en_model(raw_inputs )) )
            #full_model.summary()
            full_model.compile(optimizer=optimizer_flow_inpaint, loss=mse)

            size_str = str( raw_input_shape[0] )

            try:
                full_model.load_weights(MODEL_DIR + "flow_inpaint_" + size_str + ".h5")
                print("load saved model done")
            except:
                print("No saved model")

            estim_loss = full_model.train_on_batch ( norm_flows_masked, norm_flows_origin )
            print(str(i+1) + " epochs train done ==> total loss on this epoch : " + str(estim_loss))
            full_model.save_weights(MODEL_DIR + "flow_inpaint_" + size_str + ".h5")

    return "Train Done"

def train():
    
    EPOCH = 40000
    
    img_shape = None
    raw_data = Img_loader()

    if img_shape is None:
        img_shape = (512, 512, 3)

    raw_input_shape = img_shape
    target_output_shape = (512, 512, 2)
    
    #img_shape = (436,1024, 3) # default_size of flow_nn model input

    mask_loader = MaskGenerator(img_shape[0], img_shape[1])#._generate_mask()
    train_data, _ = Data_split(raw_data, train_test_ratio = 0.8)
    train_dataloader = data_batch_loader_forward(train_data, (img_shape[0], img_shape[1])  )

    print(train_model(mask_loader, train_dataloader,  (target_output_shape[0], target_output_shape[1]), EPOCH) )
            

if __name__ == "__main__":
    Data_dir = '../3D_model/DATASET/UCF-101/'
    Init_dataloader(Data_dir)
    train()

