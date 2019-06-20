#from model import *
from DataWeight_load import *
import matplotlib.pyplot as plt
from pconv_Dilatedconv_model import *

def test_one_epoch(mask_loader, train_dataloader, BATCH_SIZE):
    batch_size = BATCH_SIZE
    random_sample_size = 10000
    
    for i in range(randint(0, random_sample_size)):
        next(train_dataloader)

    img_train_batch = np.array( iter_to_one_batch(train_dataloader, BATCH_SIZE) )
    img_masked_batch = None
    mask_batch = None
    
    #IMAGE BATCH
    mask_batch = np.array(mask_to_one_batch(mask_loader, batch_size))
    img_masked_batch = np.array(image_masking(img_train_batch, mask_batch) )
    
    pdCN_result = pdCN_model.predict_on_batch (  [img_masked_batch, mask_batch ] )
    img_masked_batch = image_to_origin(img_masked_batch)
    pdCN_result = image_to_origin(pdCN_result)
    img_train_batch = image_to_origin(img_train_batch)

    return img_masked_batch, pdCN_result, img_train_batch

def test():
    BATCH_SIZE = 4
    LEARN_RATE = 0.001
    MODEL_DIR = "img_model_log/"
    TRAIN_LOG_DIR ="img_train_log/"

    global train_log
    train_log = None
    img_shape = None
    train_dataloader_forward = None 
    raw_data = Img_loader()

    if img_shape is None:
        img_shape = (512, 512, 3)
        print("###")
        print(img_shape)
        print("###")
    
    global pdCN_model
    pdCN_model = None

    if (img_shape[0] > 512 and img_shape[1] > 512):
        mask_loader = MaskGenerator(img_shape[0], img_shape[1])#._generate_mask()
        train_data, val_data = Data_split(raw_data, train_test_ratio = 0.8)
        train_dataloader_forward = data_batch_loader_forward(train_data)
        pdCN_model = pdCN_network_generate(data_shape= img_shape, sampling_frame=8, frame_net_mid_depth=4, learn_rate = LEARN_RATE)    
    else:
        mask_loader = MaskGenerator(512, 512)#._generate_mask()
        train_data, val_data = Data_split(raw_data, train_test_ratio = 0.8)
        train_dataloader_forward = data_batch_loader_forward(train_data, (512, 512) )
        pdCN_model = pdCN_network_generate(data_shape= (512, 512, 3), sampling_frame=8, frame_net_mid_depth=4, learn_rate = LEARN_RATE)

    try:
        pdCN_model = Weight_load(pdCN_model, MODEL_DIR + "pdCN.h5")
        print("load saved model done")
    except:
        print("No saved model")
    
    masked_in, result, raw_img = test_one_epoch(mask_loader, train_dataloader_forward,  BATCH_SIZE)   
    fig = plt.figure()
    rows = BATCH_SIZE
    cols = 3
    c = 0
    
    for j in range(BATCH_SIZE):
        c = c+1
        ax = fig.add_subplot(rows, cols, c)
        mask_ax =cv2.cvtColor(np.uint8(masked_in[j, :]), cv2.COLOR_BGR2RGB) 
        ax.imshow(mask_ax)
        cv2.imwrite(TRAIN_LOG_DIR + "img_mask_TEST_" + str(j) + ".jpg", mask_ax)

        c = c+1
        ax2 = fig.add_subplot(rows, cols, c)
        result_ax = cv2.cvtColor(np.uint8(result[j, :]), cv2.COLOR_BGR2RGB)
        cv2.imwrite(TRAIN_LOG_DIR + "img_result_TEST_" + str(j) + ".jpg", result_ax)
        ax2.imshow(result_ax)

        c = c+1
        ax3 = fig.add_subplot(rows, cols, c)
        raw_img_ax = cv2.cvtColor(np.uint8(raw_img[j, :]), cv2.COLOR_BGR2RGB)
        cv2.imwrite(TRAIN_LOG_DIR + "img_raw_TEST_" + str(j) + ".jpg", raw_img_ax)
        ax3.imshow(raw_img_ax)

    #to check the training result
    plt.show()
        
if __name__ == "__main__":
    Data_dir = '../3D_model/DATASET/UCF-101/'
    Init_dataloader(Data_dir)
    test()

