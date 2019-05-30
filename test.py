from model import *
from DataWeight_load import *
from result_plot import *
import matplotlib.pyplot as plt

def test_one_epoch(mask_loader, half_mask_loader, train_dataloader, val_dataloader, batch_size, frame_size):
    
    vid_train_batch = [ iter_to_one_batch(train_dataloader, frame_size) for i in range(batch_size) ]
    img_train_batch = np.array([ frame_per_batch[len(frame_per_batch)-1] for frame_per_batch in vid_train_batch])
    vid_train_batch = np.array([ image_to_half_size(batch) for batch in vid_train_batch])
    #for prediction task
    #img_trian_batch = iter_to_one_batch(val_dataloader, batch_size)

    vid_masked_batch = []
    img_masked_batch = None
    mask_batch = None
    
    #VIDEO BATCH
    for i in range(batch_size):
        half_mask_batch = mask_to_one_batch(half_mask_loader, frame_size)
        frame_masked_batch = image_masking(vid_train_batch[i], half_mask_batch)
        vid_masked_batch.append(frame_masked_batch)
    vid_masked_batch = np.array(vid_masked_batch)

    #IMAGE BATCH
    mask_batch = mask_to_one_batch(mask_loader, batch_size)
    img_masked_batch = image_masking(img_train_batch, mask_batch)
    
    #cn3d_loss = CN3D_model.train_on_batch(vid_masked_batch, vid_train_batch)
    
    comb_result = CombCN_model.predict_on_batch ( [img_masked_batch, vid_masked_batch] )

    #final_loss = final_model.train_on_batch ( [img_masked_batch, vid_masked_batch], [img_train_batch, vid_train_batch])

    img_masked_batch = image_to_origin(img_masked_batch)
    comb_result = image_to_origin(comb_result)

    return img_masked_batch, comb_result


def train():
    BATCH_SIZE = 4
    #SAMPLE_BATCH_SIZE = 6
    FRAME_SIZE = 8
    EPOCH = 40000
    SAVE_TERM_PER_EPOCH = 500

    MODEL_DIR = "model_log/"
    TRAIN_LOG_DIR ="train_log/"

    img_shape = None #(320, 240, 3)
    train_dataloader_forward = None 
    train_dataloader_backward = None
    val_dataloader_forward = None
    val_dataloader_backward = None

    raw_data = Img_loader()

    if img_shape is None:
        img_shape = Get_image_shape()
        print("###")
        print(img_shape)
        print("###")

    mask_loader = MaskGenerator(img_shape[0], img_shape[1])#._generate_mask()
    half_mask_loader = MaskGenerator(int(img_shape[0]/2), int(img_shape[1]/2) )#._generate_mask()
    train_data, val_data = Data_split(raw_data, train_test_ratio = 0.8)
    
    train_dataloader_forward = data_batch_loader_forward(train_data)
    train_dataloader_backward = data_batch_loader_backward(train_data)
    #FUTURE WORK
    #val_dataloader_forward = data_batch_loader_forward(val_data)
    #val_dataloader_backward = data_batch_loader_backward(val_data)
    global CN3D_model
    global CombCN_model
    '''
    global final_model
    
    CN3D_model, CombCN_model, final_model = None, None, None
    CN3D_model, CombCN_model, final_model = network_generate(sampling_frame=8, data_shape=img_shape, 
                                                            vid_net_mid_depth=3, frame_net_mid_depth=4)    
    '''
    CN3D_model, CombCN_model = None, None
    CN3D_model, CombCN_model = network_generate(sampling_frame=8, data_shape=img_shape, 
                                                            vid_net_mid_depth=3, frame_net_mid_depth=4)    
    try:
        CN3D_model = Weight_load(CN3D_model, MODEL_DIR + "CN3D.h5")
        CombCN_model = Weight_load(CombCN_model, MODEL_DIR + "CombCN.h5")
        #final_model = Weight_load(final_model, MODEL_DIR + "final.h5")
        print("load saved model done")
    except:
        print("No saved model")


    masked_in, result = test_one_epoch(mask_loader, half_mask_loader,
                                train_dataloader_forward, val_dataloader_forward, BATCH_SIZE, FRAME_SIZE)    

    fig = plt.figure()
    rows = 2
    cols = BATCH_SIZE
    c = 0
    
    for j in range(cols):
        c = c+1
        ax = fig.add_subplot(rows, cols, c)
        ax.imshow(cv2.cvtColor(np.uint8(masked_in[j, :]), cv2.COLOR_BGR2RGB) )
        ax.set_xlabel(str(j))

        c = c+1
        ax2 = fig.add_subplot(rows, cols, c)
        ax2.imshow(cv2.cvtColor(np.uint8(result[j, :]), cv2.COLOR_BGR2RGB) )
        ax2.set_xlabel(str(j))

    #to check the training result
    plt.show()
    plt.savefig(TRAIN_LOG_DIR + "plt" + str(i) + ".png")
    
if __name__ == "__main__":
    Init_dataloader()
    train()

