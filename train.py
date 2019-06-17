from model import *
from DataWeight_load import *
import matplotlib.pyplot as plt
#from Pconv_model import *
def train_one_epoch(mask_loader, half_mask_loader, train_CN3D, train_dataloader, val_dataloader, batch_size, frame_size):
    
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

    cn3d_loss = 0
    comb_loss = 0
    #final_loss = 0
    
    if train_CN3D:
        cn3d_loss = CN3D_model.train_on_batch(vid_masked_batch, vid_train_batch)
        print("CN3D_model_loss")
        print(cn3d_loss)
        train_log.write(" " + str(cn3d_loss))
    else:
        cn3d_loss = CN3D_model.test_on_batch(vid_masked_batch, vid_train_batch)
        print("CN3D_model_val_loss")
        print(cn3d_loss)
        train_log.write(" " + str(cn3d_loss))

    comb_loss = CombCN_model.train_on_batch ( [img_masked_batch, vid_masked_batch], img_train_batch )
    print("COMB_model_loss")
    print(comb_loss)
    train_log.write(" " + str(comb_loss))

    '''
    if train_CN3D:
        final_loss = final_model.train_on_batch ( [img_masked_batch, vid_masked_batch], [img_train_batch, vid_train_batch])
        print("FINAL_model_loss")
        final_loss = sum(final_loss)/3
        print(final_loss)
    else:
        final_loss = final_model.test_on_batch ( [img_masked_batch, vid_masked_batch], [img_train_batch, vid_train_batch])
        print("FINAL_model_val_loss")
        final_loss = sum(final_loss)/3
        print(final_loss)
    '''
    #if cn3d_loss != 0:
    epoch_loss = (cn3d_loss + comb_loss)/2 # + final_loss) / 3
    #else:
        #epoch_loss = comb_loss

    return epoch_loss

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
    img_train_batch = image_to_origin(img_train_batch)

    return img_masked_batch, comb_result, img_train_batch


def train():
    BATCH_SIZE = 4
    #SAMPLE_BATCH_SIZE = 6
    FRAME_SIZE = 8
    EPOCH = 40000
    SAVE_TERM_PER_EPOCH = 500
    LEARN_RATE = 0.01

    MODEL_DIR = "model_log/"
    TRAIN_LOG_DIR ="train_log/"

    global train_log
    train_log = None
    train_log = open( TRAIN_LOG_DIR + "train_log.log", "w")

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
                                                            vid_net_mid_depth=3, frame_net_mid_depth=4,learn_rate = LEARN_RATE)    
    try:
        CN3D_model = Weight_load(CN3D_model, MODEL_DIR + "CN3D.h5")
        CombCN_model = Weight_load(CombCN_model, MODEL_DIR + "CombCN.h5")
        #final_model = Weight_load(final_model, MODEL_DIR + "final.h5")
        print("load saved model done")
    except:
        print("No saved model")

    for i in range(EPOCH):
        train_log.write(str(i+1) + " ")
        try:
            if i % SAVE_TERM_PER_EPOCH == 0:
                train_log.close()
                train_log = open( TRAIN_LOG_DIR + "train_log.log", "a")
                masked_in, result, raw_img = test_one_epoch(mask_loader, half_mask_loader,
                                                train_dataloader_forward, val_dataloader_forward, BATCH_SIZE, FRAME_SIZE)    
                fig = plt.figure()

                rows = BATCH_SIZE
                cols = 3
                c = 0
                
                for j in range(BATCH_SIZE):
                    c = c+1
                    ax = fig.add_subplot(rows, cols, c)
                    mask_ax =cv2.cvtColor(np.uint8(masked_in[j, :]), cv2.COLOR_BGR2RGB) 
                    ax.imshow(mask_ax)
                    cv2.imwrite(TRAIN_LOG_DIR + "mask_" + str(i) + "_" + str(j) + ".jpg", mask_ax)
                    #ax.set_xlabel(str(j))

                    c = c+1
                    ax2 = fig.add_subplot(rows, cols, c)
                    result_ax = cv2.cvtColor(np.uint8(result[j, :]), cv2.COLOR_BGR2RGB)
                    cv2.imwrite(TRAIN_LOG_DIR + "result_" + str(i) + "_" + str(j) + ".jpg", result_ax)
                    ax2.imshow(result_ax )
                    #ax2.set_xlabel(str(j))

                    c = c+1
                    ax3 = fig.add_subplot(rows, cols, c)
                    raw_img_ax = cv2.cvtColor(np.uint8(raw_img[j, :]), cv2.COLOR_BGR2RGB)
                    cv2.imwrite(TRAIN_LOG_DIR + "raw_" + str(i) + "_" + str(j) + ".jpg", raw_img_ax)
                    ax3.imshow(raw_img_ax)
                    #ax3.set_xlabel(str(j))

                #to check the training result
                plt.show()

                Weight_save(CN3D_model, MODEL_DIR + "CN3D.h5")
                Weight_save(CombCN_model, MODEL_DIR + "CombCN.h5")
                #Weight_save(final_model, MODEL_DIR + "final.h5")
                #FUTURE WORK
                #Init_plot()
                #sample_img_batch, sample_vid_batch = Random_sampling_data(SAMPLE_BATCH_SIZE, data_batch_loader_forward)
                #sample_result= final_model.predict(sample_data)
                #result_plot(sample_data, sample_result)

            if i % 3 == 0:
                #set_train_CN3D = True
                #set_train_CN3D_backward = False
                set_train_CN3D = True
                set_train_CN3D_backward = True
            elif i % 3 == 1:
                #set_train_CN3D = False
                #set_train_CN3D_backward = True
                set_train_CN3D = True
                set_train_CN3D_backward = True
            else:
                set_train_CN3D = True
                set_train_CN3D_backward = True

            print("train forward")
            forward_loss = train_one_epoch(mask_loader, half_mask_loader,set_train_CN3D,
                                            train_dataloader_forward, val_dataloader_forward, BATCH_SIZE, FRAME_SIZE)    
            print(str(i+1) + " epochs train  forward done ==> total loss on this epoch : " + str(forward_loss))
            train_log.write(" " + str(forward_loss))

            print("train backward")
            backward_loss = train_one_epoch(mask_loader, half_mask_loader,set_train_CN3D_backward,
                                            train_dataloader_backward, val_dataloader_backward, BATCH_SIZE, FRAME_SIZE)    
            print(str(i+1) + " epochs train  backward done ==> total loss on this epoch : " + str(backward_loss))        
            train_log.write(" " + str(backward_loss) + "\n")

        except:
            continue


    
if __name__ == "__main__":
    Data_dir = '../3D_model/DATASET/UCF-101/'
    Init_dataloader(Data_dir)
    train()

