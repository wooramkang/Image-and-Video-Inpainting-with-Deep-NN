#from model import *
from DataWeight_load import *
import matplotlib.pyplot as plt
from pconv_Dilatedconv_model import *
from flow_guide_frame_inpainting import inpainting_process
from img_to_flow import img_to_optflow

def test_one_epoch(mask_loader, train_dataloader, BATCH_SIZE, target_size):
    batch_size = BATCH_SIZE
    random_sample_size = 10000
    
    for i in range(randint(0, random_sample_size)):
        next(train_dataloader)

    #img_train_batch = np.array( iter_to_one_batch(train_dataloader, BATCH_SIZE) )
    img_train_batch = np.array( iter_to_one_batch(train_dataloader, BATCH_SIZE, False) )

    _, flows_forward = img_to_optflow(img_train_batch, batch_size, target_size[0], target_size[1], direction=True, with_resizing = True)
    img_train_batch, flows_backward = img_to_optflow(img_train_batch, batch_size, target_size[0], target_size[1], direction=False, with_resizing = True)

    #IMAGE BATCH
    mask_batch = np.array(mask_to_one_batch(mask_loader, batch_size))
    img_masked_batch = np.array(image_masking(img_train_batch, mask_batch) )
    
    flow_guide_frames, masked_frames, final_framses = inpainting_process( flows_forward, flows_backward, img_masked_batch, mask_batch, batch_size)

    return flow_guide_frames, masked_frames, final_framses

def test():
    BATCH_SIZE = 8
    TRAIN_LOG_DIR ="img_train_log/"

    global train_log
    train_log = None
    img_shape = None
    train_dataloader_forward = None 
    raw_data = Img_loader()

    if img_shape is None:
        img_shape = (512, 512, 3)
    
    #img_shape = (436,1024, 3)

    mask_loader = MaskGenerator(img_shape[0], img_shape[1])#._generate_mask()
    train_data, val_data = Data_split(raw_data, train_test_ratio = 0.8)
    train_dataloader_forward = data_batch_loader_forward(train_data, (img_shape[0], img_shape[1])  )

    masked_in, result, raw_img = test_one_epoch(mask_loader, train_dataloader_forward,  BATCH_SIZE, img_shape)   
    fig = plt.figure()
    rows = BATCH_SIZE
    cols = 3
    c = 0
    
    for j in range(BATCH_SIZE):
        c = c+1
        ax = fig.add_subplot(rows, cols, c)
        mask_ax =cv2.cvtColor(np.uint8(masked_in[j, :]), cv2.COLOR_BGR2RGB) 
        cv2.imshow("mask",mask_ax)
        cv2.imwrite(TRAIN_LOG_DIR + "img_mask_TEST_" + str(j) + ".jpg", mask_ax)
        ax.imshow(mask_ax)

        c = c+1
        ax2 = fig.add_subplot(rows, cols, c)
        result_ax = cv2.cvtColor(np.uint8(result[j, :]), cv2.COLOR_BGR2RGB)
        cv2.imwrite(TRAIN_LOG_DIR + "img_result_TEST_" + str(j) + ".jpg", result_ax)
        cv2.imshow("result",result_ax)
        ax2.imshow(result_ax)

        c = c+1
        ax3 = fig.add_subplot(rows, cols, c)
        raw_img_ax = cv2.cvtColor(np.uint8(raw_img[j, :]), cv2.COLOR_BGR2RGB)
        cv2.imwrite(TRAIN_LOG_DIR + "img_raw_TEST_" + str(j) + ".jpg", raw_img_ax)
        cv2.imshow("raw",raw_img_ax)
        ax3.imshow(raw_img_ax)

        cv2.waitKey(5000)
    #to check the training result
    plt.show()
        
if __name__ == "__main__":
    Data_dir = '../3D_model/DATASET/UCF-101/'
    Init_dataloader(Data_dir)
    test()

