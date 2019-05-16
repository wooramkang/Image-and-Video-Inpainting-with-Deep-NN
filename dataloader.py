global default_dir
default_dir = "/home/rd/recognition_research/3D_model/DATASET/UCF-101/"

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