import os,math,av,subprocess,glob,time
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool  ### this uses threads

ENABLE_MULTI_THREAD_DECODE = True
# Decoding backend, options include `pyav` or `torchvision`
DECODING_BACKEND = 'pyav'

max_size=190
# root_dir='/home/hc/ds/minikinetics/data/videos/train/'
# resized_video_folder='/home/hc/ds/minikinetics/data/videos/train_160'
#
# root_dir='/home/hc/ds/temp/kinetics/320/videos_flow'
# resized_video_folder='/home/hc/ds/temp/kinetics/160/videos_flow_resized'

root_dir='/home/hcoskun_snapchat_com/hc/dsdata/kinetics/kinetics-400_original/data/320'
resized_video_folder='/home/hcoskun_snapchat_com/hc/dsdata/kinetics/kinetics-400_original/data/190'
video_splits=[-1]
video_modalityies=['videos','videos_flow']

# root_dir='//home/hcoskun_snapchat_com/hc/dsdata/ucf101/320'
# resized_video_folder='//home/hcoskun_snapchat_com/hc/dsdata/ucf101/190'
#
# root_dir='//home/hcoskun_snapchat_com/hc/dsdata/hmdb/320'
# resized_video_folder='//home/hcoskun_snapchat_com/hc/dsdata/hmdb/190'

# video_splits=[1,2,3]
# video_modalityies=['videos','videos_flow']

def get_video_container(path_to_vid, multi_thread_decode=False, backend="pyav"):
    try:
        container = av.open(path_to_vid)
    except:
        container = av.open(path_to_vid, metadata_errors="ignore")
    if multi_thread_decode:
        # Enable multiple threads for decoding.
        container.streams.video[0].thread_type = "AUTO"
    curr_fps=container.streams.video[0].frames / (
                container.streams.video[0].duration / container.streams.video[0].time_base.denominator)
    return container,curr_fps

def get_sizes(max_size,height,width, boxes=None):

    size =max_size

    if (width <= height and width == size) or (
        height <= width and height == size
    ):
        return height, width
    new_width = size
    new_height = size
    if width < height:
        new_height = int(math.floor((float(height) / width) * size))
    else:
        new_width = int(math.floor((float(width) / height) * size))
    if new_width%2 !=0: new_width=new_width+1
    if new_height%2 !=0: new_height=new_height+1

    return  new_height, new_width

def resize_video(old_vpath):
    try:
        vname = old_vpath.split('/')[-1]
        cls_name = old_vpath.split('/')[-2]
        modal_name = old_vpath.split('/')[-3]
        video_modal_name = old_vpath.split('/')[-4]
        if sum(video_splits)>0:
            split_nm = old_vpath.split('/')[-5]
            dest_vid_dir=os.path.join(resized_video_folder,split_nm,video_modal_name,modal_name, cls_name)
        else:
            dest_vid_dir = os.path.join(resized_video_folder, video_modal_name, modal_name, cls_name)

        if not os.path.exists(dest_vid_dir):
            try:
                os.makedirs(dest_vid_dir,exist_ok=True)
            except Exception as ex:
              pass

        new_vpath = os.path.join(dest_vid_dir, vname)
        if os.path.exists(new_vpath):
            return 2
        video_container, curr_fps = get_video_container(
            old_vpath,
            ENABLE_MULTI_THREAD_DECODE,
            DECODING_BACKEND,
        )
        height, width=video_container.streams.video[0].height, video_container.streams.video[0].width
        nH,nW = get_sizes(max_size, height, width, boxes=None)
        #
        subprocess.run(['ffmpeg', '-y', '-i', old_vpath, '-loglevel','error',
                        '-vf', 'scale=' + str(nW) + ":" + str(nH), new_vpath])
        return 1
    except Exception as ex:
        print(ex)
        return 0

for split_id in video_splits:
    for video_modality in video_modalityies:
        if split_id>0:
            root_dir_modal = os.path.join(root_dir,'split_'+str(split_id), video_modality)
        else:
            root_dir_modal=os.path.join(root_dir,video_modality)
        print('Converting Path: ',root_dir_modal)
        print('Dest Path: ',resized_video_folder)
        s1=time.time()
        video_lst=glob.glob(os.path.join(root_dir_modal,'*/*/*.mp4'))
        s2=time.time()
        print('Number of Videos: ', len(video_lst),' time: ',s2-s1)
        pool = ThreadPool(100)
        ret=pool.map(resize_video, video_lst)
        pool.close()
        pool.join()
        s3=time.time()
        ret=np.asarray(ret)
        print('Number of succ video conversation: ', len(ret[ret==1]),' - ',len(ret[ret==2]),' - ',len(ret[ret==0]),' - ',len(video_lst),' time: ',s3-s2,' total-time: ', s3-s1)