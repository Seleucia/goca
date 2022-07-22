import os,time,copy
import math
import sys
import cv2
import numpy as np
import argparse
import glob
import socket
use_cpu=False
nsave_thread=10
print(cv2.__version__)

from multiprocessing import Pool as ProcThreadPool
from multiprocessing.dummy import Pool as ThreadPool  ### this uses threads

parser = argparse.ArgumentParser(description="Implementation of GOCA OF extraction")
parser.add_argument("--cuid", type=int, default=0, help="Which cuda device")
parser.add_argument("--ncu", type=int, default=8, help="Number of cuda")
parser.add_argument("--nproc", type=int, default=8, help="N proccess")
parser.add_argument("--nsubsplit", type=int, default=5, help="N sub-splits")
parser.add_argument("--nclass", type=int, default=10, help="Split By Class")
parser.add_argument("--nclass_idx", type=int, default=0, help="Split By Class")
parser.add_argument("--dsmode", type=str, default='val', help="Modality")
args = parser.parse_args()

basename=args.dsmode
flow_root = '/home/hc/ds/temp/kinetics/ds/ds/gpu/flows8'
bpath = '/home/hc/ds/temp/kinetics/ds/ds/videos'
basename = 'val'

#
flow_root_real = os.path.join(flow_root, basename)
log_name=os.path.join(flow_root,'flow_export_logs_{0}_{1}_{2}.txt'.format(basename,args.cuid,args.ncu))
if args.cuid==0:
    print('Flow save to %s' % flow_root_real)
    print('Log File:'+log_name)

os.makedirs(flow_root_real, exist_ok=True)
flow_computer=cv2.cuda_OpticalFlowDual_TVL1.create()


def save_muti_full(flow_img_lst):
    def save_img(inps):
        flow_img, flow_img_path=inps
        cv2.imwrite(flow_img_path,
                    flow_img,
                    [cv2.IMWRITE_JPEG_QUALITY, 100])

    pool = ThreadPool(nsave_thread)
    pool.close()
    pool.join()


class PinnedMem(object):
    def __init__(self, size, dtype=np.uint8):
        self.array = np.empty(size,dtype)
        cv2.cuda.registerPageLocked(self.array)
        self.pinned = True
    def __del__(self):
        cv2.cuda.unregisterPageLocked(self.array)
        self.pinned = False
    def __repr__(self):
        return f'pinned = {self.pinned}'


def compute_TVL1_gpu(prev_gpu, curr_gpu, prev_pin_gpu, curr_pin_gpu,prev_gray_gpu,curr_gray_gpu,flow_calc_gpu,flow_pin_gpu,stream, bound=20):
    prev_gpu.upload(prev_pin_gpu.array,stream)
    curr_gpu.upload(curr_pin_gpu.array,stream)
    cv2.cuda.cvtColor(src=prev_gpu, dst=prev_gray_gpu, code=cv2.COLOR_BGR2GRAY, stream=stream)
    cv2.cuda.cvtColor(src=curr_gpu, dst=curr_gray_gpu, code=cv2.COLOR_BGR2GRAY, stream=stream)
    flow_computer.calc(I0=prev_gray_gpu, I1=curr_gray_gpu, flow=flow_calc_gpu, stream=stream)
    flow_calc_gpu.download(stream, flow_pin_gpu.array)
    stream.waitForCompletion();
    flow_gpu_np = np.clip(flow_pin_gpu.array, -bound, bound)

    flow_gpu_np = np.round((flow_gpu_np + bound) * (255.0 / (2 * bound))).astype('uint8')
    return flow_gpu_np

def extract_ff_opencv(v_path):

    v_class = v_path.split('/')[-2]
    v_name = os.path.basename(v_path)[0:-4]
    try:
        t1 = time.time()
        flow_out_dir = os.path.join(flow_root_real, v_class, v_name)

        vidcap = cv2.VideoCapture(v_path)
        nb_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        if os.path.exists(flow_out_dir):
            if len(glob.glob(os.path.join(flow_out_dir, '*.jpg'))) >= nb_frames - 5:  # tolerance = 3 frame difference
                print('[WARNING]', flow_out_dir, 'has finished, dropped!')
                vidcap.release()
                return [1, v_name]
        else:
            os.makedirs(flow_out_dir, exist_ok=True)

        width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
        if (width == 0) or (height == 0):
            print(width, height, v_path)

        empty_img = 128 * np.ones((int(height), int(width), 3)).astype(np.uint8)

        if use_cpu==False:
            stream = cv2.cuda_Stream()
            prev_gpu =cv2.cuda_GpuMat(empty_img.shape[0],empty_img.shape[1],cv2.CV_8UC3)
            curr_gpu = cv2.cuda_GpuMat(empty_img.shape[0],empty_img.shape[1],cv2.CV_8UC3)

            prev_gray_gpu = cv2.cuda_GpuMat(empty_img.shape[0],empty_img.shape[1],cv2.CV_8UC1)
            curr_gray_gpu = cv2.cuda_GpuMat(empty_img.shape[0],empty_img.shape[1],cv2.CV_8UC1)

            flow_calc_gpu = cv2.cuda_GpuMat(empty_img.shape[0],empty_img.shape[1],cv2.CV_32FC2)

            prev_pin_gpu = PinnedMem(size=empty_img.shape)
            curr_pin_gpu = PinnedMem(size=empty_img.shape)
            flow_pin_gpu = PinnedMem(size=(empty_img.shape[0],empty_img.shape[1],2), dtype=np.float32)
        else:
            stream = None
            prev_gpu = None
            curr_gpu = None

        success, _ = vidcap.read(curr_pin_gpu.array)
        count = 1
        flow_img_lst = []

        while success:
            if count != 1:
                flow_img = empty_img.copy()
                flow_img[:, :, 0:2] = compute_TVL1_gpu(prev_gpu,curr_gpu,prev_pin_gpu,curr_pin_gpu,prev_gray_gpu,curr_gray_gpu,flow_calc_gpu,flow_pin_gpu,stream)
                flow_img_lst.append([flow_img,os.path.join(flow_out_dir, 'flow_%05d.jpg' % (count - 1))])

            np.copyto(prev_pin_gpu.array,curr_pin_gpu.array)
            success, _ = vidcap.read(curr_pin_gpu.array)
            count += 1

        vidcap.release()
        if nb_frames < count + 3:
            save_muti_full(flow_img_lst)
            etime = time.time()
            print('Single Iter: ', etime - t1)
            return [1, v_name]
        else:
            return [0, v_name]
    except  Exception as inst:
        print(inst)
        return [-1, v_name]


def split_seq(seq, num_pieces):
    start = 0
    for i in range(num_pieces):
        stop = start + len(seq[i::num_pieces])
        yield seq[start:stop]
        start = stop

def save_logs(succ_fail_lst,ntotal):
    if os.path.exists(log_name):
        os.remove(log_name)
    nsuccess_0 = (np.asarray(succ_fail_lst)[:, 0].astype(np.int32) == 0).sum()
    nsuccess_1 = (np.asarray(succ_fail_lst)[:, 0].astype(np.int32) == 1).sum()
    nsuccess_min = (np.asarray(succ_fail_lst)[:, 0].astype(np.int32) == -1).sum()
    with open(log_name, "w") as f:
        f.write('Total Vid Files: '+str(ntotal) + "\n")
        str_logs='Success-Failures [1,0,-1]: {0}-{1}-{2}'.format(nsuccess_1,nsuccess_0,nsuccess_min)
        f.write(str_logs+ "\n")
        for s in succ_fail_lst:
            f.write(str(s) + "\n")


def main_kinetics400(v_root):
    v_root_real = v_root + '/' + basename
    if not os.path.exists(v_root_real):
        print('Wrong v_root');
        sys.exit()
    if args.nclass >1:
        cls_lst=sorted(os.listdir(v_root_real))
        sel_cls_lst=list(split_seq(cls_lst, args.nclass))[args.nclass_idx]
        full_vid_list = sum([glob.glob(os.path.join(v_root_real,cls, '*.mp4')) for cls in sel_cls_lst],[])
    else:
        full_vid_list=glob.glob(os.path.join(v_root_real, '*/*.mp4'))
    full_vid_list = sorted(full_vid_list)

    if args.ncu>1:
        vid_list_lst=list(split_seq(full_vid_list, args.ncu))[args.cuid]
    else:
        vid_list_lst=full_vid_list

    if args.nsubsplit>1:
        vid_list_lst=list(split_seq(vid_list_lst, args.nsubsplit))
    else:
        vid_list_lst=[vid_list_lst]
    completed_lst=[]
    succ_fail_lst=[]
    if args.nclass >1:
        print('Extracting: {0}/{1}, NVideos:{2}/{3}, Processing Class:{4}/{5} '.format(args.cuid,args.ncu,len(vid_list_lst[0])*args.nsubsplit,len(full_vid_list),
              args.nclass_idx,args.nclass))
    else:
        print('Extracting: {0}/{1}kinetics, NVideos:{2}/{3}'.format(args.cuid,args.ncu,len(vid_list_lst[0])*args.nsubsplit,len(full_vid_list)))
    for it,vid_list in enumerate(vid_list_lst):
        if len(vid_list)==0:
            continue

        pool = ProcThreadPool(args.nproc)
        ress_lst=pool.map(extract_ff_opencv, vid_list)
        pool.close()
        pool.join()
        for ress in ress_lst:
            succ_fail_lst.append([ress[0], ress[1]])

        nsuccess=(np.asarray(succ_fail_lst)[:,0].astype(np.int32)==1).sum()
        completed_lst.append(len(vid_list))
        print('Completed: {0}/{1}/{2}'.format(nsuccess,np.sum(completed_lst),len(full_vid_list)))
    save_logs(succ_fail_lst, len(full_vid_list))

if __name__ == '__main__':
    main_kinetics400(v_root=bpath)