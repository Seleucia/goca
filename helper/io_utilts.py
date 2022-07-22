import socket
import os
from datetime import datetime
def getMName():
    return socket.gethostname()


def get_wd_root():
    root = '<path_to_goca>/goca'
    return root

def get_log_root():
    root = '/mnt/ssd2/ds/kinetics'
    root=os.path.join(root,'training_logs')
    return root

def get_ds_root(ds_name,split_id=-1,vid_base_arch='',evaluation_type=-1):
    root=''
    if ds_name=='kinetics':
        root = '/mnt/ssd2/ds/kinetics'
    elif ds_name=='ucf101':
        root = '/mnt/ssd2/ds/ucf101'
    elif ds_name=='hmdb':
        root = '/mnt/ssd2/ds/hmdb'
    if split_id>0:
        root=os.path.join(root,'split_'+str(split_id))
    return root


def create_ifnot_exist(dir_to_create):
    try:
        if not os.path.exists(dir_to_create):
            os.makedirs(dir_to_create)
    except:
        pass

def prep_logfiles(args,logmode):
    str_time = datetime.now().strftime('%m-%d_%H_%M_%S')
    if logmode in['lineval','finetune']:
        base_dir = os.path.join(args.root_log_path, args.ds_name + '_swav', logmode,args.vid_base_arch, args.video_channels,args.ckpt_idd, str_time)
    else:
        base_dir = os.path.join(args.root_log_path, args.ds_name + '_swav',logmode,args.vid_base_arch, args.video_channels, str_time)

    args.log_path = base_dir
    args.dump_path = os.path.join(base_dir, 'dumps')
    args.dump_checkpoints = os.path.join(base_dir, 'dumps', "checkpoints")
    create_ifnot_exist(args.log_path)
    create_ifnot_exist(args.dump_path)
    create_ifnot_exist(args.dump_checkpoints)
    return args