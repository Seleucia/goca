# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import datetime,os
import helper.io_utilts as iout
from helper.my_enums import ModalMergeType,ModalMergeAlgo,ChannelModal,EvalType

def parse_arguments():
    def str2bool(v):
        v = v.lower()
        if v in ('yes', 'true', 't', '1'):
            return True
        elif v in ('no', 'false', 'f', '0'):
            return False
        raise ValueError('Boolean argument needs to be true or false. '
                         'Instead, it is %s.' % v)

    def strlist2list(v):
        v = v.lower()
        if '|' in v:
            ret_lst = []
            list_of_list_of_items=v.split('|')
            for llli in list_of_list_of_items:
                list_of_items=llli.split(',')
                itemm=[int(l) for l in list_of_items]
                ret_lst.append(itemm)
            return ret_lst
        else:
            list_of_items=v.split(',')
            return [int(l) for l in list_of_items]


    parser = argparse.ArgumentParser(description="Implementation of GOCA")
    parser.register('type', 'bool', str2bool)
    parser.register('type', 'strlist', strlist2list)

    #########################
    #### data parameters ####
    #########################

    parser.add_argument("--ds_name", type=str, default='kinetics',
                        choices=[ 'kinetics','ucf101', 'hmdb'],
                        help="name of dataset")

    parser.add_argument("--split_idx", type=int, default=-1,
                        help="Dataset Split")

    parser.add_argument("--data_root_dir", type=str, default="",
                        help="root dir of dataset")

    parser.add_argument("--data_cache_dir", type=str, default="",
                        help="path to store dataset pkl files")

    parser.add_argument("--num_data_samples", type=int, default=None,
                        help="number of dataset samples")

    parser.add_argument("--target_fps", type=int, default=30,
                        help="video fps")

    parser.add_argument("--video_channels", type=str, default=ChannelModal.rgb_flow.value,
                        help="video channels to be used",choices=[en.value for en in ChannelModal])

    parser.add_argument("--video_channels_val", type=str, default=ChannelModal.rgb_flow.value,
                        help="video channels to be used",choices=[en.value for en in ChannelModal])

    parser.add_argument("--share_proj_head", type='bool', default='False',
                        help="Should we merge projection head")

    parser.add_argument("--video_modal_merge_algo", type=int, default=ModalMergeAlgo.PriorRgbFlow.value,
                        help="How to handle merged modalities, 0: avg, 1: Mix, 1: using prior",choices=[en.value for en in ModalMergeAlgo])

    parser.add_argument("--video_modal_merge", type=int, default=ModalMergeType.ProcSeperate.value,
                        help="how to merge features of modalities, 0: avg, 1: concat, 3: using prior, 4: just merging",choices=[en.value for en in ModalMergeType])

    parser.add_argument("--video_modal_merge_val", type=int, default=ModalMergeType.Avg.value,
                        help="how to merge features of modalities, 0: avg, 1: concat, 3: using prior",choices=[en.value for en in ModalMergeType])

    parser.add_argument("--video_modal_merge_start", type=int, default=100,
                        help="When to start merging information from two modalities.")

    parser.add_argument('--subsample_with', default=1, type=int,
                        help='If we are subsampling over FPS')

    parser.add_argument("--num_train_clips", type=int, default=1,
                        help="number of clips to sample per videos")

    parser.add_argument("--train_crop_size", type=int, default=0,
                        help="crop size")

    parser.add_argument("--val_crop_size", type=int, default=0,
                        help="val crop size")

    parser.add_argument("--train_crop_range", type=list, default=[0,0],
                        help="train crop size")

    parser.add_argument("--val_crop_range", type=list, default=[0,0],
                        help="train crop size")



    parser.add_argument('--use_colorjitter', type=float, default=0.8,
                        help='use color jitter')

    parser.add_argument('--use_grayscale', type=float, default=0.2,
                        help='use grayscale augmentation')

    parser.add_argument('--use_gaussian', type=float, default=0.5,
                        help='use gaussian augmentation')

    parser.add_argument('--use_timereverse', type=float, default=0.0,
                        help='Randomly reverse list in time dimension')

    parser.add_argument('--train_center_crop', type='bool', default='False',
                        help='Central crop for train')

    #########################
    #### SK parameters ###
    #########################
    parser.add_argument('--schedulepower', default=1.3, type=float,
                        help='SK schedule power compared to linear (default: 1.5)')
    parser.add_argument('--nopts', default=100, type=int,
                        help='number of pseudo-opts (default: 100)')
    parser.add_argument('--lamb', default=20, type=int,
                        help='for pseudoopt: lambda (default:25) ')
    parser.add_argument('--dist', default=None, type=int,
                        help='use for distribution')
    parser.add_argument('--diff_dist_every', default='False', type='bool',
                        help='use a different Gaussian at every SK-iter?')
    parser.add_argument('--diff_dist_per_head', default='True', type='bool',
                        help='use a different Gaussian for every head?')

    #########################
    #### Selavi parameters ###
    #########################
    parser.add_argument('--ind_groups', default=2, type=int,
                        help='number of independent groups (default: 100)')
    parser.add_argument('--gauss_sd', default=0.1, type=float,
                        help='sd')
    parser.add_argument('--match', default='True', type='bool',
                        help='match distributions at beginning of training')
    parser.add_argument('--distribution', default='default', type=str,
                        help='distribution of SK-clustering', choices=['gauss', 'default', 'zipf'])
    #########################
    #### Sinkhorn parameters ###
    #########################

    parser.add_argument("--epsilon", default=0.05, type=float,
                        help="regularization parameter for Sinkhorn-Knopp algorithm")
    parser.add_argument("--epsilon2", default=0.01, type=float,
                        help="regularization parameter for Sinkhorn-Knopp algorithm for prior")
    parser.add_argument("--sinkhorn_iterations", default=3, type=int,
                        help="number of iterations in Sinkhorn-Knopp algorithm")

    parser.add_argument("--sinkhorn_cycle", default=1, type=int,
                        help="How many times send and receive prior")

    #########################
    #### GOCA Spevc parameters ###
    #########################
    parser.add_argument("--temperature", default=0.1, type=float,
                        help="temperature parameter in training loss")
    parser.add_argument('-temporal_for_assign', default='0,1', type='strlist',
                        help="number of frames to sample per clip")
    parser.add_argument('--num_frames_lst', default='4,2', type='strlist',
                        help="number of frames to sample per clip. Should be in descending order")
    parser.add_argument("--nmb_temporal_samples",  default='2,4', type='strlist',
                        help="list of number of crops (example: [2, 6])")

    parser.add_argument("--eval_lr", default=0.3, type=float, help="base learning rate")
    parser.add_argument("--eval_wd", default=1e-6, type=float, help="weight decay")
    parser.add_argument('--train_equal_temporal_sampling', type='bool', default='False',
                        help='Temporal sampling for train set')

    #########################
    #### Clip parameters ###
    #########################
    parser.add_argument('--num_frames_lst_val', default='8', type='strlist',
                        help="number of frames to sample per clip. Should be in descending order")

    parser.add_argument("--nmb_temporal_samples_val", default='5',type='strlist',
                        help="list of number of crops (example: [2, 6])")

    parser.add_argument("--view_lst_per_clip_val", default='0,1,2',type='strlist',
                        help="0=left_crp,1=center,2=right_crop")

    parser.add_argument('--stride', type=int, default=8,
                        help='Stride')

    parser.add_argument('--train_clip_sampling_strategy', type=int, default=1,
                        help='1: Consecutively selecting, 2: Deterministic Chunking, 3: Stochastic Chunking, 4: Uniformly Selecting')

    parser.add_argument('--train_clip_consecutive_start', type=int, default='3',
                        help='1: Start with beginning,  2: Start with Center (Make selection with center), ,  3: Random Start')

    # parser.add_argument('--train_clip_consecutive_start_distance', type=int, default=[[5,5]],
    parser.add_argument('--train_clip_consecutive_start_distance', type='strlist', default='30,-1|30,-1',
                        help='Select clip starting points with a min,max this distance')

    parser.add_argument('--train_clip_consecutive_start_distance_type', type=int, default=2,
                        help='1: Fixed with Min, 2: Randomly with Minimum Distance from others , 3: Randomly with Minimum and Max Distance from others')

    parser.add_argument('--eval_clip_sampling_strategy', type=int, default=1,
                        help='1: consecutive selecting with stride')

    parser.add_argument('--eval_clip_consecutive_start', type=int, default=3,
                        help='1: Start with beginning,  2: Start with Center (Make selection with center), ,  3: unformly selecting')


    #########################
    #### model parameters ###
    #########################
    parser.add_argument("--vid_base_arch", default="r2plus1d_18", type=str,
                        help="video architecture", choices=['s3d', 'r2plus1d_18'])
    parser.add_argument("--pool_type", default="s3d_new_avg", type=str,
                        help="Type of pooling layer")

    parser.add_argument("--norm_base_encoder_feat", default='False', type='bool', help="Normalize video encoder feats")

    #########################
    #### Prot parameters ###
    #########################
    parser.add_argument('--pre_protype_normalize', type='bool', default='True',
                        help='Normalize feature before protatypes (means also after head')

    parser.add_argument("--nmb_prototypes", default=1000, type=int,
                        help="number of prototypes")
    parser.add_argument("--hidden_mlp", default=1024 * 2, type=int,
                        help="projection head hidden units")
    parser.add_argument("--emedding_dim", default=128, type=int,
                        help="final layer dimension in projection head")

    parser.add_argument('--use_protreg',type='bool', default='True',
                        help='Use prototype regulazation....')

    parser.add_argument('--use_precomp_prot', type='bool',default='True',
                        help='Use precomputed prototypes....')

    parser.add_argument('--use_precomp_prot_path', type=str,default='prot/fls/prototypes-128d-1000c_50000.npy',
                        help='Use precomputed prototypes....')

    #########################
    #### linear classifier parameters ###
    #########################
    parser.add_argument('--evaluation_type', type=int, default=EvalType.PreTrain.value,
                        help='Model evaluation type,0=pretraining,1=linear classifier, 2= fine-tuning.')

    parser.add_argument('--use_lincls_use_bn', type='bool', default='True',
                        help='use linear classifier BN')

    parser.add_argument('--use_lincls_l2_norm', type='bool', default='False',
                        help='use linear classifier L2 norm')
    parser.add_argument('--lincls_drop', default=0, type=float, help='Linear classifier dropout usage')
    parser.add_argument('--backbone_ratio', default=10, type=float, help='Backbone network ratio')
    parser.add_argument('--lr_milestones', default=[60,  80], type=int, help='Learning rate scheduler...')
    parser.add_argument('--use_wicc_val_scheduler', default='False', type='bool', help='Learning rate scheduler...')

    #########################
    #### low ds experiments ###
    #########################
    parser.add_argument("--percent_per_cls", default=1, type=float,
                        help="number of percentages per class")
    #########################
    #### optim parameters ###
    #########################

    parser.add_argument("--epochs", default=200, type=int,
                        help="number of total epochs to run")

    parser.add_argument("--curr_final_epochs", default=200, type=int,
                        help="number of total epochs to run")

    parser.add_argument("--resume_training", default='False', type='bool',
                        help="Resume training")

    parser.add_argument("--resume_model_path", default='', type=str,
                        help="Resume training")

    parser.add_argument("--batch_size", default=2, type=int,
                        help="batch size per gpu, i.e. how many unique instances per gpu")

    parser.add_argument("--batch_size_val", default=1, type=int,
                        help="batch size per gpu, i.e. how many unique instances per gpu")

    parser.add_argument("--base_lr", default=0.075*2 , type=float, help="base learning rate")
    parser.add_argument("--final_lr", type=float, default=0.0048 , help="final learning rate")
    parser.add_argument("--fix_lr", type='bool', default=False , help="fix the learning rate for the givenfinal learning")

    parser.add_argument("--freeze_prototypes_epoch", default=1, type=int,
                        help="freeze the prototypes during this many iterations from the start")
    parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
    parser.add_argument("--start_warmup", default=0.3, type=float,
                        help="initial warmup learning rate")
    parser.add_argument("--epoch_queue_starts", type=int, default=10,
                        help="from this epoch, we start using a queue")
    parser.add_argument("--queue_length", type=int, default=0,
                        help="length of the queue (0 for no queue)")
    parser.add_argument("--sync_bn", type=str, default="pytorch", help="pytorch,apex")
    parser.add_argument('--use_LARC', type='bool', default='True',help='use LARC optim')
    parser.add_argument('--use_adam', type='bool', default='False',help='use adam optim')
    parser.add_argument('--use_WarmupMultiStepLR', type='bool', default='False',
                        help='use_WarmupMultiStepLR')
    parser.add_argument('--use_vicc_opt', type='bool', default='False',
                        help='')

    #########################
    #### dist parameters ###
    #########################
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up distributed
                        training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--world_size", default=1, type=int, help="""
                        number of processes: it is set automatically and
                        should not be passed as argument""")
    parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                        it is set automatically and should not be passed as argument""")
    parser.add_argument("--local_rank", default=0, type=int,
                        help="this argument is not used and should be ignored")

    parser.add_argument("--master_node", help="Master node")
    parser.add_argument("--master_port", help="Master port")
    parser.add_argument("--bash", type='bool',default='false', help='is calling from bash')
    parser.add_argument("--resume", default='False', type='bool', help="are we resuming mode")
    parser.add_argument("--distributed", default='False', type='bool', help="We will set in incode")

    parser.add_argument("--workers", default=6, type=int,
                        help="number of data loading workers")
    parser.add_argument("--checkpoint_freq", type=int, default=10,
                        help="Save the model periodically")

    parser.add_argument("--dump_path", type=str, default="logs",
                        help="experiment dump path for checkpoints and log")

    parser.add_argument("--root_log_path", type=str, default=iout.get_log_root(),
                        help="experiment dump path for checkpoints and log")

    parser.add_argument("--pre_train_model", type=str, default="",
                        help="pre-trained-model")

    parser.add_argument("--seed", type=int, default=31, help="seed")
    parser.add_argument("--per_iter_show", type=int, default=100, help="per_iter_show")
    parser.add_argument("--pretrained", type=bool, default=False, help="")
    # Mixed precision training parameters
    parser.add_argument('--use_fp16',default=False, type='bool',
                        help='Use apex for mixed precision training')

    parser.add_argument('--apex-opt-level', default='O1', type=str,
                        help='For apex mixed precision training'
                             'O0 for FP32 training, O1 for mixed precision training.'
                             'For further detail, see https://github.com/NVIDIA/apex/tree/master/examples/imagenet'
                        )

    parser.add_argument('--start_epoch', default=0, type=int, help='initial start epoch')

    parser.add_argument('--momentum_warmp', default=0.9, type=float, metavar='M',
                        help='momentum_warmp')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')


    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--lr-warmup-epochs', default=10, type=int, help='number of warmup epochs')

    parser.add_argument("--train_dir", type=str, default="train",
                        help="directory")
    parser.add_argument("--val_dir", type=str, default="val",
                        help="directory")

    parser.add_argument("--feats_save_dir", type=str, default="feats",
                        help="directory")

    args = parser.parse_args()
    args.data_root_dir = os.path.join(
        iout.get_ds_root(args.ds_name, args.split_idx, args.vid_base_arch, args.evaluation_type), 'videos')
    args.data_cache_dir = os.path.join(
        iout.get_ds_root(args.ds_name, args.split_idx, args.vid_base_arch, args.evaluation_type),
        'ds_cache')

    adjust_params(args)
    return args



def adjust_params(args):
    if 'r2plus1d_18' in args.vid_base_arch:
        args.train_crop_size=112
        args.train_crop_range=[128,160]
        args.val_crop_range = [112,112]
        assert 'r2plus1d_18' in args.pool_type

    elif 's3d' in args.vid_base_arch:
        args.train_crop_size=128
        args.train_crop_range=[136,190]
        args.val_crop_size = 128
        args.val_crop_range = [128,128]
        if args.vid_base_arch =='s3d_old':
            args.pool_type='s3d_old'
        assert 's3d' in args.pool_type

    if args.video_channels==ChannelModal.rgb_flow.value and args.evaluation_type==EvalType.PreTrain.value:
        if args.video_modal_merge_algo in [ModalMergeAlgo.MixRgbFlow.value,ModalMergeAlgo.PriorRgbFlow.value]:
            assert args.video_modal_merge ==ModalMergeType.ProcSeperate.value

        if args.video_modal_merge_algo ==ModalMergeAlgo.AvgRgbFlow.value:
            assert args.video_modal_merge ==ModalMergeType.Avg.value
    if args.evaluation_type in EvalType.KNN_FullOpt.value or \
            args.evaluation_type ==EvalType.LinCls.value or \
            args.evaluation_type==EvalType.FT.value: #we are checking if we are loading correct model .
        if  args.evaluation_type in EvalType.KNN_FullOpt.value:
            assert args.num_frames_lst==args.num_frames_lst_val
        assert args.vid_base_arch == args.pre_train_model.split('/')[-5]


    # ModalMergeType,ModalMergeAlgo