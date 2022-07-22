from my_ds.AVideoDatasetCont import AVideoDataset
import torch
# from my_ds import presets
from helper.my_enums import EvalType
from my_ds import presets_pt as presets
from my_ds.clip_sampler import DistributedSampler


def get_dataset_withmodality(args,modality,eval_mode=False):
    return_lst = []
    if 'train' in modality:
        # build data
        train_dataset = AVideoDataset(
            evaluation_type=args.evaluation_type,
            ds_name=args.ds_name,
            data_root_dir=args.data_root_dir,
            mode='train',
            data_cache_dir=args.data_cache_dir,
            num_frames_lst= args.num_frames_lst,
            nmb_temporal_samples= args.nmb_temporal_samples,
            clip_sampling_strategy=args.train_clip_sampling_strategy,
            clip_consecutive_start=args.train_clip_consecutive_start,
            clip_consecutive_start_distance=args.train_clip_consecutive_start_distance,
            clip_consecutive_start_distance_type=args.train_clip_consecutive_start_distance_type,
            subsample_with=args.subsample_with,
            view_lst_per_clip_val=args.view_lst_per_clip_val,
            stride=args.stride,
            video_channels=args.video_channels,
            video_channels_val=args.video_channels_val,
            percent_per_cls=args.percent_per_cls,
            rank=args.rank
        )
        if args.evaluation_type in EvalType.KNN_FullOpt.value:
            train_dataset.transform = presets.VideoClassificationPresetEval(args)
        else:
            train_dataset.transform = presets.VideoClassificationPresetTrain(args)


        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            sampler=DistributedSampler(train_dataset, shuffle=True) if args.distributed else None,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
        )
        return_lst.append(train_loader)
    if 'val' in modality :
        # build data
        # build data
        val_dataset = AVideoDataset(
            evaluation_type=args.evaluation_type,
            ds_name=args.ds_name,
            data_root_dir=args.data_root_dir,
            mode= 'test' if args.ds_name=='hmdb' else 'val',
            data_cache_dir=args.data_cache_dir,
            num_frames_lst=args.num_frames_lst_val,
            nmb_temporal_samples=args.nmb_temporal_samples_val,
            clip_sampling_strategy=args.eval_clip_sampling_strategy,
            clip_consecutive_start=args.eval_clip_consecutive_start,
            clip_consecutive_start_distance=args.train_clip_consecutive_start_distance,
            clip_consecutive_start_distance_type=args.train_clip_consecutive_start_distance_type,
            subsample_with=args.subsample_with,
            view_lst_per_clip_val=args.view_lst_per_clip_val,
            stride=args.stride,
            video_channels=args.video_channels,
            video_channels_val=args.video_channels_val,
            rank=args.rank
        )
        val_dataset.transform =  presets.VideoClassificationPresetEval(args)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            sampler=DistributedSampler(val_dataset, shuffle=False) if args.distributed else None,
            batch_size=args.batch_size_val,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
        )
        return_lst.append(val_loader)

    return return_lst

