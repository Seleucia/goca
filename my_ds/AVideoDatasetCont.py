# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import time,random,math
import copy
import av
import numpy as np
import ffmpeg
from joblib import Parallel, delayed
from helper import ds_utils as dsut
import multiprocessing
from multiprocessing import Manager
import os
import pickle
import torch
import torch.utils.data
import glob
from helper.my_enums import EvalType, ChannelModal
from  torchvision.transforms import functional as F


import my_ds.my_decoder as mdec



# Enable multi thread decoding.
ENABLE_MULTI_THREAD_DECODE = True

# Decoding backend, options include `pyav` or `torchvision`
DECODING_BACKEND = 'pyav'
check_flow_consistenncy=5 #maximum flow difference...
# check_flow_consistenncy=0 #maximum flow difference...

def valid_video(vid_idx, vid_path):
    try:
        probe = ffmpeg.probe(vid_path)
        video_stream = next((
            stream for stream in probe['streams'] if stream['codec_type'] == 'video'),
            None
        )

        if check_flow_consistenncy>0:
            probe_flow = ffmpeg.probe(vid_path.replace('/videos/','/videos_flow/'))
            video_stream_flow = next((
                stream for stream in probe_flow['streams'] if stream['codec_type'] == 'video'),
                None
            )

        if video_stream and float(video_stream['duration']) > 1.1:
            print(f"{vid_idx}: True", end='\r', flush=True)
            str_fps=video_stream['avg_frame_rate']
            fps=float(str_fps.split('/')[0])/float(str_fps.split('/')[1])
            nframes=int(video_stream['nb_frames'])

            if check_flow_consistenncy > 0:
                nflow_frames=int(video_stream_flow['nb_frames'])
                if abs(nflow_frames - nframes)> check_flow_consistenncy:
                    return False, 0, 0
                nframes=min(nflow_frames,nframes)
            return True, nframes,fps
        else:
            print(f"{vid_idx}: False (duration short/ no audio)", flush=True)
            return False,0,0
    except  Exception as e:
        print(vid_path,e)
        print(f"{vid_idx}: False", flush=True)
        return False,0,0

def valid_pyav_video(vid_idx, vid_path):
    try:
        # print('Reading: ',vid_path)
        use_multi_thread = True
        pts_unit = 'sec'
        start_pts, end_pts = 0, math.inf
        video_frames=[]
        with av.open(vid_path, metadata_errors="ignore") as container:
            if container.streams.video:
                if use_multi_thread == True:
                    container.streams.video[0].thread_type = "AUTO"
                video_frames =  mdec._read_from_stream(
                    container,
                    start_pts,
                    end_pts,
                    pts_unit,
                    container.streams.video[0],
                    {"video": 0},
                )
                fps = float(container.streams.video[0].average_rate)
                nframe = len(video_frames)
        if check_flow_consistenncy > 0:
            with av.open(vid_path.replace('/videos/','/videos_flow/'), metadata_errors="ignore") as container:
                if container.streams.video:
                    if use_multi_thread == True:
                        container.streams.video[0].thread_type = "AUTO"
                    video_flow_frames = mdec._read_from_stream(
                        container,
                        start_pts,
                        end_pts,
                        pts_unit,
                        container.streams.video[0],
                        {"video": 0},
                    )
                    fps = float(container.streams.video[0].average_rate)
                    nframe_flow = len(video_flow_frames)
            if abs(nframe_flow - nframe) > check_flow_consistenncy:
                return False, 0, 0
            nframe=min(nframe,nframe_flow)

        if video_frames and len(video_frames) > 30:
            return True,nframe ,fps
        else:
            print(f"{vid_idx}: False (duration short/ no audio)", flush=True)
            return False,0,0
    except  Exception as e:
        print(vid_path,e)
        print(f"{vid_idx}: False", flush=True)
        return False,0,0




def filter_videos(vid_paths,ds_name):
    if ds_name=='kinetics':
        vide_checker=valid_video
    else:
        vide_checker=valid_pyav_video
    all_indices = Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(vide_checker)(vid_idx, vid_paths[vid_idx]) for vid_idx in range(len(vid_paths)))
    valid_indices_withcnt = [[i,val[1],val[2]] for i, val in enumerate(all_indices) if val[0]]
    return valid_indices_withcnt

class AVideoDataset(torch.utils.data.Dataset):
    """
    Audio-video loader. Construct the video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(
            self,
            evaluation_type=0,
            ds_name='kinetics',
            data_root_dir='',
            mode='train',
            num_spatial_crops=3,
            num_ensemble_views=10,
            data_cache_dir='pkl_file_location',
            fold=1,
            nmb_temporal_samples=[2],
            num_frames_lst=[18],
            subsample_with=1,
            clip_sampling_strategy=1,
            clip_consecutive_start=3,
            clip_consecutive_start_distance=1,
            clip_consecutive_start_distance_type=1,
            view_lst_per_clip_val=[1],
            stride=60,
            video_channels='rgb',
            video_channels_val='rgb',
            percent_per_cls=1,
            rank=0
    ):
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for '{}'".format(mode, ds_name)

        # Set up dataset hyper-params
        self.num_data_samples=dsut.getNsamples(ds_name,mode)
        self.evaluation_type = evaluation_type
        self.rank = rank
        self.percent_per_cls = percent_per_cls
        self.ds_name = ds_name
        self.video_channels = video_channels
        self.video_channels_val = video_channels_val
        self.stride=stride
        self.mode = mode
        self.clip_consecutive_start_distance=clip_consecutive_start_distance
        self.clip_consecutive_start=clip_consecutive_start
        self.clip_sampling_strategy=clip_sampling_strategy
        self.clip_consecutive_start_distance_type=clip_consecutive_start_distance_type
        self.view_lst_per_clip_val=view_lst_per_clip_val

        self.subsample_with = subsample_with
        self.num_ensemble_views = num_ensemble_views
        self.num_spatial_crops = num_spatial_crops
        self.data_location = os.path.join(data_root_dir, mode)
        self.data_cache_dir = data_cache_dir
        self._video_meta = {}
        self.fold = fold  # ucf101 and hmdb51
        self.num_frames_lst = num_frames_lst
        self.nmb_temporal_samples = nmb_temporal_samples
        self.clip_samples =  list(zip(self.nmb_temporal_samples, self.num_frames_lst))

        # print('self.clip_samples',self.clip_samples)

        total_numberof_clip, required_min_video_len=sum([inff[0] for inff in self.clip_samples]),max([inff[1] for inff in self.clip_samples])

        for clip_info_idx, clip_info in enumerate(self.clip_samples):
            required_min_video_len+=self.clip_consecutive_start_distance[clip_info_idx][0]*clip_info[0]
        self.total_numberof_clip=total_numberof_clip
        self.required_min_video_len=required_min_video_len# Just relax the selection process, if we random selecting it can take time....

        self.class_to_idx = dsut.load_class_to_idx(data_cache_dir=data_cache_dir,data_root_dir=data_root_dir)

        self.manager = Manager()
        if self.rank ==0:
            print(f"Constructing {self.ds_name} {self.mode}...")
        self._construct_loader()

    def random_chunked_lst(self, frm_idx, num):
        # print(seq_ln,num)
        # if self.eval_model==True:
        #     sel_lst = np.linspace(0, seq_ln - 1, num, dtype=int)
        #     return sel_lst
        seq_ln = len(frm_idx)
        num = num
        tmp_avg = seq_ln / float(num)
        out = []
        last = 0.0
        if tmp_avg < 1:
            required_n_doubles = int(num - tmp_avg * num)
            doubles = [random.sample(range(seq_ln), 1)[0] for _ in range(required_n_doubles)]
            avg = 1
        else:
            avg = tmp_avg
            doubles = []
        # print(doubles,required_n_doubles)
        idx = 0
        # print(doubles,seq_ln,num)
        while last < seq_ln:
            # out.append(random.choice(seq[int(last):int(last + avg)]))
            sel_lst = frm_idx[int(last):int(last + avg)]
            # out.append(random.choice(sel_lst))
            if tmp_avg < 1:
                out.append(sel_lst[0])
                if idx in doubles:
                    nelements = doubles.count(idx)
                    for nn in range(nelements):
                        out.append(sel_lst[0])
            else:
                out.append(random.choice(sel_lst))
            idx += 1
            last += avg
            # print(last)
        # print(len(out),num)
        while len(out) > num:
            del out[random.choice(range(len(out)))]
        np.asarray(out)
        return list(sorted(out))

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        # Get list of paths
        os.makedirs(self.data_cache_dir, exist_ok=True)
        path_to_file = os.path.join(
            self.data_cache_dir, f"{self.ds_name}_{self.mode}.txt"
        )
        # print('**'*10,path_to_file)
        if not os.path.exists(path_to_file):
            files = list(sorted(glob.glob(os.path.join(self.data_location, '*', '*'))))
            # print('L'*50,len(files),self.data_location)
            with open(path_to_file, 'w') as f:
                for item in files:
                    f.write("%s\n" % item)

        # Get list of indices and labels
        self._path_to_videos = []
        self._labels = []
        # self._spatial_temporal_idx = []
        self._vid_indices = []
        if self.rank == 0:
            print('_path_to_videos:',path_to_file)
        with open(path_to_file, "r") as f:
            for clip_idx, path in enumerate(f.read().splitlines()):
                # for idx in range(self._num_clips):
                self._path_to_videos.append(
                    os.path.join(self.data_location, path)
                )
                class_name = path.split('/')[-2]
                label = self.class_to_idx[class_name]
                # print(label,class_name,path)
                self._labels.append(int(label))
                # self._spatial_temporal_idx.append(idx)
                self._vid_indices.append(clip_idx)
                # self._video_meta[clip_idx * self._num_clips + idx] = {}
                self._video_meta[clip_idx] = {}
        assert (
                len(self._path_to_videos) > 0
        ), "Failed to load {} split {} from {}".format(
            self.ds_name, self._split_idx, path_to_file
        )
        if self.rank == 0:
            print(
                "Constructing {} dataloader (size: {}) from {}".format(
                    self.ds_name, len(self._path_to_videos), path_to_file
                )
            )

        # Create / Load valid indices (has audio)
        if self.ds_name in ['kinetics', 'ucf101','hmdb', 'vggsound', 'ave', 'kinetics_sound']:
            vid_valid_file = f'{self.data_cache_dir}/{self.ds_name}_{self.mode}_valid_withcntfps.pkl'
            # vid_valid_file = f'{self.data_cache_dir}/{self.ds_name}_{self.mode}_valid_withcnt.pkl'
            if os.path.exists(vid_valid_file):
                with open(vid_valid_file, 'rb') as handle:
                    self.valid_indices = pickle.load(handle)
            else:
                self.valid_indices = filter_videos(self._path_to_videos,self.ds_name)
                with open(vid_valid_file, 'wb') as handle:
                    pickle.dump(
                        self.valid_indices,
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL
                    )
            if self.num_data_samples is not None:
                self.valid_indices = self.valid_indices[:self.num_data_samples]
            if self.percent_per_cls < 1 and EvalType.PreTrain.value==self.evaluation_type:
                self.valid_indices = self.update_valid_indices()
            if self.rank == 0:
                print(f"{self.ds_name}_{self.mode} >>>  Total number of videos: {len(self._path_to_videos)}, Valid videos: {len(self.valid_indices)}, NumOfClass {len(self.class_to_idx)}",
                  flush=True)

        # print(self.valid_indices)
        # print(self._path_to_videos)

        self._path_to_videos = self.manager.list(self._path_to_videos)
        self.valid_indices = self.manager.list(self.valid_indices)

    def update_valid_indices(self):
        cls_dic = {lbl: [] for lbl in self.class_to_idx}
        for idx, ind in enumerate(self.valid_indices):
            cls_dic[self._path_to_videos[ind[0]].split('/')[-2]].append(idx)
        sel_samples = []
        for cls in cls_dic:
            cls_valid_ind = cls_dic[cls]

            nsample_per_class_curr = max(int(len(cls_valid_ind) * self.percent_per_cls), 1)
            sel_samples_per_cls = random.sample(cls_valid_ind, nsample_per_class_curr)
            sel_samples.extend(sel_samples_per_cls)
        sel_samples = list(sorted(sel_samples))

        return [self.valid_indices[sidx] for sidx in sel_samples]

    def get_clips_idx(self,nframes):
        # print('*'*100,self.mode, self.evaluation_type,self.clip_sampling_strategy ,self.clip_consecutive_start)
        frames_idx_lst = list(range(nframes))
        full_list_ln=len(frames_idx_lst)
        idx_cnt=0
        dic_frames_idx={}
        selected_spints=[]
        for clip_info_idx,clip_info in enumerate(self.clip_samples):
            if self.mode in ['val', 'test'] or self.evaluation_type in EvalType.KNN_FullOpt.value:
                # print('*'*100,self.mode,self.evaluation_type)
                if clip_info[1]>=full_list_ln:
                    for _ in range(clip_info[0]):
                        frames_idx = self.random_chunked_lst(frames_idx_lst, clip_info[1])
                        dic_frames_idx[idx_cnt] = frames_idx
                        idx_cnt += 1
                else:
                    start_lst, adjusted_stride = dsut.get_clips_uniformly(self.stride, full_list_ln, clip_ln=clip_info[1], nclip=clip_info[0])
                    for start_idx in start_lst:
                        frames_idx=dsut.select_frameidx_withstride(adjusted_stride, start_idx, clip_ln=clip_info[1])
                        # if self.rank==0:
                        #     print(self.mode,start_lst,len(frames_idx),len(set(frames_idx)),clip_info[1],full_list_ln,clip_info[0],adjusted_stride)

                        dic_frames_idx[idx_cnt] = frames_idx
                        idx_cnt += 1
            else:
                for _ in range(clip_info[0]):
                    if self.clip_sampling_strategy == 4:
                        frames_idx = np.linspace(0, full_list_ln - 1, clip_info[1], dtype=int)
                        frames_idx=np.asarray(frames_idx_lst)[frames_idx]
                        frames_idx=frames_idx.tolist()
                    elif self.clip_sampling_strategy == 3:
                        frames_idx = self.random_chunked_lst(frames_idx_lst, clip_info[1])
                    elif self.clip_sampling_strategy == 1:
                        if self.clip_consecutive_start==1: #Start from beginning
                            frames_idx = frames_idx_lst[0:clip_info[1]]
                        elif self.clip_consecutive_start==3: #Random starting
                            min_max_dist=self.clip_consecutive_start_distance[clip_info_idx]
                            minn=min_max_dist[0]
                            if idx_cnt==0:
                                adjusted_minn=minn
                                if clip_info[1] * self.stride > full_list_ln:
                                    adjusted_stride = int(full_list_ln / clip_info[1] )
                                else:
                                    adjusted_stride=self.stride
                            spoint, adjusted_minn = dsut.select_with_mindistance(adjusted_stride, clip_info[1], selected_spints, full_list_ln,adjusted_minn, maxx=-1)
                            frames_idx = dsut.select_frameidx_withstride(adjusted_stride, spoint, clip_ln=clip_info[1])
                            selected_spints.append(spoint)
                    # print(frames_idx)
                    dic_frames_idx[idx_cnt]=frames_idx
                    idx_cnt+=1
        uniq_idxs = list(set(sum(dic_frames_idx.values(), [])))
        # print('Done.....')
        return uniq_idxs,dic_frames_idx

    def __getitem__(self, index_capped):

        path_index,nframes,fps = self.valid_indices[index_capped]

        start_pts, end_pts = 0, math.inf
        # full_frame_lst_flow=mdec.read_av_frames(self._path_to_videos[path_index].replace('/videos/','/videos_flow/'), start_pts, end_pts, 'sec', read_audio=False, use_multi_thread=True)
        # full_frame_lst_rgb = mdec.read_av_frames(self._path_to_videos[path_index], start_pts, end_pts, 'sec',read_audio=False, use_multi_thread=True)
        # # nframes = min(len(full_frame_lst_flow),len(full_frame_lst_rgb))
        # if self.video_channels == ChannelModal.flow.value:
        #     full_frame_lst=full_frame_lst_flow
        # else:
        #     full_frame_lst = full_frame_lst_rgb

        if self.video_channels == ChannelModal.flow.value:
            full_frame_lst=mdec.read_av_frames(self._path_to_videos[path_index].replace('/videos/','/videos_flow/'), start_pts, end_pts, 'sec', read_audio=False, use_multi_thread=True)
        else:
            full_frame_lst = mdec.read_av_frames(self._path_to_videos[path_index], start_pts, end_pts, 'sec',read_audio=False, use_multi_thread=True)

        # nframes=len(full_frame_lst)
        if self.video_channels==ChannelModal.rgb_flow.value:
            full_frame_lst_flow = mdec.read_av_frames(self._path_to_videos[path_index].replace('/videos/','/videos_flow/'), start_pts, end_pts, 'sec',
                                                 read_audio=False, use_multi_thread=True)
            # full_frame_lst = mdec.read_av_frames(
            #     self._path_to_videos[path_index].replace('/videos/', '/videos_flow/'), start_pts, end_pts, 'sec',
            #     read_audio=False, use_multi_thread=True)

            # print(nframes, len(full_frame_lst_flow),
            #       self._path_to_videos[path_index].replace('/videos/', '/videos_flow/'))
            # if abs(nframes-len(full_frame_lst_flow))>3:
            #     print(nframes,len(full_frame_lst_flow),self._path_to_videos[path_index].replace('/videos/','/videos_flow/'))
            nframes=min(len(full_frame_lst_flow),nframes)

        uniq_idxs, dic_frames_idx = self.get_clips_idx(nframes)
        sel_frame_lst = mdec.decode_videos(full_frame_lst, sel_idx=uniq_idxs)
        if self.video_channels==ChannelModal.rgb_flow.value:
            sel_frame_lst_flow = mdec.decode_videos(full_frame_lst_flow, sel_idx=uniq_idxs)

        idx_cnt = 0; V=[];L=[];P=[]
        for clip_info in self.clip_samples:
            for _ in range(clip_info[0]):
                frames_idx = dic_frames_idx[idx_cnt]
                # clip_frames = torch.as_tensor(np.stack([sel_frame_lst[fidx] for fidx in frames_idx]))
                clip_frames = torch.stack([sel_frame_lst[fidx] for fidx in frames_idx])
                # print('summ_flow:', torch.sum(clip_frames), self._path_to_videos[path_index],self.video_channels)
                if self.video_channels == ChannelModal.rgb_flow.value:
                    clip_frames_flow = torch.stack([sel_frame_lst_flow[fidx] for fidx in frames_idx])

                if self.mode in ['val','test'] or self.evaluation_type in EvalType.KNN_FullOpt.value:
                    for spaitial_idx in self.view_lst_per_clip_val:
                        transformed_frames=self.transform(clip_frames,spaitial_idx)
                        if self.video_channels ==ChannelModal.rgb_flow.value:
                            transformed_frames_flow = self.transform(clip_frames_flow, spaitial_idx)
                            V.append([transformed_frames,transformed_frames_flow])
                            # if spaitial_idx == 0 and idx_cnt==9:
                            #     print('summ_flow:', torch.sum(transformed_frames_flow), self._path_to_videos[path_index],
                            #           self.video_channels, idx_cnt)
                        else:
                            V.append(transformed_frames)
                            # if spaitial_idx == 0 and idx_cnt==9:
                            #     print('summ_flow:', torch.sum(transformed_frames), self._path_to_videos[path_index],
                            #           self.video_channels, idx_cnt,spaitial_idx)
                        L.append(self._labels[path_index])
                        P.append(path_index)
                else:
                    cropped_frames = self.transform(clip_frames)
                    if self.video_channels == ChannelModal.rgb_flow.value:
                        cropped_frames_flow  = self.transform(clip_frames_flow )
                        V.append([cropped_frames,cropped_frames_flow])
                    else:
                        V.append(cropped_frames)
                    L.append(self._labels[path_index])
                    P.append(path_index)

                idx_cnt += 1
        # print(path_index,self._labels[path_index],self._path_to_videos[path_index])
        return  V, L, index_capped, P

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self.valid_indices)