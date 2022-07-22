import av,re,os,time,glob
import torch

from multiprocessing.pool import ThreadPool


import ffmpeg
import numpy as np
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings
import gc
import random

# PyAV has some reference cycles
_CALLED_TIMES = 0
_GC_COLLECTION_INTERVAL = 10

def _read_from_stream(
    container: "av.container.Container",
    start_offset: float,
    end_offset: float,
    pts_unit: str,
    stream: "av.stream.Stream",
    stream_name: Dict[str, Optional[Union[int, Tuple[int, ...], List[int]]]],
) -> List["av.frame.Frame"]:
    global _CALLED_TIMES, _GC_COLLECTION_INTERVAL
    _CALLED_TIMES += 1
    if _CALLED_TIMES % _GC_COLLECTION_INTERVAL == _GC_COLLECTION_INTERVAL - 1:
        gc.collect()

    if pts_unit == "sec":
        start_offset = int(math.floor(start_offset * (1 / stream.time_base)))
        if end_offset != float("inf"):
            end_offset = int(math.ceil(end_offset * (1 / stream.time_base)))
    else:
        warnings.warn(
            "The pts_unit 'pts' gives wrong results and will be removed in a "
            + "follow-up version. Please use pts_unit 'sec'."
        )

    frames = {}
    should_buffer = True
    max_buffer_size = 5
    if stream.type == "video":
        # DivX-style packed B-frames can have out-of-order pts (2 frames in a single pkt)
        # so need to buffer some extra frames to sort everything
        # properly
        extradata = stream.codec_context.extradata
        # overly complicated way of finding if `divx_packed` is set, following
        # https://github.com/FFmpeg/FFmpeg/commit/d5a21172283572af587b3d939eba0091484d3263
        if extradata and b"DivX" in extradata:
            # can't use regex directly because of some weird characters sometimes...
            pos = extradata.find(b"DivX")
            d = extradata[pos:]
            o = re.search(br"DivX(\d+)Build(\d+)(\w)", d)
            if o is None:
                o = re.search(br"DivX(\d+)b(\d+)(\w)", d)
            if o is not None:
                should_buffer = o.group(3) == b"p"
    seek_offset = start_offset
    # some files don't seek to the right location, so better be safe here
    seek_offset = max(seek_offset - 1, 0)
    if should_buffer:
        # FIXME this is kind of a hack, but we will jump to the previous keyframe
        # so this will be safe
        seek_offset = max(seek_offset - max_buffer_size, 0)
    try:
        # TODO check if stream needs to always be the video stream here or not
        container.seek(seek_offset, any_frame=False, backward=True, stream=stream)
    except av.AVError:
        # TODO add some warnings in this case
        # print("Corrupted file?", container.name)
        return []
    buffer_count = 0
    try:
        for _idx, frame in enumerate(container.decode(**stream_name)):
            frames[frame.pts] = frame
            if frame.pts >= end_offset:
                if should_buffer and buffer_count < max_buffer_size:
                    buffer_count += 1
                    continue
                break
    except av.AVError:
        # TODO add a warning
        pass
    # ensure that the results are sorted wrt the pts
    result = [
        frames[i] for i in sorted(frames) if start_offset <= frames[i].pts <= end_offset
    ]
    if len(frames) > 0 and start_offset > 0 and start_offset not in frames:
        # if there is no frame that exactly matches the pts of start_offset
        # add the last frame smaller than start_offset, to guarantee that
        # we will have all the necessary data. This is most useful for audio
        preceding_frames = [i for i in frames if i < start_offset]
        if len(preceding_frames) > 0:
            first_frame_pts = max(preceding_frames)
            result.insert(0, frames[first_frame_pts])
    return result

def read_av_frames(filename,start_pts,end_pts,pts_unit,read_audio=False,use_multi_thread=True):
    info = {}
    video_frames = []
    audio_frames = []
    try:
        with av.open(filename, metadata_errors="ignore") as container:
            if container.streams.video:
                if use_multi_thread==True:
                    container.streams.video[0].thread_type = "AUTO"
                video_frames = _read_from_stream(
                    container,
                    start_pts,
                    end_pts,
                    pts_unit,
                    container.streams.video[0],
                    {"video": 0},
                )
                video_fps = container.streams.video[0].average_rate
                # guard against potentially corrupted files
                if video_fps is not None:
                    info["video_fps"] = float(video_fps)
            if read_audio==True:
                if container.streams.audio:
                    if use_multi_thread == True:
                        container.streams.audio[0].thread_type = "AUTO"
                    audio_frames = _read_from_stream(
                        container,
                        start_pts,
                        end_pts,
                        pts_unit,
                        container.streams.audio[0],
                        {"audio": 0},
                    )
                    info["audio_fps"] = container.streams.audio[0].rate

    except av.AVError:
        # TODO raise a warning?
        pass
    # print(dir(container.streams.video[0]))
    # probe = ffmpeg.probe(filename)
    # video_stream = next((
    #     stream for stream in probe['streams'] if stream['codec_type'] == 'video'),
    #     None
    # )
    # print('+'*100,len(video_frames),video_stream['nb_frames'],max(sel_idx),filename)
    return video_frames
    # if len(sel_idx)>0:
    #     video_frames = np.asarray(video_frames)
    #     # print(len(sel_idx),len(video_frames),filename)
    #     assert  len(sel_idx) <=len(video_frames)
    #     video_frames=video_frames[sel_idx]
    #     zipped_video_frames=zip(video_frames, sel_idx)
    #     video_frames={frame[1]: frame[0].to_rgb().to_ndarray() for frame in zipped_video_frames}
    #     # video_frames={frame[1]: frame[0].to_image() for frame in zipped_video_frames}
    #     return video_frames
    #     # print('Done..')

    # vframes_list=[]

    # vframes_list_mp=map(cnvt, video_frames)
    # vframes_list_mp=list(vframes_list_mp)
    # vframes_list = [cnvt(frame) for frame in video_frames]
    # vframes_list = [frame.to_rgb().to_ndarray() for frame in video_frames]
    # vframes_list = [frame.to_rgb() for frame in video_frames]
    # vframes_list = [frame.to_rgb().to_ndarray() for frame in video_frames[:int(len(video_frames)/2)]]
    # if read_audio == True:
    #     aframes_list = [frame.to_ndarray() for frame in audio_frames]
    #     return vframes_list,aframes_list
    # return vframes_list
def dec_single_frame(frame):
    nd_frame=torch.as_tensor(frame[0].to_rgb().to_ndarray())
    return nd_frame, frame[1]

def multi_thr_decode(nsample, zipped_video_frames):
    pool = ThreadPool(min(nsample,100))
    results = pool.map(dec_single_frame, zipped_video_frames)
    pool.close()
    pool.join()
    results={id:frame for frame, id in results}
    return results

def decode_videos(video_frames,sel_idx):
    video_frames = np.asarray(video_frames)
    # print(len(sel_idx),len(video_frames),filename)
    assert len(sel_idx) <= len(video_frames)
    if np.max(sel_idx)>=len(video_frames):
        print(len(video_frames),sel_idx)
    video_frames = video_frames[sel_idx]
    zipped_video_frames = zip(video_frames, sel_idx)
    video_frames = {frame[1]: torch.as_tensor(frame[0].to_rgb().to_ndarray()) for frame in zipped_video_frames}
    # video_frames = {frame[1]: torch.as_tensor(frame[0]) for frame in zipped_video_frames}
    #
    # print(video_frames[0].to_rgb().to_ndarray().shape)
    # rng_tens= torch.as_tensor(video_frames[0].to_rgb().to_ndarray())
    # video_frames = {frame: rng_tens for frame in sel_idx}

    # video_frames=multi_thr_decode(len(sel_idx),zip(video_frames, sel_idx))
    # video_frames={frame[1]: frame[0].to_image() for frame in zipped_video_frames}
    return video_frames

if __name__ == '__main__':
    filename='/media/hc/Data/ds/ucf101/320/split_1/videos/train/GolfSwing/v_GolfSwing_g22_c02.avi'

    probe = ffmpeg.probe(filename)
    use_multi_thread=True
    pts_unit='sec'
    start_pts, end_pts = 0, math.inf
    with av.open(filename, metadata_errors="ignore") as container:
        if container.streams.video:
            if use_multi_thread == True:
                container.streams.video[0].thread_type = "AUTO"
            video_frames = _read_from_stream(
                container,
                start_pts,
                end_pts,
                pts_unit,
                container.streams.video[0],
                {"video": 0},
            )
            fps=float(container.streams.video[0].average_rate)
            nframe=len(video_frames)
        print('Done...')