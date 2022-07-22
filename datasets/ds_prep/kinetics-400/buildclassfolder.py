# ------------------------------------------------------------------------------
# Adapted from https://github.com/activitynet/ActivityNet/
# Original licence: Copyright (c) Microsoft, under the MIT License.
# ------------------------------------------------------------------------------
import argparse
import glob
import json
import os,glob
import shutil
import ssl
import subprocess
import uuid
from collections import OrderedDict
import pandas as pd

def construct_video_filename(row, label_to_dir, trim_format='%06d'):
    """Given a dataset row, this function constructs the output filename for a
    given video."""
    # print(trim_format)
    basename = '%s_%s_%s.mp4' % (row['video-id'],
                                 trim_format % row['start-time'],
                                 trim_format % row['end-time'])
    if not isinstance(label_to_dir, dict):
        dirname = label_to_dir
    else:
        dirname = label_to_dir[row['label-name']]
    output_filename = os.path.join(dirname, basename)
    return output_filename


def parse_kinetics_annotations(input_csv, ignore_is_cc=False):
    """Returns a parsed DataFrame.
    arguments:
    ---------
    input_csv: str
        Path to CSV file containing the following columns:
          'YouTube Identifier,Start time,End time,Class label'
    returns:
    -------
    dataset: DataFrame
        Pandas with the following columns:
            'video-id', 'start-time', 'end-time', 'label-name'
    """
    # df = pd.read_csv(input_csv,nrows=50)
    df = pd.read_csv(input_csv)
    if 'youtube_id' in df.columns:
        columns = OrderedDict([('youtube_id', 'video-id'),
                               ('time_start', 'start-time'),
                               ('time_end', 'end-time'),
                               ('label', 'label-name')])
        df.rename(columns=columns, inplace=True)
        if ignore_is_cc:
            df = df.loc[:, df.columns.tolist()[:-1]]
    return df

#
# bdir='/home/hc/ds/minikinetics/data/train/'
# bdir_videos='/home/hc/ds/minikinetics/train/'

source_bdir='/mnt/disks/ds/datasets/kinetics/kinetics-400/flat/val/'
dest_videos='/mnt/disks/ds/datasets/kinetics/kinetics-400/data/videos_orig/val/'


input_csv='/mnt/disks/ds/datasets/kinetics/kinetics-400/files/kinetics400/validate.csv'
input_json='/mnt/disks/ds/datasets/kinetics/kinetics-400/files/kinetics400/validate.json'
# input_json='kinetics400/train.json'



list_of_videos=glob.glob(source_bdir+'*')
print('List of found: {0}'.format(len(list_of_videos)))
f = open(input_json)
jset=json.load(f)


dic_of_videos={('_').join(fl.split('/')[-1].split('_')[:-2]):fl for fl in list_of_videos}

dic_of_videos ={ky:dic_of_videos[ky] for ky in dic_of_videos if ky in jset}

dic_of_videos_lbls={ky:[jset[ky]['annotations']['label'],dic_of_videos[ky]] for ky in dic_of_videos}

for ky in dic_of_videos_lbls:
    src=dic_of_videos_lbls[ky][1]
    fname=src.split('/')[-1]
    lbl_name=dic_of_videos_lbls[ky][0]

    dest_dir=os.path.join(dest_videos,lbl_name.replace(' ','_'))
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    dest_path=os.path.join(dest_dir,fname)
    # dest=os.path.join(bdir,lbl_name.replace(' ','_'),fname)
    if not os.path.exists(dest_path):
        os.rename(src,dest_path)


#
# dataset=parse_kinetics_annotations(input_csv, ignore_is_cc=False)
# grp=dataset.groupby(['label-name'])
# lbl_names=list(grp.indices.keys())
# fnames=[lbl.replace(' ','_') for lbl in lbl_names]
print('Done...',len(dic_of_videos_lbls))
