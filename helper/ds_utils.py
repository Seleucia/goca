import helper.io_utilts as iout
import os,json,glob,math,random
import numpy as np
def getNsamples(ds_name,mode):
    if ds_name == 'vggsound':
        if mode == 'train':
            num_data_samples = 170752
        else:
            num_data_samples = 14032
    elif ds_name == 'kinetics':
        mname = iout.getMName()
        if ds_name == 'kinetics':
            if 'wscskn' == mname:
                if mode == 'train':
                    num_data_samples = 240976
                    # num_data_samples = 1000
                else:
                    num_data_samples = 35251
            elif 'ssv-vm' == mname:
                if mode == 'train':
                    num_data_samples = 240976
                    # num_data_samples = 50000
                else:
                    num_data_samples = 35251
            elif 'ssv-vm-8v100-1' == mname:
                if mode == 'train':
                    num_data_samples = 240976
                    # num_data_samples = 50000
                else:
                    num_data_samples = 35251
            elif 'ssv-vm-8v100-on-hcc-1' == mname:
                if mode == 'train':
                    num_data_samples = 240976
                    # num_data_samples = 50000
                else:
                    num_data_samples = 35251
            elif 'ssv-vm-8v100-2-from-img-1' == mname:
                if mode == 'train':
                    num_data_samples = 240976
                    # num_data_samples = 1000
                else:
                    num_data_samples = 35251
            elif 'ssv-vm-8v100-3' == mname:
                if mode == 'train':
                    num_data_samples = 230976
                    # num_data_samples = 50000
                else:
                    num_data_samples = 35251
            elif 'hcc-g1' in mname:
                if mode == 'train':
                    num_data_samples = 240976
                    # num_data_samples = 50000
                else:
                    num_data_samples = 35251
            elif 'cskn' == mname:
                if mode == 'train':
                    num_data_samples = 240976
                else:
                    num_data_samples = 18968
    elif ds_name == 'kinetics_sound':
        if mode == 'train':
            num_data_samples = 22408
        else:
            num_data_samples = 22408
    elif ds_name == 'ave':
        if mode == 'train':
            num_data_samples = 3328
        else:
            num_data_samples = 3328
    elif ds_name == 'ucf101':
        if mode == 'train':
            num_data_samples = 50000
        else:
            num_data_samples = 50000
    elif ds_name == 'hmdb':
        if mode == 'train':
            num_data_samples = 50000
            # num_data_samples = 1000
        else:
            num_data_samples = 50000
            # num_data_samples = 1000
    return num_data_samples



def load_class_to_idx(data_cache_dir,data_root_dir):
    # bpath='/media/hc/Data/ds/kinetics/full_set/data'
    class_path= os.path.join(data_cache_dir,'',"class_to_idx.json")
    os.makedirs(data_cache_dir, exist_ok=True)
    if os.path.exists(class_path):
        class_to_idx=json.load(open(class_path))
        # print('Loading class list from: {0}'.format(class_path))
    else:
        os.makedirs(data_cache_dir,exist_ok=True)
        classes = list(sorted(os.listdir(os.path.join(data_root_dir,'train'))))
        classes = [os.path.basename(i) for i in classes]
        assert len(classes)>20#just to be sure that we loaded classes...
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        json.dump( class_to_idx, open(class_path, 'w' ) )
        print('Dumping class list to: {0}, Nclass: {1}'.format(class_path,len(classes)))
    # print('#'*100,class_to_idx)
    return class_to_idx


def get_clips_uniformly(adjusted_stride,seq_len, clip_ln, nclip):
        if clip_ln * adjusted_stride > seq_len:
            adjusted_stride = int(seq_len / clip_ln)

        clip_len_with_stride = (clip_ln - 1) * adjusted_stride
        seq_lst = list(range(seq_len))
        clip_rest = len(seq_lst) - clip_len_with_stride
        step = math.ceil(clip_rest / (nclip - 1))
        start_idx = 0
        start_lst = [start_idx]
        is_exceeded = False
        for ii in range(nclip - 1):
            start_idx = start_lst[-1] + step
            if start_idx + clip_len_with_stride >= seq_len:
                if is_exceeded == True:
                    start_idx = random.choice(range(clip_rest))
                else:
                    start_idx = seq_len - clip_len_with_stride - 1
                is_exceeded = True
            start_lst.append(start_idx)
            # print(start_idx)

        start_lst = sorted(start_lst)
        return start_lst, adjusted_stride

def select_frameidx_withstride(adjusted_stride, start_idx, clip_ln):
    return [start_idx + ii * adjusted_stride for ii in range(clip_ln)]

def get_donot_select_lst(curren_lst, tmp_index_fll_lst, minn):
    forbidden_area = np.asarray(sum([list(range(c - minn, c + minn)) for c in curren_lst], []))
    forbidden_area = forbidden_area[forbidden_area >= 0]
    forbidden_area = forbidden_area[forbidden_area < len(tmp_index_fll_lst)]
    selection_proc = np.ones(len(tmp_index_fll_lst), dtype=bool)
    selection_proc[forbidden_area] = False
    left_tmp_index_fll_lst = tmp_index_fll_lst[selection_proc]
    return left_tmp_index_fll_lst.tolist()

def select_with_mindistance(adjusted_stride, clip_len, curren_lst, full_list_ln, minn, maxx):
    # print('+'*10,clip_len,full_list_ln,curren_lst,adjusted_stride,minn,len(curren_lst))
    clip_len_with_stride = (clip_len - 1) * adjusted_stride
    tmp_index_fll_lst = np.asarray(list(range(full_list_ln - clip_len_with_stride)))
    if len(curren_lst) < 1 or minn <= 0:
        selected = random.sample(tmp_index_fll_lst.tolist(), 1)[0]
        return selected, minn
    left_tmp_index_fll_lst = []
    while len(left_tmp_index_fll_lst) == 0 and minn > 0:
        left_tmp_index_fll_lst = get_donot_select_lst(curren_lst, tmp_index_fll_lst, minn)
        if len(left_tmp_index_fll_lst) == 0:
            minn = minn - 1
            if minn == 0:
                left_tmp_index_fll_lst = tmp_index_fll_lst.tolist()
    selected = random.sample(left_tmp_index_fll_lst, 1)[0]
    return selected, minn