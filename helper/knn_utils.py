from temp.feats import feats_util as fut
import pickle
from collections import defaultdict
import torch

import numpy as np

def pythorc_eval(suppert_set,query_set,kmax):
    knn_batch=torch.norm(suppert_set.unsqueeze(1)- query_set.unsqueeze(0),dim=2,p=2).permute(1,0).topk(kmax, largest=False)
    return knn_batch[1].cpu().detach().numpy().tolist()
    # print('kNN dist: {}, index: {}'.format(knn.values, knn.indices))

def cosine_distance_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


def pytorch_cosine_eval(suppert_set,query_set,kmax):
    output=cosine_distance_torch(suppert_set,query_set)
    knn_batch = output.permute(1, 0).topk(kmax, largest=False)
    return knn_batch[1].cpu().detach().numpy().tolist()
    # print('kNN dist: {}, index: {}'.format(knn.values, knn.indices))

def eval_features(train_feats,train_vidnames,train_vid_cls,val_feats,val_vidnames,val_vid_cls):
    train_feats=torch.from_numpy(train_feats).cuda()

    recall_dict = defaultdict(list)
    retrieval_dict = {}
    natch_size=5
    kmax=50
    val_id_range=list(range(len(val_feats)))
    print('Using: pythorc_cosine_eval: ')
    for i in range(0,len(val_feats),natch_size):
        # feat = np.expand_dims(val_feats[i], 0)
        feat_lst = val_feats[i:i+natch_size]
        feat_id_lst = val_id_range[i:i+natch_size]
        full_neighbor_indices_batch = pythorc_cosine_eval(train_feats, torch.from_numpy(feat_lst).cuda(), kmax)


        for bidx in range(len(feat_id_lst)):
            idx=feat_id_lst[bidx]
            vid_idx = val_vidnames[idx]
            val_vid_label = val_vid_cls[vid_idx]

            retrieval_dict[vid_idx] = {
                'label': val_vid_label,
                'recal_acc': {
                    '1': 0, '5': 0, '10': 0, '20': 0, '50': 0
                },
                'neighbors': {
                    '1': [], '5': [], '10': [], '20': [], '50': []
                }
            }
            full_neighbor_indices=full_neighbor_indices_batch[bidx]
            for recall_treshold in [1, 5, 10, 20, kmax]:
                neighbor_indices = full_neighbor_indices[:recall_treshold]
                neighbor_labels = set([train_vid_cls[train_vidnames[train_vid_index]] for train_vid_index in neighbor_indices])
                recall_value = 100 if val_vid_label in neighbor_labels else 0
                acc_value = len([1 for neigh_label in neighbor_labels if neigh_label == val_vid_label]) / float(
                    len(neighbor_labels))
                retrieval_dict[vid_idx]['recal_acc'][str(recall_treshold)] = acc_value
                retrieval_dict[vid_idx]['neighbors'][str(recall_treshold)] = neighbor_indices
                recall_dict[recall_treshold].append(recall_value)
        if i%1000==0:
            print('Total Val Videos: {0}/{1}'.format(len(retrieval_dict),len(val_feats)))

    for recall_treshold in [1, 5, 10, 20, 50]:
        mean_recall = np.mean(recall_dict[recall_treshold])
        print(f"Recall @ {recall_treshold}: {mean_recall}")
    return retrieval_dict
