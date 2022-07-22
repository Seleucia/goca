from temp.feats import feats_util as fut
import pickle
import helper.knn_utils as kut
from sklearn.manifold import TSNE


def parse_args():
    def str2bool(v):
        v = v.lower()
        if v in ('yes', 'true', 't', '1'):
            return True
        elif v in ('no', 'false', 'f', '0'):
            return False
        raise ValueError('Boolean argument needs to be true or false. '
            'Instead, it is %s.' % v)

    import argparse
    parser = argparse.ArgumentParser(description='Video Retrieval')
    parser.register('type', 'bool', str2bool)

    ### Retrieval params
    parser.add_argument('--save_pkl', default='False', type='bool',
                        help='save pickled feats')
    parser.add_argument('--compute_tsne', default='False', type='bool',
                        help='scompute TSNE, very time consuming')
    parser.add_argument('--avg_feats', default='True', type='bool',
                        help='Average features of video')
    parser.add_argument('--norm_feats', default='True', type='bool',
                        help='L2 normalize features of video')

    ### Dataset params
    parser.add_argument('--ds_name', default='kinetics', type=str,
                        choices=['kinetics', 'ucf101', 'hmdb51'],
                        help='name of dataset')
    parser.add_argument("--root_dir", type=str, default="/home/hcoskun_snapchat_com/hc/dsdata/feats/r2plus1d_18_custom/rgb/06-28_21_48_38/ckp-47/r2plus1d_18_new_avg",
                        help="root dir of dataset")

    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args=parse_args()
    print('Evaluation for: {0}'.format(args.root_dir),' Compute TSNE: ',args.compute_tsne)
    # load_multi_feats(fbase_path, ds_name, data_modality, do_l2_norm, do_vid_avg)
    train_vidnames, train_vid_cls, train_feats= fut.load_multi_feats(fbase_path=args.root_dir,ds_name=args.ds_name,data_modality='train',do_l2_norm=args.norm_feats, do_vid_avg=args.avg_feats)
    val_vidnames, val_vid_cls, val_feats= fut.load_multi_feats(fbase_path=args.root_dir,ds_name=args.ds_name,data_modality='val',do_l2_norm=args.norm_feats, do_vid_avg=args.avg_feats)
    assert len(train_vidnames) ==len(train_vid_cls)
    assert len(train_vidnames) ==len(train_feats)
    assert len(val_vidnames) ==len(val_vid_cls)
    assert len(val_vidnames) ==len(val_feats)
    retrieval_dict=kut.eval_features(train_feats, train_vidnames, train_vid_cls, val_feats, val_vidnames, val_vid_cls)

    print('Writing predictions.')
    with open(args.root_dir + '/retrieval_dict.p', 'wb') as handle:
        pickle.dump(retrieval_dict, handle)

    if args.compute_tsne==True:
        print('TSNE Fitting for Train:',len(train_feats))
        train_feats_embedded = TSNE(n_components=2).fit_transform(train_feats)
        with open(args.root_dir+'/train_feats_embedded.p', 'wb') as handle:
            pickle.dump(train_feats_embedded,handle)

        print('TSNE Fitting for Val:',len(val_feats))
        val_feats_embedded = TSNE(n_components=2).fit_transform(val_feats)
        with open(args.root_dir+'/val_feats_embedded.p', 'wb') as handle:
            pickle.dump(val_feats_embedded,handle)
