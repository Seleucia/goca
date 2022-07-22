from torch import nn
import torchvision
from my_models.s3dg import S3D
from my_models.my_r2plus1d_18 import VideoResNet as VideoResNetR2plus1d


def random_weight_init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                    nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)


def get_video_dim(vid_base_arch='r2plus1d_18'):
    if vid_base_arch == 's3d_old':
        return 1024
    elif vid_base_arch =='r2plus1d_18':
        return 2048
    else:
        assert("Video Architecture is not supported")

class MultiPrototypes(nn.Module):
    def __init__(self, output_dim, nmb_prototypes):
        super(MultiPrototypes, self).__init__()
        self.nmb_heads = len(nmb_prototypes)
        for i, k in enumerate(nmb_prototypes):
            self.add_module("prototypes" + str(i), nn.Linear(output_dim, k, bias=False))

    def forward(self, x):
        out = []
        for i in range(self.nmb_heads):
            out.append(getattr(self, "prototypes" + str(i))(x))
        return out



def get_pooltype(pool_type):
    if pool_type=='r2plus1d_18_old_max':
        return nn.AdaptiveMaxPool3d((1, 1, 1))
    elif pool_type=='r2plus1d_18_old_avg':
        return nn.AdaptiveAvgPool3d((1, 1, 1))
    elif pool_type=='r2plus1d_18_new_avg':
        return nn.AdaptiveAvgPool3d((1, 2, 2))
    elif pool_type=='r2plus1d_18_new_max':
        return nn.AdaptiveMaxPool3d((1, 2, 2))
    elif pool_type=='r2plus1d_18_sel_avg':
        return nn.AvgPool3d((2, 2, 2), stride=(2, 2, 2))
    elif pool_type=='r2plus1d_18_sel_max':
        return nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
    elif pool_type=='s3d_old_max':
        return nn.AdaptiveMaxPool3d((1, 1, 1))
    elif pool_type=='s3d_old_avg':
        return nn.AdaptiveAvgPool3d((1, 1, 1))
    elif pool_type=='s3d_new_avg':
        return nn.AdaptiveAvgPool3d((1, 2, 2))
    elif pool_type=='s3d_new_max':
        return nn.AdaptiveMaxPool3d((1, 2, 2))
    elif pool_type=='s3d_sel_avg':
        return nn.AvgPool3d((2, 2, 2), stride=(2, 2, 2))
    elif pool_type=='s3d_sel_max':
        return nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
    elif pool_type=='s3d_old':
        return nn.MaxPool3d((1, 1, 1))
    elif pool_type=='slow_avg':
        return nn.AdaptiveAvgPool3d(1)
    elif pool_type=='slow_max':
        return nn.AdaptiveMaxPool3d(1)
    elif pool_type=='slow_feats_avg':
        return nn.AdaptiveAvgPool3d((1, 2, 2))
    elif pool_type=='slow_feats_max':
        return nn.AdaptiveMaxPool3d((1, 2, 2))
    else:
        raise Exception('Select proper pooling')

def get_video_feature_extractor(vid_base_arch='r2plus1d_18',pool_type=None, pretrained=False, rank=0):
    avg_pool=get_pooltype(pool_type)
    if vid_base_arch=='r2plus1d_18':
        model = torchvision.models.video.resnet.r2plus1d_18(pretrained=pretrained, progress=True)
    elif vid_base_arch == 'r2plus1d_18_custom':
        assert 'r2plus1d_18' in pool_type
        model = VideoResNetR2plus1d(pool_type=avg_pool)
    elif vid_base_arch=='s3d':
        model =  S3D()

    if not pretrained:
        if rank == 0:
            print("Randomy initializing models")
        random_weight_init(model)

    return model
