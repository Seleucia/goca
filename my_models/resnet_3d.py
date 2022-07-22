import torch.nn as nn
from pytorchvideo.models.resnet import create_resnet


class VideoSlowResNet(nn.Module):
    def __init__(self, pool_type):

        super(VideoSlowResNet, self).__init__()
        self.model = create_resnet(head=None)
        self.adp_pool = pool_type

    def forward(self, inp):
        x=self.model(inp)
        x=self.adp_pool(x)
        x = x.flatten(1)
        return x
