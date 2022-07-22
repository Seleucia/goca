import torch
from helper.my_enums import EvalType
from torchvision.transforms import transforms
from my_ds.transforms import \
    ConvertTHWCtoTCHW, ConvertTCHWtoCTHW,MyRandomResizeAndCrop,\
    MyFixResizeAndCenterCrop,MyFixResizeAndLeftCrop,MyFixResizeAndRightCrop,ReverseForTimeForTCHW



# MEAN=[0.45, 0.45, 0.45]
# STD=[0.225, 0.225, 0.225]
#
MEAN=[0.485, 0.456, 0.406]
STD=[0.229, 0.224, 0.225]


class VideoClassificationPresetTrain:
    def __init__(self,args):
        trans=[]
        trans.append(ConvertTHWCtoTCHW())
        trans.append(MyRandomResizeAndCrop(crop_size=args.train_crop_size, min_size=args.train_crop_range[0], max_size=args.train_crop_range[1]))
        trans.append(transforms.RandomHorizontalFlip(0.5))
        if args.use_timereverse > 0:
            trans.append(ReverseForTimeForTCHW(args.use_timereverse))
        if args.use_colorjitter > 0:
            trans.append(
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=args.use_colorjitter))
        if args.use_grayscale > 0:
            trans.append(transforms.RandomGrayscale(p=args.use_grayscale))
        if args.evaluation_type!=EvalType.LinCls.value:
            if args.use_gaussian > 0:
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=args.train_crop_size // 20 * 2 + 1, sigma=(0.1, 2.0))],p=args.use_gaussian)
        trans.append(transforms.ConvertImageDtype(torch.float32))
        # trans.append(TorchRandCrop(spatial_idx=spatial_idx,min_scale=args.train_crop_range[0], max_scale=args.train_crop_range[1],crop_size=args.train_crop_size)) #There is flip also

        trans.append(transforms.Normalize(MEAN,STD))
        trans.append(ConvertTCHWtoCTHW())

        self.transforms = transforms.Compose(trans)

    def __call__(self, x):
        # print(x.shape)
        # vid = [self.transforms(xx) for xx in x]
        vid=self.transforms(x)
        return vid


class VideoClassificationPresetEval:
    def __init__(self, args):
        #  spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
        #     or 2, perform left, center, right crop if width is larger than
        #     height, and perform top, center, buttom crop if height is larger
        #     than width.

        spatial_idx = 1 #U
        # self.transforms = transforms.Compose([
        #     transforms.ConvertImageDtype(torch.float32),
        #     TorchNorm(MEAN=MEAN, STD=STD),
        #     ConvertBHWCtoBCHW(),
        #     TorchRandCrop(spatial_idx=spatial_idx, min_scale=args.test_crop_range[0],
        #                   max_scale=args.test_crop_range[1],
        #                   crop_size=args.crop_size),
        #     ConvertBCHWtoCBHW()
        # ])
        self.left_transforms = transforms.Compose([
            ConvertTHWCtoTCHW(),
            MyFixResizeAndLeftCrop(crop_size=args.val_crop_size, min_size=args.val_crop_range[0], max_size=args.val_crop_range[1]),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(MEAN, STD),
            ConvertTCHWtoCTHW()
        ])
        self.center_transforms = transforms.Compose([
            ConvertTHWCtoTCHW(),
            MyFixResizeAndCenterCrop(crop_size=args.val_crop_size, min_size=args.val_crop_range[0],
                                   max_size=args.val_crop_range[1]),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(MEAN, STD),
            ConvertTCHWtoCTHW()
        ])
        self.right_transforms = transforms.Compose([
            ConvertTHWCtoTCHW(),
            MyFixResizeAndRightCrop(crop_size=args.val_crop_size, min_size=args.val_crop_range[0],
                                   max_size=args.val_crop_range[1]),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(MEAN, STD),
            ConvertTCHWtoCTHW()
        ])

        self.transforms_lst={}
        self.transforms_lst[0]=self.left_transforms
        self.transforms_lst[1]=self.center_transforms
        self.transforms_lst[2]=self.right_transforms
    def __call__(self, x,spatial_id):
        return self.transforms_lst[spatial_id](x)
