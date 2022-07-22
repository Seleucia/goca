import torch
import torch.nn as nn
import my_ds.crop_utils as cut
import numpy as np
import numbers
from torchvision.transforms import functional as F
from collections.abc import Sequence
from torch import Tensor
import math
from torchvision.transforms.functional import InterpolationMode, _interpolation_modes_from_int




class ConvertTHWCtoTCHW(nn.Module):
    def forward(self, vid: torch.Tensor) -> torch.Tensor:
        return vid.permute(0, 3, 1, 2)



class ConvertTCHWtoCTHW(nn.Module):

    def forward(self, vid: torch.Tensor) -> torch.Tensor:
        return vid.permute(1, 0, 2, 3)


class ReverseForTimeForTCHW(nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p
    def forward(self, vid: torch.Tensor) -> torch.Tensor:
        # print('shape:',vid.shape)
        if torch.rand(1) < self.p:
            return reversed(vid)
        else:
            return vid
        # return vid.permute(1, 0, 2, 3)

class TorchResize(object):

    def __init__(self,resize_size,mode="bilinear"):
        self.resize_size=resize_size
        self.mode=mode

    def __call__(self, vid: torch.Tensor) -> torch.Tensor:
        # print('resizing: ',vid.shape,self.resize_size)
        return torch.nn.functional.interpolate(vid, size=self.resize_size, mode=self.mode,align_corners=True)

class TorchNorm(object):

    def __init__(self,MEAN,STD):
        self.MEAN=torch.tensor(MEAN)
        self.STD=torch.tensor(STD)

    def __call__(self, vid: torch.Tensor) -> torch.Tensor:
        # frames = vid.float()
        # vid = vid / 255.0
        # vid.sub_(self.MEAN).div_(self.STD)

        # print("vid.shape:",vid.shape)
        vid = vid - self.MEAN
        vid = vid / self.STD
        return vid

class RandomGrayscale(torch.nn.Module):
    """Randomly convert image to grayscale with a probability of p (default 0.1).
    If the image is torch Tensor, it is expected
    to have [..., 3, H, W] shape, where ... means an arbitrary number of leading dimensions

    Args:
        p (float): probability that image should be converted to grayscale.

    Returns:
        PIL Image or Tensor: Grayscale version of the input image with probability p and unchanged
        with probability (1-p).
        - If input image is 1 channel: grayscale version is 1 channel
        - If input image is 3 channel: grayscale version is 3 channel with r == g == b

    """

    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be converted to grayscale.

        Returns:
            PIL Image or Tensor: Randomly grayscaled image.
        """
        # print(img.shape)
        num_output_channels = F._get_image_num_channels(img)
        # num_output_channels =3
        if torch.rand(1) < self.p:
            return F.rgb_to_grayscale(img, num_output_channels=num_output_channels)
        return img

def _check_sequence_input(x, name, req_sizes):
    msg = req_sizes[0] if len(req_sizes) < 2 else " or ".join([str(s) for s in req_sizes])
    if not isinstance(x, Sequence):
        raise TypeError("{} should be a sequence of length {}.".format(name, msg))
    if len(x) not in req_sizes:
        raise ValueError("{} should be sequence of length {}.".format(name, msg))


def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size

class GaussianBlur(torch.nn.Module):
    """Blurs image with randomly chosen Gaussian blur.
    If the image is torch Tensor, it is expected
    to have [..., C, H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        kernel_size (int or sequence): Size of the Gaussian kernel.
        sigma (float or tuple of float (min, max)): Standard deviation to be used for
            creating kernel to perform blurring. If float, sigma is fixed. If it is tuple
            of float (min, max), sigma is chosen uniformly at random to lie in the
            given range.

    Returns:
        PIL Image or Tensor: Gaussian blurred version of the input image.

    """

    def __init__(self, kernel_size, sigma=(0.1, 2.0)):
        super().__init__()
        self.kernel_size = _setup_size(kernel_size, "Kernel size should be a tuple/list of two integers")
        for ks in self.kernel_size:
            if ks <= 0 or ks % 2 == 0:
                raise ValueError("Kernel size value should be an odd and positive number.")

        if isinstance(sigma, numbers.Number):
            if sigma <= 0:
                raise ValueError("If sigma is a single number, it must be positive.")
            sigma = (sigma, sigma)
        elif isinstance(sigma, Sequence) and len(sigma) == 2:
            if not 0. < sigma[0] <= sigma[1]:
                raise ValueError("sigma values should be positive and of the form (min, max).")
        else:
            raise ValueError("sigma should be a single number or a list/tuple with length 2.")

        self.sigma = sigma

    @staticmethod
    def get_params(sigma_min: float, sigma_max: float) -> float:
        """Choose sigma for random gaussian blurring.

        Args:
            sigma_min (float): Minimum standard deviation that can be chosen for blurring kernel.
            sigma_max (float): Maximum standard deviation that can be chosen for blurring kernel.

        Returns:
            float: Standard deviation to be passed to calculate kernel for gaussian blurring.
        """
        return torch.empty(1).uniform_(sigma_min, sigma_max).item()


    def forward(self, img: Tensor) -> Tensor:
        """
        Args:
            img (PIL Image or Tensor): image to be blurred.

        Returns:
            PIL Image or Tensor: Gaussian blurred image
        """
        sigma = self.get_params(self.sigma[0], self.sigma[1])
        return F.gaussian_blur(img, self.kernel_size, [sigma, sigma])


    def __repr__(self):
        s = '(kernel_size={}, '.format(self.kernel_size)
        s += 'sigma={})'.format(self.sigma)
        return self.__class__.__name__ + s



    def _setup_size(size, error_msg):
        if isinstance(size, numbers.Number):
            return int(size), int(size)

        if isinstance(size, Sequence) and len(size) == 1:
            return size[0], size[0]

        if len(size) != 2:
            raise ValueError(error_msg)

        return size


    def _check_sequence_input(x, name, req_sizes):
        msg = req_sizes[0] if len(req_sizes) < 2 else " or ".join([str(s) for s in req_sizes])
        if not isinstance(x, Sequence):
            raise TypeError("{} should be a sequence of length {}.".format(name, msg))
        if len(x) not in req_sizes:
            raise ValueError("{} should be sequence of length {}.".format(name, msg))


    def _setup_angle(x, name, req_sizes=(2, )):
        if isinstance(x, numbers.Number):
            if x < 0:
                raise ValueError("If {} is a single number, it must be positive.".format(name))
            x = [-x, x]
        else:
            _check_sequence_input(x, name, req_sizes)

        return [float(d) for d in x]

class RandomColorJitt(object):
    def __init__(self,p,img_brightness,img_contrast,img_saturation):
        self.img_brightness=img_brightness
        self.img_contrast=img_contrast
        self.img_saturation=img_saturation
        self.p=p
        jitter = []
        if img_brightness != 0:
            jitter.append("brightness")
        if img_contrast != 0:
            jitter.append("contrast")
        if img_saturation != 0:
            jitter.append("saturation")
        self.jitter=jitter

    def blend(self,images1, images2, alpha):
        """
        Blend two images with a given weight alpha.
        Args:
            images1 (tensor): the first images to be blended, the dimension is
                `num frames` x `channel` x `height` x `width`.
            images2 (tensor): the second images to be blended, the dimension is
                `num frames` x `channel` x `height` x `width`.
            alpha (float): the blending weight.
        Returns:
            (tensor): blended images, the dimension is
                `num frames` x `channel` x `height` x `width`.
        """
        return images1 * alpha + images2 * (1 - alpha)


    def grayscale(self,images):
        """
        Get the grayscale for the input images. The channels of images should be
        in order BGR.
        Args:
            images (tensor): the input images for getting grayscale. Dimension is
                `num frames` x `channel` x `height` x `width`.
        Returns:
            img_gray (tensor): blended images, the dimension is
                `num frames` x `channel` x `height` x `width`.
        """
        # R -> 0.299, G -> 0.587, B -> 0.114.
        img_gray = torch.zeros_like(images)
        gray_channel = (
                0.299 * images[:, 2] + 0.587 * images[:, 1] + 0.114 * images[:, 0]
        )
        img_gray[:, 0] = gray_channel
        img_gray[:, 1] = gray_channel
        img_gray[:, 2] = gray_channel
        return img_gray

    def brightness_jitter(self,var, images):
        """
        Perfrom brightness jittering on the input images. The channels of images
        should be in order BGR.
        Args:
            var (float): jitter ratio for brightness.
            images (tensor): images to perform color jitter. Dimension is
                `num frames` x `channel` x `height` x `width`.
        Returns:
            images (tensor): the jittered images, the dimension is
                `num frames` x `channel` x `height` x `width`.
        """
        alpha = 1.0 + np.random.uniform(-var, var)

        img_bright = torch.zeros(images.shape)
        images = self.blend(images, img_bright, alpha)
        return images

    def contrast_jitter(self,var, images):
        """
        Perfrom contrast jittering on the input images. The channels of images
        should be in order BGR.
        Args:
            var (float): jitter ratio for contrast.
            images (tensor): images to perform color jitter. Dimension is
                `num frames` x `channel` x `height` x `width`.
        Returns:
            images (tensor): the jittered images, the dimension is
                `num frames` x `channel` x `height` x `width`.
        """
        alpha = 1.0 + np.random.uniform(-var, var)

        img_gray = self.grayscale(images)
        img_gray[:] = torch.mean(img_gray, dim=(1, 2, 3), keepdim=True)
        images = self.blend(images, img_gray, alpha)
        return images

    def saturation_jitter(self,var, images):
        """
        Perfrom saturation jittering on the input images. The channels of images
        should be in order BGR.
        Args:
            var (float): jitter ratio for saturation.
            images (tensor): images to perform color jitter. Dimension is
                `num frames` x `channel` x `height` x `width`.
        Returns:
            images (tensor): the jittered images, the dimension is
                `num frames` x `channel` x `height` x `width`.
        """
        alpha = 1.0 + np.random.uniform(-var, var)
        img_gray = self.grayscale(images)
        images = self.blend(images, img_gray, alpha)

        return images

    def __call__(self, images):
        if torch.rand(1) > self.p:
            return images
        if len(self.jitter) > 0:
            order = np.random.permutation(np.arange(len(self.jitter)))
            for idx in range(0, len(self.jitter)):
                if self.jitter[order[idx]] == "brightness":
                    images = self.brightness_jitter(self.img_brightness, images)
                elif self.jitter[order[idx]] == "contrast":
                    images = self.contrast_jitter(self.img_contrast, images)
                elif self.jitter[order[idx]] == "saturation":
                    images = self.saturation_jitter(self.img_saturation, images)
        return images

class TorchRandCrop(object):
    """Convert tensor from (B, H, W, C) to (B, C, H, W)
    """

    def __init__(self,   spatial_idx=-1,
            min_scale=256,
            max_scale=320,
            crop_size=224):
        self.spatial_idx = spatial_idx
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.crop_size = crop_size

    def spatial_sampling(self, frames):
        """
        Perform spatial sampling on the given video frames. If spatial_idx is
        -1, perform random scale, random crop, and random flip on the given
        frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
        with the given spatial_idx.
        Args:
            frames (tensor): frames of images sampled from the video. The
                dimension is `num frames` x `height` x `width` x `channel`.
            spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
                or 2, perform left, center, right crop if width is larger than
                height, and perform top, center, buttom crop if height is larger
                than width.
            min_scale (int): the minimal size of scaling.
            max_scale (int): the maximal size of scaling.
            crop_size (int): the size of height and width used to crop the
                frames.
        Returns:
            frames (tensor): spatially sampled frames.
        """
        assert self.spatial_idx in [-1, 0, 1, 2, 3, 4, 5]
        if self.spatial_idx == -1:
            frames, _ = cut.random_short_side_scale_jitter(
                frames, self.min_scale, self.max_scale
            )
            # print(frames[0].shape,self.crop_size)
            frames, _ = cut.random_crop(frames, self.crop_size)
            frames, _ = cut.horizontal_flip(0.5, frames)
        else:
            frames, _ = cut.random_short_side_scale_jitter(frames, self.min_scale, self.max_scale)
            idx_mapping = {0: 0, 1: 1, 2: 2, 3: 0, 4: 1, 5: 2}
            frames, _ = cut.uniform_crop(frames, self.crop_size, idx_mapping[self.spatial_idx])
            if self.spatial_idx in [3, 4, 5]:
                frames, _ = cut.horizontal_flip(1, frames)
        return frames

    def __call__(self, vid: torch.Tensor) -> torch.Tensor:
        vid=self.spatial_sampling(vid)
        return vid


def get_random_size(min_size, max_size,curr_height, curr_width,fix_with_min=False):

    if fix_with_min==True:
        size=min_size
    else:
        size = int(round(np.random.uniform(min_size, max_size)))

    height = curr_height
    width = curr_width
    if (width <= height and width == size) or (
            height <= width and height == size
    ):
        return (height, width)
    new_width = size
    new_height = size
    if width < height:
        new_height = int(math.floor((float(height) / width) * size))
    else:
        new_width = int(math.floor((float(width) / height) * size))
    # print('*' * 100, images.shape)
    return (new_height, new_width)

class MyRandomResizeAndCrop(object):
    def __init__(self,crop_size,min_size, max_size):
        self.crop_size=crop_size
        self.min_size=min_size
        self.max_size=max_size

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        curr_width=frames.shape[-1]
        curr_height=frames.shape[-2]
        size=get_random_size(self.min_size, self.max_size,curr_height, curr_width)
        frames=F.resize(frames, size, InterpolationMode.BILINEAR)
        frames, _ = cut.random_crop(frames, self.crop_size)
        return frames

class MyFixResizeAndLeftCrop(object):
    def __init__(self,crop_size,min_size, max_size):
        self.crop_size=crop_size
        self.min_size=min_size
        self.max_size=max_size

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        curr_width=frames.shape[-1]
        curr_height=frames.shape[-2]
        size = get_random_size(self.min_size, self.max_size, curr_height, curr_width,fix_with_min=True)
        frames=F.resize(frames, size, InterpolationMode.BILINEAR)
        frames, _ = cut.uniform_crop(frames, self.crop_size, 0)
        # frames, _ = cut.random_crop(frames, self.crop_size)
        return frames

class MyFixResizeAndCenterCrop(object):
    def __init__(self,crop_size,min_size, max_size):
        self.crop_size=crop_size
        self.min_size=min_size
        self.max_size=max_size

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        curr_width=frames.shape[-1]
        curr_height=frames.shape[-2]
        size = get_random_size(self.min_size, self.max_size, curr_height, curr_width,fix_with_min=True)
        frames=F.resize(frames, size, InterpolationMode.BILINEAR)
        frames, _ = cut.uniform_crop(frames, self.crop_size, 1)
        return frames

class MyFixResizeAndRightCrop(object):
    def __init__(self,crop_size,min_size, max_size):
        self.crop_size=crop_size
        self.min_size=min_size
        self.max_size=max_size

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        curr_width=frames.shape[-1]
        curr_height=frames.shape[-2]
        size = get_random_size(self.min_size, self.max_size, curr_height, curr_width,fix_with_min=True)
        frames=F.resize(frames, size, InterpolationMode.BILINEAR)
        frames, _ = cut.uniform_crop(frames, self.crop_size, 2)
        return frames

