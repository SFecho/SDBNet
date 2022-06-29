from enum import Enum, auto
from functools import partial
from random import randrange
from typing import Union

# import cv2
import imageio
import torch as t
import torch.nn as nn
from torch import Tensor
from numpy import ndarray
import torch.nn.functional as F
import transplant
from torch.autograd import Variable
from math import ceil, floor, pi, sqrt
import numpy as np
from scipy.io import loadmat
import torchvision as tv
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from torch.nn.modules.utils import _pair, _quadruple

class ImageOption(Enum):
    @staticmethod
    class Formal(Enum):
        HWC = auto()
        BHWC = auto()
        BCHW = auto()
        CHW = auto()

    @staticmethod
    class Type(Enum):
        numpy = auto()
        pytorch = auto()



def convert_otf2psf(otf, size):
    psf = t.fft.ifft2(otf)
    ker = t.zeros(size, dtype=t.complex64, device=otf.device)
    centre = size[-1] // 2  + 1

    ker[:, :, (centre - 1):, (centre - 1):] = psf[:, :, :centre, :centre]
    # print(ker[:, :, (centre - 1):, :(centre - 1)].shape , psf[:, :, :centre, -(centre - 1):].shape)
    ker[:, :, (centre - 1):, :(centre - 1)] = psf[:, :, :centre, -(centre - 1):]
    ker[:, :, : (centre - 1), (centre - 1):] = psf[:, :, -(centre - 1):, :centre]
    ker[:, :, :(centre - 1), :(centre - 1)] = psf[:, :, -(centre - 1):, -(centre - 1):]

    return t.real(ker)

class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.

    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """

    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding

    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd,
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x


def tensor2npy(tensor, dtype=np.float32):
    if tensor.get_device() >= 0:
        npy = tensor.cpu().detach().numpy().astype(dtype)
    else:
        npy = tensor.numpy().astype(dtype)

    del tensor

    return npy

def psnr(test, label, rgb_range=1.):
    sum_psnr = 0

    b, _, _, _ = test.shape

    test_npy = tensor2npy(test, np.float32)
    label_npy = tensor2npy(label, np.float32)

    for i in range(b):
        test_item = test_npy[i].transpose([1, 2, 0])
        label_item = label_npy[i].transpose([1, 2, 0])

        hl, wl, _ = label_item.shape
        ht, wt, _ = test_item.shape

        # if hl != ht or wl != wt:
        #     test_item = resize(test_item, [hl, wl])

        sum_psnr += peak_signal_noise_ratio(label_item, test_item, data_range=rgb_range).item()

    return sum_psnr / b

def ssim(test, label, rgb_range=1.):
    sum_ssim = 0

    b, _, _, _ = test.shape

    test_npy = tensor2npy(test, np.float32)
    label_npy = tensor2npy(label, np.float32)

    for i in range(b):
        test_item = test_npy[i].transpose([1, 2, 0])
        label_item = label_npy[i].transpose([1, 2, 0])

        hl, wl, _ = label_item.shape
        ht, wt, _ = test_item.shape

        # if hl != ht or wl != wt:
        #     test_item = resize(test_item, [hl, wl])

        sum_ssim += structural_similarity(label_item, test_item, multichannel=True, data_range=rgb_range).item()

    return sum_ssim / b

def imread(filename, data_format: Union[ImageOption.Formal]=ImageOption.Formal.HWC, data_type: Union[ImageOption.Type]=ImageOption.Type.numpy):

    img = imageio.imread(filename)
    if img.ndim < 3:
        img = np.expand_dims(img, 2)

    if data_format is ImageOption.Formal.CHW:
        img = np.transpose(img, axes=[1, 2, 0])

    if data_format in [ImageOption.Formal.BCHW, ImageOption.Formal.BHWC]:
        img = np.expand_dims(img, 0)

    if data_type is ImageOption.Type.pytorch:
        img = t.tensor(img)

    return img


# def imwrite(filename, img, data_format='BCHW', rgb_range=255, gray=False, batch_idx=0):
#     if isinstance(img, np.ndarray):
#         if data_format == 'BCHW':
#             img = img[batch_idx].transpose(1, 2, 0) * rgb_range
#             # imwrite(filename, img)
#         elif data_format == 'BHWC':
#             img = img[batch_idx] * rgb_range
#             # img = np.clip(img, 0, rgb_range).astype(np.uint8)
#         else:
#             raise Exception('image tensor format error! Please select "BHWC" or "BCHW"')
#     elif isinstance(img, t.Tensor):
#         img = tensor2npy(img)
#         if data_format == 'BCHW':
#
#             img = img[batch_idx].transpose(1, 2, 0) * rgb_range
#         elif data_format == 'BHWC':
#             img = img[batch_idx] * rgb_range
#         elif data_format == 'CHW' * rgb_range
#         # img = np.clip(img, 0, rgb_range).astype(np.uint8)
#         # imwrite(filename, img)
#
#     else:
#         raise Exception('The type of image tensor error! Please input ndarray or torch tensor')
#
#     img = np.clip(img, 0, rgb_range).astype(np.uint8)
#     if gray:
#         img = img[:, :, 0]
#     imageio.imwrite(filename, img)

def imwrite(
        filename: Union[str],
        img: Union[ndarray, Tensor],
        data_format:Union[ImageOption.Formal]=ImageOption.Formal.BCHW,
        rgb_range=1,
        batch_idx=0):

    if isinstance(img, Tensor):
        img = tensor2npy(img)
    # assert data_format in ['BCHW', 'BHWC', 'CHW', 'HWC']
    if data_format in [ImageOption.Formal.BCHW, ImageOption.Formal.BHWC]:
        img = img[batch_idx]


    if data_format in [ImageOption.Formal.BCHW, ImageOption.Formal.CHW]:
        img = img.transpose(1, 2, 0)
    img = np.clip(img * 255 / rgb_range, 0, 255).astype(np.uint8)

    c = img.shape[-1]
    gray = True if c == 1 else False
    if gray:
        img = np.squeeze(img, axis=-1)

    imageio.imwrite(filename, img)

def rgb2gray(image: object, data_format:Union[ImageOption.Formal]=ImageOption.Formal.BCHW) -> object:
    if len(image.shape) < 3: #or type not in [ImageOption.Formal.BCHW, ImageOption.Formal.BCHW]:
        raise Exception('image error!')
    if data_format is ImageOption.Formal.BCHW:
        b, c, h, w = image.shape
        if c == 3:
            R = image[:, 0, :, :]
            G = image[:, 1, :, :]
            B = image[:, 2, :, :]
            gray = t.unsqueeze(0.299 * R + 0.587 * G + 0.114 * B, dim=1)
        elif c == 1:
            gray = image
        else:
            raise Exception('image error!')
    elif data_format is ImageOption.Formal.CHW:
        c, h, w = image.shape
        if c == 3:
            R = image[0, :, :]
            G = image[1, :, :]
            B = image[2, :, :]
            gray = t.unsqueeze(0.299 * R + 0.587 * G + 0.114 * B, dim=0)
        else:
            gray = image
    else:
        raise Exception('image error!')

    return gray

def image_edge_rmap(image_edge, window_size):
    abs_edge = t.abs(image_edge)
    filter = t.ones([window_size, window_size], dtype=image_edge.dtype, device=image_edge.device)
    Denom = imfilter(abs_edge, filter, filter_format="HW", padding_mode='zeros') + 0.5
    Numer = imfilter(image_edge, filter, filter_format="HW", padding_mode='zeros')
    return t.abs(Numer) / Denom

# def image_edge_rmask(image, window_size, psf_size, threshold_r=None, threshold_s=None):
#     threshold_pxpy()
#     rmap = image_edge_rmap(image_edge, window_size)
#     h, w = image_edge.shape[2:]
#     if threshold_r is None:
#         threshold_r = 2 * psf_size
#
#     if threshold_s is None:
#         threshold_s = 2 * sqrt(h * w) * psf_size
#
#
#     mask = rmap - threshold_r
#     mask[mask < 0] = 0
#     mask[mask >= 0] = 1
#
#     # image_edge =  mask * image_edge
#
#     return mask



def fspecial_gaussian(hsize, sigma, device=t.device("cpu")):
    hsize = [hsize, hsize]
    siz = [(hsize[0] - 1.0) / 2.0, (hsize[1] - 1.0) / 2.0]
    std = sigma
    # [x, y] = np.meshgrid(np.arange(-siz[1], siz[1]+1), np.arange(-siz[0], siz[0]+1))
    [y, x] = t.meshgrid(t.arange(-siz[0], siz[0] + 1, device=device), t.arange(-siz[1], siz[1] + 1, device=device))

    arg = -(x*x + y*y)/(2*std*std)
    h = t.exp(arg)
    h[h < t.finfo(t.float32).eps * h.max()] = 0

    sumh = h.sum()
    if sumh != 0:
        h = h/sumh
    return h

def imfilter(img, filter, padding_mode='circular', filter_mode='corr', img_format="BCHW", filter_format="BHW"):
    _filter = filter.clone()
    if filter_mode not in ['corr', 'conv']:
        raise Exception('')

    if img_format not in ["CHW", "BCHW", "HWC"]:
        raise Exception('')

    if filter_format not in ["HW", "BHW"]:
        raise Exception('')

    if img_format == "CHW":
        img.unsqueeze_(0)
    if img_format == "HWC":
        img = img.permute(2, 0, 1)
        img.unsqueeze_(0)

    if img.ndim != 4:
        raise Exception('')


    b, c, h, w = img.shape
    # print(filter.shape)
    fh, fw = _filter.shape if filter_format == "HW" else _filter.shape[1:]

    if padding_mode != 'valid':
        # pad_up = fh // 2
        # pad_left = fw // 2
        # if fh % 2 == 0:
        #     pad_right = fw // 2 - 1
        #     pad_down = fh // 2 - 1
        # else:
        #     pad_right = fw // 2
        #     pad_down = fh // 2

        pad_up = fh // 2
        pad_left = fw // 2
        if fh % 2 == 0:
            pad_right = fw // 2 - 1
            pad_down = fh // 2 - 1
        else:
            pad_right = fw // 2
            pad_down = fh // 2


        pad = (pad_left, pad_right, pad_up, pad_down)
        # pad = (0, pad_left + pad_right, 0,  pad_up + pad_down)
        # pads = [
        #     (0, pad_left + pad_right, 0, pad_up + pad_down),
        #     (pad_left + pad_right, 0, 0, pad_up + pad_down),
        #     (0, pad_left + pad_right, pad_up + pad_down, 0),
        #     (pad_left + pad_right, 0, pad_up + pad_down, 0),
        #     (pad_left, pad_right, pad_up, pad_down)
        # ]
        # padidx = randrange(0, 4)
        # pad = pads[padidx]
        if padding_mode == 'zeros':
            img = F.pad(img, pad=pad, mode="constant", value=0)
        else:
            img = F.pad(img, pad=pad, mode=padding_mode)


    if filter_mode == 'conv':
        _filter = t.rot90(_filter, 2, dims=(-2, -1))



    if filter_format == "HW":
        _filter.unsqueeze_(0).unsqueeze_(0)
        _filter = _filter.repeat([c, 1, 1, 1])
        out = F.conv2d(img, _filter, groups=c, bias=None, stride=1, padding=0)
    else:
        _filter.unsqueeze_(1)
        out = F.conv2d(img.permute(1, 0, 2, 3), _filter, groups=b, bias=None, stride=1, padding=0)
        out = out.permute(1, 0, 2, 3)

    if format == "CHW":
        out.squeeze_(0)

    if format == "HWC":
        out.squeeze_(0)
        out = out.permute(2, 0, 1)

    return out


def mat2gray(image, min=None, max=None):
    b, c, h, w = image.shape
    if min is None or max is None:
        min, _ = t.min(image.flatten(1), dim=1)
        max, _ = t.max(image.flatten(1), dim=1)
        min = min.view(b, 1, 1, 1)
        max = max.view(b, 1, 1, 1)

    for i in range(b):
        if min[i].item() == max[i].item():
            if image.dtype != t.float32:
                image = image.float()
        else:
            delta = 1 / (max[i] - min[i])
            image[i] = (image[i] - min[i]) * delta#imlincomb(delta, image, -min * delta, 'double')

    return t.clip(image, 0, 1)

# def fspecial_laplacian(alpha):
#     alpha = max([0, min([alpha,1])])
#     h1 = alpha/(alpha+1)
#     h2 = (1-alpha)/(alpha+1)
#     h = [[h1, h2, h1], [h2, -4/(alpha+1), h2], [h1, h2, h1]]
#     h = np.array(h)
#     return h


# def fspecial_log(hsize, sigma):
#     raise(NotImplemented)
#
#
# def fspecial_motion(motion_len, theta):
#     raise(NotImplemented)
#
#
# def fspecial_prewitt():
#     return np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
#
#
# def fspecial_sobel():
#     return np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])


# def fspecial(filter_type, *args, **kwargs):
#     '''
#     python code from:
#     https://github.com/ronaldosena/imagens-medicas-2/blob/40171a6c259edec7827a6693a93955de2bd39e76/Aulas/aula_2_-_uniform_filter/matlab_fspecial.py
#     '''
#     if filter_type == 'average':
#         return fspecial_average(*args, **kwargs)
#     if filter_type == 'disk':
#         return fspecial_disk(*args, **kwargs)
#     if filter_type == 'gaussian':
#         return fspecial_gaussian(*args, **kwargs)
#     if filter_type == 'laplacian':
#         return fspecial_laplacian(*args, **kwargs)
#     if filter_type == 'log':
#         return fspecial_log(*args, **kwargs)
#     if filter_type == 'motion':
#         return fspecial_motion(*args, **kwargs)
#     if filter_type == 'prewitt':
#         return fspecial_prewitt(*args, **kwargs)
#     if filter_type == 'sobel':
#         return fspecial_sobel(*args, **kwargs)


def psf2otf(psf, shape):
    eps = t.finfo(psf.dtype).eps
    psf_shape = t.tensor(psf.shape)
    otf = t.zeros(psf.shape[:-2] + shape).type_as(psf)
    otf[...,:psf.shape[2],:psf.shape[3]].copy_(psf)
    ndim = psf_shape.shape[0]
    # print(ndim)
    for axis, axis_size in enumerate(psf.shape[2:]):
        otf = t.roll(otf, -int(floor(axis_size / 2)), dims=axis + 2)

    otf = t.fft.fft2(otf, dim=(-2, -1))

    otf_real = t.real(otf)
    otf_img = t.imag(otf)

    n_ops = 0
    n_elem = t.prod(psf_shape[2: ])
    for k in range(2, ndim):
        nffts = n_elem / psf_shape[k]
        n_ops = n_ops + psf.shape[k] * t.log2(psf_shape[k].float()) * nffts

    max_val, idx = t.max(t.abs(otf_img.flatten(2)), dim=2, keepdim=True)
    max_abs_val, abs_idx = t.max(t.abs(otf.flatten(2)), dim=2, keepdim=True)
    rate_map = (max_val / max_abs_val) <= n_ops * eps# 2.2204e-16

    for i in range(psf_shape[0].item()):
        for j in range(psf_shape[1].item()):
            if rate_map[i, j] == True:
                otf_img[i, j, :, :] = 0

    otf = t.complex(otf_real, otf_img)

    return otf

def otf2psf(otf, outsize):
    eps = t.finfo(otf.dtype).eps
    insize = t.tensor(otf.shape[2: ], device=otf.device)
    #     outsize = t.tensor(outsize)
    #     otf_shape = t.tensor(otf.shape)

    otf_shape = t.tensor(otf.shape, device=otf.device)
    ndim = otf_shape.shape[0]

    psf = t.fft.ifft2(otf, dim=(-2, -1))

    psf_real = t.real(psf)
    psf_img = t.imag(psf)
    n_ops = 0
    n_elem = t.prod(otf_shape[2:])
    for k in range(2, ndim):
        nffts = n_elem / otf_shape[k]
        n_ops = n_ops + otf.shape[k] * t.log2(otf_shape[k].float()) * nffts

    max_val, idx = t.max(t.abs(psf_img.flatten(2)), dim=2, keepdim=True)
    max_abs_val, abs_idx = t.max(t.abs(psf.flatten(2)), dim=2, keepdim=True)


    rate_map = (max_val / max_abs_val) <= n_ops * eps

    for i in range(otf_shape[0].item()):
        for j in range(otf_shape[1].item()):
            if rate_map[i, j] == True:
                psf_img[i, j, :, :] = 0

    psf = t.complex(psf_real, psf_img)
    for axis, axis_size in enumerate(outsize):
        psf = t.roll(psf, int(floor(axis_size / 2)), dims=axis + 2)

    psf = psf[:, :, 0: outsize[0], 0: outsize[1]]

    return psf

class EstimatePSF(nn.Module):
    def __init__(self, n_iter=10, tol=1e-5, **kwargs):
        super(EstimatePSF, self).__init__()
        self.n_iter = n_iter
        self.tol = tol

    def forward(self, psf_size, blurx, blury, latentx, latenty, gamma=1):
        blur = t.sqrt(blurx ** 2 + blury ** 2)
        latent = t.sqrt(latentx ** 2 + latenty ** 2)
        b, c, h, w = blur.shape
        # img_size = [h, w]
        kernel = t.ones(
            [b, 1, psf_size, psf_size],
            dtype=blur.dtype,
            device=blur.device) * (1 / float(psf_size ** 2))


        blur_f = t.fft.fft2(blur)
        latent_f = t.fft.fft2(latent)
        blur_f = t.conj(latent_f) * blur_f
        latent_ft = t.conj(latent_f) * latent_f
        blur_otf = t.real(otf2psf(blur_f, (psf_size, psf_size)))
            # m = t.conj(latent_f) * latent_f
        kwargs = {
                'image_size': (h, w),
                'psf_size': (psf_size, psf_size),
                'gamma': gamma
        }
        kernel = conjgrad(kernel, latent_ft, blur_otf, self.n_iter, self.tol, **kwargs)


        kernel_flat = kernel.flatten(2)
        max_var, idx = t.max(kernel_flat, dim=2, keepdim=True)


        kernel_flat[kernel_flat < 0.05 * max_var] = 0
        kernel = kernel_flat.view([b, c, psf_size, psf_size])
        kernel[kernel < 0] = 0
        kernel = kernel / t.sum(kernel, dim=(-2, -1), keepdim=True)


        return kernel

def conjgrad(x, latent_ft, b, n_iter, tol, visfunc=None, **kwargs):
    ax1 = compute_ax(x, latent_ft, **kwargs)
    r = b - ax1
    p = r.clone()
    rsold = t.sum(r ** 2, dim=(-2, -1), keepdim=True)

    for i in range(n_iter):
        # bs = x.shape[0]
        Ap = compute_ax(p, latent_ft, **kwargs)
        alpha = rsold / t.sum(p * Ap, dim=(-2, -1), keepdim=True)
        x = x + alpha * p
        if visfunc is not None:
            visfunc(x, i, **kwargs)



        r = r - alpha * Ap
        rsnew = t.sum(r * r, dim=(-2, -1), keepdim=True)

        if t.sqrt(rsnew) < tol:
            break

        p = r + rsnew / rsold * p
        rsold = rsnew

    return x


def compute_ax(psf, latent_ft, image_size=None, psf_size=None, gamma=1, **kwargs):
    xf = psf2otf(psf, image_size)

    tmp = latent_ft * xf
    y = t.real(otf2psf(tmp, psf_size))
    return y + gamma * psf

def convert_psf2otf(ker, size):
    psf = t.zeros(size, device=ker.device)# .cuda()
    # circularly shift
    # print(ker.shape)
    centre = ker.shape[2]//2 + 1
    psf[:, :, :centre, :centre] = ker[:, :, (centre-1):, (centre-1):]
    psf[:, :, :centre, -(centre-1):] = ker[:, :, (centre-1):, :(centre-1)]
    psf[:, :, -(centre-1):, :centre] = ker[:, :, : (centre-1), (centre-1):]
    psf[:, :, -(centre-1):, -(centre-1):] = ker[:, :, :(centre-1), :(centre-1)]
    # compute the otf

    otf = t.fft.fft2(psf)
    return otf



class ImageGrad(nn.Module):
    def __init__(self, in_channels=1, pad_mode='reflect'):
        super(ImageGrad, self).__init__()
        filter_x = t.tensor([[[[0., 0.], [1., -1.]]]]).repeat([in_channels, 1, 1, 1])
        filter_y = t.tensor([[[[0., 1.], [0., -1.]]]]).repeat([in_channels, 1, 1, 1])
        self.pad = partial(F.pad, mode='reflect', pad=(0, 1, 0, 1))
        # self.pad_y = partial(F.pad, mode='circular', pad=())
        self.filter_x = nn.Parameter(
            filter_x, requires_grad=False
        )

        self.filter_y = nn.Parameter(
            filter_y, requires_grad=False
        )
        self.pad_mode = pad_mode
        self.in_channels = in_channels

    def forward(self, x):
        if self.pad_mode != 'valid':
            x = self.pad(x)
        grad_x = F.conv2d(x, self.filter_x, stride=1, groups=self.in_channels)
        grad_y = F.conv2d(x, self.filter_y, stride=1, groups=self.in_channels)
        return grad_x, grad_y

class L0Smoothing(nn.Module):
    def __init__(self, in_channels, kappa=2.0):
        super(L0Smoothing, self).__init__()
        self.fx = nn.Parameter(t.tensor([[[[1, -1]]]], dtype=t.float32), requires_grad=False)
        self.fy = nn.Parameter(t.tensor([[[[1], [-1]]]], dtype=t.float32), requires_grad=False)

        self.betamax = 1e5
        # self.padx = partial(F.pad, mode=self.pad_mode, pad=(0, 1, 0, 0))
        # self.pady = partial(F.pad, mode=self.pad_mode, pad=(0, 0, 0, 1))
        self.kappa = kappa
        self.in_channels = in_channels

    def forward(self, image, lambda_):
        S = image
        B, _, N, M = image.shape
        sizeI2D = (N, M)
        with t.no_grad():
            otfFx = psf2otf(self.fx, sizeI2D)
            otfFy = psf2otf(self.fy, sizeI2D)
        Normin1 = t.fft.fft2(image)
        Denormin2 = t.abs(otfFx).pow(2) + t.abs(otfFy).pow(2)

        if self.in_channels > 1:
            Denormin2 = Denormin2.repeat([1, self.in_channels, 1, 1])

        beta = 2 * lambda_

        while beta < self.betamax:
            Denormin = 1 + beta * Denormin2
            h = t.cat(
                [
                    S[:, :, :, 1:] - S[:, :, :, 0:M-1],
                    (S[:, :, :, 0] - S[:, :, :, -1]).unsqueeze(3)
                ],
                dim=3)

            v = t.cat(
                [
                    S[:, :, 1:, :] - S[:, :, 0: N - 1, :],
                    (S[:, :, 0, :] - S[:, :, -1, :]).unsqueeze(2)
                ],
                dim=2)

            # idx_mask = t.sum(h.pow(2) + v.pow(2), dim=1, keepdim=True)
            if self.in_channels == 1:
                idx_mask = (h.pow(2) + v.pow(2)) < (lambda_ / beta)
            else:
                idx_mask = t.sum(h.pow(2) + v.pow(2), dim=1, keepdim=True)
                idx_mask = idx_mask < (lambda_ / beta)
                idx_mask = idx_mask.repeat([1, self.in_channels, 1, 1])

            h[idx_mask] = 0
            v[idx_mask] = 0

            Normin2 = t.cat(
                [
                    (h[:, :, :, -1] - h[:, :, :, 0]).unsqueeze(3),
                    h[:, :, :, 0:M - 1] - h[:, :, :, 1:],
                ],
                dim=3
            )

            # F.conv2d(self.padx(h),  self.fx, stride=1, groups=self.in_channels)
            Normin2 = Normin2 + t.cat(
                [
                    (v[:, :, -1, :] - v[:, :, 0, :]).unsqueeze(2),
                    v[:, :, 0: N - 1, :] - v[:, :, 1:, :],
                ],
                dim=2
            )

            Fs = (Normin1 + beta * t.fft.fft2(Normin2)) / Denormin

            S = t.real(t.fft.ifft2(Fs))
            beta *= self.kappa


        return S

class FastPSFEstimate(nn.Module):
    def __init__(self):
        super(FastPSFEstimate, self).__init__()

        # self.matlab_handle = transplant.Matlab(jvm=False)

    def forward(self, psf_size, blur_grad_x, blur_grad_y, sharp_grad_x, sharp_grad_y):
        # kernel_size = (psf_size, psf_size)
        blur_grad_x = blur_grad_x.detach()
        blur_grad_y = blur_grad_y.detach()
        sharp_grad_x = sharp_grad_x.detach()
        sharp_grad_y = sharp_grad_y.detach()

        # h, w = blur_grad_x.shape[2:]
        # blur_grad_x = wrap_boundary_liu(blur_grad_x, opt_fft_size([h + psf_size - 1, w +psf_size - 1]))
        # blur_grad_y = wrap_boundary_liu(blur_grad_y, opt_fft_size([h + psf_size - 1, w + psf_size - 1]))
        # sharp_grad_x = wrap_boundary_liu(sharp_grad_x, opt_fft_size([h + psf_size - 1, w + psf_size - 1]))
        # sharp_grad_y = wrap_boundary_liu(sharp_grad_y, opt_fft_size([h + psf_size - 1, w + psf_size - 1]))
        #
        batch_size = blur_grad_x.shape[0]

        gamma = 1

        with t.no_grad():
            blur_fft_x = t.fft.fft2(blur_grad_x)
            blur_fft_y = t.fft.fft2(blur_grad_y)

            sharp_fft_x = t.fft.fft2(sharp_grad_x)
            sharp_fft_y = t.fft.fft2(sharp_grad_y)

            a = t.conj(sharp_fft_x) * blur_fft_x + t.conj(sharp_fft_y) * blur_fft_y
            b = t.abs(sharp_fft_x) ** 2 + t.abs(sharp_fft_y) ** 2 + gamma

            # out = t.real(otf2psf(a / b, (psf_size, psf_size)))
            out = t.real(otf2psf(a / b, (psf_size, psf_size)))
            # kernel_npy = out.cpu().squeeze(1).numpy()
            # res_k = []
            #
            # # 调用matlab库函数 bwconncomp，用于降噪
            # for i in range(b):
            #     cur_CC = self.matlab_handle.bwconncomp(kernel_npy[i, ...], 8.)
            #     kernel_flat = kernel_npy[i].T.flatten()
            #.cuda()
            #     for i in range(0, int(cur_CC['NumObjects'])):
            #         idx = cur_CC['PixelIdxList'][i]
            #         if isinstance(idx, np.ndarray):
            #             idx = idx.astype(np.int) - 1
            #         else:
            #             idx = int(idx) - 1
            #
            #         currsum = np.sum(kernel_flat[idx])
            #         if currsum < 0.01:
            #             kernel_flat[idx] = 0
            #
            #     tmp = t.from_numpy(kernel_flat.reshape([h, w]).T).unsqueeze(0).unsqueeze(0)
            #
            #
            #     res_k.append(tmp)
            # out = t.cat(res_k, dim=0).to(0)
            #
            # print(out)
            #


            # out[out < 0] = 0
            # out_sum = t.sum(t.flatten(out, 2), dim=2).unsqueeze(0)
            # out = out / out_sum

            # out_max = t.max(out)
            # out[out < out_max / 20] = 0
            # out_sum = t.sum(t.flatten(out, 2), dim=2).unsqueeze(0)
            # out = out / out_sum

        return out

def pixmax(img):
    max_v, _ = t.max(img.flatten(2), dim=2, keepdim=True)
    # max_v[max_v <= 0] = 0
    return img / max_v.unsqueeze(2)

@t.no_grad()
def adjust_psf_center(psf, matlab=None):
    # print(psf)
    b, c, h, w = psf.shape
    if c !=  1:
        raise Exception("psf channel must be one!")
    X, Y = t.meshgrid(t.arange(1, w + 1), t.arange(1, h + 1))
    if psf.get_device() >= 0:
        X = X.to(psf.get_device())
        Y = Y.to(psf.get_device())
    xc1 = t.sum(psf * X.permute([1, 0]).view(1, 1, h, w), dim=(-2, -1)).unsqueeze(-1)
    yc1 = t.sum(psf * Y.permute([1, 0]).view(1, 1, h, w), dim=(-2, -1)).unsqueeze(-1)
    xc2 = (w + 1) / 2
    yc2 = (h + 1) / 2

    xshift = t.round(xc2 - xc1)
    yshift = t.round(yc2 - yc1)

    shift = t.cat([-xshift, -yshift], dim=-2)
    eye = t.eye(2, device=psf.device).unsqueeze(0).repeat([b, 1, 1])

    M = t.cat([eye, shift], dim=-1)
    # print('M', M)
    return _warpimage(psf, M, matlab)

def _warpimage(img, M, matlab):
    warped = _warp_projective2(img, M, matlab)
    warped[t.isnan(warped)] = 0
    return warped

def _warp_projective2(im, A, matlab):
    b, c, h, w = im.shape

    if A.shape[1] > 2:
        A = A[:, 0: 3, :]

    x, y = t.meshgrid(t.arange(w), t.arange(h))
    x = x.permute(1, 0).flatten()
    y = y.permute(1, 0).flatten()

    coords = t.stack([x, y], dim=0) # .unsqueeze(0).repeat([b, 1, 1])
    homogeneousCoords = t.cat([coords, t.ones([1, h * w])], dim=0)
    if im.get_device() >= 0:
        homogeneousCoords = homogeneousCoords.to(im.get_device())
    # print('homogeneousCoords：', homogeneousCoords.device)
    warpedCoords = t.einsum('bik, kh->bih', A, homogeneousCoords) # A @ homogeneousCoords
    xprime = warpedCoords[: ,0, :]
    yprime = warpedCoords[: ,1, :]
    if matlab is None:
        f = interp2linear(im, xprime - 1, yprime - 1)
    else:
        pass

    return  f.reshape(b, c, h, w)  #.permute([0, 1, 3, 2])


# @nb.jit
def interp2linear(z, xi, yi):
    """
    Linear interpolation equivalent to interp2(z, xi, yi,'linear') in MATLAB

    @param z: function defined on square lattice [0..width(z))X[0..height(z))
    @param xi: matrix of x coordinates where interpolation is required, (b, hidden)
    @param yi: matrix of y coordinates where interpolation is required, (b, hidden)
    @param extrapval: value for out of range positions. default is numpy.nan
    @return: interpolated values in [xi,yi] points
    @raise Exception:
    """
    # print(z.get_device())
    extrapval = t.tensor(float('nan'),device=z.device)# .to(z.get_device())
    x = xi
    y = yi
    b, c, nrows, ncols = z.shape

    if nrows < 2 or ncols < 2:
        raise Exception("z shape is too small")

    if not x.shape == y.shape:
        raise Exception(""
                        "sizes of X indexes and Y-indexes must match")


    # find x values out of range
    x_bad = ( (x < 0) | (x > ncols - 1))

    if x_bad.any():
        x[x_bad] = 0

    # find y values out of range
    y_bad = ((y < 0) | (y > nrows - 1))

    if y_bad.any():
        y[y_bad] = 0

    # linear indexing. z must be in 'C' order
    ndx = t.floor(y) * ncols + t.floor(x)
    ndx = ndx.long()#ndx.astype('int32')
    # fix parameters on x border
    d = (x == ncols - 1)
    x = (x - t.floor(x))
    if d.any():
        x[d] += 1
        ndx[d] -= 1

    # fix parameters on y border
    d = (y == nrows - 1)
    y = (y - t.floor(y))

    if d.any():
        y[d] += 1
        ndx[d] -= ncols

    # interpolate
    one_minus_t = 1 - y
    # z = z.ravel(2).unsqueeze(2)
    z = z.flatten(1) #.unsqueeze(2)
    b, h = z.shape
    # bidx = t.arange(b).unsqueeze_(-1).repeat([1, h]).ravel()
    ndx = ndx.ravel()
    ndxh = ndx.shape[0]
    bidx = t.arange(b).unsqueeze(-1).repeat([1, ndxh]).ravel()
    # print('ndx', ndx)


    f = (z[bidx, ndx] * one_minus_t + z[bidx, ndx + ncols] * y) * (1 - x) + (z[bidx, ndx + 1] * one_minus_t + z[bidx, ndx + ncols + 1] * y) * x

    # Set out of range positions to extrapval
    if x_bad.any():
        f[x_bad] = extrapval
    if y_bad.any():
        f[y_bad] = extrapval

    return f

@t.no_grad()
def downsample_imc(img, ret):
    if ret == 1:
        return img

    sigma = 1 / pi * ret
    g0 = t.arange(-50, 50 + 1)  * 2 * pi
    sf=t.exp(-0.5 * (g0 ** 2) * (sigma ** 2))
    sf = sf / t.sum(sf)


    csf= t.cumsum(sf, dim=0)
    csf= t.min(csf, t.flip(csf, dims=[0]))

    ii = csf > 0.05
    sf = sf[ii].unsqueeze(0)
    b, c, ih, iw = img.shape

    sf = t.matmul(sf.transpose(1, 0), sf)
    sh, sw = sf.shape
    sf = sf.view(1, 1, sw, sw).repeat(c, 1, 1, 1) #  .to(img.get_device())
    if img.get_device() >= 0:
        sf = sf.to(img.get_device())
    # sum(sf)

    I = F.conv2d(img, sf, stride=1, padding=0, groups=c)

    sI = F.interpolate(I, scale_factor=ret, mode='bilinear')#interp2linear(I, gx, gy)

    return sI #sI.reshape([b, c, nh, nw])

@t.no_grad()
def resize_kernel(kernel, ret, k1, k2):
    kernel = F.interpolate(kernel, scale_factor=ret, mode='bicubic', align_corners=True)#mode='nearest')

    kernel[kernel < 0] =  0
    # print(kernel.shape)
    kernel = fixsize(kernel, k1, k2)
    # print(kernel.shape)
    kernel = kernel / t.sum(kernel, dim=(-2, -1), keepdim=True)
    return kernel



def adjust_kernel_center(k):
    b, kc, kh, kw = k.shape


    if kc != 1:
        raise Exception('')


    mu_y = t.sum(t.arange(1, kh + 1, dtype=k.dtype, device=k.device) * t.sum(k[:, 0, :, :], dim=-1), dim=-1)
    mu_x = t.sum(t.arange(1, kw + 1, dtype=k.dtype, device=k.device) * t.sum(k[:, 0, :, :], dim=-2), dim=-1)

    # print(mu_y)

    offset_x = t.round(kw // 2 + 1 - mu_x)
    offset_y = t.round(kh // 2 + 1 - mu_y)
    offset_x = offset_x.int()
    offset_y = offset_y.int()
    kshifts = []
    # xshifts = []
    # yshifts = []

    for i in range(b):
        # shift_kernel = t.zeros(1, 1, abs(offset_y[i].item() * 2) + 1, abs(offset_x[i].item() * 2) + 1, device=k.device)
        shift_kernel = t.zeros(1, 1, abs(offset_y[i].item() * 2) + 1, abs(offset_x[i].item() * 2) + 1, device=k.device)
        shift_kernel[:, :, abs(offset_y[i].item()) + offset_y[i].item(), abs(offset_x[i].item()) + offset_x[i].item()] = 1

        padding = (shift_kernel.shape[-2] // 2, shift_kernel.shape[-1] // 2)

        kshift = F.conv2d(k[i: i + 1, :, :, :], t.rot90(shift_kernel, 2, dims=(-2, -1)), stride=1, padding=padding)
        kshifts.append(kshift)

        # xshift = F.conv2d(x[i: i + 1, :, :, :], t.rot90(shift_kernel, 2, dims=(-2, -1)), stride=1, padding=padding)
        # xshifts.append(xshift)
        # yshift = F.conv2d(y[i: i + 1, :, :, :], t.rot90(shift_kernel, 2, dims=(-2, -1)), stride=1, padding=padding)
        # yshifts.append(yshift)

        # xshift = F.conv2d(x[i: i + 1, :, :, :], shift_kernel, stride=1, padding=padding)
        # xshifts.append(xshift)
        # yshift = F.conv2d(y[i: i + 1, :, :, :], shift_kernel, stride=1, padding=padding)
        # yshifts.append(yshift)

    return t.cat(kshifts, dim=0) #t.cat(xshifts, dim=0), t.cat(yshifts, dim=0), t.cat(kshifts, dim=0)


def fixsize(f, nk1, nk2):
    # [k1, k2] = size(f);
    b, c, k1, k2 = f.shape

    if c != 1:
        raise Exception('error!')
    while (k1 != nk1) or (k2 != nk2):
        if k1 > nk1:
            s = t.sum(f, dim=-1)
            fs = []
            for i in range(b):
                if (s[i, 0, 0].item() < s[i, 0, -1].item()):
                    fs.append(f[i: i + 1, :, 1: , :])
                else:
                    fs.append(f[i: i + 1, :, 0: -1, :])
            f = t.cat(fs, dim=0)

        if k1 < nk1:
            s = t.sum(f, dim=-1)
            fs = []
            for i in range(b):
                if s[i, 0, 0].item() < s[i, 0, -1].item():
                    tf = t.zeros([1, c, k1 + 1, f.shape[-1]], device=f.device)
                    tf[0:1, :, 0: k1,:] = f[i: i+1, ...]
                    fs.append(tf)
                    # f = tf
                else:
                    tf = t.zeros([1, c, k1 + 1, f.shape[-1]], device=f.device)
                    tf[0: 1, :, 1: k1 + 1, :] = f[i: i + 1, ...]
                    fs.append(tf)
            f = t.cat(fs, dim=0)

        if k2 > nk2:
            s = t.sum(f, dim=-2)
            fs = []
            for i in range(b):
                if s[i, 0, 0].item() < s[i, 0, -1].item():
                    fs.append(f[i:i+1, :, :, 1: ])
                else:
                    fs.append(f[i:i+1, :, :, 0: -1])# f(:, 1: end - 1);
            f = t.cat(fs, dim=0)

        if k2 < nk2:
            s = t.sum(f, dim=-2)
            fs = []
            for i in range(b):
                if s[i, 0, 0].item() < s[i, 0, -1].item():
                    tf = t.zeros([1, c, f.shape[-2], k2 + 1], device=f.device)
                    tf[0: 1, :, :, 0: k2] = f[i: i + 1, ...]
                    fs.append(tf)
                else:
                    tf = t.zeros([1, c, f.shape[-2], k2 + 1], device=f.device)
                    tf[0: 1, :, :, 1: k2 + 1] = f[i: i + 1, ...]
                    fs.append(tf)
            f = t.cat(fs, dim=0)
        b, c, k1, k2 = f.shape
    return f

@t.no_grad()
def wrap_boundary_liu(img, img_size):
    """
    Reducing boundary artifacts in image deconvolution
    Renting Liu, Jiaya Jia
    ICIP 2008
    """
    b, c, h, w = img.shape
    if c == 1:
        ret = wrap_boundary(img, img_size)
    elif c == 3:
        ret = [wrap_boundary(img[:, i: i+1, :, :], img_size) for i in range(3)]

        ret = t.cat(ret, dim=1)
    else:
        raise Exception('wrap_boundary_liu error!')
    # ret = wrap_boundary(img, img_size)
    return ret


def wrap_boundary(img, img_size):
    """
    python code from:
    https://github.com/ys-koshelev/nla_deblur/blob/90fe0ab98c26c791dcbdf231fe6f938fca80e2a0/boundaries.py
    Reducing boundary artifacts in image deconvolution
    Renting Liu, Jiaya Jia
    ICIP 2008
    """
    (b, c, H, W) = img.shape
    H_w = int(img_size[0]) - H
    W_w = int(img_size[1]) - W
    # ret = np.zeros((img_size[0], img_size[1]));
    alpha = 1
    # HG = img[:, :]
    HG = img
    r_A = t.zeros((b, c, alpha * 2 + H_w, W), device=img.device)
    r_A[:, :, :alpha, :] = HG[:, :, -alpha:, :]
    r_A[:, :, -alpha:, :] = HG[:, :, :alpha, :]
    a = t.arange(H_w, device=img.device) / (H_w - 1)
    # r_A(alpha+1:end-alpha, 1) = (1-a)*r_A(alpha,1) + a*r_A(end-alpha+1,1)

    r_A[:, :, alpha:-alpha, 0] = ((1 - a) * r_A[:, :, alpha - 1, 0] + a * r_A[:, :, -alpha, 0]).unsqueeze(1)
    # r_A(alpha+1:end-alpha, end) = (1-a)*r_A(alpha,end) + a*r_A(end-alpha+1,end)
    r_A[:, :, alpha:-alpha, -1] = ((1 - a) * r_A[:, :, alpha - 1, -1] + a * r_A[:, :, -alpha, -1]).unsqueeze(1)




    r_B = t.zeros((b, c, H, alpha * 2 + W_w), device=img.device)
    r_B[:, :, :, :alpha] = HG[:, :, :, -alpha:]
    r_B[:, :, :, -alpha:] = HG[:, :, :, :alpha]
    a = t.arange(W_w, device=img.device) / (W_w - 1)



    r_B[:, :, 0, alpha:-alpha] = ((1 - a) * r_B[:, :, 0, alpha - 1] + a * r_B[:, :, 0, -alpha]).unsqueeze(1)
    r_B[:, :, -1, alpha:-alpha] = ((1 - a) * r_B[:, :, -1, alpha - 1] + a * r_B[:, :, -1, -alpha]).unsqueeze(1)



    if alpha == 1:
        A2 = solve_min_laplacian(r_A[:, :, alpha - 1:, :])
        B2 = solve_min_laplacian(r_B[:, :, :, alpha - 1:])
        r_A[:, :, alpha - 1:, :] = A2
        r_B[:, :, :, alpha - 1:] = B2
    else:
        A2 = solve_min_laplacian(r_A[:, :, alpha - 1:-alpha + 1, :])
        r_A[:, :, alpha - 1:-alpha + 1, :] = A2
        B2 = solve_min_laplacian(r_B[:, :, :, alpha - 1:-alpha + 1])
        r_B[:, :, :, alpha - 1:-alpha + 1] = B2
    A = r_A
    B = r_B

    r_C = t.zeros((b, c, alpha * 2 + H_w, alpha * 2 + W_w), device=img.device)
    r_C[:, :, :alpha, :] = B[:, :, -alpha:, :]
    r_C[:, :, -alpha:, :] = B[:, :, :alpha, :]
    r_C[:, :, :, :alpha] = A[:, :, :, -alpha:]
    r_C[:, :, :, -alpha:] = A[:, :, :, :alpha]

    if alpha == 1:
        C2 = solve_min_laplacian(r_C[:, :, alpha - 1:, alpha - 1:])
        r_C[:, :, alpha - 1:, alpha - 1:] = C2
    else:
        C2 = solve_min_laplacian(r_C[:, :, alpha - 1:-alpha + 1, alpha - 1:-alpha + 1])
        r_C[:, :, alpha - 1:-alpha + 1, alpha - 1:-alpha + 1] = C2
    C = r_C
    # return C
    A = A[:, :, alpha - 1:-alpha - 1, :]
    B = B[:, :, :, alpha:-alpha]
    C = C[:, :, alpha:-alpha, alpha:-alpha]


    ret = t.cat((t.cat((img, B), dim=-1), t.cat((A, C), dim=-1)), dim=-2)
    return ret


def solve_min_laplacian(boundary_image):
    b, c, H, W = boundary_image.shape

    # Laplacian
    f = t.zeros((b, c, H, W), device=boundary_image.device)
    # boundary image contains image intensities at boundaries
    boundary_image[:, :, 1:-1, 1:-1] = 0
    j = t.arange(2, H) - 1
    k = t.arange(2, W) - 1
    f_bp = t.zeros((b, c, H, W), device=boundary_image.device)
    f_bp[:, :, 1: H - 1, 1: W - 1] = -4 * boundary_image[:, :, 1: H - 1, 1: W - 1] + boundary_image[:, :, 1: H - 1, 2: W] + boundary_image[
        :, :, 1: H - 1, 0: W - 2] + boundary_image[:, :, 0: H - 2, 1: W - 1] + boundary_image[:, :, 2: H, 1: W - 1]

    del (j, k)
    f1 = f - f_bp  # subtract boundary points contribution
    del (f_bp, f)

    # DST Sine Transform algo starts here
    f2 = f1[:, :, 1:-1, 1:-1]
    del (f1)

    # tt = dst(f2)
    # compute sine tranform
    # if f2.shape[-1] == 1:
    #     tt = dst(f2)
    # else:
    #     tt = dst(f2)
    #
    # if tt.shape[-2] == 1:
    #     f2sin = t.transpose(dst(t.transpose(tt, -1, -2)), -1, -2)
    # else:
    #     f2sin = t.transpose(dst(t.transpose(tt, -1, -2)), -1, -2)

    tt = dst(f2)
    f2sin = t.transpose(dst(t.transpose(tt, -1, -2)), -1, -2)
    del (f2)

    # compute Eigen Values
    [y, x] = t.meshgrid(t.arange(1, H - 1, device=boundary_image.device), t.arange(1, W - 1, device=boundary_image.device))
    denom = (2 * t.cos(pi * x / (W - 1)) - 2) + (2 * t.cos(pi * y / (H - 1)) - 2)

    # divide
    f3 = f2sin / denom
    del (f2sin, x, y)

    # compute Inverse Sine Transform
    # if f3.shape[0] == 1:
    #     tt = idst(f3 * 2, type=1, axis=1) / (2 * (f3.shape[1] + 1))
    # else:
    #     tt = idst(f3 * 2, type=1, axis=0) / (2 * (f3.shape[0] + 1))
    # del (f3)
    tt = idst(f3)
    del (f3)

    img_tt = t.transpose(idst(t.transpose(tt, -2, -1)), -2, -1)
    # if tt.shape[1] == 1:
    #     img_tt = np.transpose(idst(np.transpose(tt) * 2, type=1) / (2 * (tt.shape[0] + 1)))
    # else:
    #     img_tt = np.transpose(idst(np.transpose(tt) * 2, type=1, axis=0) / (2 * (tt.shape[1] + 1)))
    del (tt)

    # put solution in inner points; outer points obtained from boundary image
    img_direct = boundary_image
    img_direct[:, :, 1:-1, 1:-1] = 0
    img_direct[:, :, 1:-1, 1:-1] = img_tt
    return img_direct


def dst(a, n=None):
    b, c, h, w = a.shape
    if min(h, w) == 1:
        if w > 1:
            do_trans = True
        else:
            do_trans = False
        a = a.flatten(2)
    else:
        do_trans = False

    if n is None:
        n = h
    m = w

    if h < n:
        aa = t.zeros([b, c, n, m], device=a.device)
        aa[:, :, 0: h, :] = a
    else:
        aa = a[:, :, 0: n, :]

    y = t.zeros([b, c, 2 * (n + 1), m], device=a.device)
    y[:, :, 1: n + 1, :] = aa
    y[:, :, n + 2: 2 * (n + 1), :] = -t.flip(aa, dims=[-2])
    # print('y', y.shape)
    yy = t.fft.fft(y.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
    b = yy[:, :, 1: n + 1, :] / (-2 * t.complex(t.tensor(0., device=a.device, dtype=a.dtype), t.tensor(1., device=a.device, dtype=a.dtype)))

    if t.isreal(a).all():
        b = t.real(b)

    if do_trans:
        b = b.transpose(-1, -2)

    return b

def opt_fft_size(n):
    '''
    Kai Zhang (github: https://github.com/cszn)
    03/03/2019
    #  opt_fft_size.m
    # compute an optimal data length for Fourier transforms
    # written by Sunghyun Cho (sodomau@postech.ac.kr)
    # persistent opt_fft_size_LUT;
    '''

    LUT_size = 4096
    # print("generate opt_fft_size_LUT")
    opt_fft_size_LUT = t.zeros(LUT_size)

    e2 = 1
    while e2 <= LUT_size:
        e3 = e2
        while e3 <= LUT_size:
            e5 = e3
            while e5 <= LUT_size:
                e7 = e5
                while e7 <= LUT_size:
                    if e7 <= LUT_size:
                        opt_fft_size_LUT[e7-1] = e7
                    if e7*11 <= LUT_size:
                        opt_fft_size_LUT[e7*11-1] = e7*11
                    if e7*13 <= LUT_size:
                        opt_fft_size_LUT[e7*13-1] = e7*13
                    e7 = e7 * 7
                e5 = e5 * 5
            e3 = e3 * 3
        e2 = e2 * 2

    nn = 0
    for i in range(LUT_size, 0, -1):
        if opt_fft_size_LUT[i-1] != 0:
            nn = i - 1
        else:
            opt_fft_size_LUT[i-1] = nn + 1

    m = []#t.zeros(len(n), device=device)
    for c in range(len(n)):
        nn = n[c]
        if nn <= LUT_size:
            m.append(opt_fft_size_LUT[nn-1].item())
            # m[c] = opt_fft_size_LUT[nn-1]
        else:
            m.append(-1)
            # m[c] = -1
    return m


def idst(a, n=None):
    b, c, h, w = a.shape
    if n is None:
        if min(h, w) == 1:
            n = max(h, w)
        else:
            n = h

    n2 = n + 1
    b = 2 / n2 * dst(a, n)
    return b

def init_kernel(batch_size, psf_size, device=t.device("cpu")):
    k = t.zeros([batch_size, 1, psf_size, psf_size], device=device)
    k[:, :, psf_size // 2, psf_size // 2: psf_size // 2 + 2] = 1/2
    return k


def im2uint8(img):
    if img.dtype == t.uint8:
        return img
    img = t.clip(img * 255, 0., 255.)
    img = img.type(dtype=t.uint8)
    return img


def threshold_pxpy(image, kernel_size, threshold=None):
    if threshold is None:
        b_estimate_threshold = True
        threshold = 0.0
    else:
        if isinstance(threshold, float):
            b_estimate_threshold = False
        elif isinstance(threshold, int):
            b_estimate_threshold = False
            threshold = float(threshold)
        else:
            raise Exception('')

    b, c, h, w = image.shape
    kernel_size = [kernel_size] if not (isinstance(kernel_size, list) or isinstance(kernel_size, tuple))  else  kernel_size

    if c == 3:
        image = rgb2gray(image)
    elif c == 1:
        pass
    else:
        raise  Exception('')

    dx = t.tensor([[-1, 1], [0, 0]], dtype=image.dtype, device=image.device)
    dy = t.tensor([[-1, 0], [1, 0]], dtype=image.dtype, device=image.device)

    px = imfilter(image, filter=dx, padding_mode='valid', filter_mode='conv', filter_format="HW")
    py = imfilter(image, filter=dy, padding_mode='valid', filter_mode='conv', filter_format="HW")


    pm = px ** 2 + py ** 2

    if b_estimate_threshold:
        pd = t.arctan(py / (px + 1e-7))
        pm_steps = t.arange(0, 2 + 0.00006, 0.00006,  device=image.device)
        n_pm_steps = pm_steps.shape[0]#int(2 / 0.00006) + 1
        # print('n_pm_steps:', n_pm_steps)
        H1 = t.cumsum(t.flipud(t.histc(pm[t.logical_and(pd >= 0, pd < pi/4)], min=0, max=2, bins=n_pm_steps)), dim=0)
        H2 = t.cumsum(t.flipud(t.histc(pm[t.logical_and(pd >= pi/4, pd < pi/2)], min=0, max=2, bins=n_pm_steps)), dim=0)
        H3 = t.cumsum(t.flipud(t.histc(pm[t.logical_and(pd >= -pi/4, pd < 0)], min=0, max=2, bins=n_pm_steps)), dim=0)
        H4 = t.cumsum(t.flipud(t.histc(pm[t.logical_and(pd >= -pi/2, pd < -pi/4)], min=0, max=2, bins=n_pm_steps)), dim=0)

        th = max([max(kernel_size) * 20, 10])
        # print(pm_steps.shape, H1.shape, H2.shape, H3.shape, H4.shape)
        # n_pm_steps = len(pm_steps)

        for idx in range(0, n_pm_steps - 1):
            min_h = min(H1[idx].item(), H2[idx].item(), H3[idx].item(), H4[idx].item())
            if min_h >= th:
                threshold = pm_steps[n_pm_steps - idx - 1].item()
                break
    m = pm < threshold

    while t.all(m == True):
        threshold *= 0.81
        m = pm < threshold
    px[m] = 0
    py[m] = 0

    if b_estimate_threshold == False:
        threshold /= 1.1

    # edge = t.sqrt(px **2 + py ** 2)

    return px, py, threshold


def im2relief(image):
    image = rgb2gray(image)
    grad = ImageGrad(1)
    if image.get_device() >= 0:
        grad = grad.to(image.get_device())
    px, py = grad(image)
    px = mat2gray(px)
    py = mat2gray(py)

    return px, py

def salt_imnoise(img, p, max_val=1):
    x = t.rand_like(img)
    b = img
    # b = t.where(x < p / 2, t.zeros_like(img), max_val * t.ones_like(img))
    b[x < p / 2] = 0
    mask_ = t.logical_and(x >= p /2, x < p)
    b[mask_] = max_val


    return b






if __name__ == '__main__':
    a = 0.05 * t.randn([1, 3, 256, 256]) + 0.5
    print(a)
    a = t.clip(a, 0, 1)
    # a = mat2gray(a)
    a = a.permute(0, 2, 3, 1)[0]
    a = im2uint8(a)
    imageio.imwrite('aaa.png', a)
    # imwrite('aaa.png', a)
