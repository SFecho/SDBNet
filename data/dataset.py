import glob
import math
import os
from math import ceil, sqrt
from random import randrange, random

import cv2
import imageio
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as tvf
import numpy as np
from torchvision.transforms.transforms import _setup_size

from utils.image import imread, fspecial_gaussian, rgb2gray, ImageOption, imfilter, imwrite, salt_imnoise, \
    FastPSFEstimate, ImageGrad
import torchvision as tv
import scipy.io as io
import torch.nn.functional as F
import torchvision.transforms.functional as tvfF

class IRDataset(Dataset):
    def __init__(self, data_dir, patch_size, file_format, **kwargs):
        super(IRDataset, self).__init__()
        # self.cfg = cfg
        # self.mode = mode
        # self.task = cfg.task
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.file_format = file_format
        if patch_size >= 0:
            transform = tvf.Compose([
                tvf.ToTensor(),
                tvf.RandomCrop(patch_size),
                tvf.RandomHorizontalFlip(0.6),
                tvf.RandomVerticalFlip(0.6)
            ])
        elif patch_size == -1:
            transform = tvf.Compose([tvf.ToTensor()])
        else:
            raise ValueError(f"invalid dataloader patch size {patch_size}")

        self.transform = transform
        self.data_dir = data_dir

        self.degrads, self.labels = self.load_dataset(self.data_dir)

    def __len__(self):
        return len(self.degrads)

    def load_dataset(self, data_dir):
        degrad_dir = os.path.join(data_dir, 'degrad')
        label_dir = os.path.join(data_dir, 'label')

        degrads = glob.glob(os.path.join(label_dir, self.file_format))
        labels = []

        for filepath in degrads:
            filename = filepath.split(filepath, os.sep)[-1]
            labels.append(os.path.join(degrad_dir, filename))

        return degrads, labels

    def __getitem__(self, item):
        degrad_path = self.degrads[item]
        label_path = self.labels[item]
        filename = label_path.split(os.sep)[-1]
        degrad = imread(degrad_path)
        label = imread(label_path)

        dl = self.transform(np.concatenate([degrad, label], axis=-1))
        degrad_tensor, label_tensor = torch.split(dl, split_size_or_sections=2)

        return degrad_tensor, label_tensor, filename


class DenoiseDataset(Dataset):
    def __init__(self, data_dir, patch_size, file_format, sigma, is_gray=False, **kwargs):
        super(DenoiseDataset, self).__init__()

        self.file_format = file_format
        self.patch_size = patch_size
        self.data_dir = data_dir
        self.sigma = sigma
        self.is_gray = is_gray

        if patch_size >= 0:
            transform = tvf.Compose([
                tvf.ToTensor(),
                tvf.RandomCrop(size=patch_size),
                tvf.RandomHorizontalFlip(0.6),
                tvf.RandomVerticalFlip(0.6)
            ])
        elif patch_size == -1:
            transform = tvf.Compose([tvf.ToTensor()])
        else:
            raise ValueError(f"invalid dataloader patch size is {patch_size}")

        self.transform = transform
        self.data_dir = data_dir

        self.labels = self.load_dataset(self.data_dir)

    def __len__(self):
        return len(self.labels)

    def load_dataset(self, data_dir):
        label_dir = os.path.join(data_dir, 'label')
        labels = []
        for item in self.file_format:
            labels += glob.glob(os.path.join(label_dir, item))
        return labels

    def __getitem__(self, item):
        label_path = self.labels[item]
        filename = label_path.split(os.sep)[-1]
        label = imread(label_path)
        label_tensor = self.transform(label)
        degrad_tensor = label_tensor + self.sigma / 255. * torch.randn_like(label_tensor)

        if self.is_gray == True:
            label_tensor = rgb2gray(label_tensor, ImageOption.Formal.CHW)
            degrad_tensor = rgb2gray(degrad_tensor, ImageOption.Formal.CHW)

        return degrad_tensor, label_tensor, filename




class TBlurDataset(Dataset):
    def __init__(self, patch_size, data_dir, csv_filename, file_format, sigma, min_psf_size = 19, max_psf_size=59, batch_size=8, outliner=True, repeat=4, **kwargs):
        super(TBlurDataset, self).__init__()
        # self.batch_size = batch_size
        self.data_dir = data_dir
        # print('data_dir:', data_dir)
        self.csv_filename = csv_filename
        self.patch_size = patch_size
        self.max_psf_size = max_psf_size
        self.min_psf_size = min_psf_size
        self.batch_size = batch_size
        self.outliner = outliner
        self.file_format = file_format
        self.sigma = sigma
        if patch_size >= 0:
            transform = tvf.Compose([
                tvf.ToTensor(),
                tvf.RandomCrop(patch_size),
                tvf.RandomHorizontalFlip(0.6),
                tvf.RandomVerticalFlip(0.6)
            ])
        else:
            raise Exception('error!')
        self.transform = transform

        self.repeat = repeat
        self.sharps = self.load_csv(data_dir, self.csv_filename)
        self.point_size = list(range(15, 30, 2))

    @torch.no_grad()
    def generate_outliner(self, image, point_size, start_pos, end_pos, n_point):

        bias_sigma = 1.0
        center = point_size // 2

        h_pos = [randrange(start_pos[0] + center, end_pos[0] - center - 1) for _ in range(n_point)]
        w_pos = [randrange(start_pos[1] + center, end_pos[1] - center - 1) for _ in range(n_point)]


        for i in range(n_point):
            light = 2 + 0.5 * random()

            sigma = bias_sigma * (1 + random() * 0.6)  # randrange(2, 3)
            kernel = fspecial_gaussian(point_size, sigma, image.device)
            kernel = (kernel / torch.max(kernel)) * light

            for j in range(-center, center + 1, 1):
                for k in range(-center, center + 1, 1):
                    image[:, h_pos[i] + j, w_pos[i] + k] += kernel[j + center, k + center]

        return image

    def load_csv(self, data_path, csv_filename):
        csv_file = os.path.join(data_path, csv_filename)

        if not os.path.exists(csv_file):
            sharp = []
            for ext in self.file_format:
                sharp_path = os.path.join(data_path, 'sharp', ext)
                sharp += glob.glob(sharp_path)

            trainset_info = pd.DataFrame({
                'sharp': sharp,
            })
            trainset_info.to_csv(csv_file, index=False)

        else:
            trainset_info = pd.read_csv(csv_file)

            sharp = trainset_info['sharp'].tolist()

        return sharp * self.repeat

    def __len__(self):
        return len(self.sharps)

    def crop_patch(self, img, size):
        h, w = img.shape[:2]
        x = randrange(0, h - size + 1)
        y = randrange(0, w - size + 1)

        return img[x: x + size, y: y + size, :]


    def __getitem__(self, item):
        # print(item)
        sharp_filename = self.sharps[item]
        sharp = imread(sharp_filename)
        grid = (item // self.batch_size) % (self.max_psf_size - self.min_psf_size)
        grid += self.min_psf_size

        kpath = os.path.join(self.data_dir, 'kernel', 'kernel_g{}.mat'.format(grid))
        kernel = io.loadmat(kpath)['kernel']
        c = kernel.shape[-1]
        kernel = kernel[:, :, randrange(0, c - 1)]

        h, w, c = sharp.shape

        if c == 1:
            sharp = sharp.repeat([1, 1, 3])
        elif c > 3:
            sharp = sharp[:, :, :3]

        real_ksize = kernel.shape[-1]
        pad = real_ksize // 2
        crop_size = self.patch_size + pad * 2

        self.transform.transforms[1].size = tuple(_setup_size(
            crop_size, error_msg="Please provide only two dimensions (h, w) for size."
        ))
        # print(self.sigma)
        sigmas = list(range(1, int(self.sigma * 100) + 1, 1))
        # print(sigmas)
        sigmas = [item / 100  for item in sigmas]
        if self.patch_size > 0:
            sharp = self.transform(sharp)
            if self.outliner:
                n_point = randrange(1, 3)
                sharp = self.generate_outliner(sharp, self.point_size[randrange(0, len(self.point_size) - 1)], (65, 65), (self.patch_size - 65, self.patch_size - 65), n_point) * 1.3

            kernel = torch.from_numpy(kernel).unsqueeze(0)
            # blur = cv2.filter2D(sharp, ddepth=-1,kernel=kernel)
            with torch.no_grad():
                i = randrange(0, len(sigmas) - 1, 1)
                blur = imfilter(sharp.unsqueeze(0), kernel, padding_mode="valid").squeeze(0)
                blur = blur + sigmas[i] * torch.randn_like(blur)
                blur = torch.clip(blur, 0, 1).detach_()
                sharp = torch.clip(sharp[:, pad: -pad, pad: -pad], 0, 1)

        else:
            raise Exception('patch size must > 0!')


        # print(blur.shape, sharp[pad: -pad, pad: -pad, :].shape)

        return blur, kernel, sharp, sharp_filename.split(os.sep)[-1]


class VBlurDataset(Dataset):
    def __init__(self, data_dir, csv_filename, file_format, kernel_type, sigma, is_gray=True, **kwargs):
        super(VBlurDataset, self).__init__()
        self.data_format = file_format
        self.n2t = tv.transforms.ToTensor()
        self.kernels, self.sharps = self.load_csv(data_dir, kernel_type, csv_filename, file_format)
        self.sigma = sigma
        self.is_gray = is_gray

    def load_kernel(self, filename):
        ext = filename.split('.')[-1]
        if ext == 'mat':
            kernel = io.loadmat(filename)['kernel'].astype(np.float32)
            kernel = self.n2t(np.expand_dims(kernel, axis=-1))
            kernel /= torch.sum(kernel)
        else:
            kernel = imread(filename)
            kernel = self.n2t(kernel)
            kernel = rgb2gray(kernel, ImageOption.Formal.CHW)

        return kernel

    def load_csv(self, data_dir, kernel_type, csv_filename, data_format):
        csv_file = os.path.join(data_dir, csv_filename)
        if not os.path.exists(csv_file):
            tmp_sharps = []
            tmp_kernels = []
            kernels = []
            sharps = []
            for ext in data_format:
                kernel_path = os.path.join(data_dir, kernel_type, ext)
                tmp_kernels += glob.glob(kernel_path)
                sharp_path = os.path.join(data_dir, 'sharp', ext)
                tmp_sharps += glob.glob(sharp_path)

            for sharp in tmp_sharps:
                for kernel in tmp_kernels:
                    sharps.append(sharp)
                    kernels.append(kernel)

            del tmp_sharps, tmp_kernels

            trainset_info = pd.DataFrame({
                'sharp': sharps,
                'kernel': kernels
            })

            trainset_info.to_csv(csv_file, index=False)

        else:
            trainset_info = pd.read_csv(csv_file)
            sharps = trainset_info['sharp'].tolist()
            kernels = trainset_info['kernel'].tolist()
        return kernels, sharps

    def __len__(self):
        return len(self.sharps)

    def __getitem__(self, item):
        sharp_filename = self.sharps[item]
        kernel_filename = self.kernels[item]
        sharp = self.n2t(imread(sharp_filename))
        if self.is_gray == True:
            sharp = rgb2gray(sharp, ImageOption.Formal.CHW)
        kernel = self.load_kernel(kernel_filename)
        with torch.no_grad():
            blur = imfilter(sharp.unsqueeze(0), kernel, padding_mode='replicate', filter_mode='conv').squeeze(0)
            noise = self.sigma * torch.randn_like(blur)
            blur = blur + noise
            blur = torch.clip(blur, 0, 1).detach_().mul(255).round().div(255)
            # blur = salt_imnoise(blur, self.salt_p ,1)
        # blur, kernel, sharp, sharp_filename.split(os.sep)[-1]
        return blur, kernel, sharp, sharp_filename.split(os.sep)[-1]


class IRTeacherDataset(Dataset):
    def __init__(self, patch_size, data_dir, file_format, csv_filename, repeat=4, is_gray=False):
        super(IRTeacherDataset, self).__init__()
        self.data_dir = data_dir
        self.csv_filename = csv_filename
        self.patch_size = patch_size
        self.file_format = file_format
        self.is_gray = is_gray

        if patch_size >= 0:
            transform = tvf.Compose([
                tvf.ToTensor(),
                tvf.RandomCrop(size=patch_size),
                tvf.RandomHorizontalFlip(0.6),
                tvf.RandomVerticalFlip(0.6)
            ])
        elif patch_size == -1:
            transform = tvf.Compose([tvf.ToTensor()])
        else:
            raise ValueError(f"invalid dataloader patch size is {patch_size}")
        self.transform = transform

        self.repeat = repeat
        self.sharps = self.load_csv(data_dir, self.csv_filename)
        self.patch_size = patch_size


    def __len__(self):
        return len(self.sharps)

    def load_csv(self, data_path, csv_filename):
        csv_file = os.path.join(data_path, csv_filename)

        if not os.path.exists(csv_file):
            sharp = []
            for ext in self.file_format:
                sharp_path = os.path.join(data_path, 'sharp', ext)
                sharp += glob.glob(sharp_path)

            trainset_info = pd.DataFrame({
                'sharp': sharp,
            })

            trainset_info.to_csv(csv_file, index=False)

        else:
            trainset_info = pd.read_csv(csv_file)
            sharp = trainset_info['sharp'].tolist()[0:10000]

        sharp = sharp * self.repeat
        return sharp

    def __getitem__(self, item):
        label_path = self.sharps[item]
        filename = label_path.split(os.sep)[-1]

        label = imread(label_path)
        h, w = label.shape[:2]

        if min(h, w) < self.patch_size:
            # print(h, w)
            factor = ceil(float(self.patch_size) / float(min(h, w)))
            label = cv2.resize(label, dsize=(h * factor, w * factor))
            # h, w = label.shape[:2]
            # print(h, w)


        label_tensor = self.transform(label)



        c = label_tensor.shape[1]
        if c > 3:
            label_tensor = label_tensor[:3, :, :]

        light_factor = [1.2, 1.3, 1.4]
        idx = randrange(0, 3)

        label_tensor = label_tensor.div(1.2).mul(light_factor[idx]).clip(0, 1).mul(255).round().div(255)

        if self.is_gray == True:
            label_tensor = rgb2gray(label_tensor, ImageOption.Formal.CHW)

        return label_tensor, label_tensor, filename

class NoiseTeacherDataset(Dataset):
    def __init__(self, data_dir, file_format, csv_filename, repeat=4, is_gray=False):
        super(NoiseTeacherDataset, self).__init__()
        self.data_dir = data_dir
        self.csv_filename = csv_filename
        self.file_format = file_format
        self.is_gray = is_gray


        self.repeat = repeat
        self.sharps = self.load_csv(data_dir, self.csv_filename)

    def __len__(self):
        return len(self.sharps)

    def load_csv(self, data_path, csv_filename):
        csv_file = os.path.join(data_path, csv_filename)

        if not os.path.exists(csv_file):
            sharp = []
            for ext in self.file_format:
                sharp_path = os.path.join(data_path, 'sharp', ext)
                sharp += glob.glob(sharp_path)

            trainset_info = pd.DataFrame({
                'sharp': sharp,
            })

            trainset_info.to_csv(csv_file, index=False)

        else:
            trainset_info = pd.read_csv(csv_file)
            sharp = trainset_info['sharp'].tolist()

        sharp = sharp * self.repeat
        return sharp

    def __getitem__(self, item):
        label_path = self.sharps[item]
        filename = label_path.split(os.sep)[-1]
        label_tensor = torch.load(label_path)
        # label = imread(label_path)
        # h, w = label.shape[:2]
        #
        # if min(h, w) < self.patch_size:
        #     # print(h, w)
        #     factor = ceil(float(self.patch_size) / float(min(h, w)))
        #     label = cv2.resize(label, dsize=(h * factor, w * factor))
        #     # h, w = label.shape[:2]
        #     # print(h, w)
        #
        #
        # label_tensor = self.transform(label)
        # c = label_tensor.shape[1]
        # if c > 3:
        #     label_tensor = label_tensor[:3, :, :]
        #
        #
        #
        # if self.is_gray == True:
        #     label_tensor = rgb2gray(label_tensor, ImageOption.Formal.CHW)

        return label_tensor, label_tensor, filename

class Deblur5144(Dataset):
    def __init__(self, data_dir, file_format, csv_filename, padmode='replicate'):
        super(Deblur5144, self).__init__()
        self.file_format = file_format
        self.padmode = padmode

        self.blur_replicate, self.blur_circular, self.kernel, self.sharp, self.noise, self.sigma = self.load_csv(data_dir, csv_filename)

        self.compose = tvf.Compose([
            tvf.ToTensor()
        ])

    def __len__(self):
        return len(self.sharp)

    # def load_csv(self, data_path, csv_filename):
    #     csv_file = os.path.join(data_path, csv_filename)
    #
    #     if not os.path.exists(csv_file):
    #         sharp = []
    #         blur_replicate = []
    #         blur_circular = []
    #         noise = []
    #         sigma = []
    #         kernel = []
    #         for ext in self.file_format:
    #             sharp_path = os.path.join(data_path, 'sharp', ext)
    #             sharp += glob.glob(sharp_path)
    #
    #         for sharppath in sharp:
    #             filename, ext = os.path.splitext(sharppath.split('/')[-1])
    #             noise.append(os.path.join(data_path, 'noise', filename + '.pt'))
    #             sigma.append(os.path.join(data_path, 'sigma', filename + '.pt'))
    #             kernel.append(os.path.join(data_path, 'kernel', filename + '.pt'))
    #             blur_circular.append(os.path.join(data_path, 'blur_circular', filename + ext))
    #             blur_replicate.append(os.path.join(data_path, 'blur_replicate', filename + ext))
    #         trainset_info = pd.DataFrame({
    #             'sharp': sharp,
    #             'blur_replicate': blur_replicate,
    #             'blur_circular': blur_circular,
    #             'noise': noise,
    #             'sigma': sigma,
    #             'kernel': kernel
    #         })
    #
    #         trainset_info.to_csv(csv_file, index=False)
    #
    #     else:
    #         trainset_info = pd.read_csv(csv_file)
    #         sharp = trainset_info['sharp'].tolist()
    #         blur_replicate = trainset_info['blur_replicate'].tolist()
    #         blur_circular = trainset_info['blur_circular'].tolist()
    #         noise = trainset_info['noise'].tolist()
    #         sigma = trainset_info['sigma'].tolist()
    #         kernel = trainset_info['kernel'].tolist()
    #
    #     # sharp = sharp * self.repeat
    #     return blur_replicate, blur_circular, kernel, sharp, noise, sigma
    def load_csv(self, data_path, csv_filename):
        csv_file = os.path.join(data_path, csv_filename)

        if not os.path.exists(csv_file):
            sharp = []
            blur_replicate = []
            blur_circular = []
            noise = []
            sigma = []

            kernels = glob.glob(os.path.join(data_path,  'kernel', '*.pt'))
            kernel_dict = {}

            for kernel_path in kernels:
                cur_k = torch.load(kernel_path)
                ksize = cur_k.shape[-1]
                if not kernel_dict.__contains__(ksize):
                    kernel_dict[ksize] = [kernel_path]
                else:
                    kernel_dict[ksize].append(kernel_path)

            kernel = []

            for key in kernel_dict.keys():
                print(len(kernel_dict[key]))

            for key in kernel_dict.keys():
                kernel = kernel + kernel_dict[key]
            # print(kernel)
            for kernelpath in kernel:
                filename, _ = os.path.splitext(kernelpath.split('/')[-1])
                noise.append(os.path.join(data_path, 'noise', filename + '.pt'))
                sigma.append(os.path.join(data_path, 'sigma', filename + '.pt'))

                for ext in self.file_format:
                    ext = ext[1:]
                    sharppath = os.path.join(data_path, 'sharp', filename + ext)
                    if os.path.exists(sharppath):
                        sharp.append(sharppath)
                        blur_circular.append(os.path.join(data_path, 'blur_circular', filename + ext))
                        blur_replicate.append(os.path.join(data_path, 'blur_replicate', filename + ext))

            trainset_info = pd.DataFrame({
                'sharp': sharp,
                'blur_replicate': blur_replicate,
                'blur_circular': blur_circular,
                'noise': noise,
                'sigma': sigma,
                'kernel': kernel
            })
            trainset_info.to_csv(csv_file, index=False)

        else:
            trainset_info = pd.read_csv(csv_file)
            sharp = trainset_info['sharp'].tolist()
            blur_replicate = trainset_info['blur_replicate'].tolist()
            blur_circular = trainset_info['blur_circular'].tolist()
            noise = trainset_info['noise'].tolist()
            sigma = trainset_info['sigma'].tolist()
            kernel = trainset_info['kernel'].tolist()

        return blur_replicate, blur_circular, kernel, sharp, noise , sigma


    def __getitem__(self, item):

        sharp_path = self.sharp[item]
        filename = sharp_path.split(os.sep)[-1]
        sharp = self.compose(imread(sharp_path))
        kernel = torch.load(self.kernel[item]).float()
        blurs = self.blur_circular if self.padmode == 'circular' else self.blur_replicate #'replicate'
        blur = self.compose(imread(blurs[item]))
        noise = torch.load(self.noise[item]).float() / 255.
        # sigma = torch.load(self.sigma[item])
        return blur, kernel, noise, sharp, filename



class LevinDataset(Dataset):
    def __init__(self, data_dir, sigma, csv_filename):
        super(LevinDataset, self).__init__()
        self.data_dir = data_dir
        self.imgmat = self.load_csv(data_dir, csv_filename)
        self.sigma = sigma

    def __len__(self):
        return len(self.imgmat)

    @torch.no_grad()
    def __getitem__(self, index):
        filepath = self.imgmat[index]
        matdata = io.loadmat(filepath)

        label = matdata['x']
        blur = matdata['y']
        kernel = matdata['f']

        blur = torch.from_numpy(blur).type(torch.float32).unsqueeze(0)
        label = torch.from_numpy(label).type(torch.float32).unsqueeze(0)
        kernel = torch.from_numpy(kernel).type(torch.float32).unsqueeze(0)
        kernel = torch.rot90(kernel, k=2, dims=(-2, -1))
        noise = self.sigma * torch.randn_like(blur)
        blur = blur + noise

        return blur, kernel, label, filepath.split('/')[-1]

    def load_csv(self, data_dir, csv_filename):
        csv_file = os.path.join(data_dir, csv_filename)
        if not os.path.exists(csv_file):
           img_mats = glob.glob(os.path.join(data_dir, 'im*.mat'))

           trainset_info = pd.DataFrame({
               'imgmat': img_mats
           })
           trainset_info.to_csv(csv_file, index=False)

        else:
            trainset_info = pd.read_csv(csv_file)
            img_mats = trainset_info['imgmat'].tolist()
        return img_mats


# class LevinEDDataset(Dataset):
#     def __init__(self, data_dir, sigma, csv_filename):
#         super(LevinEDDataset, self).__init__()
#         self.data_dir = data_dir
#         self.imgmat = self.load_csv(data_dir, csv_filename)
#         self.sigma = sigma
#
#     def __len__(self):
#         return len(self.imgmat)
#
#     @torch.no_grad()
#     def __getitem__(self, index):
#         filepath = self.imgmat[index]
#         matdata = io.loadmat(filepath)
#
#         label = matdata['x']
#         blur = matdata['y']
#         kernel = matdata['f']
#
#         blur = torch.from_numpy(blur).type(torch.float32).unsqueeze(0)
#         label = torch.from_numpy(label).type(torch.float32).unsqueeze(0)
#         kernel = torch.from_numpy(kernel).type(torch.float32).unsqueeze(0)
#         kernel = torch.rot90(kernel, k=2, dims=(-2, -1))
#         noise = self.sigma * torch.randn_like(blur)
#         blur = blur + noise
#
#         return blur, kernel, label, filepath.split('/')[-1]
#
#     def load_csv(self, data_dir, csv_filename):
#         csv_file = os.path.join(data_dir, csv_filename)
#         if not os.path.exists(csv_file):
#            img_mats = glob.glob(os.path.join(data_dir, 'im*.mat'))
#
#            trainset_info = pd.DataFrame({
#                'imgmat': img_mats
#            })
#            trainset_info.to_csv(csv_file, index=False)
#
#         else:
#             trainset_info = pd.read_csv(csv_file)
#             img_mats = trainset_info['imgmat'].tolist()
#         return img_mats

class TDeblurAEDataset(Dataset):
    def __init__(self, data_dir, file_format, csv_filename, padmode='replicate'):
        super(TDeblurAEDataset, self).__init__()
        self.file_format = file_format
        self.padmode = padmode

        self.blur_replicate, self.blur_circular, self.kernel, self.sharp, self.noise, self.sigma = self.load_csv(
            data_dir, csv_filename)

        self.compose = tvf.Compose([
            tvf.ToTensor()
        ])

    def __len__(self):
        return len(self.sharp)

    def __getitem__(self, item):

        noise_path = self.noise[item]
        noise = torch.load(noise_path).type(torch.float32) / 255.

        blurs = self.blur_circular if self.padmode == 'circular' else self.blur_replicate  # 'replicate'
        blur = self.compose(imread(blurs[item]))

        return blur, noise, [blur, noise, noise, blur - noise], blurs[item].split('/')[-1]

    def load_csv(self, data_path, csv_filename):
        csv_file = os.path.join(data_path, csv_filename)

        if not os.path.exists(csv_file):
            sharp = []
            blur_replicate = []
            blur_circular = []
            noise = []
            sigma = []

            kernels = glob.glob(os.path.join(data_path,  'kernel', '*.pt'))
            kernel_dict = {}

            for kernel_path in kernels:
                cur_k = torch.load(kernel_path)
                ksize = cur_k.shape[-1]
                if not kernel_dict.__contains__(ksize):
                    kernel_dict[ksize] = [kernel_path]
                else:
                    kernel_dict[ksize].append(kernel_path)

            kernel = []

            for key in kernel_dict.keys():
                print(len(kernel_dict[key]))

            for key in kernel_dict.keys():
                kernel = kernel + kernel_dict[key]
            # print(kernel)
            for kernelpath in kernel:
                filename, _ = os.path.splitext(kernelpath.split('/')[-1])
                noise.append(os.path.join(data_path, 'noise', filename + '.pt'))
                sigma.append(os.path.join(data_path, 'sigma', filename + '.pt'))

                for ext in self.file_format:
                    ext = ext[1:]
                    sharppath = os.path.join(data_path, 'sharp', filename + ext)
                    if os.path.exists(sharppath):
                        sharp.append(sharppath)
                        blur_circular.append(os.path.join(data_path, 'blur_circular', filename + ext))
                        blur_replicate.append(os.path.join(data_path, 'blur_replicate', filename + ext))

            trainset_info = pd.DataFrame({
                'sharp': sharp,
                'blur_replicate': blur_replicate,
                'blur_circular': blur_circular,
                'noise': noise,
                'sigma': sigma,
                'kernel': kernel
            })
            trainset_info.to_csv(csv_file, index=False)

        else:
            trainset_info = pd.read_csv(csv_file)
            sharp = trainset_info['sharp'].tolist()
            blur_replicate = trainset_info['blur_replicate'].tolist()
            blur_circular = trainset_info['blur_circular'].tolist()
            noise = trainset_info['noise'].tolist()
            sigma = trainset_info['sigma'].tolist()
            kernel = trainset_info['kernel'].tolist()

        return blur_replicate, blur_circular, kernel, sharp, noise, sigma

class VBlurAEDataset(Dataset):
    def __init__(self, data_dir, csv_filename, file_format, kernel_type, sigma, is_gray=True, **kwargs):
        super(VBlurAEDataset, self).__init__()
        self.data_format = file_format
        self.n2t = tv.transforms.ToTensor()
        self.kernels, self.sharps = self.load_csv(data_dir, kernel_type, csv_filename, file_format)
        self.sigma = sigma
        self.is_gray = is_gray

    def load_kernel(self, filename):
        ext = filename.split('.')[-1]
        if ext == 'mat':
            kernel = io.loadmat(filename)['kernel'].astype(np.float32)
            kernel = self.n2t(np.expand_dims(kernel, axis=-1))
            kernel /= torch.sum(kernel)
        else:
            kernel = imread(filename)
            kernel = self.n2t(kernel)
            kernel = rgb2gray(kernel, ImageOption.Formal.CHW)

        return kernel

    def load_csv(self, data_dir, kernel_type, csv_filename, data_format):
        csv_file = os.path.join(data_dir, csv_filename)
        if not os.path.exists(csv_file):
            tmp_sharps = []
            tmp_kernels = []
            kernels = []
            sharps = []
            for ext in data_format:
                kernel_path = os.path.join(data_dir, kernel_type, ext)
                tmp_kernels += glob.glob(kernel_path)
                sharp_path = os.path.join(data_dir, 'sharp', ext)
                tmp_sharps += glob.glob(sharp_path)

            for sharp in tmp_sharps:
                for kernel in tmp_kernels:
                    sharps.append(sharp)
                    kernels.append(kernel)

            del tmp_sharps, tmp_kernels

            trainset_info = pd.DataFrame({
                'sharp': sharps,
                'kernel': kernels
            })

            trainset_info.to_csv(csv_file, index=False)

        else:
            trainset_info = pd.read_csv(csv_file)
            sharps = trainset_info['sharp'].tolist()
            kernels = trainset_info['kernel'].tolist()
        return kernels, sharps

    def __len__(self):
        return len(self.sharps)

    def __getitem__(self, item):
        sharp_filename = self.sharps[item]
        kernel_filename = self.kernels[item]
        sharp = self.n2t(imread(sharp_filename))
        if self.is_gray == True:
            sharp = rgb2gray(sharp, ImageOption.Formal.CHW)
        kernel = self.load_kernel(kernel_filename)
        with torch.no_grad():

            blur = imfilter(sharp.unsqueeze(0), kernel, padding_mode='replicate', filter_mode='conv').squeeze(0)
            noise = self.sigma * torch.randn_like(blur)
            blur = blur + noise
            blur = torch.clip(blur, 0, 1).detach_()
        # blur, kernel, sharp, sharp_filename.split(os.sep)[-1]
        return blur, noise, [blur, noise, noise, blur - noise], sharp_filename.split(os.sep)[-1]


class BlurEDDataset(Dataset):
    def __init__(self, patch_size, data_dir, csv_filename, file_format, min_psf_size = 15, max_psf_size=38, batch_size=8, **kwargs):
        super(BlurEDDataset, self).__init__()
        # self.batch_size = batch_size
        self.data_path = data_dir
        # csv = csv_filename.split('.')[0]
        # ext = csv_filename.split('.')[-1]
        self.csv_filename = csv_filename# csv + '_k{}-{}'.format(min_psf_size, max_psf_size) + ext

        self.max_psf_size = max_psf_size
        self.min_psf_size = min_psf_size
        self.batch_size = batch_size
        self.file_format = file_format
        self.hflip = tv.transforms.RandomHorizontalFlip(p=0.6)
        self.vflip = tv.transforms.RandomVerticalFlip(p=0.6)
        self.n2t = tv.transforms.ToTensor()
        # self.crop = tv.transforms.RandomCrop(size=patch_size)

        self.sharps = self.load_csv(data_dir, self.csv_filename)
        self.patch_size = patch_size


    def load_csv(self, data_path, csv_filename):
        csv_file = os.path.join(data_path, csv_filename)

        if not os.path.exists(csv_file):
            sharp = []
            for i in self.file_format:
                sharp_path = os.path.join(data_path, 'sharp', self.file_format)[0:10000]
                tmp = glob.glob(sharp_path)
                sharp = sharp + tmp

            trainset_info = pd.DataFrame({
                'sharp': sharp,
            })
            trainset_info.to_csv(csv_file, index=False)

        else:
            trainset_info = pd.read_csv(csv_file)

            sharp = trainset_info['sharp'].tolist()[0:10000]

        return sharp

    def __len__(self):
        return len(self.sharps)

    def __getitem__(self, item):
        n_sharp = len(self)
        sharp_filename = self.sharps[item]
        sharp = self.n2t(imread(sharp_filename))
        # blur = self.n2t(imread(blur_filename))
        # grid_size = randrange(self.min_psf_size, self.max_psf_size)

        grid = (item // self.batch_size) % (self.max_psf_size - self.min_psf_size)
        grid += self.min_psf_size
        kpath = os.path.join(self.data_path, 'kernel_mat', 'kernel_g{}.mat'.format(grid))
        kernel = io.loadmat(kpath)['kernel']
        idx_kernel = randrange(0, 4999)
        kernel = torch.from_numpy(kernel[:, :, idx_kernel])

        c, h, w = sharp.shape

        if c == 1:
            sharp = sharp.repeat([3, 1, 1])
            # blur = blur.repeat([3, 1, 1])
        real_ksize = kernel.shape[-1]

        crop_size = self.patch_size + real_ksize // 2 * 2

        if h < crop_size:
            factor = float(crop_size) / float(h)
            sharp = F.interpolate(sharp.unsqueeze(0), scale_factor=factor).detach().squeeze(0)
        if w < crop_size:
            factor = float(crop_size) / float(w)
            sharp = F.interpolate(sharp.unsqueeze(0), scale_factor=factor).detach().squeeze(0)


        # patch = self.crop(t.cat([blur, sharp], dim=0))
        crop = tv.transforms.RandomCrop(size=(self.patch_size + real_ksize // 2 * 2))
        patch = crop(sharp) # self.crop(t.cat([sharp], dim=0))
        sharp = self.vflip(self.hflip(patch))

        sigma = randrange(0, 6) * 0.01

        with torch.no_grad():
            blur = imfilter(sharp.unsqueeze(0), kernel.unsqueeze(0), padding_mode='valid').squeeze(0)
            blur = blur + sigma * torch.randn_like(blur)
            blur = blur.mul(255.).round().div(255.)

        return blur, blur, sharp_filename.split(os.sep)[-1]



class OutlineDataset(Dataset):
    def __init__(self, data_dir, csv_filename, file_format, sigma_max, is_pretrain=True):
        super(OutlineDataset, self).__init__()
        self.data_dir = data_dir
        self.csv_filename = csv_filename
        self.file_format = file_format
        self.sigma_max = sigma_max
        self.is_pretrain = is_pretrain
        self.transform = tvf.Compose([
            tvf.ToTensor()
        ])
        # self.batch_size = batch_size

        self.blurs, self.kernels, self.sharps = self.load_csv(data_dir, csv_filename, file_format)

    def load_csv(self, data_dir, csv_filename, data_format):
        csv_file = os.path.join(data_dir, csv_filename)
        if not os.path.exists(csv_file):
            tmp_sharps = []

            kernels = []
            sharps = []
            blurs = []
            blurs_unclip = []
            for ext in data_format:
                sharp_path = os.path.join(data_dir, 'sharp', ext)
                tmp_sharps += glob.glob(sharp_path)

            n_sharp = len(tmp_sharps)

            for i in range(n_sharp):
                sharps.append(os.path.join(data_dir, 'sharp', '{:>05}.png'.format(i)))
                blurs.append(os.path.join(data_dir, 'blur', '{:>05}.png'.format(i)))
                # blurs_unclip.append(os.path.join(data_dir, 'blur_unclip', '{:>05}.pt'.format(i)))
                kernels.append(os.path.join(data_dir, 'kernel', '{:>05}.pt'.format(i)))


            trainset_info = pd.DataFrame({
                'sharp': sharps,
                'kernel': kernels,
                'blur': blurs,
                # 'blur_unclip': blurs_unclip
            })

            trainset_info.to_csv(csv_file, index=False)

        else:
            trainset_info = pd.read_csv(csv_file)
            sharps = trainset_info['sharp'].tolist()[0:10000]
            kernels = trainset_info['kernel'].tolist()[0:10000]
            blurs = trainset_info['blur'].tolist()[0:10000]
            # blurs_unclip = trainset_info['blur_unclip'].tolist()[0:10000]
        return blurs, kernels, sharps

    def __len__(self):
        return len(self.sharps)

    def random_patch(self, blur, sharp, kernel):
        random_num_h = torch.rand(1)
        random_num_v = torch.rand(1)
        # tvf.RandomVerticalFlip
        if random_num_h > 0.5:
            blur = tvfF.hflip(blur)
            sharp = tvfF.hflip(sharp)
            kernel = tvfF.hflip(kernel)

        if random_num_v > 0.5:
            blur = tvfF.vflip(blur)
            sharp = tvfF.vflip(sharp)
            kernel = tvfF.vflip(kernel)

        return blur, sharp, kernel


    def __getitem__(self, item):
        filename = self.blurs[item].split('/')[-1]
        blur = self.transform(imread(self.blurs[item]))

        # rand = random()
        # if rand >= 0.9: rand = 1.
        # sigma = self.sigma_max * rand
        # light_factor = [1.2, 1.3, 1.4]
        # idx = randrange(0, 3)
        sigma = self.sigma_max

        blur = torch.clip(sigma * torch.randn_like(blur) + blur, 0, 1)
        blur = torch.round(blur * 255) / 255
        sharp = self.transform(imread(self.sharps[item]))
        kernel = torch.load(self.kernels[item])

        if self.is_pretrain:
            blur_unclip = torch.load(self.blurs_unclip[item])
            blur_unclip = sigma * torch.randn_like(blur_unclip) + blur_unclip

            return blur, [blur, blur_unclip], filename



        # return blur, kernel, (blur_unclip, sharp), filename
        return blur, kernel, sharp, filename


class OutlierTestDataset(Dataset):
    def __init__(self, data_dir, csv_filename, file_format, sigma_max, is_pretrain=True):
        super(OutlierTestDataset, self).__init__()
        self.data_dir = data_dir
        self.csv_filename = csv_filename
        self.file_format = file_format
        self.sigma_max = sigma_max
        self.is_pretrain = is_pretrain
        self.transform = tvf.Compose([
            tvf.ToTensor()
        ])
        self.kernel_type = self.data_dir.split('/')[-2]
        self.blurs, self.kernels, self.sharps = self.load_csv(data_dir, csv_filename, file_format)

    def load_csv(self, data_dir, csv_filename, data_format):
        csv_file = os.path.join(data_dir, csv_filename)
        if not os.path.exists(csv_file):
            tmp_sharps = []

            kernels = []
            blurs = []


            for ext in data_format:
                sharp_path = os.path.join(data_dir, 'sharp', ext)
                tmp_sharps += glob.glob(sharp_path)

            sharps = tmp_sharps
            n_sharp = len(sharps)
            kernel_ext = 'mat' if self.kernel_type == 'levin' else 'png'
            for i in range(n_sharp):
                sharp = sharps[i]
                filename = sharp.split('/')[-1].split('.')[0]
                img_ext = sharp.split('/')[-1].split('.')[-1]

                kernel = sharp.split('_')[-2] + '_' + sharp.split('_')[-1]
                kernel = kernel.split('.')[0]
                blurs.append(os.path.join(data_dir, 'blur', filename + '.{}'.format(img_ext)))
                kernels.append(os.path.join(data_dir, 'kernel', kernel + '.{}'.format(kernel_ext)))

            trainset_info = pd.DataFrame({
                'sharp': sharps,
                'kernel': kernels,
                'blur': blurs
            })

            trainset_info.to_csv(csv_file, index=False)

        else:
            trainset_info = pd.read_csv(csv_file)
            sharps = trainset_info['sharp'].tolist()#[0: 200]
            kernels = trainset_info['kernel'].tolist()#[0: 200]
            blurs = trainset_info['blur'].tolist()#[0: 200]
        return blurs, kernels, sharps

    def __len__(self):
        return len(self.sharps)

    # def robust(self, x, a=0.0):
    #     a = 459 / sqrt(2 * math.pi)
    #     b = -2601 / 2
    #     return 1 -  1 / (1 + a * torch.exp(b * x.pow(2)))

    def robust(self, x, a=0.0):
        a = 0.039#459 / sqrt(2 * math.pi)
        # b = -2601 / 2
        return 1 - torch.tanh(F.softshrink(x, a).div(a).pow(2))#1 -  1 / (1 + a * torch.exp(b * x.pow(2)))

    def __getitem__(self, item):
        filename = self.blurs[item].split('/')[-1]
        blur = self.transform(imread(self.blurs[item]))

        # rand = random()
        # if rand >= 0.9: rand = 1.
        # sigma = self.sigma_max * rand
        sigma = self.sigma_max
        blur = torch.clip(sigma * torch.randn_like(blur) + blur, 0, 1)
        blur = torch.round(blur * 255) / 255
        sharp = self.transform(imread(self.sharps[item]))
        ext = self.kernels[item].split('.')[-1]

        if ext == 'mat':
            kernel = io.loadmat(self.kernels[item])['kernel'].astype(np.float32)
            kernel = torch.from_numpy(kernel).unsqueeze(0)
        else:
            kernel = rgb2gray(self.transform(imread(self.kernels[item])), data_format=ImageOption.Formal.CHW)

        kernel = kernel / kernel.sum()
########################
        reb = imfilter(sharp.unsqueeze(0), kernel, filter_mode='conv', padding_mode='replicate').squeeze(0)
        weight = self.robust(reb - blur)

        return blur, kernel, sharp, filename



class DeblurMaskDataset(Dataset):
    def __init__(self, data_dir, csv_filename, file_format, sigma_max, is_pretrain=True):
        super(DeblurMaskDataset, self).__init__()
        self.data_dir = data_dir
        self.csv_filename = csv_filename
        self.file_format = file_format
        self.sigma_max = sigma_max
        self.is_pretrain = is_pretrain
        self.transform = tvf.Compose([
            tvf.ToTensor()
        ])
        # self.batch_size = batch_size

        self.blurs, self.blurs_unclip, self.kernels, self.sharps = self.load_csv(data_dir, csv_filename, file_format)

    def load_csv(self, data_dir, csv_filename, data_format):
        csv_file = os.path.join(data_dir, csv_filename)
        if not os.path.exists(csv_file):
            tmp_sharps = []

            kernels = []
            sharps = []
            blurs = []
            blurs_unclip = []
            for ext in data_format:
                sharp_path = os.path.join(data_dir, 'sharp', ext)
                tmp_sharps += glob.glob(sharp_path)

            n_sharp = len(tmp_sharps)

            for i in range(n_sharp):
                sharps.append(os.path.join(data_dir, 'sharp', '{:>05}.png'.format(i)))
                blurs.append(os.path.join(data_dir, 'blur', '{:>05}.png'.format(i)))
                blurs_unclip.append(os.path.join(data_dir, 'blur_unclip', '{:>05}.pt'.format(i)))
                kernels.append(os.path.join(data_dir, 'kernel', '{:>05}.pt'.format(i)))


            trainset_info = pd.DataFrame({
                'sharp': sharps,
                'kernel': kernels,
                'blur': blurs,
                'blur_unclip': blurs_unclip
            })

            trainset_info.to_csv(csv_file, index=False)

        else:
            trainset_info = pd.read_csv(csv_file)
            sharps = trainset_info['sharp'].tolist()[0:10000]
            kernels = trainset_info['kernel'].tolist()[0:10000]
            blurs = trainset_info['blur'].tolist()[0:10000]
            blurs_unclip = trainset_info['blur_unclip'].tolist()[0:10000]
        return blurs, blurs_unclip, kernels, sharps

    def __len__(self):
        return len(self.sharps)

    def random_patch(self, blur, sharp, kernel):
        random_num_h = torch.rand(1)
        random_num_v = torch.rand(1)
        # tvf.RandomVerticalFlip
        if random_num_h > 0.5:
            blur = tvfF.hflip(blur)
            sharp = tvfF.hflip(sharp)
            kernel = tvfF.hflip(kernel)

        if random_num_v > 0.5:
            blur = tvfF.vflip(blur)
            sharp = tvfF.vflip(sharp)
            kernel = tvfF.vflip(kernel)

        return blur, sharp, kernel


    def __getitem__(self, item):
        filename = self.blurs[item].split('/')[-1]
        blur = self.transform(imread(self.blurs[item]))

        # rand = random()
        # if rand >= 0.9: rand = 1.
        # sigma = self.sigma_max * rand
        sigma = self.sigma_max
        blur = torch.clip(sigma * torch.randn_like(blur) + blur, 0, 1)
        blur = torch.round(blur * 255) / 255
        sharp = self.transform(imread(self.sharps[item]))
        kernel = torch.load(self.kernels[item])
        # blur, sharp, kernel = self.random_patch(blur, sharp, kernel)
        if self.is_pretrain:
            blur_unclip = torch.load(self.blurs_unclip[item])
            blur_unclip = sigma * torch.randn_like(blur_unclip) + blur_unclip
            # blur = self.transform(imread(self.blurs[item]))
            # blur_unclip = torch.load(self.blurs_unclip[item])
            return blur, [blur, blur_unclip], filename

        # reb = imfilter(sharp.unsqueeze(0), kernel, padding_mode='replicate', filter_mode='conv').squeeze(0)#.clip(0, 1).mul(255.).round().div(255.)


        # return blur, kernel, (blur_unclip, sharp), filename
        return blur, kernel, blur, filename


class DeblurMaskDataset2(Dataset):
    def __init__(self, data_dir, csv_filename, file_format, sigma_max, is_pretrain=True):
        super(DeblurMaskDataset2, self).__init__()
        self.data_dir = data_dir
        self.csv_filename = csv_filename
        self.file_format = file_format
        self.sigma_max = sigma_max
        self.is_pretrain = is_pretrain
        self.transform = tvf.Compose([
            tvf.ToTensor()
        ])
        # self.batch_size = batch_size

        self.blurs, self.blurs_unclip, self.kernels, self.sharps = self.load_csv(data_dir, csv_filename, file_format)

    def load_csv(self, data_dir, csv_filename, data_format):
        csv_file = os.path.join(data_dir, csv_filename)
        if not os.path.exists(csv_file):
            tmp_sharps = []

            kernels = []
            sharps = []
            blurs = []
            blurs_unclip = []
            for ext in data_format:
                sharp_path = os.path.join(data_dir, 'sharp', ext)
                tmp_sharps += glob.glob(sharp_path)

            n_sharp = len(tmp_sharps)

            for i in range(n_sharp):
                sharps.append(os.path.join(data_dir, 'sharp', '{:>05}.png'.format(i)))
                blurs.append(os.path.join(data_dir, 'blur', '{:>05}.png'.format(i)))
                blurs_unclip.append(os.path.join(data_dir, 'blur_unclip', '{:>05}.pt'.format(i)))
                kernels.append(os.path.join(data_dir, 'kernel', '{:>05}.pt'.format(i)))


            trainset_info = pd.DataFrame({
                'sharp': sharps,
                'kernel': kernels,
                'blur': blurs,
                'blur_unclip': blurs_unclip
            })

            trainset_info.to_csv(csv_file, index=False)

        else:
            trainset_info = pd.read_csv(csv_file)
            sharps = trainset_info['sharp'].tolist()[0:10000]
            kernels = trainset_info['kernel'].tolist()[0:10000]
            blurs = trainset_info['blur'].tolist()[0:10000]
            blurs_unclip = trainset_info['blur_unclip'].tolist()[0:10000]
        return blurs, blurs_unclip, kernels, sharps

    def __len__(self):
        return len(self.sharps)

    def random_patch(self, blur, sharp, kernel):
        random_num_h = torch.rand(1)
        random_num_v = torch.rand(1)
        # tvf.RandomVerticalFlip
        if random_num_h > 0.5:
            blur = tvfF.hflip(blur)
            sharp = tvfF.hflip(sharp)
            kernel = tvfF.hflip(kernel)

        if random_num_v > 0.5:
            blur = tvfF.vflip(blur)
            sharp = tvfF.vflip(sharp)
            kernel = tvfF.vflip(kernel)

        return blur, sharp, kernel


    def __getitem__(self, item):
        filename = self.blurs[item].split('/')[-1]
        blur = self.transform(imread(self.blurs[item]))

        # rand = random()
        # if rand >= 0.9: rand = 1.
        # sigma = self.sigma_max * rand
        sigma = self.sigma_max
        blur = torch.clip(sigma * torch.randn_like(blur) + blur, 0, 1)
        blur = torch.round(blur * 255) / 255
        sharp = self.transform(imread(self.sharps[item]))
        kernel = torch.load(self.kernels[item])
        # blur, sharp, kernel = self.random_patch(blur, sharp, kernel)
        if self.is_pretrain:
            blur_unclip = torch.load(self.blurs_unclip[item])
            blur_unclip = sigma * torch.randn_like(blur_unclip) + blur_unclip
            # blur = self.transform(imread(self.blurs[item]))
            # blur_unclip = torch.load(self.blurs_unclip[item])
            return blur, [blur, blur_unclip], filename

        reb = imfilter(sharp.unsqueeze(0), kernel, padding_mode='replicate', filter_mode='conv').squeeze(0)#.clip(0, 1).mul(255.).round().div(255.)


        # return blur, kernel, (blur_unclip, sharp), filename
        return blur, kernel, [blur, reb], filename


class DeblurMaskEDDataset(Dataset):
    def __init__(self, data_dir, csv_filename, file_format, sigma_max, is_pretrain=True):
        super(DeblurMaskEDDataset, self).__init__()
        self.data_dir = data_dir
        self.csv_filename = csv_filename
        self.file_format = file_format
        self.sigma_max = sigma_max
        self.is_pretrain = is_pretrain
        self.transform = tvf.Compose([
            tvf.ToTensor()
        ])
        # self.batch_size = batch_size

        self.blurs, self.kernels, self.sharps = self.load_csv(data_dir, csv_filename, file_format)

    def load_csv(self, data_dir, csv_filename, data_format):
        csv_file = os.path.join(data_dir, csv_filename)
        if not os.path.exists(csv_file):
            tmp_sharps = []

            kernels = []
            sharps = []
            blurs = []
            blurs_unclip = []
            for ext in data_format:
                sharp_path = os.path.join(data_dir, 'sharp', ext)
                tmp_sharps += glob.glob(sharp_path)

            n_sharp = len(tmp_sharps)

            for i in range(n_sharp):
                sharps.append(os.path.join(data_dir, 'sharp', '{:>05}.png'.format(i)))
                blurs.append(os.path.join(data_dir, 'blur', '{:>05}.png'.format(i)))
                # blurs_unclip.append(os.path.join(data_dir, 'blur_unclip', '{:>05}.pt'.format(i)))
                kernels.append(os.path.join(data_dir, 'kernel', '{:>05}.pt'.format(i)))


            trainset_info = pd.DataFrame({
                'sharp': sharps,
                'kernel': kernels,
                'blur': blurs,
                # 'blur_unclip': blurs_unclip
            })

            trainset_info.to_csv(csv_file, index=False)

        else:
            trainset_info = pd.read_csv(csv_file)
            sharps = trainset_info['sharp'].tolist()[0:10000]
            kernels = trainset_info['kernel'].tolist()[0:10000]
            blurs = trainset_info['blur'].tolist()[0:10000]
            # blurs_unclip = trainset_info['blur_unclip'].tolist()[0:10000]
        return blurs, kernels, sharps

    def __len__(self):
        return len(self.sharps)

    def random_patch(self, blur, sharp, kernel):
        random_num_h = torch.rand(1)
        random_num_v = torch.rand(1)
        # tvf.RandomVerticalFlip
        if random_num_h > 0.5:
            blur = tvfF.hflip(blur)
            sharp = tvfF.hflip(sharp)
            kernel = tvfF.hflip(kernel)

        if random_num_v > 0.5:
            blur = tvfF.vflip(blur)
            sharp = tvfF.vflip(sharp)
            kernel = tvfF.vflip(kernel)

        return blur, sharp, kernel


    def __getitem__(self, item):
        filename = self.blurs[item].split('/')[-1]
        blur = self.transform(imread(self.blurs[item]))
        sharp = self.transform(imread(self.sharps[item]))
        # ls = [1.2, 1.3, 1.4]
        # idx = randrange(0, 3)
        # factor = ls[idx]
        # blur = blur * factor / 1.2
        # sharp = sharp * factor / 1.2
        # blur = blur.clip(0, 1).mul(255).round().div(255)
        # sharp = sharp.clip(0, 1).mul(255).round().div(255)
        # rand = random()
        # if rand >= 0.9: rand = 1.
        # sigma = self.sigma_max * rand
        sigma = self.sigma_max
        blur = torch.clip(sigma * torch.randn_like(blur) + blur, 0, 1)
        blur = torch.round(blur * 255) / 255

        kernel = torch.load(self.kernels[item])
        # blur, sharp, kernel = self.random_patch(blur, sharp, kernel)
        if self.is_pretrain:
            blur_unclip = torch.load(self.blurs_unclip[item])
            blur_unclip = sigma * torch.randn_like(blur_unclip) + blur_unclip
            # blur = self.transform(imread(self.blurs[item]))
            # blur_unclip = torch.load(self.blurs_unclip[item])
            return blur, [blur, blur_unclip], filename

        reb = imfilter(sharp.unsqueeze(0), kernel, padding_mode='replicate', filter_mode='conv').squeeze(0)#.clip(0, 1).mul(255.).round().div(255.)


        # return blur, kernel, (blur_unclip, sharp), filename
        return blur, reb, [blur, reb], filename


class DeblurMaskDisstillDataset(Dataset):
    def __init__(self, data_dir, csv_filename, file_format, sigma_max, is_pretrain=True):
        super(DeblurMaskDisstillDataset, self).__init__()
        self.data_dir = data_dir
        self.csv_filename = csv_filename
        self.file_format = file_format
        self.sigma_max = sigma_max
        self.is_pretrain = is_pretrain
        self.transform = tvf.Compose([
            tvf.ToTensor()
        ])
        # self.batch_size = batch_size

        self.blurs, self.kernels, self.sharps = self.load_csv(data_dir, csv_filename, file_format)

    def load_csv(self, data_dir, csv_filename, data_format):
        csv_file = os.path.join(data_dir, csv_filename)
        if not os.path.exists(csv_file):
            tmp_sharps = []

            kernels = []
            sharps = []
            blurs = []
            blurs_unclip = []
            for ext in data_format:
                sharp_path = os.path.join(data_dir, 'sharp', ext)
                tmp_sharps += glob.glob(sharp_path)

            n_sharp = len(tmp_sharps)

            for i in range(n_sharp):
                sharps.append(os.path.join(data_dir, 'sharp', '{:>05}.png'.format(i)))
                blurs.append(os.path.join(data_dir, 'blur', '{:>05}.png'.format(i)))
                # blurs_unclip.append(os.path.join(data_dir, 'blur_unclip', '{:>05}.pt'.format(i)))
                kernels.append(os.path.join(data_dir, 'kernel', '{:>05}.pt'.format(i)))


            trainset_info = pd.DataFrame({
                'sharp': sharps,
                'kernel': kernels,
                'blur': blurs,
                'blur_unclip': blurs_unclip
            })

            trainset_info.to_csv(csv_file, index=False)

        else:
            trainset_info = pd.read_csv(csv_file)
            sharps = trainset_info['sharp'].tolist()[0:10000]
            kernels = trainset_info['kernel'].tolist()[0:10000]
            blurs = trainset_info['blur'].tolist()[0:10000]
            # blurs_unclip = trainset_info['blur_unclip'].tolist()[0:10000]
        return blurs, kernels, sharps

    def __len__(self):
        return len(self.sharps)

    def random_patch(self, blur, sharp, kernel):
        random_num_h = torch.rand(1)
        random_num_v = torch.rand(1)
        # tvf.RandomVerticalFlip
        if random_num_h > 0.5:
            blur = tvfF.hflip(blur)
            sharp = tvfF.hflip(sharp)
            kernel = tvfF.hflip(kernel)

        if random_num_v > 0.5:
            blur = tvfF.vflip(blur)
            sharp = tvfF.vflip(sharp)
            kernel = tvfF.vflip(kernel)

        return blur, sharp, kernel


    def __getitem__(self, item):
        filename = self.blurs[item].split('/')[-1]
        blur = self.transform(imread(self.blurs[item]))

        # rand = random()
        # if rand >= 0.9: rand = 1.
        # sigma = self.sigma_max * rand
        sigma = self.sigma_max
        # light_factor = [1.2, 1.3, 1.3]
        # idx = randrange(0, 3)
        # factor = light_factor[idx]
        # blur = blur.div(1.2).mul(factor).clip(0, 1).mul(255).round().div(255)
        blur = torch.clip(sigma * torch.randn_like(blur) + blur, 0, 1)
        blur = torch.round(blur * 255) / 255
        sharp = self.transform(imread(self.sharps[item]))

        # sharp = sharp.div(1.2).mul(factor).clip(0, 1).mul(255).round().div(255)

        kernel = torch.load(self.kernels[item])
        # blur, sharp, kernel = self.random_patch(blur, sharp, kernel)

        reb = imfilter(sharp.unsqueeze(0), kernel, padding_mode='replicate', filter_mode='conv').squeeze(0)#.clip(0, 1).mul(255.).round().div(255.)
        # return blur, kernel, (blur_unclip, sharp), filename
        return blur, kernel, [blur, reb], filename



class DeblurMaskTestDataset(Dataset):
    def __init__(self, data_dir, csv_filename, file_format, sigma_max, is_pretrain=False):
        super(DeblurMaskTestDataset, self).__init__()
        self.data_dir = data_dir
        self.csv_filename = csv_filename
        self.file_format = file_format
        self.sigma_max = sigma_max
        self.is_pretrain = is_pretrain
        self.transform = tvf.Compose([
            tvf.ToTensor()
        ])
        self.kernel_type = self.data_dir.split('/')[-2]
        self.blurs, self.kernels, self.sharps = self.load_csv(data_dir, csv_filename, file_format)

    def load_csv(self, data_dir, csv_filename, data_format):
        csv_file = os.path.join(data_dir, csv_filename)
        if not os.path.exists(csv_file):
            tmp_sharps = []

            kernels = []
            blurs = []


            for ext in data_format:
                sharp_path = os.path.join(data_dir, 'sharp', ext)
                tmp_sharps += glob.glob(sharp_path)

            sharps = tmp_sharps
            n_sharp = len(sharps)
            kernel_ext = 'mat' if self.kernel_type == 'levin' else 'png'
            for i in range(n_sharp):
                sharp = sharps[i]
                filename = sharp.split('/')[-1].split('.')[0]
                img_ext = sharp.split('/')[-1].split('.')[-1]

                kernel = sharp.split('_')[-2] + '_' + sharp.split('_')[-1]
                kernel = kernel.split('.')[0]
                blurs.append(os.path.join(data_dir, 'blur', filename + '.{}'.format(img_ext)))
                kernels.append(os.path.join(data_dir, 'kernel', kernel + '.{}'.format(kernel_ext)))

            trainset_info = pd.DataFrame({
                'sharp': sharps,
                'kernel': kernels,
                'blur': blurs
            })

            trainset_info.to_csv(csv_file, index=False)

        else:
            trainset_info = pd.read_csv(csv_file)
            sharps = trainset_info['sharp'].tolist()[0: 200]
            kernels = trainset_info['kernel'].tolist()[0: 200]
            blurs = trainset_info['blur'].tolist()[0: 200]
        return blurs, kernels, sharps

    def __len__(self):
        return len(self.sharps)

    def __getitem__(self, item):
        filename = self.blurs[item].split('/')[-1]
        blur = self.transform(imread(self.blurs[item]))

        # rand = random()
        # if rand >= 0.9: rand = 1.
        # sigma = self.sigma_max * rand
        sigma = self.sigma_max
        blur = torch.clip(sigma * torch.randn_like(blur) + blur, 0, 1)
        blur = torch.round(blur * 255) / 255
        sharp = self.transform(imread(self.sharps[item]))
        ext = self.kernels[item].split('.')[-1]

        if ext == 'mat':
            kernel = io.loadmat(self.kernels[item])['kernel'].astype(np.float32)
            kernel = torch.from_numpy(kernel).unsqueeze(0)
        else:
            kernel = rgb2gray(self.transform(imread(self.kernels[item])), data_format=ImageOption.Formal.CHW)

        kernel = kernel / kernel.sum()

        reb = imfilter(sharp.unsqueeze(0), kernel, padding_mode='replicate', filter_mode='conv').squeeze(
            0)  # .clip(0, 1).mul(255.).round().div(255.)
        # print('reb:', reb.shape)
        # return blur, kernel, (blur_unclip, sharp), filename
        return blur, kernel, [blur, reb], filename

        # return blur, kernel, (blur_unclip, sharp), filename
        # return blur, kernel, [blur, ], filename

# class DeblurMaskTestDataset(Dataset):
#     def __init__(self, data_dir, csv_filename, file_format, sigma_max, is_pretrain=False):
#         super(DeblurMaskTestDataset, self).__init__()
#         self.data_dir = data_dir
#         self.csv_filename = csv_filename
#         self.file_format = file_format
#         self.sigma_max = sigma_max
#         self.is_pretrain = is_pretrain
#         self.transform = tvf.Compose([
#             tvf.ToTensor()
#         ])
#         # self.batch_size = batch_size
#
#         self.blurs, self.blurs_unclip, self.kernels, self.sharps = self.load_csv(data_dir, csv_filename, file_format)
#
#     def load_csv(self, data_dir, csv_filename, data_format):
#         csv_file = os.path.join(data_dir, csv_filename)
#         if not os.path.exists(csv_file):
#             tmp_sharps = []
#
#             kernels = []
#             sharps = []
#             blurs = []
#             blurs_unclip = []
#             for ext in data_format:
#                 sharp_path = os.path.join(data_dir, 'sharp', ext)
#                 tmp_sharps += glob.glob(sharp_path)
#
#             n_sharp = len(tmp_sharps)
#
#             for i in range(n_sharp):
#                 sharps.append(os.path.join(data_dir, 'sharp', '{:>05}.png'.format(i)))
#                 blurs.append(os.path.join(data_dir, 'blur', '{:>05}.png'.format(i)))
#                 blurs_unclip.append(os.path.join(data_dir, 'blur_unclip', '{:>05}.pt'.format(i)))
#                 kernels.append(os.path.join(data_dir, 'kernel', '{:>05}.pt'.format(i)))
#
#
#             trainset_info = pd.DataFrame({
#                 'sharp': sharps,
#                 'kernel': kernels,
#                 'blur': blurs,
#                 'blur_unclip': blurs_unclip
#             })
#
#             trainset_info.to_csv(csv_file, index=False)
#
#         else:
#             trainset_info = pd.read_csv(csv_file)
#             sharps = trainset_info['sharp'].tolist()[0:1000]
#             kernels = trainset_info['kernel'].tolist()[0:1000]
#             blurs = trainset_info['blur'].tolist()[0:1000]
#             blurs_unclip = trainset_info['blur_unclip'].tolist()[0:1000]
#         return blurs, blurs_unclip, kernels, sharps
#
#     def __len__(self):
#         return len(self.sharps)
#
#     def random_patch(self, blur, sharp, kernel):
#         random_num_h = torch.rand(1)
#         random_num_v = torch.rand(1)
#         # tvf.RandomVerticalFlip
#         if random_num_h > 0.5:
#             blur = tvfF.hflip(blur)
#             sharp = tvfF.hflip(sharp)
#             kernel = tvfF.hflip(kernel)
#
#         if random_num_v > 0.5:
#             blur = tvfF.vflip(blur)
#             sharp = tvfF.vflip(sharp)
#             kernel = tvfF.vflip(kernel)
#
#         return blur, sharp, kernel
#
#
#     def __getitem__(self, item):
#         filename = self.blurs[item].split('/')[-1]
#         blur = self.transform(imread(self.blurs[item]))
#
#         # rand = random()
#         # if rand >= 0.9: rand = 1.
#         # sigma = self.sigma_max * rand
#         sigma = self.sigma_max
#         blur = torch.clip(sigma * torch.randn_like(blur) + blur, 0, 1)
#         blur = torch.round(blur * 255) / 255
#         sharp = self.transform(imread(self.sharps[item]))
#         kernel = torch.load(self.kernels[item])
#         # blur, sharp, kernel = self.random_patch(blur, sharp, kernel)
#         # if self.is_pretrain:
#         #     blur_unclip = torch.load(self.blurs_unclip[item])
#         #     blur_unclip = sigma * torch.randn_like(blur_unclip) + blur_unclip
#         #     # blur = self.transform(imread(self.blurs[item]))
#         #     # blur_unclip = torch.load(self.blurs_unclip[item])
#         #     return blur, [blur, blur_unclip], filename
#
#         reb = imfilter(sharp.unsqueeze(0), kernel, padding_mode='replicate', filter_mode='conv').squeeze(0)#.clip(0, 1).mul(255.).round().div(255.)
#
#
#         # return blur, kernel, (blur_unclip, sharp), filename
#         return blur, kernel, [blur, reb], filename

class PanDataset(Dataset):
    def __init__(self, data_dir, csv_filename, file_format, sigma_max, img_no):
        super(PanDataset, self).__init__()
        self.data_dir = data_dir
        self.csv_filename = csv_filename
        self.file_format = file_format
        self.sigma_max = sigma_max
        self.img_no = img_no
        self.transform = tvf.Compose([
            tvf.ToTensor()
        ])

        self.blurs, self.kernels, self.sharps = self.load_csv(data_dir, csv_filename)

    def load_csv(self, data_dir, csv_filename):
        csv_file = os.path.join(data_dir, csv_filename)
        if not os.path.exists(csv_file):

            kernels = []
            sharps = []
            sharp_path =  os.path.join(data_dir, 'gt', f'saturated_img{self.img_no}.png')

            blurs = glob.glob(os.path.join(data_dir, 'blur', f'saturated_img{self.img_no}*.png'))

            for blur in blurs:
                sharps.append(sharp_path)
                kernel_no = blur.split('_')[-2]
                kernels.append(os.path.join(data_dir, 'kernel', f'kernel_0{kernel_no}.mat'))

            trainset_info = pd.DataFrame({
                'sharp': sharps,
                'kernel': kernels,
                'blur': blurs
            })

            trainset_info.to_csv(csv_file, index=False)

        else:
            trainset_info = pd.read_csv(csv_file)
            sharps = trainset_info['sharp'].tolist()
            kernels = trainset_info['kernel'].tolist()
            blurs = trainset_info['blur'].tolist()
        return blurs, kernels, sharps

    def __len__(self):
        return len(self.sharps)

    def __getitem__(self, item):
        filename = self.blurs[item].split('/')[-1]
        blur = self.transform(imread(self.blurs[item]))

        sigma = self.sigma_max
        blur = torch.clip(sigma * torch.randn_like(blur) + blur, 0, 1)
        blur = torch.round(blur * 255) / 255
        sharp = self.transform(imread(self.sharps[item]))
        ext = self.kernels[item].split('.')[-1]

        if ext == 'mat':
            kernel = io.loadmat(self.kernels[item])['kernel'].astype(np.float32)
            kernel = torch.from_numpy(kernel).unsqueeze(0)
        else:
            kernel = rgb2gray(self.transform(imread(self.kernels[item])), data_format=ImageOption.Formal.CHW)

        kernel = kernel / kernel.sum()


        return blur, kernel.rot90(k=2, dims=(-2, -1)), sharp, filename

# class MaskL2CGDeblur(Dataset)
#     def

if __name__ == '__main__':
    # deblur = OutlierTestDataset('/home/fubo/data/deblur/outliers/lai', csv_filename='lai.csv', file_format=['*.png', '*.jpg', '*.bmp'], sigma_max=0.01, is_pretrain=True)
    #
    # blur, kernel, sharp, filename = deblur[10]
    # imwrite('kernel.png', kernel / kernel.max(), ImageOption.Formal.CHW)
    # imwrite('sharp.png', sharp, ImageOption.Formal.CHW)
    # imwrite('blur.png', blur, ImageOption.Formal.CHW)
    # estpsf = FastPSFEstimate()
    # imagegrad = ImageGrad()
    # bx, by = imagegrad(rgb2gray(blur.unsqueeze(0)))
    # ix, iy = imagegrad(rgb2gray(sharp.unsqueeze(0)))
    # psf = estpsf(31, bx, by, ix, iy)
    # imwrite('psf.png', psf / psf.max(), ImageOption.Formal.BCHW)
    dataset = PanDataset(
        '/home/echo/dataset/deblur/BlurDataset/train-multioptim-deblur/jspan-low-illumination',
        'im01.csv',
        ['*.png'],
        0.01,
        1
    )

    blur, kernel, sharp, filename = dataset[1]
    imwrite('blur.png', blur.unsqueeze(0))
    imwrite('sharp.png', sharp.unsqueeze(0))
    # imwrite('blur.png', blur.unsqueeze(0))