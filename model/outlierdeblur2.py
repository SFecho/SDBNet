import math
from math import sqrt

import torch as t
import torch.nn as nn
import torch.nn.functional as F

from model.deblurmask import DeblurMask
from utils.image import MedianPool2d, psf2otf, imfilter, imwrite, rgb2gray
import numpy as np

class OutlierDeblur2(nn.Module):
    def __init__(self, in_channels, out_channels, mask_path):
        super(OutlierDeblur2, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.g1_kernel = nn.Parameter(t.from_numpy(
            np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]], dtype="float32").reshape((1, 3, 3))),
            requires_grad=False)
        # .cuda()#to(self.device)
        self.g2_kernel = nn.Parameter(t.from_numpy(
            np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]], dtype="float32").reshape((1, 3, 3))),
            requires_grad=False)

        self.g3_kernel = nn.Parameter(t.from_numpy(
            np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]], dtype="float32").reshape((1, 3, 3))),
            requires_grad=False)

        self.g4_kernel = nn.Parameter(t.from_numpy(
            np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]], dtype="float32").reshape((1, 3, 3))),
            requires_grad=False)

        self.g5_kernel = nn.Parameter(
            t.from_numpy(
                np.array([[-1, 1, 0], [1, -1, 0], [0, 0, 0]], dtype="float32").reshape((1, 3, 3))),
            requires_grad=False)


        # self.media = MedianPool2d(kernel_size=5, same=True)

        self.mask = DeblurMask(in_channels, kernel_size=5, n_feat=32)
        if mask_path is not None:
            state_dict = t.load(mask_path)
            self.mask.load_state_dict(state_dict)




    def auto_crop_kernel(self, kernel):
        end = 0
        for i in range(kernel.size()[2]):
            if kernel[0, 0, end, 0] == -1:
                break
            end += 1
        kernel = kernel[:, :, :end, :end]
        return kernel

    def conv_func(self, input, kernel, padding='same'):
        # kernel = kernel.unsqueeze(1)
        b, c, h, w = input.size()
        _, _, ksize, ksize = kernel.size()
        if padding == 'same':
            pad = ksize // 2
        elif padding == 'valid':
            pad = 0
        else:
            raise Exception("not support padding flag!")

        with t.no_grad():
            otf = psf2otf(t.rot90(kernel, k=2, dims=(-2, -1)), shape=input.shape[-2:])
        conv_result_tensor = t.real(t.fft.ifft2(t.fft.fft2(input) * otf))

        return conv_result_tensor

    def dual_conv(self, input, kernel, mask):
        kernel_numpy = kernel.cpu().numpy()[:, :, ::-1, ::-1]
        kernel_numpy = np.ascontiguousarray(kernel_numpy)
        kernel_flip = t.from_numpy(kernel_numpy).to(input.device)
        result = self.conv_func(input, kernel_flip, padding='same')
        result = self.conv_func(result * mask, kernel, padding='same')
        return result

    def dual_conv_grad(self, input, kernel):
        kernel_numpy = kernel.cpu().numpy()[:, :, ::-1, ::-1]
        kernel_numpy = np.ascontiguousarray(kernel_numpy)
        kernel_flip = t.from_numpy(kernel_numpy).to(input.device)
        result = self.conv_func(input, kernel_flip, padding='same')
        return result

    def vector_inner_product3(self, x1, x2):
        b, c, h, w = x1.size()
        re = x1.mul(x2).sum(dim=(-2, -1), keepdim=True)
        return re

    def compute_reg_l2(self, x, lambd=0.003):
        b = x.shape[0]
        # w1 = 0.1
        # w2 = w1
        # w3, w4, w5 = 0.25 * w1, 0.25 * w1, 0.25 * w1

        x1 = imfilter(x, self.g1_kernel.repeat(b, 1, 1))
        x1 = imfilter(x1, self.g1_kernel.repeat(b, 1, 1), filter_mode='conv')

        x2 = imfilter(x, self.g2_kernel.repeat(b, 1, 1))
        x2 = imfilter(x2, self.g2_kernel.repeat(b, 1, 1), filter_mode='conv')

        x3 = imfilter(x, self.g3_kernel.repeat(b, 1, 1))
        x3 = imfilter(x3, self.g3_kernel.repeat(b, 1, 1), filter_mode='conv')

        x4 = imfilter(x, self.g4_kernel.repeat(b, 1, 1))
        x4 = imfilter(x4, self.g4_kernel.repeat(b, 1, 1), filter_mode='conv')

        x5 = imfilter(x, self.g5_kernel.repeat(b, 1, 1))
        x5 = imfilter(x5, self.g5_kernel.repeat(b, 1, 1), filter_mode='conv')

        return lambd * x1 + lambd * x2 + lambd * x3 + lambd * x4 + lambd * x5

    def get_l08_weight(self, x):
        b = x.shape[0]
        w1 = 0.1
        w2 = w1
        w3, w4, w5 = 0.25 * w1, 0.25 * w1, 0.25 * w1
        thr_e = 0.01
        x1 = imfilter(x, self.g1_kernel.repeat(b, 1, 1))
        w_x1 = w1 * t.where(x1.abs() > thr_e, x1.abs(), thr_e * t.ones_like(x1)).pow(-1.2)

        x2 = imfilter(x, self.g2_kernel.repeat(b, 1, 1))
        w_x2 = w2 * t.where(x2.abs() > thr_e, x2.abs(), thr_e * t.ones_like(x2)).pow(-1.2)

        x3 = imfilter(x, self.g3_kernel.repeat(b, 1, 1))
        w_x3 = w3 * t.where(x3.abs() > thr_e, x3.abs(), thr_e * t.ones_like(x3)).pow(-1.2)

        x4 = imfilter(x, self.g4_kernel.repeat(b, 1, 1))
        w_x4 = w4 * t.where(x4.abs() > thr_e, x4.abs(), thr_e * t.ones_like(x4)).pow(-1.2)

        x5 = imfilter(x, self.g5_kernel.repeat(b, 1, 1))
        w_x5 = w5 * t.where(x5.abs() > thr_e, x5.abs(), thr_e * t.ones_like(x5)).pow(-1.2)
        return w_x1, w_x2, w_x3, w_x4, w_x5

    def compute_reg_l08(self, x, w_x1, w_x2, w_x3, w_x4, w_x5, lambd=0.003):
        b = x.shape[0]
        w1 = 0.1
        # w2 = w1
        # w3, w4, w5 = 0.25 * w1, 0.25 * w1, 0.25 * w1
        # thr_e = 0.01
        x1 = imfilter(x, self.g1_kernel.repeat(b, 1, 1))
        x1 = imfilter(w_x1 * x1, self.g1_kernel.repeat(b, 1, 1), filter_mode='conv')

        x2 = imfilter(x, self.g2_kernel.repeat(b, 1, 1))
        x2 = imfilter(x2 * w_x2, self.g2_kernel.repeat(b, 1, 1), filter_mode='conv')

        x3 = imfilter(x, self.g3_kernel.repeat(b, 1, 1))
        x3 = imfilter(x3 * w_x3, self.g3_kernel.repeat(b, 1, 1), filter_mode='conv')

        x4 = imfilter(x, self.g4_kernel.repeat(b, 1, 1))
        x4 = imfilter(x4 * w_x4, self.g4_kernel.repeat(b, 1, 1), filter_mode='conv')

        x5 = imfilter(x, self.g5_kernel.repeat(b, 1, 1))
        x5 = imfilter(x5 * w_x5, self.g5_kernel.repeat(b, 1, 1), filter_mode='conv')

        return lambd * (x1 + x2 + x3 + x4 + x5)



    def deconv_func_l2(self, mask, input, kernel):  # , beta11, beta22, beta12
        b, c, h, w = input.size()
        kb, kc, ksize, ksize = kernel.size()
        psize = ksize // 2
        input = self.pad(input, ksize)
        # input = F.pad(input, pad=(psize, psize, psize, psize), mode='replicate')
        input_ori = input
        # assert b == 1, "only support one image deconv operation!"

        kernel = self.auto_crop_kernel(kernel)
        assert ksize % 2 == 1, "only support odd kernel size!"

        # mask = t.zeros_like(input).to(input.device)
        # mask[:, :, psize:-psize, psize:-psize] = 1.
        mask = F.pad(mask, pad=(psize, psize, psize, psize), mode='constant', value=0)
        mask_beta = t.ones_like(input).to(input.device)

        x = input

        b = self.conv_func(input_ori * mask, kernel, padding='same')

        Ax = self.dual_conv(x, kernel, mask)
        # Ax = Ax + sigma * self.dual_conv(x, self.g1_kernel, mask_beta) \
        #      + sigma * self.dual_conv(x, self.g2_kernel, mask_beta)
        Ax = Ax + self.compute_reg_l2(x)#sigma * self.dual_conv(x, self.g1_kernel, mask_beta) \
             #sigma * self.dual_conv(x, self.g2_kernel, mask_beta)

        r = b - Ax
        for i in range(15):
            rho = self.vector_inner_product3(r, r)
            if i == 0:
                p = r
            else:
                beta = rho / rho_1
                p = r + beta * p

            Ap = self.dual_conv(p, kernel, mask)
            # Ap = Ap + sigma * self.dual_conv(p, self.g1_kernel, mask_beta) \
            #      + sigma * self.dual_conv(p, self.g2_kernel, mask_beta)
            Ap = Ap + self.compute_reg_l2(p)

            q = Ap
            alp = rho / self.vector_inner_product3(p, q)
            x = x + alp * p
            r = r - alp * q
            rho_1 = rho

        deconv_result = x[:, :, psize: -psize, psize: -psize]
        # deconv_result = x
        return deconv_result

    def deconv_func_l08(self, mask, input, x, kernel):  # , beta11, beta22, beta12
        b, c, h, w = input.size()
        kb, kc, ksize, ksize = kernel.size()
        psize = ksize // 2
        input = self.pad(input, ksize)
        x = self.pad(x, ksize)
        # input = F.pad(input, pad=(psize, psize, psize, psize), mode='replicate')
        input_ori = input
        # assert b == 1, "only support one image deconv operation!"

        kernel = self.auto_crop_kernel(kernel)
        assert ksize % 2 == 1, "only support odd kernel size!"

        # mask = t.zeros_like(input).to(input.device)
        # mask[:, :, psize:-psize, psize:-psize] = 1.
        mask = F.pad(mask, pad=(psize, psize, psize, psize), mode='constant', value=0)
        mask_beta = t.ones_like(input).to(input.device)

        w_x1, w_x2, w_x3, w_x4, w_x5 = self.get_l08_weight(x)
        # x = input

        b = self.conv_func(input_ori * mask, kernel, padding='same')

        Ax = self.dual_conv(x, kernel, mask)
        # Ax = Ax + sigma * self.dual_conv(x, self.g1_kernel, mask_beta) \
        #      + sigma * self.dual_conv(x, self.g2_kernel, mask_beta)
        Ax = Ax + self.compute_reg_l08(x, w_x1, w_x2, w_x3, w_x4, w_x5)#sigma * self.dual_conv(x, self.g1_kernel, mask_beta) \
             #sigma * self.dual_conv(x, self.g2_kernel, mask_beta)

        r = b - Ax
        for i in range(15):
            rho = self.vector_inner_product3(r, r)
            if i == 0:
                p = r
            else:
                beta = rho / rho_1
                p = r + beta * p

            Ap = self.dual_conv(p, kernel, mask)
            # Ap = Ap + sigma * self.dual_conv(p, self.g1_kernel, mask_beta) \
            #      + sigma * self.dual_conv(p, self.g2_kernel, mask_beta)
            Ap = Ap + self.compute_reg_l08(p, w_x1, w_x2, w_x3, w_x4, w_x5)

            q = Ap
            alp = rho / self.vector_inner_product3(p, q)
            x = x + alp * p
            r = r - alp * q
            rho_1 = rho

        # deconv_result = x[:, :, psize: -psize, psize: -psize]
        deconv_result = x[:, :, psize: -psize, psize: -psize]
        return deconv_result

    def dual_l08conv(self, input, kernel, mask):
        kernel_numpy = kernel.cpu().numpy()[:, :, ::-1, ::-1]
        kernel_numpy = np.ascontiguousarray(kernel_numpy)
        kernel_flip = t.from_numpy(kernel_numpy).to(input.device)
        result = self.conv_func(input, kernel_flip, padding='same')
        weight = result.abs()
        weight = t.where(weight > 1e-4, weight, t.ones_like(weight) * 1e-4)
        weight = weight.pow(-1.2)
        result = self.conv_func(result * mask * weight, kernel, padding='same')
        return result

    def pad(self, blur, ksize):
        # h, w = blur.shape[-2:]

        pad = ksize // 2
        pads = [pad for _ in range(4)]


        return F.pad(blur, pad=pads, mode='replicate')



    def forward(self, blur, kernel):

        mask = self.mask(blur, kernel)
        mask = mask[-1]

        imwrite('mask.png', mask)
        outs = []
        with t.no_grad():
            deconv1 = self.deconv_func_l2(t.ones_like(blur), blur, kernel)
        for i in range(5):
            deconv1 = self.deconv_func_l08(mask, blur, deconv1, kernel)
            outs.append(deconv1)

        return outs
