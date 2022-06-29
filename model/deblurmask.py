import math
from collections import OrderedDict

import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from einops import rearrange
from torch.autograd import Variable

from layer.conv import Conv, Deconv, ResBlock, RCAB
from utils.image import imfilter, psf2otf, MedianPool2d, convert_psf2otf, imwrite, imread, rgb2gray
import numpy as np
from layer.ionv import involution_cuda
from math import ceil, floor, sqrt
from modules.deform_feature import DeformFeature
import scipy.io as io
# class GenerateWeight(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, n_feat):
#         super(GenerateWeight, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#
#         self.head = nn.Conv2d(in_channels, n_feat, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
#         self.block1 =  RCAB(n_feat, kernel_size)
#         self.block2 = RCAB(n_feat, kernel_size)
#         self.block3 = RCAB(n_feat, kernel_size)
#         self.tail = nn.Conv2d(n_feat, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
#
#
#     def forward(self, x):
#         x1 = self.head(x)
#         x2 = self.block1(x1)
#         x3 = self.block2(x2)
#         out = self.block3(x3)
#         out = self.tail(out)
#         out = F.sigmoid(out)
#         out = t.clamp(out, 0.02, 1)
#
#         return out
#
# class GenerateWeight(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, n_feat):
#         super(GenerateWeight, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#
#         self.body = nn.Sequential(*[
#             nn.Conv2d(in_channels, n_feat, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
#             RCAB(n_feat, kernel_size),
#             nn.Conv2d(n_feat, n_feat * 2, kernel_size=kernel_size, stride=2, padding=kernel_size // 2),
#             RCAB(n_feat * 2, kernel_size),
#             nn.Conv2d(n_feat * 2, n_feat * 4, kernel_size=kernel_size, stride=2, padding=kernel_size // 2),
#             RCAB(n_feat * 4, kernel_size),
#             nn.AdaptiveMaxPool2d(1),
#             nn.Conv2d(n_feat * 4, 1, kernel_size=1, stride=1, padding=0),
#             nn.Sigmoid()
#         ])
#         # self.head = nn.Conv2d(in_channels, n_feat, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
#         # self.block1 =  RCAB(n_feat, kernel_size)
#         # self.block2 = RCAB(n_feat, kernel_size)
#         # self.block3 = RCAB(n_feat, kernel_size)
#         # self.tail = nn.Conv2d(n_feat, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
#
#
#     def forward(self, x):
#         out = self.body(x)
#         return t.clamp(out, 0.02, 1)
#
# # class NonUniform2d(nn.Module):
# #     def __init__(self):
# #         super(NonUniform2d, self).__init__()
# #         self.involution2d = involution_cuda
# #
# #     def forward(self, x, weight):
# #         pass
#
# def soft_threshold(x, thres):
#     return t.sign(x) * t.relu(t.abs(x) - thres)
#
# class ChannelAffine(nn.Module):
#     def __init__(self, in_channels, n_kernel):
#         super(ChannelAffine, self).__init__()
#         self.in_channels = in_channels
#         self.n_kernel = n_kernel
#     def forward(self, x, weight):
#         weight_ = rearrange(weight, 'b (k1 k2 n) h w -> b k1 k2 n h w', k1=self.in_channels, k2=self.in_channels, n=self.n_kernel)
#
#         return t.einsum('bcnhw,bcrnhw -> brnhw', x, weight_) / self.in_channels
#
# class ChannelAffineTranspose(nn.Module):
#     def __init__(self, in_channels, n_kernel):
#         super(ChannelAffineTranspose, self).__init__()
#         self.in_channels = in_channels
#         self.n_kernel = n_kernel
#     def forward(self, x, weight):
#         # weight = t.flip(weight, dims=[1])
#         weight_ = rearrange(weight, 'b (k1 k2 n) h w -> b k1 k2 n h w', k1=self.in_channels, k2=self.in_channels, n=self.n_kernel)
#         weight_ = weight_.permute(0, 2, 1, 3, 4, 5)
#         # print('weight:', weight_.shape)
#         out = t.einsum('bcnhw,bcrnhw -> brnhw', x, weight_) / self.in_channels
#         # print('out:', out.shape)
#         return out
#
# class NonUnformConvTranspose2d(nn.Module):
#     def __init__(self, kernel_size, n_kernel):
#         super(NonUnformConvTranspose2d, self).__init__()
#         self.kernel_size = kernel_size
#         self.n_kernel = n_kernel
#         self.non_conv2d = involution_cuda
#
#     def forward(self, x, kernel):
#
#         kernel = kernel.rot90(k=2, dims=(2, 3))
#         return self.non_conv2d(x, kernel)
#
# class DualNonLinearConv2d(nn.Module):
#     def __init__(self, in_channels, kernel_size, n_kernel):
#         super(DualNonLinearConv2d, self).__init__()
#         self.n_kernel = n_kernel
#         self.kernel_size = kernel_size
#         self.padding = kernel_size // 2
#         self.non_conv2d = involution_cuda#kernel_size, n_kernel)
#         # self.non_convtrans_2d = NonUnformConvTranspose2d(kernel_size, n_kernel)
#         self.in_channels = in_channels
#         self.p_conv = OffsetNetwork(in_channels, kernel_size)
#         self.zero_padding = nn.ZeroPad2d(self.padding)
#         self.deform_feature = DeformFeature(in_channels, in_channels, kernel_size, 1, kernel_size // 2)
#         # self.p_conv.register_backward_hook(self._set_lr)
#         self.stride = 1
#
#     @staticmethod
#     def _set_lr(module, grad_input, grad_output):
#         grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
#         grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))
#
#     def get_offset_image(self, x, offset):
#         # print('offset:', offset.shape)
#
#         dtype = offset.data.type()
#         ks = self.kernel_size
#         N = offset.size(1) // 2
#
#         if self.padding:
#             x = self.zero_padding(x)
#
#         # (b, 2N, h, w)
#         p = self._get_p(offset, dtype)
#         # print('p:', p.shape)
#         # (b, h, w, 2N)
#         p = p.contiguous().permute(0, 2, 3, 1)
#         q_lt = p.detach().floor()
#         q_rb = q_lt + 1
#
#         q_lt = t.cat([t.clamp(q_lt[..., :N], 0, x.size(2)-1), t.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
#         q_rb = t.cat([t.clamp(q_rb[..., :N], 0, x.size(2)-1), t.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
#         q_lb = t.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
#         q_rt = t.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)
#
#         # clip p
#         p = t.cat([t.clamp(p[..., :N], 0, x.size(2) - 1), t.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)
#
#         # bilinear kernel (b, h, w, N)
#         g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
#         g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
#         g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
#         g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))
#
#         # (b, c, h, w, N)
#         x_q_lt = self._get_x_q(x, q_lt, N)
#         x_q_rb = self._get_x_q(x, q_rb, N)
#         x_q_lb = self._get_x_q(x, q_lb, N)
#         x_q_rt = self._get_x_q(x, q_rt, N)
#         # print(x_q_lt.shape, x_q_rb.shape, x_q_lb.shape, x_q_rt.shape)
#         # (b, c, h, w, N)
#         x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
#                    g_rb.unsqueeze(dim=1) * x_q_rb + \
#                    g_lb.unsqueeze(dim=1) * x_q_lb + \
#                    g_rt.unsqueeze(dim=1) * x_q_rt
#
#         x_offset = self._reshape_x_offset(x_offset, ks)
#         return x_offset
#
#     def _get_p_n(self, N, dtype):
#         p_n_x, p_n_y = t.meshgrid(
#             t.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
#             t.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
#         # (2N, 1)
#         p_n = t.cat([t.flatten(p_n_x), t.flatten(p_n_y)], 0)
#         p_n = p_n.view(1, 2*N, 1, 1).type(dtype)
#
#         return p_n
#
#     def _get_p_0(self, h, w, N, dtype):
#         p_0_x, p_0_y = t.meshgrid(
#             t.arange(1, h*self.stride+1, self.stride),
#             t.arange(1, w*self.stride+1, self.stride))
#         p_0_x = t.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
#         p_0_y = t.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
#         p_0 = t.cat([p_0_x, p_0_y], 1).type(dtype)
#
#         return p_0
#
#     def _get_p(self, offset, dtype):
#         N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)
#
#         # (1, 2N, 1, 1)
#         p_n = self._get_p_n(N, dtype)
#         # (1, 2N, h, w)
#         p_0 = self._get_p_0(h, w, N, dtype)
#         p = p_0 + p_n + offset
#         return p
#
#     def _get_x_q(self, x, q, N):
#         b, h, w, _ = q.size()
#         padded_w = x.size(3)
#         c = x.size(1)
#         # (b, c, h*w)
#         x = x.contiguous().view(b, c, -1)
#
#         # (b, h, w, N)
#         index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
#         # (b, c, h*w*N)
#         index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
#
#         x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
#
#         return x_offset
#
#     @staticmethod
#     def _reshape_x_offset(x_offset, ks):
#         b, c, h, w, N = x_offset.size()
#         x_offset = t.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
#         x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)
#         # print('x_offset:', x_offset.shape)
#         return x_offset
#
#     # def forward(self, x, kernel):
#     #     kernel = rearrange(kernel, 'b (c n k1 k2) h w -> b c n k1 k2 h w', c=self.in_channels, k1=self.kernel_size, k2=self.kernel_size, n=self.n_kernel)
#     #     kernels = t.chunk(kernel, self.n_kernel, dim=2)#kernel.contiguous()
#     #     # print(len(kernels), kernels[0].shape)
#     #     outs = []
#     #     pad =  self.kernel_size // 2
#     #     for cur_k in kernels:
#     #         # cur_k = cur_k
#     #         cur_k = cur_k.squeeze(2)  / (self.kernel_size ** 2)
#     #         fk = self.non_conv2d(x, cur_k, padding=pad)
#     #         ftfk = self.non_conv2d(fk, cur_k.rot90(k=2, dims=(2, 3)), padding=pad)
#     #         outs.append(ftfk)
#     #
#     #     return sum(item for item in outs)
#
#
#     def forward(self, x, kernel):
#         kernel = rearrange(kernel, 'b (c n k1 k2) h w -> b c n (k1 k2) h w', c=self.in_channels, k1=self.kernel_size, k2=self.kernel_size, n=self.n_kernel)
#         kernels = t.chunk(kernel, self.n_kernel, dim=2)#kernel.contiguous()
#         # print(len(kernels), kernels[0].shape)
#         outs = []
#         pad =  self.kernel_size // 2
#         # x__ = Variable(x, requires_grad=True)
#         offset = self.p_conv(x)
#         x = self.deform_feature(x, offset)
#         # x = self.get_offset_image(x, offset)
#
#         # print(x_.grad,)
#         reg_loss = 0
#         for cur_k in kernels:
#             # print('x:', x.shape, cur_k.shape)
#             cur_k = cur_k.squeeze(2)
#             fk = t.einsum('bckhw,bckhw->bchw', cur_k, x) / (self.kernel_size ** 2)#cur_k * x
#             fk = self.deform_feature(fk, offset)
#             # print('offset：', offset.shape)
#             ftfk = t.einsum('bckhw,bckhw->bchw', cur_k.flip(2), fk) / (self.kernel_size ** 2)
#             # print('ftfk:', ftfk.shape)
#             # cur_k = cur_k
#             # cur_k = cur_k.squeeze(2) / (self.kernel_size ** 2)
#             # fk = self.non_conv2d(x, cur_k, padding=0, stride=self.kernel_size)
#             # fk = self.get_offset_image(fk, offset)
#             # ftfk = self.non_conv2d(fk, cur_k.rot90(k=2, dims=(2, 3)), padding=0, stride=self.kernel_size)
#             outs.append(ftfk)
#
#         return sum(item for item in outs)
#
# class L2NonDeblurCG(nn.Module):
#
#     def __init__(self, in_channels, out_channels, window_size):
#         super(L2NonDeblurCG, self).__init__()
#
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.window_size = window_size
#         self.g1_kernel = nn.Parameter(t.from_numpy(
#             np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]], dtype="float32").reshape((1, 1, 3, 3))),
#             requires_grad=False)
#         # .cuda()#to(self.device)
#         self.g2_kernel = nn.Parameter(t.from_numpy(
#             np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]], dtype="float32").reshape((1, 1, 3, 3))),
#             requires_grad=False)
#
#         self.media = MedianPool2d(kernel_size=5, same=True)
#
#     @t.no_grad()
#     def kx_y(self, x, psf, blur):
#         kx = self.conv_blur(t.clamp(x, 0, 1), psf)
#         out = rgb2gray(kx - blur)
#         return out
#
#     def auto_crop_kernel(self, kernel):
#         end = 0
#         for i in range(kernel.size()[2]):
#             if kernel[0, 0, end, 0] == -1:
#                 break
#             end += 1
#         kernel = kernel[:, :, :end, :end]
#         return kernel
#
#     def conv_func(self, input, kernel, padding='same'):
#         # kernel = kernel.unsqueeze(1)
#         b, c, h, w = input.size()
#         _, _, ksize, ksize = kernel.size()
#         if padding == 'same':
#             pad = ksize // 2
#         elif padding == 'valid':
#             pad = 0
#         else:
#             raise Exception("not support padding flag!")
#
#         with t.no_grad():
#             otf = psf2otf(t.rot90(kernel, k=2, dims=(-2, -1)), shape=input.shape[-2:])
#         conv_result_tensor = t.real(t.fft.ifft2(t.fft.fft2(input) * otf))
#
#         return conv_result_tensor
#
#     def dual_conv(self, input, kernel, mask):
#         kernel_numpy = kernel.cpu().numpy()[:, :, ::-1, ::-1]
#         kernel_numpy = np.ascontiguousarray(kernel_numpy)
#         kernel_flip = t.from_numpy(kernel_numpy).to(input.device)
#         result = self.conv_func(input, kernel_flip, padding='same')
#         result = self.conv_func(result * mask, kernel, padding='same')
#         return result
#
#     def dual_conv_grad(self, input, kernel):
#         kernel_numpy = kernel.cpu().numpy()[:, :, ::-1, ::-1]
#         kernel_numpy = np.ascontiguousarray(kernel_numpy)
#         kernel_flip = t.from_numpy(kernel_numpy).to(input.device)
#         result = self.conv_func(input, kernel_flip, padding='same')
#         return result
#
#     def vector_inner_product3(self, x1, x2):
#         b, c, h, w = x1.size()
#         x1 = x1.view(b, c, -1)
#         x2 = x2.view(b, c, -1)
#         re = x1 * x2
#         re = t.sum(re, dim=2)
#         re = re.view(b, c, 1, 1)
#         return re
#
#     def deconv_func(self, input, kernel, sigma):  # , beta11, beta22, beta12
#         b, c, h, w = input.size()
#         kb, kc, ksize, ksize = kernel.size()
#         psize = ksize // 2
#         input = self.pad(input, ksize)
#         # input = F.pad(input, pad=(psize, psize, psize, psize), mode='replicate')
#         input_ori = input
#         # assert b == 1, "only support one image deconv operation!"
#
#         kernel = self.auto_crop_kernel(kernel)
#         assert ksize % 2 == 1, "only support odd kernel size!"
#
#         mask = t.zeros_like(input).to(input.device)
#         mask[:, :, psize:-psize, psize:-psize] = 1.
#         mask_beta = t.ones_like(input).to(input.device)
#
#         x = input
#
#         # weight = self.robust(self.conv_func(x, t.rot90(kernel, k=2, dims=(-2, -1))) - input, 0.02)
#
#         b = self.conv_func(input_ori * mask, kernel, padding='same')
#
#         Ax = self.dual_conv(x, kernel, mask)
#         Ax = Ax + sigma * self.dual_conv(x, self.g1_kernel, mask_beta) \
#              + sigma * self.dual_conv(x, self.g2_kernel, mask_beta)
#
#         r = b - Ax
#         for i in range(25):
#             rho = self.vector_inner_product3(r, r)
#             if i == 0:
#                 p = r
#             else:
#                 beta = rho / rho_1
#                 p = r + beta * p
#
#             Ap = self.dual_conv(p, kernel, mask)
#             Ap = Ap + sigma * self.dual_conv(p, self.g1_kernel, mask_beta) \
#                  + sigma * self.dual_conv(p, self.g2_kernel, mask_beta)
#
#             q = Ap
#             alp = rho / self.vector_inner_product3(p, q)
#             x = x + alp * p
#             r = r - alp * q
#             rho_1 = rho
#
#         # deconv_result = x[:, :, psize: -psize, psize: -psize]
#         deconv_result = x
#         return deconv_result
#
#     def pad(self, blur, ksize):
#         # h, w = blur.shape[-2:]
#
#         pad = ksize // 2
#
#         pads = []
#         img_size = blur.shape[-2:]
#         for size in reversed(img_size):
#             psize = pad * 2 + size
#             if psize // self.window_size == ceil(psize / self.window_size):
#                 pads.extend([pad, pad])
#             else:
#                 pads.append(pad)
#                 padr = ceil(psize / self.window_size) * self.window_size
#                 padr = padr - pad - size
#                 pads.append(padr)
#
#         return F.pad(blur, pad=pads, mode='replicate')
#
#
#     def forward(self, blur, kernel):
#         # noise = blur - self.media(blur)
#         # c, h, w = noise.shape[1:]
#         # mean = t.mean(noise, dim=(-3, -2, -1), keepdim=True)
#         # sigma = t.sqrt(t.sum((noise - mean).pow(2), dim=(-3, -2, -1), keepdim=True) / (c * h * w - 1))
#         # print(sigma)
#         sigma = 0.01
#         deconv1 = self.deconv_func(blur, kernel, sigma)
#
#         return deconv1
#
#
# class NonKernelGenerator(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, n_kernel):
#         super(NonKernelGenerator, self).__init__()
#         out_feats = out_channels * n_kernel * (kernel_size ** 2)
#         # print('out_feats:', out_feats, out_channels, n_kernel, kernel_size)
#         hidden_feats = 64
#         _ksize = 5
#         padding = _ksize // 2
#         self.body = nn.Sequential(
#             nn.Conv2d(in_channels, hidden_feats, kernel_size=_ksize, padding=padding, stride=1),
#             nn.ReLU(True),
#             nn.Conv2d(hidden_feats, hidden_feats, kernel_size=_ksize, padding=padding, stride=1),
#             nn.ReLU(True),
#             nn.Conv2d(hidden_feats, hidden_feats, kernel_size=_ksize, padding=padding, stride=1),
#             nn.ReLU(True),
#             nn.Conv2d(hidden_feats, out_feats, kernel_size=_ksize, padding=padding, stride=1)
#         )
#
#     def forward(self, x):
#         return self.body(x)
#
#
# class PatchMerge(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride, padding, window_size):
#         super(PatchMerge, self).__init__()
#         in_channels = in_channels * window_size * window_size
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
#         self.window_size = window_size
#
#         self.window_size = window_size
#
#     def forward(self, x):
#         h, w = x.shape[-2:]
#
#         if self.window_size > 1:
#             ih = h // self.window_size
#             iw = w // self.window_size
#             x = F.unfold(x, kernel_size=self.window_size, stride=self.window_size, padding=0)
#             x = rearrange(x, 'b k (h w) -> b k h w', h=ih, w=iw)
#
#         return self.conv(x)
#
#
# class LocalKernelGenerator(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, n_kernel, window_size):
#         super(LocalKernelGenerator, self).__init__()
#
#         out_feats = out_channels * n_kernel * (kernel_size ** 2)
#         hidden_feats = 64
#         self.window_size = window_size
#         padding = kernel_size // 2
#         s1, s2 = self.get_kernels()
#         if s1 == 1:
#             k1 = 3
#             p1 = 1
#         else:
#             k1 = s1
#             p1 = 0
#         if s2 == 1:
#             k2 = 3
#             p2 = 1
#         else:
#             k2 = s2
#             p2 = 0
#
#         patch_feat1 = hidden_feats # hidden_feats * 2
#         patch_feat2 = hidden_feats #patch_feat1 * 2
#
#
#         self.body = nn.Sequential(
#             nn.Conv2d(in_channels, hidden_feats, kernel_size=kernel_size, padding=padding, stride=1),
#             nn.ReLU(True),
#             # nn.Conv2d(hidden_feats, hidden_feats, kernel_size=kernel_size, padding=padding, stride=1),
#             # nn.ReLU(True),
#             PatchMerge(hidden_feats, patch_feat1, kernel_size=kernel_size, padding=kernel_size // 2, stride=1, window_size=s1),
#             nn.ReLU(True),
#             nn.Conv2d(patch_feat1, patch_feat1, kernel_size=kernel_size, padding=kernel_size // 2, stride=1),
#             nn.ReLU(True),
#             PatchMerge(patch_feat1, patch_feat2, kernel_size=kernel_size, padding=kernel_size // 2, stride=1, window_size=s2),
#             nn.ReLU(True),
#             nn.Conv2d(patch_feat2, out_feats, kernel_size=kernel_size, padding=padding, stride=1),
#         )
#
#     def forward(self, x):
#         out = self.body(x)
#         return out
#
#     def get_kernels(self):
#         s1 = floor(sqrt(self.window_size))
#         while s1 >= 1:
#             s2 = self.window_size // s1
#             if s1 * s2 == self.window_size:
#                 return s1, s2
#             else:
#                 s1 -= 1
#         return self.window_size, 1
#
#
# class AffineGenerate(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, n_feat):
#         super(AffineGenerate, self).__init__()
#         self.body = nn.Sequential(
#             nn.Conv2d(in_channels, n_feat, kernel_size=kernel_size, padding=kernel_size // 2),
#             nn.ReLU(True),
#             nn.Conv2d(n_feat, n_feat, kernel_size=kernel_size, padding=kernel_size // 2),
#             nn.ReLU(True),
#             nn.Conv2d(n_feat, n_feat, kernel_size=kernel_size, padding=kernel_size // 2),
#             nn.ReLU(True),
#             nn.Conv2d(n_feat, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
#         )
#
#     def forward(self, x):
#         return self.body(x)
#
#
# class FineTuneDeblur(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, window_size=4, n_feat=64, n_kernel=1):
#         super(FineTuneDeblur, self).__init__()
#
#         self.localkernel_reg = LocalKernelGenerator(in_channels, out_channels, kernel_size=3, window_size=window_size,
#                                                      n_kernel=n_kernel)
#         # self.dual_localconv2d = DualLocalLinearConv2d(kernel_size=kernel_size, n_kernel=1, window_size=window_size)
#         # self.local_conv = LocalUnformConv2d(kernel_size, n_kernel=n_kernel, window_size=window_size)
#         # self.local_conv_transpose = LocalUnformConvTranspose2d(kernel_size, n_kernel=n_kernel, window_size=window_size)
#
#
#         self.dual_local_conv = DualNonLinearConv2d(in_channels, kernel_size, n_kernel)
#         self.generate_weight = GenerateWeight(1, 1, kernel_size, 16)
#
#         self.media = MedianPool2d(kernel_size=5, same=True)
#         self.window_size = window_size
#         self.inner_iter = 1
#
#
#     def auto_crop_kernel(self, kernel):
#         end = 0
#         for i in range(kernel.size()[2]):
#             if kernel[0, 0, end, 0] == -1:
#                 break
#             end += 1
#         kernel = kernel[:, :, :end, :end]
#         return kernel
#
#     def conv_func(self, input, kernel, padding='same'):
#         # kernel = kernel.unsqueeze(1)
#         # b, c, h, w = input.size()
#         # print(input.shape)
#         _, _, ksize, ksize = kernel.size()
#         if padding == 'same':
#             pad = ksize // 2
#         elif padding == 'valid':
#             pad = 0
#         else:
#             raise Exception("not support padding flag!")
#
#         with t.no_grad():
#             otf = psf2otf(t.rot90(kernel, k=2, dims=(-2, -1)), shape=input.shape[-2:])
#         conv_result_tensor = t.real(t.fft.ifft2(t.fft.fft2(input) * otf))
#
#         return conv_result_tensor
#
#     def dual_conv(self, input, kernel, mask):
#         kernel_numpy = kernel.cpu().numpy()[:, :, ::-1, ::-1]
#         kernel_numpy = np.ascontiguousarray(kernel_numpy)
#         kernel_flip = t.from_numpy(kernel_numpy).to(input.device)
#         result = self.conv_func(input, kernel_flip, padding='same')
#         result = self.conv_func(result * mask, kernel, padding='same')
#         return result
#
#     def conv_blur(self, input, kernel):
#         kernel_numpy = kernel.cpu().numpy()[:, :, ::-1, ::-1]
#         kernel_numpy = np.ascontiguousarray(kernel_numpy)
#         kernel_flip = t.from_numpy(kernel_numpy).to(input.device)
#         result = self.conv_func(input, kernel_flip, padding='same')
#         return result
#
#     def conv_blur_transpose(self, input, kernel, mask):
#         result = self.conv_func(input * mask, kernel, padding='same')
#         return result
#
#     def dual_conv_grad(self, input, kernel):
#         kernel_numpy = kernel.cpu().numpy()[:, :, ::-1, ::-1]
#         kernel_numpy = np.ascontiguousarray(kernel_numpy)
#         kernel_flip = t.from_numpy(kernel_numpy).to(input.device)
#         result = self.conv_func(input, kernel_flip, padding='same')
#         return result
#
#     def vector_inner_product3(self, x1, x2):
#         b, c, h, w = x1.size()
#         x1 = x1.view(b, c, -1)
#         x2 = x2.view(b, c, -1)
#         re = x1 * x2
#         re = t.sum(re, dim=2)
#         re = re.view(b, c, 1, 1)
#         return re
#
#
#
#     def ax(self, x, kernel, weight, local_weight_reg, mask, coff):
#         kx = self.conv_blur(x, kernel)
#         ktkx = self.conv_blur_transpose(kx * weight, kernel, mask)
#
#         # fx = self.local_conv(x, local_weight_reg)
#         # ftfx = self.local_conv_transpose(fx, local_weight_reg).sum(dim=2)
#         ftfx = self.dual_local_conv(x, local_weight_reg)
#         # print(',ktkx')
#         return ktkx + coff * ftfx
#
#
#     def kx_y(self, x, psf, blur):
#         kx = self.conv_blur(t.clamp(x, 0, 1), psf)
#         out = rgb2gray(kx - blur)
#         return out
#
#     def robust(self, x, w):
#         # print(x.dtype, w.dtype)
#
#         # y = t.where(t.abs(rgb2gray(x)) <= w, t.zeros_like(x), x)
#         y = soft_threshold(x, w)
#         y = 1 - F.tanh(y.div(w).pow(2))
#
#         # print(y.min())
#         return y
#
#
#     def b(self, blur, kernel, mask, weight):
#         b = self.conv_blur_transpose(blur * weight, kernel, mask)
#
#         return b
#
#     def ktkx_kty(self, x, psf, blur, mask):
#         kx = self.conv_blur(x, psf)
#         ktkx = self.conv_blur_transpose(kx, psf, mask)
#         kty = self.conv_blur_transpose(blur, psf, mask)
#
#         return ktkx - kty
#
#     def deconv_func(self, blur, x, kernel):  # , beta11, beta22, beta12
#         b, c, h, w = blur.size()
#         kb, kc, ksize, ksize = kernel.size()
#         psize = ksize // 2
#
#
#         kernel = self.auto_crop_kernel(kernel)
#         assert ksize % 2 == 1, "only support odd kernel size!"
#
#
#
#         mask = t.zeros_like(x).to(x.device)
#         mask[:, :, psize:psize + h, psize: psize + w] = 1.
#
#         blur = self.pad(blur, ksize=ksize)
#         kx_y = self.kx_y(x, kernel, blur)
#         w =  self.generate_weight(kx_y)
#         # print('threshold:', w)
#         # w = 0.02
#         weight = self.robust(kx_y, w)
#
#         local_weight_reg = self.localkernel_reg(x)
#
#
#         b = self.b(blur, kernel, mask, weight)
#
#         Ax = self.ax(x, kernel, weight, local_weight_reg, mask, 1)
#         r = b - Ax
#         for i in range(5):
#             rho = self.vector_inner_product3(r, r)
#             if i == 0:
#                 p = r
#             else:
#                 beta = rho / rho_1
#                 p = r + beta * p
#
#             # Ap = self.dual_conv(p, kernel, mask)
#             # Ap = Ap + sigma * self.dual_conv(p, self.g1_kernel, mask_beta) \
#             #      + sigma * self.dual_conv(p, self.g2_kernel, mask_beta)
#             Ap = self.ax(p, kernel, weight, local_weight_reg, mask, 1)
#             q = Ap
#             alp = rho / self.vector_inner_product3(p, q)
#             x = x + alp * p
#             r = r - alp * q
#             rho_1 = rho
#
#         deconv_result = x
#
#         return deconv_result
#
#     @t.no_grad()
#     def get_sigma(self, x):
#         noise = x - self.media(x)
#         c, h, w = noise.shape[1:]
#         mean = t.mean(noise, dim=(-3, -2, -1), keepdim=True)
#         # print(t.sum((noise - mean).pow(2), dim=(-3, -2, -1), keepdim=True) / (c * h * w - 1))
#
#         sigma = t.sqrt(t.sum((noise - mean).pow(2), dim=(-3, -2, -1), keepdim=True) / (c * h * w - 1))
#         return sigma
#
#     def pad(self, blur, ksize):
#         # h, w = blur.shape[-2:]
#
#         pad = ksize // 2
#
#         pads = []
#         img_size = blur.shape[-2:]
#         for size in reversed(img_size):
#             psize = pad * 2 + size
#             if psize // self.window_size == ceil(psize / self.window_size):
#                 pads.extend([pad, pad])
#             else:
#                 pads.append(pad)
#                 padr = ceil(psize / self.window_size) * self.window_size
#                 padr = padr - pad - size
#                 pads.append(padr)
#
#         return F.pad(blur, pad=pads, mode='replicate')
#
#     def forward(self, blur, x, kernel):
#         h, w = blur.shape[-2:]
#         ksize = kernel.shape[-1]
#         x = self.deconv_func(blur, x, kernel)
#         deconv1 = x
#
#         return deconv1
#
# class OffsetNetwork(nn.Module):
#     def __init__(self, in_channels, kernel_size, n_feat=64):
#         super(OffsetNetwork, self).__init__()
#         self.body = nn.Sequential(
#             nn.Conv2d(in_channels, n_feat, kernel_size=3, padding=1, stride=1),
#             nn.ReLU(True),
#             nn.Conv2d(n_feat, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=1)
#         )
#
#     def forward(self, x):
#         return self.body(x)
#
# class Deblur6(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, window_size=4, n_feat=64, n_kernel=1):
#         super(Deblur6, self).__init__()
#         self.l2deblur = L2NonDeblurCG(in_channels, out_channels, window_size=window_size)
#         self.finetune1 = FineTuneDeblur(in_channels, out_channels, kernel_size=kernel_size, window_size=window_size, n_feat=n_feat, n_kernel=n_kernel)
#         self.finetune2 = FineTuneDeblur(in_channels, out_channels, kernel_size=kernel_size, window_size=window_size, n_feat=n_feat, n_kernel=n_kernel)
#         # self.finetune3 = FineTuneDeblur(in_channels, out_channels, kernel_size=kernel_size, window_size=window_size, n_feat=n_feat, n_kernel=n_kernel)
#         self.window_size = window_size
#     def pad(self, blur, ksize):
#         # h, w = blur.shape[-2:]
#
#         pad = ksize // 2
#
#         pads = []
#         img_size = blur.shape[-2:]
#         for size in reversed(img_size):
#             psize = pad * 2 + size
#             if psize // self.window_size == ceil(psize / self.window_size):
#                 pads.extend([pad, pad])
#             else:
#                 pads.append(pad)
#                 padr = ceil(psize / self.window_size) * self.window_size
#                 padr = padr - pad - size
#                 pads.append(padr)
#
#         return F.pad(blur, pad=pads, mode='replicate')
#
#     def forward(self, blur, kernel):
#         h, w = blur.shape[-2:]
#         # print('h, w:', h, w)
#         ksize = kernel.shape[-1]
#         pad = ksize // 2
#         sigma = 1#self.get_sigma(blur)
#         # x = self.pad(blur, ksize)
#         x = self.l2deblur(blur, kernel)
#         # x = x.detach()
#         x1 = self.finetune1(blur, x, kernel)
#         # x = self.finetune2(blur, x1, kernel)
#         x2 = x1
#         out = [blur, x1[:, :, pad: pad + h, pad: pad + w]]
#         x2 = self.finetune2(blur, x2, kernel)
#         out.append(x2[:, :, pad: pad + h, pad: pad + w])
#         # x3 = self.finetune3(blur, x2, kernel)
#         # out.append(x3[:, :, pad: pad + h, pad: pad + w])
#
#
#         return out #x1[:, :, pad: pad + h, pad: pad + w], deconv1
#
#
# class TVDeblur2d(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(TVDeblur2d, self).__init__()
#
#         self.g1_kernel = nn.Parameter(t.from_numpy(
#             np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]], dtype="float32").reshape((1, 1, 3, 3))),
#             requires_grad=False)
#         # .cuda()#to(self.device)
#         self.g2_kernel = nn.Parameter(t.from_numpy(
#             np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]], dtype="float32").reshape((1, 1, 3, 3))),
#             requires_grad=False)
#
#     @t.no_grad()
#     def forward(self, blur, otf, rho):
#         otft = t.conj(otf)
#         with t.no_grad():
#             g1 = convert_psf2otf(self.g1_kernel, blur.shape)
#             g2 = convert_psf2otf(self.g1_kernel, blur.shape)
#             g1t = t.conj(g1)
#             g2t = t.conj(g2)
#             sumg = g1 * g1t + g2 * g2t
#         a = otft * blur
#         b = otft * otf + rho * sumg
#         return t.real(t.fft.ifft2(a / b))


class L2NonDeblurCG(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(L2NonDeblurCG, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.g1_kernel = nn.Parameter(t.from_numpy(
            np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]], dtype="float32").reshape((1, 1, 3, 3))),
            requires_grad=False)
        # .cuda()#to(self.device)
        self.g2_kernel = nn.Parameter(t.from_numpy(
            np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]], dtype="float32").reshape((1, 1, 3, 3))),
            requires_grad=False)

        self.media = MedianPool2d(kernel_size=5, same=True)

    @t.no_grad()
    def kx_y(self, x, psf, blur):
        kx = self.conv_blur(t.clamp(x, 0, 1), psf)
        out = rgb2gray(kx - blur)
        return out

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
        # x1 = x1.view(b, c, -1)
        # x2 = x2.view(b, c, -1)
        # re = x1 * x2
        # re = t.sum(re, dim=2)
        # re = re.view(b, c, 1, 1)
        re = x1.mul(x2).sum(dim=(-2, -1), keepdim=True)
        return re

    def deconv_func(self, input, kernel, sigma):  # , beta11, beta22, beta12
        b, c, h, w = input.size()
        kb, kc, ksize, ksize = kernel.size()
        psize = ksize // 2
        input = self.pad(input, ksize)
        # input = F.pad(input, pad=(psize, psize, psize, psize), mode='replicate')
        input_ori = input
        # assert b == 1, "only support one image deconv operation!"

        kernel = self.auto_crop_kernel(kernel)
        assert ksize % 2 == 1, "only support odd kernel size!"

        mask = t.zeros_like(input).to(input.device)
        mask[:, :, psize:-psize, psize:-psize] = 1.
        mask_beta = t.ones_like(input).to(input.device)

        x = input

        # weight = self.robust(self.conv_func(x, t.rot90(kernel, k=2, dims=(-2, -1))) - input, 0.02)

        b = self.conv_func(input_ori * mask, kernel, padding='same')

        Ax = self.dual_conv(x, kernel, mask)
        Ax = Ax + sigma * self.dual_conv(x, self.g1_kernel, mask_beta) \
             + sigma * self.dual_conv(x, self.g2_kernel, mask_beta)

        r = b - Ax
        for i in range(20):
            rho = self.vector_inner_product3(r, r)
            if i == 0:
                p = r
            else:
                beta = rho / rho_1
                p = r + beta * p

            Ap = self.dual_conv(p, kernel, mask)
            Ap = Ap + sigma * self.dual_conv(p, self.g1_kernel, mask_beta) \
                 + sigma * self.dual_conv(p, self.g2_kernel, mask_beta)

            q = Ap
            alp = rho / self.vector_inner_product3(p, q)
            x = x + alp * p
            r = r - alp * q
            rho_1 = rho

        # deconv_result = x[:, :, psize: -psize, psize: -psize]
        deconv_result = x[:, :, psize: -psize, psize: -psize]
        return deconv_result

    def pad(self, blur, ksize):
        # h, w = blur.shape[-2:]

        pad = ksize // 2

        pads = [pad for _ in range(4)]
        # img_size = blur.shape[-2:]
        # for size in reversed(img_size):
        #     psize = pad * 2 + size
        #     if psize // self.window_size == ceil(psize / self.window_size):
        #         pads.extend([pad, pad])
        #     else:
        #         pads.append(pad)
        #         padr = ceil(psize / self.window_size) * self.window_size
        #         padr = padr - pad - size
        #         pads.append(padr)

        return F.pad(blur, pad=pads, mode='replicate')


    def forward(self, blur, kernel):
        noise = blur - self.media(blur)
        c, h, w = noise.shape[1:]
        mean = t.mean(noise, dim=(-3, -2, -1), keepdim=True)
        sigma = t.sqrt(t.sum((noise - mean).pow(2), dim=(-3, -2, -1), keepdim=True) / (c * h * w - 1))
        # print(sigma)
        # sigma = 0.01
        deconv1 = self.deconv_func(blur, kernel, sigma)

        return deconv1

class DeblurMask(nn.Module):
    def  __init__(self, in_channels, kernel_size=3, n_feat=16):
        super(DeblurMask, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels * 3, n_feat, kernel_size=kernel_size, padding=kernel_size // 2, stride=1)
        )

        self.body1 = RCAB(n_feat, kernel_size)
        self.body2 = RCAB(n_feat, kernel_size)
        self.body3 = RCAB(n_feat, kernel_size)

        self.tail = nn.Sequential(
            nn.Conv2d(n_feat, in_channels, kernel_size=kernel_size, padding=kernel_size // 2, stride=1),
            nn.Sigmoid()
        )

        self.l2deblur = L2NonDeblurCG(in_channels, out_channels=in_channels)

    # def robust(self, x, w):
    #     # print(x.dtype, w.dtype)
    #
    #     # y = t.where(t.abs(rgb2gray(x)) <= w, t.zeros_like(x), x)
    #     y = F.softshrink(x, 0.05)
    #     y = 1 - F.tanh(y.div(w).pow(2))
    #
    #     # print(y.min())
    #     return y

    def robust(self, x, a=0.0):
        a = 459 / sqrt(2 * math.pi)
        b = -2601 / 2
        return 1 -  1 / (1 + a * t.exp(b * x.pow(2)))


    def forward(self, x, psf):
        ksize = psf.shape[-1]
        pad = ksize // 2
        with t.no_grad():
            # deb = self.l2deblur(x, psf)
            # deb = t.clip(deb, 0, 1).mul(255.).round().div(255.)
            deb_pad = F.pad(x, (pad, pad, pad, pad), mode='replicate')
            reb = self.l2deblur.conv_func(deb_pad, psf)

            reb = reb[:, :, pad: -pad, pad: -pad]

        out = t.cat([x, t.abs(x - reb), reb], dim=1)
        a = self.head(out)
        a1 = self.body1(a)
        a2 = self.body2(a1)
        a3 = self.body3(a2)
        out = self.tail(a3)

        return a, a1, a2, a3, out
        # if self.training:
        #     return a, a1, a2, a3, out
        # else:
        #     return out

        # return reb2 , out


# class DeblurMaskED(nn.Module):
#     def __init__(self, in_channels, kernel_size=3, n_feat=16):
#         super(DeblurMaskED, self).__init__()
#         self.head = nn.Sequential(
#             nn.Conv2d(in_channels * 2, n_feat, kernel_size=kernel_size, padding=kernel_size // 2, stride=1)
#         )
#
#         self.body1 = RCAB(n_feat, 3)
#         self.body2 = RCAB(n_feat, 3)
#         self.body3 = RCAB(n_feat, 3)
#
#         self.tail = nn.Sequential(
#             nn.Conv2d(n_feat, in_channels, kernel_size=kernel_size, padding=kernel_size // 2, stride=1),
#             nn.Sigmoid()
#         )
#
#         self.l2deblur = L2NonDeblurCG(in_channels, out_channels=in_channels)
#
#     # def robust(self, x, w):
#     #     # print(x.dtype, w.dtype)
#     #
#     #     # y = t.where(t.abs(rgb2gray(x)) <= w, t.zeros_like(x), x)
#     #     y = F.softshrink(x, 0.05)
#     #     y = 1 - F.tanh(y.div(w).pow(2))
#     #
#     #     # print(y.min())
#     #     return y
#
#
#     # def robust(self, x, a=0.0):
#     #     a = 459 / sqrt(2 * math.pi)
#     #     b = -2601 / 2
#     #     return 1 -  1 / (1 + a * t.exp(b * x.pow(2)))
#     def forward(self, x, psf):
#         ksize = psf.shape[-1]
#         pad = ksize // 2
#         # with t.no_grad():
#         #     deb = self.l2deblur(x, psf)
#         #     deb = t.clip(deb, 0, 1).mul(255.).round().div(255.)
#         #     deb_pad = F.pad(deb, (pad, pad, pad, pad), mode='replicate')
#         #     reb = self.l2deblur.conv_func(deb_pad, psf)
#         #
#         #     reb = reb[:, :, pad: -pad, pad: -pad]
#         #     # deb_pad = F.pad(deb, (pad, pad, pad, pad), mode='replicate')
#         #     reb2 = self.l2deblur.conv_func(deb_pad, psf)
#         #     reb2 = reb2[:, :, pad: -pad, pad: -pad]
#
#             # print(reb2.shape)
#
#         # out = t.cat([], dim=1)
#
#
#         deb_pad = F.pad(x, (pad, pad, pad, pad), mode='replicate')
#
#         reb = self.l2deblur.conv_func(deb_pad, psf)[:,:, pad: -pad, pad: -pad]
#         out = t.cat([reb, x], dim=1)
#
#         a = self.head(out)
#         a1 = self.body1(a)
#         a2 = self.body2(a1)
#         a3 = self.body3(a2)
#         out = self.tail(a3)
#
#         if self.training:
#             return a, a1, a2, a3, out
#         else:
#             return out



if __name__ == '__main__':
    # deblur = Deblur(3, 3, 3, 4).cuda()
    # x = t.randn([8, 3, 256, 256]).cuda()
    # psf = t.randn([8, 3, 15, 15]).cuda()
    # out = deblur(x, psf)
    # print(out.shape)
    n2t = torchvision.transforms.ToTensor()

    deblur = DeblurMask(3, kernel_size=3).cuda()
    # state = '/home/echo/code/python/deblur/outliner_deblur/experiment/Deblur4/outputs/training-2022-01-27_01-29-17/chkpt/training_latest.state'
    # state_dict = t.load(state, map_location=t.device("cuda:0"))['model']
    # deblur.load_state_dict(state_dict)

    x = t.randn([4, 3, 256, 256]).cuda()
    psf = t.randn([4, 1, 31, 31]).cuda()

    outs = deblur(x, psf)
    print(outs[-1].shape)
    # blur = n2t(imread('/home/echo/code/python/deblur/outliner_deblur/model/19021_blur.png')).unsqueeze(0).cuda()
    # # psf = n2t(imread('/home/echo/code/python/deblur/outliner_deblur/model/my_test_car6_kernel.png')).unsqueeze(0).cuda()
    # psf = t.from_numpy(io.loadmat('/home/echo/code/python/deblur/outliner_deblur/model/19021_kernel.mat')['kernel'].astype(np.float32)).unsqueeze(0).unsqueeze(0).cuda()
    # psf = t.rot90(psf, dims=(-2, -1), k=2)
    # with t.no_grad():
    #     psf = psf / psf.sum()
    #     out = deblur(blur, psf)
    #     print(len(out))
    #     imwrite('19021_deblur.png', out[-1])


