import math
from typing import Union

import torch as t
import torch.nn as nn

from math import exp, sqrt
import numpy as np
from torch.nn.modules.loss import _Loss
from torchvision import models

from layer.discriminator import MSDiscriminator
from model import make_model
from model.deblurmasked import DeblurMaskED
# from model.noiseed import NoiseEncoder, NoiseDecoder
from model.unet import UNet
from model.vgg import vgg19
import torch.nn.functional as F

from utils.image import imwrite, psf2otf


class PerceptualLoss(nn.Module):
    def __init__(self, model='vgg19', pretrain=True, is_gray=False, **kwargs):
        super(PerceptualLoss, self).__init__()
        if model == 'vgg19':
            features = vgg19(pretrained=pretrain, is_gray=is_gray).features

        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()
        self.mse = nn.MSELoss()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        h_x = self.to_relu_1_2(x)
        h_relu_1_2_x = h_x
        h_x = self.to_relu_2_2(h_x)
        h_relu_2_2_x = h_x
        h_x = self.to_relu_3_3(h_x)
        h_relu_3_3_x = h_x
        h_x = self.to_relu_4_3(h_x)
        h_relu_4_3_x = h_x
        out_x = (h_relu_1_2_x, h_relu_2_2_x, h_relu_3_3_x, h_relu_4_3_x)

        h_y = self.to_relu_1_2(y)
        h_relu_1_2_y = h_y
        h_y = self.to_relu_2_2(h_y)
        h_relu_2_2_y = h_y
        h_y = self.to_relu_3_3(h_y)
        h_relu_3_3_y = h_y
        h_y = self.to_relu_4_3(h_y)
        h_relu_4_3_y = h_y
        out_y = (h_relu_1_2_y, h_relu_2_2_y, h_relu_3_3_y, h_relu_4_3_y)

        sum_loss = 0

        for i in range(len(out_x)):
            sum_loss += self.mse(out_y[i], out_x[i])

        return sum_loss

class MSELoss(nn.MSELoss):
    def __init__(self, **kwargs):
        super(MSELoss, self).__init__()

class L1Loss(nn.L1Loss):
    def __init__(self, **kwargs):
        super(L1Loss, self).__init__()

class ML1Loss(nn.Module):
    def __init__(self):
        super(ML1Loss, self).__init__()

    def forward(self, input, target):
        return t.mean(t.abs(input - target))


# 计算一维的高斯分布向量
def gaussian(window_size, sigma):
    gauss = t.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


# 创建高斯核，通过两个一维高斯分布向量进行矩阵乘法得到
# 可以设定channel参数拓展为3通道
def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


# 计算SSIM
# 直接使用SSIM的公式，但是在计算均值时，不是直接求像素平均值，而是采用归一化的高斯核卷积来代替。
# 在计算方差和协方差时用到了公式Var(X)=E[X^2]-E[X]^2, cov(X,Y)=E[XY]-E[X]E[Y].
# 正如前面提到的，上面求期望的操作采用高斯核卷积代替。
def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if t.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if t.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = t.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret

def ssim_map(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if t.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if t.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = t.mean(v1 / v2)  # contrast sensitivity

    _ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    # if size_average:
    #     ret = ssim_map.mean()
    # else:
    #     ret = ssim_map.mean(1).mean(1).mean(1)
    #
    # if full:
    #     return ret, cs
    return _ssim_map

# Classes to re-use window
class SSIMLoss(_Loss):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)



class CharbonnierLoss(_Loss):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = t.mean(t.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss


class L2Loss(_Loss):
    """Charbonnier Loss (L1)"""

    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = t.sum(diff.pow(2)) #t.mean(t.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class SelfDistillConstrastLoss(_Loss):
    def __init__(self, gammas:Union[tuple, list], extractor, urls=None, ablation=False, eps=1e-7):
        super(SelfDistillConstrastLoss, self).__init__()
        teacher = make_model(extractor)
        if urls is not None:
            state_dict = t.load(urls)
            teacher.load_state_dict(state_dict)
        self.ablation = ablation
        self.gammas = gammas
        self.inBlock1_1 = nn.Sequential()
        self.encoders = nn.ModuleList()

        self.l1loss = nn.L1Loss()
        self.eps = eps

        if extractor.model =='UNet':
            self.inBlock1_1.add_module('inBlock1_1', teacher.inBlock1_1)
            self.encoders.append(
                teacher.encoder_first_1,
            )
            self.encoders.append(
                teacher.encoder_second_1
            )
            # self.encoder_first_1.add_module('encoder_first_1', teacher.encoder_first_1)
            # self.encoder_second_1.add_module('encoder_second_1', teacher.encoder_second_1)

        for param in self.parameters():
            param.requires_grad = False
            # print(param.names, param.requires_grad)


    def forward(self, x: Union[list, tuple], y):
        negative_feat, hqs_deblurs, *deblur_feats  = x

        positive_feat = self.inBlock1_1(y)

        negative_feat_encs = []
        negative_feat_enc = negative_feat

        positive_feat_encs = []

        positive_feat_enc = positive_feat
        # print('negative_feat_enc:', negative_feat_enc.shape)
        for i, (module) in enumerate(self.encoders):
            negative_feat_enc = module(negative_feat_enc)
            negative_feat_encs.append(negative_feat_enc)
            # print(positive_feat_enc.shape)
            positive_feat_enc = module(positive_feat_enc)
            # print(positive_feat_enc.shape)
            # print('======================')
            positive_feat_encs.append(positive_feat_enc)

        loss = 0
        for deblur in hqs_deblurs:
            # print('hqs_deblurs:', deblur.shape,  positive_feat.shape, negative_feat.shape)
            loss += self.compute_loss(1, deblur,  positive_feat, negative_feat)

        for (gamma, anchor, positive, negative) in zip(self.gammas, deblur_feats, positive_feat_encs, negative_feat_encs):
            # print(anchor.shape, positive_feat.shape, negative_feat.shape)
            loss += self.compute_loss(gamma, anchor, positive, negative)

        return loss


    def compute_loss(self, gamma, anchor, positive, negative):

        d_ap = self.l1loss(anchor, positive)
        if self.ablation:
            d_an = self.l1loss(anchor, negative ) + self.eps
            loss = d_ap / d_an
        else:
            loss = d_ap

        return gamma * loss

# class DeblurLoss(_Loss):
#     def __init__(self, reg_coefficients: Union[tuple, list], gammas, extractor, urls=None, ablation=False, eps=1e-7):
#         super(DeblurLoss, self).__init__()
#         self.loss_1 = CharbonnierLoss()
#         self.constrast = SelfDistillConstrastLoss(gammas, extractor, urls=urls, ablation=ablation, eps=eps)
#         self.reg_coefficients = reg_coefficients
#         assert len(reg_coefficients) == 2, 'length of reg_coefficients must be 2!'
#
#     def forward(self, x, y):
#         *constrast_input, deblur = x
#         # print('deblur, y:', deblur.shape, y.shape)
#         loss_1 = self.reg_coefficients[0] * self.loss_1(deblur, y) + self.reg_coefficients[1] * self.constrast(constrast_input, y)
#         return loss_1

# class DeblurLoss(_Loss):
#     def __init__(self, noiseed_model, noiseed_url):
#         super(DeblurLoss, self).__init__()
#         noiseed = make_model(noiseed_model)
#         if noiseed_url is not None:
#             noiseed.load_state_dict(noiseed_url)
#         self.noisee = noiseed.encoder
#
#     def forward(self, x, y):
#         out,

class AELoss(_Loss):
    def __init__(self):
        super(AELoss, self).__init__()

    def forward(self, x, y):
        #[blur_o, noise_gto, noise_o, blur_nonoise]
        blur, noise_gto, noise_o, blur_no_noise = x
        blur_gt, noise_gt, noise_gt_, blur_no_noise_gt = y
        loss = F.l1_loss(blur, blur_gt) + F.l1_loss(noise_gto, noise_gt) + F.l1_loss(blur_no_noise, blur_no_noise_gt) + F.l1_loss(noise_o, noise_gt_)

        return loss

# class ADMMLoss(_Loss):
#     def __init__(self, in_channels, out_channels, kernel_size, n_feat, pre_noiseed=None):
#         super(ADMMLoss, self).__init__()
#         self.noise_enc = NoiseEncoder(in_channels, n_feat, kernel_size)
#         self.noise_dec = NoiseDecoder(n_feat, out_channels, kernel_size)
#
#         if pre_noiseed is not None:
#             state = t.load(pre_noiseed)
#             self.noise_enc.load_state_dict(state, strict=False)
#             self.noise_dec.load_state_dict(state, strict=False)
#
#
#     def forward(self, x, y):
#         blur, reblur, noise_estfeat, out = x
#         noise_gt, sharp = y
#         # loss_1 = F.l1_loss(blur, reblur)
#         loss_2 = F.l1_loss(sharp, out)
#         noise_gtfeat = self.noise_enc(noise_gt)
#         # noise_out = self.noise_dec(noise_gtfeat)
#         # loss_3 = F.l1_loss(noise_gt, noise_out)
#
#         loss_4 = F.l1_loss(noise_gtfeat, noise_estfeat)
#
#         # return loss_1 +loss_2 + loss_3 + loss_4
#         return loss_2 + loss_4




class DeblurL1Loss(_Loss):
    def __init__(self):
        super(DeblurL1Loss, self).__init__()
        # self.contrast_loss = ContrastLoss(True)

    def forward(self, x, y):
        blur, *x = x
        loss = 0
        i = 0
        for x_ in x:
            # print(i, type(x_))
            i += 1
            loss += F.l1_loss(x_, y) #+ self.contrast_loss(x_, y, blur))
        return loss


class DWDNLoss(_Loss):
    def __init__(self):
        super(DWDNLoss, self).__init__()

    def forward(self, x, y):
        loss = 0
        for x_ in x:
            y_ = F.interpolate(y, size=x_.shape[-2:], mode='bilinear')
            loss += F.l1_loss(x_, y_)

        return loss

class DeblurMaskLoss(_Loss):
    def __init__(self):
        super(DeblurMaskLoss, self).__init__()

        self.g1_kernel = nn.Parameter(t.from_numpy(
            np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]], dtype="float32").reshape((1, 1, 3, 3))),
            requires_grad=False)
        # .cuda()#to(self.device)
        self.g2_kernel = nn.Parameter(t.from_numpy(
            np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]], dtype="float32").reshape((1, 1, 3, 3))),
            requires_grad=False)
        self.p_loss = PerceptualLoss(pretrain=False)

    def robust(self, x, w):
        # print(x.dtype, w.dtype)

        # y = t.where(t.abs(rgb2gray(x)) <= w, t.zeros_like(x), x)
        y = F.softshrink(x, 0.05)
        y = 1 - F.tanh(y.div(w).pow(2))

        # print(y.min())
        return y

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


    def forward(self, x, y):
        reb, mask_out = x
        blur = y
        gt_mask = self.robust(t.abs(reb - blur), 0.05)
        # print(gt_mask.shape)
        imwrite('gt_mask.png', t.cat([gt_mask, mask_out], dim=-1))

        # print(mask_out.shape, reb.shape, y.shape)

        # mx = self.conv_func(mask_out, self.g1_kernel)
        # my = self.conv_func(mask_out, self.g2_kernel)
        #
        # mgx = self.conv_func(1 - gt_mask, self.g1_kernel)
        # mgy = self.conv_func(1 - gt_mask, self.g2_kernel)

        return F.l1_loss(mask_out, gt_mask) # + (self.p_loss(mask_out, 1 - gt_mask))  #+ 0.05 * t.abs(mask_out).sum(dim=(-3, -2, -1)).mean()#t.abs(mask_out * (reb - y)).sum()
        # with t.no_grad():
        #     det = y - reb
        #     t.where(det < 0.05)

class DeblurMaskLoss2(_Loss):
    def __init__(self):
        super(DeblurMaskLoss2, self).__init__()

        self.g1_kernel = nn.Parameter(t.from_numpy(
            np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]], dtype="float32").reshape((1, 1, 3, 3))),
            requires_grad=False)
        # .cuda()#to(self.device)
        self.g2_kernel = nn.Parameter(t.from_numpy(
            np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]], dtype="float32").reshape((1, 1, 3, 3))),
            requires_grad=False)
        self.p_loss = PerceptualLoss(pretrain=False)

    def robust(self, x, a):
        return 1 - t.tanh(F.softshrink(x, a).div(a).pow(2))

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


    def forward(self, x, y):
        mask_out = x[-1]
        blur, reb = y
        gt_mask = self.robust(t.abs(reb - blur), 0.039)
        # c = gt_mask.shape[1]
        # gt_mask, _ = t.min(gt_mask, dim=1, keepdim=True)
        # gt_mask = gt_mask.repeat(1, c, 1, 1)
        # gt_mask = t.where(gt_mask < 0.6, t.zeros_like(gt_mask), t.ones_like(gt_mask))
        imwrite('mask.png', t.cat([gt_mask, mask_out], dim=-1))
        return F.mse_loss(mask_out, gt_mask)  #+ F.l1_loss(mask_out * reb, mask_out * blur)  #+ 0.05 * t.abs(mask_out).sum(dim=(-3, -2, -1)).mean()#t.abs(mask_out * (reb - y)).sum()


class DeblurMaskSelfDistillLoss(_Loss):
    def __init__(self, teacher_path, in_channels, kernel_size, n_feat):
        super(DeblurMaskSelfDistillLoss, self).__init__()
        self.teacher = DeblurMaskED(in_channels, kernel_size, n_feat)

        state = t.load(teacher_path)#['model']
        self.teacher.load_state_dict(state)

        for param in self.teacher.parameters():
            param.requires_grad = False

    def robust(self, x, a):
        return 1 - t.tanh(F.softshrink(x, a).div(a).pow(2))

    def forward(self, x, y):
        blur, reblur = y
        res = blur - reblur
        label_gt = self.robust(res.abs(), 0.039)
        # a, a1, a2, a3, out = x
        # print(blur.shape, reblur.shape)
        label_outs = self.teacher(blur, reblur)
        *label_features, _ = label_outs
        *x_, x_out = x
        loss = 0


        # imwrite('gt.png', _)
        imwrite('output_mask.png', t.cat([label_gt, _, x_out], dim=-1))
        for (a, b) in zip(x_, label_features):
            # print(a.shape, b.shape)
            loss += F.mse_loss(a, b.detach())

        loss += F.mse_loss(x_out, label_gt)

        return loss


class DeblurMaskSelfDistillTestLoss(_Loss):
    def __init__(self, teacher_path, in_channels, kernel_size, n_feat):
        super(DeblurMaskSelfDistillTestLoss, self).__init__()
        self.teacher = DeblurMaskED(in_channels, kernel_size, n_feat)

        state = t.load(teacher_path)['model']
        self.teacher.load_state_dict(state)

        for param in self.teacher.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        self.teacher.eval()
        blur, reblur = y
        a, a1, a2, a3, out = x
        # print(blur.shape, reblur.shape)
        label_outs = self.teacher(blur, reblur)
        # loss = 0
        imwrite('label_outs.png', t.cat([label_outs[-1], out], dim=-1))
        # print(type(label_outs[-1]), type(x))

        # for (a, b) in zip(x, label_outs):
        #     print(a.shape, b.shape)
        #     loss += F.mse_loss(a, b)
        loss = F.mse_loss(label_outs[-1], out)
        return loss

class DeblurMaskRefineLoss(_Loss):
    def __init__(self):
        super(DeblurMaskRefineLoss, self).__init__()

    def robust(self, x, a):
        return 1 - t.tanh(F.softshrink(x, a).div(a).pow(2))#1 -  1 / (1 + a * torch.exp(b * x.pow(2)))

    def forward(self, x, y):
        blur, reblur = y
        label_map = self.robust(blur - reblur, 0.039)
        imwrite('mask.png', t.cat([label_map, x[-1]], dim=-1))
        loss = F.l1_loss(x[-1], label_map)
        return loss

class Deblurl08nonLoss(_Loss):
    def __init__(self):
        super(Deblurl08nonLoss, self).__init__()

    def forward(self, x, y):
        outputs = x
        loss = 0
        for output in outputs:
            loss += F.l1_loss(output, y)
        return loss

class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]

class ContrastLoss(nn.Module):
    def __init__(self, ablation=False):

        super(ContrastLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.l1 = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.ab = ablation

    def forward(self, a, p, n):
        a_vgg, p_vgg, n_vgg = self.vgg(a), self.vgg(p), self.vgg(n)
        loss = 0

        d_ap, d_an = 0, 0
        for i in range(len(a_vgg)):
            d_ap = self.l1(a_vgg[i], p_vgg[i].detach())
            if not self.ab:
                d_an = self.l1(a_vgg[i], n_vgg[i].detach())
                contrastive = d_ap / (d_an + 1e-7)
            else:
                contrastive = d_ap

            loss += self.weights[i] * contrastive
        return loss

class DeblurFrameLoss(_Loss):
    def __init__(self):
        super(DeblurFrameLoss, self).__init__()
        # self.is_constrast = True
        # self.constrast = ContrastLoss(True)
        # self.teacher = UNet(3, 3, 5, 32, 3)
        # state = t.load('/home/echo/code/python/lowlight/outlierdeblur/experiment/pretrain/training_epoch_30.state')['model']
        # self.teacher.load_state_dict(state)
        # for param in self.teacher.parameters():
        #     param.requires_grad = False

    def forward(self, x, y):
        deconvs, *outputs = x
        loss = 0
        # label_outs = self.teacher(y)
        # print(type(outputs))
        imwrite('tmp.png', t.cat([deconvs[-1], outputs[-1],  y], dim=-1))
        loss = F.l1_loss(outputs[-1], y)

        for deconv in deconvs:
            loss += loss + F.l1_loss(deconv, y)

        return loss

class DeblurFrame2Loss(_Loss):
    def __init__(self):
        super(DeblurFrame2Loss, self).__init__()
        # self.is_constrast = True
        # self.constrast = ContrastLoss(True)
        # self.teacher = UNet(3, 3, 5, 32, 3)
        # state = t.load('/home/echo/code/python/lowlight/outlierdeblur/experiment/pretrain/training_epoch_30.state')['model']
        # self.teacher.load_state_dict(state)
        # for param in self.teacher.parameters():
        #     param.requires_grad = False

    def forward(self, x, y):
        deconvs = x
        loss = 0
        imwrite('tmp.png', t.cat([deconvs[-1], deconvs[-2]], dim=-1))
        for deconv in deconvs:
            loss += loss + F.l1_loss(deconv, y)

        return loss

#
# if __name__ == '__main__':
#     n2t = tvf.ToTensor()
#     img1 = n2t(imread('1461_kernel_01.png')).unsqueeze(0)
#
#     n2t = tvf.ToTensor()
#     img2 = n2t(imread('1461_kernel_04.png')).unsqueeze(0)
#
#     map = ssim_map(img1, img2)
#     imwrite('map.png', map)
#     print(map.min(), map.max())
#     sigma = 20
#     mu = -1
#     map2 = 2 * t.exp(-1 * ((map - mu).pow(2) / (sigma ** 2)))
#     print(map2.min(), map2.max())
#     imwrite('map2.png', map2)