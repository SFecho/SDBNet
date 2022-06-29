import torch as t
import torch.nn as nn

from model.maskl2cgdeblur import MaskL2CGDeblur
from model.msuformer import MSUformer
from model.outlierdeblur2 import OutlierDeblur2
from model.unet import UNet
import torchvision.transforms as tvf
import imageio
import numpy as np


class DeblurFrame2(nn.Module):
    def __init__(self, in_channels, out_channels, mask_path=None):
        super(DeblurFrame2, self).__init__()
        self.deconv = OutlierDeblur2(in_channels, out_channels, mask_path)


    def forward(self, blur, psf):
        deconv = self.deconv(blur, psf)

        return deconv



# class UNet(nn.Module):
#     def __init__(self):
#         super(UNet, self).__init__()
#         self.he
#
#     def forward(self, x):
#


if __name__ == '__main__':
    model = DeblurFrame2(3, 3, None).cuda()

    n2t = tvf.ToTensor()
    img = n2t(imageio.imread('1102019.png')).unsqueeze(0).cuda()
    psf = n2t(imageio.imread('1102019_kernel.png')).unsqueeze(0).cuda()
    psf = psf / psf.sum()
    state_dict = '/home/echo/code/python/lowlight/outlierdeblur/experiment/DeblurFrame/outputs/training-2022-03-05_17-52-31/chkpt/training_epoch_31.state'
    state_dict = t.load(state_dict)['model']
    model.load_state_dict(state_dict)
    # x = t.randn([4, 3, 256, 256]).cuda()
    # psf = t.randn([4, 3, 31, 31]).cuda()
    with t.no_grad():
        out = model(img, psf)
        out = out[-1].cpu().squeeze(0).permute(1, 2, 0).clip(0, 1).mul(255).round().numpy()
        out = out.astype(np.uint8)
        imageio.imwrite('1102019_car6-3.png', out)
