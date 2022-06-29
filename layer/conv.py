import torch as t
import torch.nn as nn
from torch.autograd import Function
# import non_uniform_conv
from torch.nn import LayerNorm
from torch.nn.utils import weight_norm
import torch.nn.functional as F
from layer.norm import spectral_norm, AdaptiveInstanceNorm2d

import torchvision.ops.deform_conv as deform_conv2d

class CrossCorrelation(nn.Module):
    def __init__(self, max_shiftx, max_shifty, is_add_flips=False, **kwargs):
        super(CrossCorrelation, self).__init__()
        self.max_shift_x = max_shiftx
        self.max_shift_y = max_shifty
        self.is_add_flips = is_add_flips


    def forward(self, x):
        b, c, h, w = x.shape
        x_std = self._standartize(x)
        x_std_f = t.fft.fft2(x_std)
        x_std_f_conj = t.conj(x_std_f)

        cc_freq_domain_list = []

        for i in range(c):
            cc_freq_domain_list.append(
                    x_std_f[:, i, :, :].unsqueeze(1) * x_std_f_conj[:, i:, :, :]
            )

        cc_freq_domain = t.real(t.fft.ifft2(
            t.cat(cc_freq_domain_list, dim=1)
        ))

        tl = cc_freq_domain[:, :, :self.max_shift_y + 1, :self.max_shift_x + 1]
        tr = cc_freq_domain[:, :, :self.max_shift_y + 1, -self.max_shift_x:]
        bl = cc_freq_domain[:, :, -self.max_shift_y:, :self.max_shift_x + 1]
        br = cc_freq_domain[:, :, -self.max_shift_y:, -self.max_shift_x:]

        # h = 2*self.max_shift_y + 1, w = 2*self.max_shift_x + 1
        cc_freq_domain = t.cat((t.cat((br, bl), dim=3), t.cat((tr, tl), dim=3)), dim=2)

        if self.is_add_flips:
            cc_freq_domain = t.cat([cc_freq_domain, cc_freq_domain[:, :, ::-1, ::-1]], dim=1)

        cc_freq_domain = cc_freq_domain.view(self.compute_output_shape(cc_freq_domain.shape))

        return cc_freq_domain


    def _standartize(self, x, dims=(2, 3), std_eps=1e-9):

        N = t.prod(t.gather(x.shape, dim=dims))
        x = x - t.mean(x, dim=dims, keepdim=True)
        stds = t.std(x, dim=dims, keepdim=True)
        stds = t.where(stds < std_eps, t.fill_(t.zeros_like(stds), t.tensor(float('inf'))), stds)
        x.div_(stds * t.sqrt(N.float()))
        return x

    def compute_output_shape(self, input_shape):
        n_features = input_shape[1]
        out_features = (n_features * (n_features + 1)) // 2

        if self.is_add_flips:
            out_features *= 2

        return input_shape[0], out_features, self.max_shift_y * 2 + 1, self.max_shift_x * 2 + 1


class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, n_feature, ksize=3, norm=None, act=nn.ReLU(True), groups=1, bias=True, **kwargs):
        super(ResBlock, self).__init__()
        self.act = act
        pad = ksize // 2

        self.norm = nn.BatchNorm2d(n_feature) if norm is not None else nn.Identity()

        self.conv1 = nn.Conv2d(n_feature, n_feature, kernel_size=ksize, padding=pad, stride=1, groups=groups, bias=bias)
        self.conv2 = nn.Conv2d(n_feature, n_feature, kernel_size=ksize, padding=pad, stride=1, groups=groups, bias=bias)

    def forward(self, inputs):
        out1 = self.conv1(inputs)
        out1 = self.norm(out1)
        out1 = self.act(out1)
        out2 = self.conv2(out1)

        return out2 + inputs

class Conv(nn.Module):
    def __init__(self , input_channels , n_feats , kernel_size , stride = 1 ,padding=0 , groups=1, bias=True , bn=False , act=False):
        super(Conv , self).__init__()
        m = []
        assert n_feats % groups == 0
        m.append(nn.Conv2d(input_channels , n_feats, kernel_size , stride , padding , bias=bias, groups=groups))
        if bn: m.append(nn.BatchNorm2d(n_feats))
        if act:m.append(nn.ReLU(True))

        self.body = nn.Sequential(*m)

    def forward(self, input):
        return self.body(input)

class Deconv(nn.Module):
    def __init__(self, input_channels, n_feats, kernel_size, stride=2, padding=0, output_padding=0 , bias=True, act=False):
        super(Deconv, self).__init__()
        m = []
        m.append(nn.ConvTranspose2d(input_channels, n_feats, kernel_size, stride=stride, padding=padding,output_padding=output_padding, bias=bias))
        if act: m.append(nn.ReLU(True))
        self.body = nn.Sequential(*m)

    def forward(self, input):
        return self.body(input)


class UpsampleConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, interpolation_mode='bilinear', **kwargs):
        super(UpsampleConv2d, self).__init__()
        # assert scale_factor > 1, 'scale_factor must in (1, +infty)'
        self.body = nn.Sequential(
            nn.Upsample(scale_factor=2, mode=interpolation_mode),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=kernel_size // 2, stride=1)
        )

    def forward(self, x):
        return self.body(x)

class DownsampleConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, interpolation_mode='bilinear', **kwargs):
        super(DownsampleConv2d, self).__init__()
        # assert scale_factor > 0 and scale_factor < 1, 'scale_factor must in (0, 1)'
        self.body = nn.Sequential(
            nn.Upsample(scale_factor=0.5, mode=interpolation_mode),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=kernel_size // 2, stride=1)
        )

    def forward(self, x):
        return self.body(x)


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, n_feature, kernel_size, reduction=16,
        bias=True, norm=None, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(nn.Conv2d(n_feature, n_feature, kernel_size, 1, kernel_size // 2, bias=bias))
            if norm is not None: modules_body.append(nn.BatchNorm2d(n_feature))
            if i == 0: modules_body.append(act)

        modules_body.append(CALayer(n_feature, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

# class NonUniformConv2dFunction(Function):
#     @staticmethod
#     def forward(ctx, inputs, kernel):
#         outputs = non_uniform_conv.forward(inputs, kernel)
#         variables = [inputs, kernel]
#         ctx.save_for_backward(*variables)
#         return outputs
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         inputs, kernel = ctx.saved_variables
#         d_input, d_kernel = non_uniform_conv.backward(
#             grad_output.contiguous(), inputs, kernel)
#         return d_input, d_kernel


# class NonUniformConv2d(nn.Module):
#     def __init__(self, **kwargs):
#         super(NonUniformConv2d, self).__init__()
#
#     def forward(self, inputs, kernel):
#         return NonUniformConv2dFunction.apply(inputs, kernel)

class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)
        # print('offset:', offset.shape)
        if self.modulation:
            m = t.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)
        # print('p:', p.shape)
        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = t.cat([t.clamp(q_lt[..., :N], 0, x.size(2)-1), t.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = t.cat([t.clamp(q_rb[..., :N], 0, x.size(2)-1), t.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = t.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = t.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = t.cat([t.clamp(p[..., :N], 0, x.size(2) - 1), t.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = t.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = t.meshgrid(
            t.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            t.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
        # (2N, 1)
        p_n = t.cat([t.flatten(p_n_x), t.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = t.meshgrid(
            t.arange(1, h*self.stride+1, self.stride),
            t.arange(1, w*self.stride+1, self.stride))
        p_0_x = t.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = t.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = t.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = t.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)
        # print('x_offset:', x_offset.shape)
        return x_offset



class Conv2dBlock(nn.Module):
    def __init__(self, input_dim , output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero', **kwargs):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)
        self.norm_type = norm
        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'wn':
            self.conv = weight_norm(self.conv)
        elif norm is None:
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm_type != 'wn' and self.norm != None:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x
    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = t.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset


class LeakyReLUConv2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding=0, norm='None', sn=False, **kwargs):
        super(LeakyReLUConv2d, self).__init__()
        model = []
        model += [nn.ReflectionPad2d(padding)]
        if sn:
            model += [spectral_norm(nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True))]
        else:
            model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True)]
        if 'norm' == 'Instance':
            model += [nn.InstanceNorm2d(n_out, affine=False)]
        model += [nn.LeakyReLU(inplace=True)]
        self.model = nn.Sequential(*model)
        # self.model.apply(gaussian_weights_init)
        #elif == 'Group'
    def forward(self, x):
        return self.model(x)


class SepConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,act_layer=nn.ReLU):
        super(SepConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels,
                                         in_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act_layer = act_layer() if act_layer is not None else nn.Identity()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        x = self.depthwise(x)
        x = self.act_layer(x)
        x = self.pointwise(x)
        return x

    def flops(self, H, W):
        flops = 0
        flops += H*W*self.in_channels*self.kernel_size**2/self.stride**2
        flops += H*W*self.in_channels*self.out_channels
        return flops


if __name__ == '__main__':
    model = DeformConv2d(3, 3).cuda()
    x = t.randn([4, 3, 256, 256], device=t.device('cuda:0'))
    out = model(x)
    print(out.shape)