import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from models.spade_models.networks import BaseNetwork, get_nonspade_norm_layer
import torch.nn as nn
import torch
from models.spade_models.networks.normalization import SPADE
from models.spade_models.networks.sync_batchnorm import SynchronizedBatchNorm2d




class FlowsGenerator(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        obj_vecs_act_embed = self.opt.gconv_dim
        num_conditional_timesteps = self.opt.n_frames_G - 1
        input_nc = (obj_vecs_act_embed*4) * self.opt.n_frames_G + num_conditional_timesteps * 3

        self.flow_multiplier = opt.flow_multiplier
        nf = opt.nff
        n_blocks = opt.n_blocks_F
        n_downsample_F = opt.n_downsample_F
        nf_max = 1024
        ch = [min(nf_max, nf * (2 ** i)) for i in range(n_downsample_F + 1)]

        norm = opt.norm_F
        norm_layer = get_nonspade_norm_layer(opt, norm)
        activation = nn.LeakyReLU(0.2, True)

        down_flow = [norm_layer(nn.Conv2d(input_nc, nf, kernel_size=3, padding=1)), activation]
        for i in range(n_downsample_F):
            down_flow += [norm_layer(nn.Conv2d(ch[i], ch[i + 1], kernel_size=3, padding=1, stride=2)), activation]
            ### resnet blocks

        res_flow = []
        ch_r = min(nf_max, nf * (2 ** n_downsample_F))
        for i in range(n_blocks):
            res_flow += [SPADEResnetBlock(ch_r, ch_r, norm=norm)]

        ### upsample
        up_flow = []
        for i in reversed(range(n_downsample_F)):
            if opt.flow_deconv:
                up_flow += [norm_layer(
                    nn.ConvTranspose2d(ch[i + 1], ch[i], kernel_size=3, stride=2, padding=1, output_padding=1)),
                    activation]
            else:
                up_flow += [nn.Upsample(scale_factor=2),
                            norm_layer(nn.Conv2d(ch[i + 1], ch[i], kernel_size=3, padding=1)), activation]

        conv_flow = [nn.Conv2d(nf, 2, kernel_size=3, padding=1)]
        conv_w = [nn.Conv2d(nf, 1, kernel_size=3, padding=1), nn.Sigmoid()]

        self.down_flow = nn.Sequential(*down_flow)
        self.res_flow = nn.Sequential(*res_flow)
        self.up_flow = nn.Sequential(*up_flow)
        self.conv_flow = nn.Sequential(*conv_flow)
        self.conv_w = nn.Sequential(*conv_w)

    def forward(self, label):
        downsample = self.down_flow(label)
        res = self.res_flow(downsample)
        flow_feat = self.up_flow(res)
        flow = self.conv_flow(flow_feat) * self.flow_multiplier
        weight = self.conv_w(flow_feat)
        return weight, flow


class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, norm='batch', hidden_nc=0, ks=3, stride=1, conv_params_free=False,
                 norm_params_free=False):
        super().__init__()
        fhidden = min(fin, fout)
        self.learned_shortcut = (fin != fout)
        self.stride = stride
        Conv2d = generalConv(adaptive=conv_params_free)
        sn_ = spectral_norm if not conv_params_free else lambda x: x

        # Submodules
        self.conv_0 = sn_(Conv2d(fin, fhidden, 3, stride=stride, padding=1))
        self.conv_1 = sn_(Conv2d(fhidden, fout, 3, padding=1))
        if self.learned_shortcut:
            self.conv_s = sn_(Conv2d(fin, fout, 1, stride=stride, bias=False))

        Norm = generalNorm(norm)
        self.bn_0 = Norm(fin, hidden_nc=hidden_nc, norm=norm, ks=ks, params_free=norm_params_free)
        self.bn_1 = Norm(fhidden, hidden_nc=hidden_nc, norm=norm, ks=ks, params_free=norm_params_free)
        if self.learned_shortcut:
            self.bn_s = Norm(fin, hidden_nc=hidden_nc, norm=norm, ks=ks, params_free=norm_params_free)

    def forward(self, x, label=None, conv_weights=[], norm_weights=[]):
        if not conv_weights: conv_weights = [None] * 3
        if not norm_weights: norm_weights = [None] * 3
        x_s = self._shortcut(x, label, conv_weights[2], norm_weights[2])
        dx = self.conv_0(actvn(self.bn_0(x, label, norm_weights[0])), conv_weights[0])
        dx = self.conv_1(actvn(self.bn_1(dx, label, norm_weights[1])), conv_weights[1])
        out = x_s + 1.0 * dx
        return out

    def _shortcut(self, x, label, conv_weights, norm_weights):
        if self.learned_shortcut:
            x_s = self.conv_s(self.bn_s(x, label, norm_weights), conv_weights)
        elif self.stride != 1:
            x_s = nn.AvgPool2d(3, stride=2, padding=1)(x)
        else:
            x_s = x
        return x_s


def generalNorm(norm):
    if 'spade' in norm: return SPADE

    def get_norm(norm):
        if 'instance' in norm:
            return nn.InstanceNorm2d
        elif 'syncbatch' in norm:
            return SynchronizedBatchNorm2d
        elif 'batch' in norm:
            return nn.BatchNorm2d

    norm = get_norm(norm)

    class NormalNorm(norm):
        def __init__(self, *args, hidden_nc=0, norm='', ks=1, params_free=False, **kwargs):
            super(NormalNorm, self).__init__(*args, **kwargs)

        def forward(self, input, label=None, weight=None):
            return super(NormalNorm, self).forward(input)

    return NormalNorm


def actvn(x):
    out = F.leaky_relu(x, 2e-1)
    return out


def concat(a, b, dim=0):
    if isinstance(a, list):
        return [concat(ai, bi, dim) for ai, bi in zip(a, b)]
    if a is None:
        return b
    return torch.cat([a, b], dim=dim)


def batch_conv(x, weight, bias=None, stride=1, group_size=-1):
    if weight is None: return x
    if isinstance(weight, list) or isinstance(weight, tuple):
        weight, bias = weight
    padding = weight.size()[-1] // 2
    groups = group_size//weight.size()[2] if group_size != -1 else 1
    if bias is None: bias = [None] * x.size()[0]
    y = None
    for i in range(x.size()[0]):
        if stride >= 1:
            yi = F.conv2d(x[i:i+1], weight=weight[i], bias=bias[i], padding=padding, stride=stride, groups=groups)
        else:
            yi = F.conv_transpose2d(x[i:i+1], weight=weight[i], bias=bias[i,:weight.size(2)],
                                    padding=padding, stride=int(1/stride), output_padding=padding, groups=groups)
        y = concat(y, yi)
    return y


def generalConv(adaptive=False, transpose=False):
    class NormalConv2d(nn.Conv2d):
        def __init__(self, *args, **kwargs):
            super(NormalConv2d, self).__init__(*args, **kwargs)

        def forward(self, input, weight=None, bias=None, stride=1):
            return super(NormalConv2d, self).forward(input)

    class NormalConvTranspose2d(nn.ConvTranspose2d):
        def __init__(self, *args, output_padding=1, **kwargs):
            # kwargs['output_padding'] = 1
            super(NormalConvTranspose2d, self).__init__(*args, **kwargs)

        def forward(self, input, weight=None, bias=None, stride=1):
            return super(NormalConvTranspose2d, self).forward(input)

    class AdaptiveConv2d(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, input, weight=None, bias=None, stride=1):
            return batch_conv(input, weight, bias, stride)

    if adaptive: return AdaptiveConv2d
    return NormalConv2d if not transpose else NormalConvTranspose2d
