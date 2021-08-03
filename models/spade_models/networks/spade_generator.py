import torch.nn as nn
import torch.nn.functional as F
from torch import tanh
from models.spade_models.networks import BaseNetwork, get_nonspade_norm_layer
from models.spade_models.networks.architecture import SPADEResnetBlock as SPADEResnetBlock


class SPADEGenerator(BaseNetwork):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf
        self.sw, self.sh = self.compute_latent_vector_size(opt)

        self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1)

        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, opt)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, opt)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, opt)

        final_nc = nf

        if opt.num_upsampling_layers == 'most':
            self.up_4 = SPADEResnetBlock(1 * nf, nf // 2, opt)
            final_nc = nf // 2

        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt.num_upsampling_layers)

        sw = opt.image_size[0] // (2 ** num_up_layers)
        sh = round(sw / opt.aspect_ratio)
        return sw, sh

    def forward(self, layout):
        # we downsample segmap and run convolution
        x = F.interpolate(layout, size=(self.sh, self.sw))
        x = self.fc(x)
        x = self.head_0(x, layout)
        x = self.up(x)
        x = self.G_middle_0(x, layout)

        if self.opt.num_upsampling_layers == 'more' or self.opt.num_upsampling_layers == 'most':
            x = self.up(x)

        x = self.G_middle_1(x, layout)

        x = self.up(x)
        x = self.up_0(x, layout)
        x = self.up(x)
        x = self.up_1(x, layout)
        x = self.up(x)
        x = self.up_2(x, layout)
        x = self.up(x)
        x = self.up_3(x, layout)

        if self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4(x, layout)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        img_raw = tanh(x)
        return img_raw
