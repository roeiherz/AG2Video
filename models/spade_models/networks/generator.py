import torch
import torch.nn as nn
from models.attribute_embed import AttributeEmbeddings
from models.layout import boxes_to_layout, boxes_to_mask
from models.spade_models.networks.flows_generator import FlowsGenerator
from models.spade_models.networks.spade_generator import SPADEGenerator
from models.utils import remove_dummy_objects, resample
from models.spade_models.networks import BaseNetwork, get_nonspade_norm_layer


class Layout2VidGenerator(BaseNetwork):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.attribute_embedding = AttributeEmbeddings(opt.vocab['attributes'], 384//len(opt.vocab['attributes']))
        self.netG = SPADEGenerator(opt)
        self.bs = self.opt.batch_size
        self.n_frames_G = self.opt.n_frames_G
        self.n_scales = 1
        self.n_frames_bp = 1
        self.finetune_all = False
        self.no_initial_frame = False
        self.flows_network = FlowsGenerator(opt)

        norm = opt.norm_F
        norm_layer = get_nonspade_norm_layer(opt, norm)
        activation = nn.LeakyReLU(0.2, True)
        down_flow = [norm_layer(
            nn.Conv2d((self.opt.gconv_dim * 4) * self.n_frames_G + 3, self.opt.semantic_nc, kernel_size=3, padding=1)),
                     activation]

        self.conv_dim_in = nn.Sequential(*down_flow)

    def forward(self, imgs_gt, objs, obj_vecs, layout, imgs_prev=None, test_mode=False):
        obj_vecs_att = self.attribute_embedding.forward(objs)  # [B, N, d']
        seg_batches = []
        for b in range(obj_vecs.size(0)):
            mask = remove_dummy_objects(objs[b], self.opt.vocab)
            seg_frames = []
            objs_vecs_att_batch = obj_vecs_att[b][mask]
            for t in range(layout.size(1)):
                layout_boxes_batch = layout[b][t][mask]
                objs_vecs_batch = obj_vecs[b][t][mask]
                objs_vecs_batch = torch.cat([objs_vecs_att_batch, objs_vecs_batch], dim=1)

                # Boxes Layout
                seg = boxes_to_layout(objs_vecs_batch, layout_boxes_batch, self.opt.image_size[0],
                                      self.opt.image_size[0])
                seg_frames.append(seg)
            seg_batches.append(torch.cat(seg_frames, dim=0))
        # padding layouts
        seg_batches = [torch.cat([seg_b, seg_b[-1].unsqueeze(0)]) for seg_b in seg_batches]
        seg = torch.stack(seg_batches, dim=0)

        num_conditional_timesteps = self.n_frames_G - 1
        # nb_frames = layout.size(1) if not test_mode else layout.size(1) - 1
        imgs_prev = imgs_gt[:, :num_conditional_timesteps, ...]
        conf_prev = torch.zeros(imgs_gt.size(0), layout.size(1), 1, self.opt.image_size[0], self.opt.image_size[0]).cuda(imgs_prev.get_device())
        flow_prev = torch.zeros(imgs_gt.size(0), layout.size(1), 2, self.opt.image_size[0], self.opt.image_size[0]).cuda(imgs_prev.get_device())
        # Sequentially generate each frame
        for t in range(num_conditional_timesteps, layout.size(1)):
            ### prepare inputs
            # 1. input labels
            bs, n, d, h, w = seg.size()
            seg_t = seg[:, t - num_conditional_timesteps:t + 1].view(bs, -1, h, w)

            if test_mode or self.opt.bp_prev:
                imgs_prev_t = imgs_prev[:, -num_conditional_timesteps:]
            else:
                imgs_prev_t = imgs_gt[:, t - num_conditional_timesteps:t]
            imgs_prev_t = imgs_prev_t.view(bs, -1, h, w)

            input_flow = torch.cat([seg_t, imgs_prev_t], dim=1)
            pred_weight, pred_flow = self.flows_network(input_flow)
            img_prev_warp = resample(imgs_prev_t[:, -3:], pred_flow)
            pred_conf = (self.norm(imgs_prev_t[:, -3:] - img_prev_warp) < 0.02).float()
            conf_prev[:, t - 1] = pred_conf
            flow_prev[:, t - 1] = pred_flow

            # Reduce dimension
            input = torch.cat([seg_t, img_prev_warp], dim=1)
            input = self.conv_dim_in(input)  # Lower dim

            ### network forward
            img_raw = self.netG.forward(input) + img_prev_warp
            imgs_prev = torch.cat([imgs_prev, img_raw.unsqueeze(1)], dim=1)

        return imgs_prev, flow_prev, conf_prev

    def norm(self, t):
        return torch.sum(t * t, dim=1, keepdim=True)

