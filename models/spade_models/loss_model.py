"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import models.spade_models.networks as networks
import torch.nn.functional as F
from models.losses import get_gan_losses
from models.utils import resample


class LossModel(torch.nn.Module):

    def __init__(self, opt, discriminator):
        super().__init__()
        self.opt = opt
        self.n_frames_G = self.opt.n_frames_G
        self.n_frames_D = self.opt.n_frames_D
        device = torch.device("cuda:{gpu}".format(gpu=self.opt.gpu_ids[0]) if self.opt.use_cuda else "cpu")
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() else torch.ByteTensor
        self.discriminator = discriminator

        if hasattr(discriminator, 'img_discriminator'):
            self.netD_img = discriminator.img_discriminator
        if hasattr(discriminator, 'obj_discriminator'):
            self.netD_obj = discriminator.obj_discriminator
        if hasattr(discriminator, 'temporal_discriminator'):
            self.netD_temp = discriminator.temporal_discriminator

        # set loss functions
        self.criterionGAN = networks.GANLoss(opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
        self.criterionFeat = torch.nn.L1Loss()
        self.gan_g_loss, self.gan_d_loss = get_gan_losses(opt.gan_loss_type)
        if not opt.no_vgg_loss:
            self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids).to(device)

        self.criterionWarp = torch.nn.L1Loss()

    def compute_graph_loss(self, batch, boxes_pred):
        imgs, objs, boxes, triplets, actions, video_id = batch

        G_losses = {}
        _boxes_pred = boxes_pred[:, 1:].contiguous().view(-1, 4)
        _boxes_gt = boxes[:, 1:].contiguous().view(-1, 4)
        flattened_bbox_pred = F.smooth_l1_loss(_boxes_pred, _boxes_gt, reduction='none') * self.opt.bbox_pred_loss_weight
        flattened_objs = objs.unsqueeze(1).repeat(1, boxes.size(1)-1, 1, 1)
        flattened_objs = flattened_objs.view(-1, objs.size(-1))
        if objs.size(-1) > 1:
            # Objs contain multiple attributes (CLEVR), dummy are [0, 0...0]
            masked_object = (flattened_objs.sum(1) != 0).repeat(4, 1).permute(1, 0)
        else:
            # Objs contain only single attribute (VG/COCO), dummy is 0
            masked_object = (flattened_objs != 0)

        is_real_object_mask = masked_object.type(torch.FloatTensor).to(flattened_objs.device)
        G_losses["bbox_pred"] = (flattened_bbox_pred * is_real_object_mask).mean()
        G_losses['total_loss'] = torch.stack(list(G_losses.values()), dim=0).sum()
        return G_losses

    def compute_generator_loss(self, batch, model_out):
        imgs, objs, boxes, triplets, actions, video_id = batch
        imgs_pred, boxes_pred, flows_pred, conf_pred, actions_data = model_out
        nb_conditional_frames = self.n_frames_G - 1

        G_losses = {}
        relevant_imgs, relevant_boxes, relevant_imgs_pred, relevant_boxes_pred = \
            [item[:, nb_conditional_frames:] for item in [imgs, boxes, imgs_pred, boxes_pred]]
        relevant_actions_data = [item[:, nb_conditional_frames:] for item in actions_data]

        # Img disc
        pred_img_fake = self.netD_img(relevant_imgs_pred, objs, relevant_boxes, relevant_actions_data)
        pred_img_fake_loss = self.criterionGAN(pred_img_fake, True, for_discriminator=False).squeeze(0)
        G_losses['GAN_Img'] = pred_img_fake_loss * self.opt.discriminator_img_loss_weight
        if not self.opt.no_ganFeat_loss:
            pred_img_real = self.netD_img(relevant_imgs, objs, relevant_boxes, relevant_actions_data)
            num_D = len(pred_img_fake)
            GAN_Feat_loss = self.FloatTensor(1).fill_(0)
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_img_fake[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = self.criterionFeat(pred_img_fake[i][j], pred_img_real[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
            G_losses['GAN_Feat'] = GAN_Feat_loss.squeeze(0)

        # Flow loss
        relevant_imgs_, relevant_flows_pred_ = \
            [item[:, nb_conditional_frames - 1:-1] for item in [imgs, flows_pred]]

        # warped prev image should be close to current image
        b, t, c, h, w = imgs.size()
        relevant_imgs_ = relevant_imgs_.contiguous().view(-1, c, h, w)
        relevant_imgs_next = imgs[:, nb_conditional_frames:].contiguous().view(-1, c, h, w)
        pred_B_warp = resample(relevant_imgs_, relevant_flows_pred_.contiguous().view(-1, 2, h, w))
        G_losses["loss_F_Warp"] = self.criterionWarp(pred_B_warp, relevant_imgs_next) * self.opt.lambda_F_warp

        if not self.opt.no_vgg_loss:
            G_losses['VGG'] = self.criterionVGG(relevant_imgs_pred, relevant_imgs) * self.opt.lambda_vgg

        G_losses['total_loss'] = torch.stack(list(G_losses.values()), dim=0).sum()
        return G_losses

    def compute_discriminator_loss(self, batch, model_out):
        imgs, objs, boxes, triplets, actions, temporal_disc = batch
        imgs_pred, boxes_pred, flows_pred, conf_pred, actions_data = model_out

        nb_conditional_frames = self.n_frames_G - 1

        relevant_imgs, relevant_boxes, relevant_imgs_pred, relevant_boxes_pred = \
            [item[:, nb_conditional_frames:] for item in [imgs, boxes, imgs_pred, boxes_pred]]
        relevant_actions_data = [item[:, nb_conditional_frames:] for item in actions_data]

        # Img disc
        D_img_losses = {}
        # Fake images; Real layout
        pred_fake = self.netD_img(relevant_imgs_pred.detach(), objs, relevant_boxes, relevant_actions_data)
        # Real images; Real layout
        gt_real = self.netD_img(relevant_imgs, objs, relevant_boxes, relevant_actions_data)
        # Update Loss
        D_img_losses.update({"D_img_fake": self.criterionGAN(pred_fake, False, for_discriminator=True)})
        D_img_losses.update({"D_img_real": self.criterionGAN(gt_real, True, for_discriminator=True)})
        D_img_losses.update({"total_img_loss": torch.stack(list(D_img_losses.values()), dim=0).sum()})

        # Temp disc
        D_temp_losses = {}
        D_obj_losses = {}

        # Merge
        D_losses = {**D_img_losses, **D_temp_losses, **D_obj_losses}
        return D_losses

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0

    def norm(self, t):
        return torch.sum(t * t, dim=1, keepdim=True)

    def forward(self, batch, model_out, mode):

        if mode == "compute_discriminator_loss":
            return self.compute_discriminator_loss(batch, model_out)

        if mode == "compute_generator_loss":
            return self.compute_generator_loss(batch, model_out)

        if mode == "compute_graph_loss":
            return self.compute_graph_loss(batch, model_out)
