import torch
import torch.nn as nn
from models.graph_models.model import models
from models.spade_models.networks import MultiscaleActionDiscriminator
from models.spade_models.networks.generator import Layout2VidGenerator
from models.spade_models.networks.sync_batchnorm import DataParallelWithCallback


class AG2VideoModel(nn.Module):
    def __init__(self, opt, device):
        super(AG2VideoModel, self).__init__()
        self.args = vars(opt)
        self.vocab = self.args["vocab"]

        # Graph Model
        self.acts_to_boxes = DataParallelWithCallback(models[self.args["layout_arch"]](opt),
                                                      device_ids=self.args['gpu_ids']).to(device)

        if "coupled_motion_apperance" not in self.args or not self.args["coupled_motion_apperance"]:
            objs_model = "graph" if self.args["layout_arch"] == 'gt_layout' else self.args["layout_arch"]
            self.acts_to_objs = DataParallelWithCallback(models[objs_model](opt),
                                                         device_ids=self.args['gpu_ids']).to(device)

        # Ag2Vid Generator
        self.layout_to_video = Layout2VidGenerator(opt)
        self.layout_to_video = DataParallelWithCallback(self.layout_to_video, device_ids=self.args['gpu_ids']).to(
            device)

    def forward(self, imgs, objs, triplets, actions, boxes_gt=None, test_mode=False, use_gt=False,
                graph_only=False):
        """
        Required Inputs:
        - objs: LongTensor of shape (O,) giving categories for all objects
        - triplets: LongTensor of shape (T, 3) where triplets[t] = [s, p, o]
          means that there is a triple (objs[s], p, objs[o])

        Optional Inputs:
        - obj_to_img: LongTensor of shape (O,) where obj_to_img[o] = i
          means that objects[o] is an object in image i. If not given then
          all objects are assumed to belong to the same image.
        - boxes_gt: FloatTensor of shape (O, 4) giving boxes to use for computing
          the spatial layout; if not given then use predicted boxes.
        """

        obj_vecs, boxes_pred, actions_data = self.acts_to_boxes(objs, triplets, actions, boxes_gt, test_mode=test_mode)

        if graph_only:
            return boxes_pred

        if "coupled_motion_apperance" not in self.args or not self.args["coupled_motion_apperance"]:
            # this is the default - decouple motion and apperance
            obj_vecs, _, actions_data = self.acts_to_objs(objs, triplets, actions, boxes_gt, test_mode=test_mode)
        generation_boxes_input = boxes_gt if use_gt else boxes_pred.detach()
        imgs_pred, flows_pred, conf_pred = self.layout_to_video(imgs, objs, obj_vecs, generation_boxes_input,
                                                                test_mode=test_mode)

        return imgs_pred, boxes_pred, flows_pred, conf_pred, actions_data


class MetaDiscriminatorModel(nn.Module):
    def __init__(self, opt):
        super(MetaDiscriminatorModel, self).__init__()
        self.args = vars(opt)
        self.init_act_discriminator(opt)

    def init_act_discriminator(self, opt):
        self.img_discriminator = MultiscaleActionDiscriminator(opt)
        self.img_discriminator.type(torch.cuda.FloatTensor)
        self.img_discriminator.train()
        self.optimizer_d_img = torch.optim.Adam(list(self.img_discriminator.parameters()),
                                                lr=opt.learning_rate,
                                                betas=(opt.beta1, 0.999))
