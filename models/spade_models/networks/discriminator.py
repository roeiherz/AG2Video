import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attribute_embed import AttributeEmbeddings
from models.bilinear import crop_bbox_batch
from models.graph_models.graph import GraphTripleConv
from models.layers import build_cnn, GlobalAvgPool
from models.layout import boxes_to_layout
from models.spade_models.networks.base_network import BaseNetwork
from models.utils import remove_dummy_objects
from models.spade_models.networks.normalization import get_nonspade_norm_layer


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminatorTM(BaseNetwork):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.n_frames_G = self.opt.n_frames_G
        self.n_frames_D = self.opt.n_frames_D

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = opt.ndf
        input_nc = self.compute_D_input_nc()
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_D)
        sequence = [[nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, False)]]

        for n in range(1, opt.n_layers_D):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == opt.n_layers_D - 1 else 2
            sequence += [[norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw,
                                               stride=stride, padding=padw)),
                          nn.LeakyReLU(0.2, False)
                          ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    def compute_D_input_nc(self):
        return self.opt.frames_per_action*3

    def forward(self, input):
        results = [input]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        get_intermediate_features = not self.opt.no_ganFeat_loss
        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]


class MultiscaleDiscriminatorT(BaseNetwork):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        for i in range(opt.num_D):
            subnetD = NLayerDiscriminatorTM(opt)
            self.add_module('discriminator_%d' % i, subnetD)

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3, stride=2, padding=[1, 1], count_include_pad=False)

    # Returns list of lists of discriminator outputs.
    # The final result is of size opt.num_D x opt.n_layers_D
    def forward(self, img):
        bs, nt, ch, h, w = img.size()
        input = img.view(bs, ch * nt, h, w)  # [2 ,4 * d, h, w]
        result = []
        get_intermediate_features = not self.opt.no_ganFeat_loss
        for name, D in self.named_children():
            if name.startswith('discriminator'):
                out = D(input)
                if not get_intermediate_features:
                    out = [out]
                result.append(out)
                input = self.downsample(input)
        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminatorT(BaseNetwork):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.attribute_embedding = AttributeEmbeddings(self.opt.vocab['attributes'], self.opt.embedding_dim,
                                                       use_attr_fc_gen=True)
        self.n_frames_total = self.opt.n_frames
        self.n_frames_G = self.opt.n_frames_G

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = opt.ndf
        input_nc = self.compute_D_input_nc()
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_D)
        sequence = [[nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, False)]]

        for n in range(1, opt.n_layers_D):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == opt.n_layers_D - 1 else 2
            sequence += [[norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=stride, padding=padw)),
                          nn.LeakyReLU(0.2, False)]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module('discriminator_t_model' + str(n), nn.Sequential(*sequence[n]))

    def compute_D_input_nc(self):
        # return (self.opt.semantic_nc + 3) * self.n_frames_total
        return (self.opt.semantic_nc + 3) * (self.n_frames_total - self.n_frames_G + 1)

    def forward(self, imgs, objs, layout_boxes):
        obj_vecs = self.attribute_embedding.forward(objs)  # [B, N, d']
        seg_batches = []
        for b in range(obj_vecs.size(0)):
            mask = remove_dummy_objects(objs[b], self.opt.vocab)
            seg_frames = []
            objs_vecs_batch = obj_vecs[b][mask]
            for t in range(layout_boxes.size(1)):
                layout_boxes_batch = layout_boxes[b][t][mask]
                # Boxes Layout
                seg = boxes_to_layout(objs_vecs_batch, layout_boxes_batch, self.opt.image_size[0],
                                      self.opt.image_size[0])
                seg_frames.append(seg)
            seg_batches.append(torch.cat(seg_frames, dim=0))

        seg = torch.stack(seg_batches, dim=0)
        input = torch.cat([imgs, seg], dim=2)  # [2 ,4 ,3 + 512, h, w]

        bs, nt, ch, h, w = input.size()
        input = input.view(-1, ch * nt, h, w)  # [2 ,4 * 515, h, w]

        results = [input]
        for name, submodel in self.named_children():
            if name.startswith('discriminator'):
                intermediate_output = submodel(results[-1])
                results.append(intermediate_output)

        get_intermediate_features = not self.opt.no_ganFeat_loss
        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]


class MultiscaleDiscriminator(BaseNetwork):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.attribute_embedding = AttributeEmbeddings(self.opt.vocab['attributes'], self.opt.embedding_dim,
                                                       use_attr_fc_gen=True)

        for i in range(opt.num_D):
            subnetD = NLayerDiscriminator(opt)
            self.add_module('discriminator_%d' % i, subnetD)

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3, stride=2, padding=[1, 1], count_include_pad=False)

    # Returns list of lists of discriminator outputs.
    # The final result is of size opt.num_D x opt.n_layers_D
    def forward(self, img, objs, layout_boxes, actions_data=None):
        obj_vecs = self.attribute_embedding.forward(objs)  # [B, N, d']

        seg_batches = []
        for b in range(obj_vecs.size(0)):
            mask = remove_dummy_objects(objs[b], self.opt.vocab)
            seg_frames = []
            objs_vecs_batch = obj_vecs[b][mask]
            for t in range(layout_boxes.size(1)):
                layout_boxes_batch = layout_boxes[b][t][mask]
                # Boxes Layout
                seg = boxes_to_layout(objs_vecs_batch, layout_boxes_batch, self.opt.image_size[0],
                                      self.opt.image_size[0])
                seg_frames.append(seg)
            seg_batches.append(torch.cat(seg_frames, dim=0))

        seg = torch.stack(seg_batches, dim=0)  # [B, T, N, d']
        input = torch.cat([img, seg], dim=2)

        bs, nt, ch, h, w = input.size()
        input = input.view(bs * nt, ch, h, w)

        result = []
        get_intermediate_features = not self.opt.no_ganFeat_loss
        for name, D in self.named_children():
            if name.startswith('discriminator'):
                out = D(input)
                if not get_intermediate_features:
                    out = [out]
                result.append(out)
                input = self.downsample(input)
        return result


class MultiscaleActionDiscriminator(BaseNetwork):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.attribute_embedding = AttributeEmbeddings(self.opt.vocab['attributes'], self.opt.embedding_dim,
                                                       use_attr_fc_gen=True)

        for i in range(opt.num_D):
            subnetD = NLayerActionDiscriminator(opt)
            self.add_module('discriminator_%d' % i, subnetD)

        self.attribute_embedding = AttributeEmbeddings(opt.vocab['attributes'], opt.embedding_dim)
        num_preds = len(opt.vocab['pred_idx_to_name'])
        self.pred_embeddings = nn.Embedding(num_preds, opt.embedding_dim)
        num_acts = len(opt.vocab['action_idx_to_name'])
        self.acts_embeddings = nn.Embedding(num_acts, opt.embedding_dim)
        num_attributes = len(opt.vocab['attributes'].keys())
        obj_input_dim = len(opt.vocab['attributes'].keys()) * opt.embedding_dim

        first_graph_conv_layer = {
            "obj_input_dim": obj_input_dim,
            "object_output_dim": opt.gconv_dim,
            "predicate_input_dim": opt.embedding_dim,
            "predicate_output_dim": opt.gconv_dim,
            "hidden_dim": opt.gconv_hidden_dim,
            "num_attributes": num_attributes,
            "mlp_normalization": opt.mlp_normalization,
            "pooling": opt.gconv_pooling,
            "loc_dim": 4
        }
        general_graph_conv_layer = first_graph_conv_layer.copy()
        general_graph_conv_layer.update(
            {"obj_input_dim": first_graph_conv_layer["object_output_dim"], "predicate_input_dim": opt.gconv_dim})
        layers = [first_graph_conv_layer] + [general_graph_conv_layer]

        self.gconvs = nn.ModuleList()
        for layer in layers:
            self.gconvs.append(GraphTripleConv(**layer))

        self.obj_vecs_net = nn.Sequential(nn.Linear(opt.embedding_dim + 4, obj_input_dim, bias=False),
                                          # nn.BatchNorm1d(obj_input_dim),
                                          nn.ReLU(inplace=True),
                                          nn.Linear(obj_input_dim, obj_input_dim, bias=False),
                                          # nn.BatchNorm1d(obj_input_dim),
                                          nn.ReLU()
                                          )
        self.pre_obj_vecs_net = nn.Sequential(nn.Linear(obj_input_dim, opt.embedding_dim, bias=False),
                                          # nn.BatchNorm1d(obj_input_dim),
                                          nn.ReLU(inplace=True),
                                          nn.Linear(opt.embedding_dim, opt.embedding_dim, bias=False),
                                          # nn.BatchNorm1d(obj_input_dim),
                                          nn.ReLU()
                                          )

        obj_vecs_act_embed = self.opt.gconv_dim
        self.fc_objs_vecs = nn.Linear(obj_vecs_act_embed + self.opt.semantic_nc, obj_vecs_act_embed * 2)

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3, stride=2, padding=[1, 1], count_include_pad=False)

    def get_obj_vecs(self, objs, layout_boxes, actions_data):
        triplets, temporal_triplets, relative_timesteps, locs = actions_data
        x_end = locs[:, :, :, 0]  # [B, F, A]
        y_end = locs[:, :, :, 1]  # [B, F, A]

        timesteps = layout_boxes.size(1)
        obj_vecs = self.attribute_embedding.forward(objs)  # [B, N, d']
        obj_vecs = self.pre_obj_vecs_net(obj_vecs)  # Reduce dimension

        all_objs_vecs = []
        for t in range(timesteps):
            action_triplets_t = temporal_triplets[:, t]
            boxes_condition_t = layout_boxes[:, t]
            r_t = relative_timesteps[:, t]

            # Concat locations with objects features
            obj_vecs = torch.cat([obj_vecs, boxes_condition_t], dim=-1)
            obj_vecs = self.obj_vecs_net(obj_vecs)  # Reduce dimension

            # Actions
            s, a, o = action_triplets_t.chunk(3, dim=-1)
            s, a, o = [x.squeeze(-1) for x in [s, a, o]]  # Now have shape [B, A]
            action_edges = torch.stack([s, o], dim=-1)  # Shape is [B, A, 2]
            action_indicators = a != self.opt.vocab["action_name_to_idx"]["__padding__"]
            acts_vecs = self.acts_embeddings(a.long())  # [B, A, d']
            acts_vecs[:, :, -3] = x_end[:, t]
            acts_vecs[:, :, -2] = y_end[:, t]
            acts_vecs[:, :, -1] = r_t

            # Combined between spatial and temporal edges
            only_temporal = self.opt.only_temporal
            indicators = action_indicators
            edges = action_edges
            pred_vecs = acts_vecs

            for i in range(len(self.gconvs)):
                obj_vecs, pred_vecs = self.gconvs[i](obj_vecs, pred_vecs, edges.type(torch.long), indicators)

            all_objs_vecs.append(obj_vecs)
        all_objs_vecs = torch.stack(all_objs_vecs, dim=1)
        return all_objs_vecs

    # Returns list of lists of discriminator outputs.
    # The final result is of size opt.num_D x opt.n_layers_D
    def forward(self, img, objs, layout_boxes, actions_data):
        obj_vecs = self.get_obj_vecs(objs, layout_boxes, actions_data)
        obj_vecs_att = self.attribute_embedding.forward(objs)  # [B, N, d']

        seg_batches = []
        for b in range(obj_vecs.size(0)):
            mask = remove_dummy_objects(objs[b], self.opt.vocab)
            seg_frames = []
            objs_vecs_att_batch = obj_vecs_att[b][mask]
            for t in range(layout_boxes.size(1)):
                layout_boxes_batch = layout_boxes[b][t][mask]
                objs_vecs_batch = obj_vecs[b][t][mask]
                objs_vecs_batch = torch.cat([objs_vecs_att_batch, objs_vecs_batch], dim=1)
                objs_vecs_batch = self.fc_objs_vecs(objs_vecs_batch)

                # Boxes Layout
                seg = boxes_to_layout(objs_vecs_batch, layout_boxes_batch, self.opt.image_size[0],
                                      self.opt.image_size[0])
                seg_frames.append(seg)
            seg_batches.append(torch.cat(seg_frames, dim=0))

        seg = torch.stack(seg_batches, dim=0)  # [B, T, N, d']
        input = torch.cat([img, seg], dim=2)

        bs, nt, ch, h, w = input.size()
        input = input.view(bs*nt, ch, h, w)

        result = []
        get_intermediate_features = not self.opt.no_ganFeat_loss
        for name, D in self.named_children():
            if name.startswith('discriminator'):
                out = D(input)
                if not get_intermediate_features:
                    out = [out]
                result.append(out)
                input = self.downsample(input)
        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerActionDiscriminator(BaseNetwork):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = opt.ndf
        input_nc = self.compute_D_input_nc()
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_D)
        sequence = [[nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, False)]]

        for n in range(1, opt.n_layers_D):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == opt.n_layers_D - 1 else 2
            sequence += [[norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw,
                                               stride=stride, padding=padw)),
                          nn.LeakyReLU(0.2, False)
                          ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    def compute_D_input_nc(self):
        obj_vecs_act_embed = self.opt.gconv_dim if self.opt.use_actions_loss else self.opt.semantic_nc
        return obj_vecs_act_embed * 2 + 3

    def forward(self, input):
        results = [input]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        get_intermediate_features = not self.opt.no_ganFeat_loss
        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(BaseNetwork):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = opt.ndf
        input_nc = self.compute_D_input_nc()
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_D)
        sequence = [[nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, False)]]

        for n in range(1, opt.n_layers_D):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == opt.n_layers_D - 1 else 2
            sequence += [[norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw,
                                               stride=stride, padding=padw)),
                          nn.LeakyReLU(0.2, False)
                          ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    def compute_D_input_nc(self):
        obj_vecs_act_embed = self.opt.gconv_dim if self.opt.use_actions_loss else self.opt.semantic_nc
        return obj_vecs_act_embed + 3

    def forward(self, input):
        results = [input]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        get_intermediate_features = not self.opt.no_ganFeat_loss
        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]


class AcAttDiscriminator(nn.Module):
    def __init__(self, vocab, arch, normalization='none', activation='relu',
                 padding='same', pooling='avg'):
        super(AcAttDiscriminator, self).__init__()
        self.vocab = vocab

        attributes = list(self.vocab['attributes'].keys())
        sorted(attributes)
        self.attributes_to_index = {attributes[i]: i for i in range(len(attributes))}
        self.is_attributes = len(vocab["attributes"]) > 1

        cnn_kwargs = {
            'arch': arch,
            'normalization': normalization,
            'activation': activation,
            'pooling': pooling,
            'padding': padding,
        }
        cnn, D = build_cnn(**cnn_kwargs)
        self.cnn = nn.Sequential(cnn, GlobalAvgPool(), nn.Linear(D, 1024))

        if self.is_attributes:
            # attributes
            self.real_classifier = nn.Linear(1024, 1)
            for att_name, att_val in vocab["attributes"].items():
                self.add_module("obj_classifier_{}".format(att_name), nn.Linear(1024, len(att_val)))
        else:
            # no attributes; only objs class labels
            num_objects = max(vocab['object_name_to_idx'].values()) + 1
            self.real_classifier = nn.Linear(1024, 1)
            self.obj_classifier = nn.Linear(1024, num_objects)

    def forward(self, x, y):
        if x.dim() == 3:
            x = x[:, None]

        vecs = self.cnn(x)
        real_scores = self.real_classifier(vecs)

        if self.is_attributes:
            # attributes
            ac_loss_att = []
            for att_name, att_val in self.vocab["attributes"].items():
                obj_scores = self._modules["obj_classifier_{}".format(att_name)](vecs)
                ac_loss_att.append(F.cross_entropy(obj_scores, y[:, self.attributes_to_index[att_name]]))
            ac_loss = sum(ac_loss_att)
        else:
            # no attributes; only objs class labels
            obj_scores = self.obj_classifier(vecs)
            ac_loss = F.cross_entropy(obj_scores, y.view(-1))

        return real_scores, ac_loss


class AcCropDiscriminator(nn.Module):
    def __init__(self, vocab, arch, normalization='none', activation='relu',
                 object_size=64, padding='same', pooling='avg'):
        super(AcCropDiscriminator, self).__init__()
        self.vocab = vocab
        self.att_discriminator = AcAttDiscriminator(vocab, arch, normalization, activation, padding, pooling)
        self.object_size = object_size

    def forward(self, imgs, objs, boxes):
        crops, objss = crop_bbox_batch(imgs, objs, boxes, self.object_size, vocab=self.vocab)

        B, F, _, _, _ = imgs.size()
        real_scores_b = []
        ac_loss_b = []
        for b in range(B):
            try:
                # mask = remove_dummy_objects(objs[b], self.vocab)
                # objs_b = objs[b][mask].unsqueeze(0).repeat(F, 1, 1).view(-1, objs.size(-1))
                objs_b = objss[b]
                crops_b = crops[b]
                real_scores, ac_loss = self.att_discriminator(crops_b, objs_b)

                ac_loss_b.append(ac_loss)
                real_scores_b.append(real_scores)
            except Exception as e:
                print("debug")

        ac_loss = sum(ac_loss_b)
        real_scores = torch.cat(real_scores_b)
        return real_scores, ac_loss, crops


class AcDiscriminator(nn.Module):
    def __init__(self, vocab, arch, normalization='none', activation='relu', padding='same', pooling='avg', att_name=''):
        super(AcDiscriminator, self).__init__()
        self.vocab = vocab

        cnn_kwargs = {
            'arch': arch,
            'normalization': normalization,
            'activation': activation,
            'pooling': pooling,
            'padding': padding,
        }
        cnn, D = build_cnn(**cnn_kwargs)
        self.cnn = nn.Sequential(cnn, GlobalAvgPool(), nn.Linear(D, 1024))
        num_objects = len(vocab["attributes"][att_name])
        # num_objects = max(vocab['object_name_to_idx'].values()) + 1

        self.real_classifier = nn.Linear(1024, 1)
        self.att_classifier = nn.Linear(1024, num_objects)

    def forward(self, x, y):
        if x.dim() == 3:
            x = x[:, None]
        vecs = self.cnn(x)
        real_scores = self.real_classifier(vecs)
        obj_scores = self.att_classifier(vecs)
        ac_loss = F.cross_entropy(obj_scores, y)
        return real_scores, ac_loss


class AcAttCropDiscriminator(nn.Module):
    def __init__(self, vocab, arch, normalization='none', activation='relu',
                 object_size=64, padding='same', pooling='avg'):
        super(AcAttCropDiscriminator, self).__init__()
        self.vocab = vocab
        self.object_size = object_size

        attributes = list(self.vocab['attributes'].keys())
        sorted(attributes)
        self.attributes_to_index = {attributes[i]: i for i in range(len(attributes))}
        self.is_attributes = len(vocab["attributes"]) > 1

        for att_name, att_val in vocab["attributes"].items():
            self.add_module("{}_discriminator".format(att_name),
                            AcDiscriminator(vocab, arch, normalization, activation, padding, pooling, att_name))

    def forward(self, imgs, objs, boxes):
        crops = crop_bbox_batch(imgs, objs, boxes, self.object_size, vocab=self.vocab)

        B, F, _, _, _ = imgs.size()
        real_scores_b = []
        ac_loss_b = []
        for b in range(B):
            mask = remove_dummy_objects(objs[b], self.vocab)
            objs_b = objs[b][mask].unsqueeze(0).repeat(F, 1, 1).view(-1, objs.size(-1))
            crops_b = crops[b]

            # attributes
            real_scores_att = []
            ac_loss_att = []
            for att_name, att_val in self.vocab["attributes"].items():
                att_discriminator = self._modules["{}_discriminator".format(att_name)]
                att_objs_b = objs_b[:, self.attributes_to_index[att_name]]
                real_scores, ac_loss = att_discriminator(crops_b, att_objs_b)
                ac_loss_att.append(ac_loss)
                real_scores_att.append(real_scores)

            # ac_loss_b.append(sum(ac_loss_att))
            ac_loss_b.append(torch.stack(ac_loss_att))
            real_scores_b.append(torch.stack(real_scores_att))

        # ac_loss = sum(ac_loss_b)
        ac_loss = torch.cat(ac_loss_b)
        real_scores = torch.cat(real_scores_b)
        return real_scores, ac_loss, crops

