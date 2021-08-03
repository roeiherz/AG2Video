import torch
import torch.nn as nn
from models.attribute_embed import AttributeEmbeddings
from models.graph_models.graph import GraphTripleConv
from models.layers import build_mlp, Interpolate


a_map = [
    [0,0,0,0],
    [0., -0.05, 0, 0],
    [0.05, 0, 0, 0],
    [0., 0.05, 0, 0],
    [-0.05, 0.00, 0, 0],
    [0,0,0,0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
]





class Acts2LayoutModel(nn.Module):
    def __init__(self, opt):
        super(Acts2LayoutModel, self).__init__()
        args = vars(opt)
        self.vocab = args["vocab"]
        self.image_size = args["image_size"]
        self.mask_noise_dim = args.get("mask_noise_dim")
        self.args = args
        self.attribute_embedding = AttributeEmbeddings(self.vocab['attributes'], args["embedding_dim"])
        num_preds = len(self.vocab['pred_idx_to_name'])
        self.pred_embeddings = nn.Embedding(num_preds, args["embedding_dim"])
        num_acts = len(self.vocab['action_idx_to_name'])
        self.acts_embeddings = nn.Embedding(num_acts, args["embedding_dim"])
        num_attributes = len(self.vocab['attributes'].keys())
        obj_input_dim = len(self.vocab['attributes'].keys()) * args["embedding_dim"]
        first_graph_conv_layer = {
            "obj_input_dim": obj_input_dim,
            "object_output_dim": args["gconv_dim"],
            "predicate_input_dim": args["embedding_dim"],
            "predicate_output_dim": args["gconv_dim"],
            "hidden_dim": args["gconv_hidden_dim"],
            "num_attributes": num_attributes,
            "mlp_normalization": args["mlp_normalization"],
            "pooling": args["gconv_pooling"],
            "loc_dim": 4
        }
        general_graph_conv_layer = first_graph_conv_layer.copy()
        general_graph_conv_layer.update(
            {"obj_input_dim": first_graph_conv_layer["object_output_dim"], "predicate_input_dim": args["gconv_dim"]})
        layers = [first_graph_conv_layer] + [general_graph_conv_layer] * (args["gconv_num_layers"] - 1)

        self.gconvs = nn.ModuleList()

        for layer in layers:
            self.gconvs.append(GraphTripleConv(**layer))

        object_output_dim = layers[-1]["object_output_dim"]
        box_net_dim = 4
        box_net_layers = [object_output_dim, args["gconv_hidden_dim"], box_net_dim]
        self.box_net = build_mlp(box_net_layers, batch_norm=args["mlp_normalization"], final_nonlinearity=None)
        self.obj_vecs_net = nn.Sequential(nn.Linear(obj_input_dim + 4, obj_input_dim, bias=False),
                                          # nn.BatchNorm1d(obj_input_dim),
                                          nn.ReLU(inplace=True),
                                          nn.Linear(obj_input_dim, obj_input_dim, bias=False),
                                          # nn.BatchNorm1d(obj_input_dim),
                                          nn.ReLU()
                                          )

        # masks generation
        self.mask_net = None
        if args["mask_size"] is not None and args["mask_size"] > 0:
            self.mask_net = self._build_mask_net(args['g_mask_dim'], args["mask_size"])

    def _build_mask_net(self, dim, mask_size):
        output_dim = 1
        layers, cur_size = [], 1
        while cur_size < mask_size:
            layers.append(Interpolate(scale_factor=2, mode='nearest'))
            layers.append(nn.Conv2d(dim, dim, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(dim))
            layers.append(nn.ReLU())
            cur_size *= 2
        if cur_size != mask_size:
            raise ValueError('Mask size must be a power of 2')
        layers.append(nn.Conv2d(dim, output_dim, kernel_size=1))
        return nn.Sequential(*layers)

    def create_mask_vecs(self, objs, obj_vecs):
        B = objs.size(0)
        O = objs.size(1)
        mask_vecs = obj_vecs
        layout_noise = torch.randn((1, self.mask_noise_dim), dtype=mask_vecs.dtype, device=mask_vecs.device).repeat(
            (B, O, 1)).view(B, O, self.mask_noise_dim)
        mask_vecs = torch.cat([mask_vecs, layout_noise], dim=-1)
        return mask_vecs

    def forward(self, objs, triplets, actions, boxes_gt=None, test_mode=False):
        """
        - objs: LongTensor of shape (O,) giving categories for all objects
        - triplets: LongTensor of shape (B, F, T, 3) where triplets[t] = [s, p, o]
          means that there is a triple (objs[s], p, objs[o])
        - actions:
        - test_mode: train or test mode
        """

        # List of positions and objects
        timesteps = triplets.size(1)
        actions_reshaped = actions.unsqueeze(1).expand(actions.size(0), timesteps, actions.size(1), actions.size(2))
        sa, a, oa, f1, f2, x_end, y_end = actions_reshaped.chunk(7, dim=-1)  # All have shape [B, F, A, 1]
        sa, a, oa, f1, f2, x_end, y_end = [x.squeeze(-1) for x in
                                           [sa, a, oa, f1, f2, x_end, y_end]]  # Now have shape [B, F, A]
        t = torch.Tensor(range(0, timesteps)).view(1, timesteps, 1).to(actions_reshaped)
        t = t.type(torch.float32)
        f1 = f1.type(torch.float32)
        f2 = f2.type(torch.float32)
        relative_timesteps = (t / timesteps) * (f2 - f1 + 1e-6) + f1
        is_included_t = (relative_timesteps >= 0) * (relative_timesteps <= 1)
        a = a.clone()  # Be careful to change the original actions
        a[~is_included_t] = self.vocab["action_name_to_idx"]["__padding__"]
        temporal_triplets = torch.stack([sa, a, oa], dim=-1).long()
        boxes_pred = [boxes_gt[:, 0]]
        obj_vecs_embedding = self.attribute_embedding.forward(objs)
        temporal_obj_vecs = [torch.zeros(objs.size(0), objs.size(1), self.args["embedding_dim"]).to(obj_vecs_embedding)]
        for t in range(1, timesteps):
            spatial_triplets_t = triplets[:, t]
            action_triplets_t = temporal_triplets[:, t]
            boxes_condition_t = boxes_pred[-1]

            r_t = relative_timesteps[:, t]

            # [B, N, d']
            # Concat locations with objects features
            obj_vecs = torch.cat([obj_vecs_embedding, boxes_condition_t], dim=-1)
            obj_vecs = self.obj_vecs_net(obj_vecs)  # Reduce dimension

            # Actions
            s, a, o = action_triplets_t.chunk(3, dim=-1)
            s, a, o = [x.squeeze(-1) for x in [s, a, o]]  # Now have shape [B, A]
            action_edges = torch.stack([s, o], dim=-1)  # Shape is [B, A, 2]
            action_indicators = a != self.vocab["action_name_to_idx"]["__padding__"]
            acts_vecs = self.acts_embeddings(a.long())  # [B, A, d']
            acts_vecs[:, :, -3] = x_end[:, t]
            acts_vecs[:, :, -2] = y_end[:, t]
            acts_vecs[:, :, -1] = r_t

            # Combined between spatial and temporal edges
            if not self.args.get("only_temporal", False):
                s, p, o = spatial_triplets_t.chunk(3, dim=-1)  # All have shape [B, T, 1]
                s, p, o = [x.squeeze(-1) for x in [s, p, o]]  # Now have shape [B, T]
                spatial_edges = torch.stack([s, o], dim=-1)  # Shape is [B, T, 2]
                pred_indicators = p != self.vocab["pred_name_to_idx"]["__padding__"]
                spatial_pred_vecs = self.pred_embeddings(p)  # [B, T, d']
                indicators = torch.cat([pred_indicators, action_indicators], dim=1)
                edges = torch.cat([spatial_edges, action_edges.long()], dim=1)
                pred_vecs = torch.cat([spatial_pred_vecs, acts_vecs], dim=1)
            else:
                indicators = action_indicators
                edges = action_edges
                pred_vecs = acts_vecs

            for i in range(len(self.gconvs)):
                obj_vecs, pred_vecs = self.gconvs[i](obj_vecs, pred_vecs, edges, indicators)

            # Generate Boxes per frames
            temporal_obj_vecs.append(obj_vecs)
            boxes_pred_t = boxes_condition_t + self.box_net(obj_vecs)
            boxes_pred.append(boxes_pred_t)

        boxes_pred = torch.stack(boxes_pred).permute(1, 0, 2, 3)  # [B, F, O, 4]
        temporal_obj_vecs = torch.stack(temporal_obj_vecs, dim=1)
        locs = torch.stack([x_end, y_end], dim=-1)  # [B, F, A, 2]
        return temporal_obj_vecs, boxes_pred, [triplets, temporal_triplets, relative_timesteps, locs]



models = {
    'graph': Acts2LayoutModel,
}
