import torch
import torch.nn as nn
from models.layers import build_mlp

"""
PyTorch modules for dealing with graphs.
"""


def _init_weights(module):
    if hasattr(module, 'weight'):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)


class GraphTripleConv(nn.Module):
    """
    A single layer of scene graph convolution.
    """

    def __init__(self, obj_input_dim, object_output_dim, predicate_input_dim, predicate_output_dim, hidden_dim,
                 num_attributes, loc_dim=4, pooling='avg', mlp_normalization='none', return_new_p_vecs=True):
        super(GraphTripleConv, self).__init__()

        self.return_new_p_vecs = return_new_p_vecs
        self.hidden_dim = hidden_dim
        self.num_attributes = num_attributes
        self.predicate_output_dim = predicate_output_dim
        assert pooling in ['sum', 'avg'], 'Invalid pooling "%s"' % pooling
        self.pooling = pooling

        net1_layers = [2 * obj_input_dim + predicate_input_dim, hidden_dim,
                       2 * hidden_dim + self.predicate_output_dim]
        self.net1 = build_mlp(net1_layers, batch_norm=mlp_normalization, final_nonlinearity='relu')
        self.net1.apply(_init_weights)

        net2_layers = [hidden_dim, hidden_dim, object_output_dim]
        self.net2 = build_mlp(net2_layers, batch_norm=mlp_normalization, final_nonlinearity='relu')
        self.net2.apply(_init_weights)

    def forward(self, obj_vecs, pred_vecs, edges, pred_indicators):
        """
        Inputs:
        - obj_vecs: FloatTensor of shape (O, D) giving vectors for all objects
        - pred_vecs: FloatTensor of shape (T, D) giving vectors for all predicates
        - edges: LongTensor of shape (T, 2) where edges[k] = [i, j] indicates the
          presence of a triple [obj_vecs[i], pred_vecs[k], obj_vecs[j]]
        - pred_indicators:
        - positions:
        - acts_vecs:
        - temporal_edges:
        - acts_indicators:

        Outputs:
        - new_obj_vecs: FloatTensor of shape (O, D) giving new vectors for objects
        - new_pred_vecs: FloatTensor of shape (T, D) giving new vectors for predicates
        - new_acts_vecs:
        """
        dtype, device = obj_vecs.dtype, obj_vecs.device
        B, O, T = obj_vecs.size(0), obj_vecs.size(1), pred_vecs.size(1)

        # Objects and locations
        # Break apart indices for subjects and objects; these have shape (T,)
        s_idx = edges[:, :, 0].contiguous()
        o_idx = edges[:, :, 1].contiguous()

        cur_s_vecs = torch.stack([obj_vecs[b, s_idx[b], :] for b in range(B)])  # [B, N, d]
        cur_o_vecs = torch.stack([obj_vecs[b, o_idx[b], :] for b in range(B)])  # [B, N, d]

        cur_t_vecs = torch.cat([cur_s_vecs, pred_vecs, cur_o_vecs], dim=-1)  # [B, N, 3d]
        new_t_vecs = self.net1(cur_t_vecs)  # [B, T, d]

        new_s_vecs = new_t_vecs[:, :, :self.hidden_dim]
        new_p_vecs = new_t_vecs[:, :, self.hidden_dim:(self.hidden_dim + self.predicate_output_dim)]
        new_o_vecs = new_t_vecs[:, :, (self.hidden_dim + self.predicate_output_dim):]

        # important. for each batch, we mask the redundant triplets and don't add them up to avg object representation
        pooled_obj_vecs_batches = []
        for b in range(B):
            sample_predicates_indicator = pred_indicators[b]
            sample_s_idx = s_idx[b][sample_predicates_indicator]
            sample_o_idx = o_idx[b][sample_predicates_indicator]
            sample_new_s_vecs = new_s_vecs[b][sample_predicates_indicator]
            sample_new_o_vecs = new_o_vecs[b][sample_predicates_indicator]

            s_idx_exp = sample_s_idx.view(-1, 1).expand_as(sample_new_s_vecs)
            o_idx_exp = sample_o_idx.view(-1, 1).expand_as(sample_new_o_vecs)

            pooled_obj_vecs = torch.zeros(O, self.hidden_dim, dtype=dtype, device=device)
            pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, s_idx_exp, sample_new_s_vecs)
            pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, o_idx_exp, sample_new_o_vecs)

            obj_counts = torch.zeros(O, dtype=dtype, device=device)
            ones = torch.ones(T, dtype=dtype, device=device)
            obj_counts = obj_counts.scatter_add(0, sample_s_idx, ones)
            obj_counts = obj_counts.scatter_add(0, sample_o_idx, ones)

            obj_mask = obj_counts > 0
            pooled_obj_vecs[obj_mask] = pooled_obj_vecs[obj_mask] / obj_counts[obj_mask].view(-1, 1)
            pooled_obj_vecs_batches.append(pooled_obj_vecs)

        pooled_obj_vecs_batches = torch.stack(pooled_obj_vecs_batches, dim=0)
        new_obj_vecs = self.net2(pooled_obj_vecs_batches)
        if not self.return_new_p_vecs:
            new_p_vecs = pred_vecs

        return new_obj_vecs, new_p_vecs


