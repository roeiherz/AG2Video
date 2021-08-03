import torch
import torch.nn as nn


class AttributeEmbedding(nn.Embedding):

    def __init__(self, num_embeddings, embedding_dim):
        super(AttributeEmbedding, self).__init__(num_embeddings, embedding_dim)

    def forward(self, x):
        ebmd = torch.matmul(x, self.weight)
        avg_embd = ebmd / torch.sum(x, dim=1, keepdim=True)
        return avg_embd


class AttributeEmbeddings(nn.Module):

    def __init__(self, attributes, embedding_dim, use_attr_fc_gen=False):
        super(AttributeEmbeddings, self).__init__()
        sorted(attributes)
        num_attr = len(attributes)
        if num_attr > 1 or use_attr_fc_gen:
            self.attribute_fc_gen = nn.Linear(num_attr * embedding_dim, num_attr * embedding_dim)
        self.attribute_embedding_lst = []
        for i in range(len(attributes)):
            self.add_module("att_emb_{}".format(str(i)),
                            nn.Embedding(max(attributes[list(attributes)[i]].values())+1, embedding_dim))

    def forward(self, x):
        """

        :param x: [B, O, A]
        :return:
        """

        obj_vecs = []

        for k in range(x.size(-1)):
            embedding_layer = self._modules["att_emb_{}".format(k)]
            embedding = embedding_layer(x[:, :, k])
            obj_vecs.append(embedding)

        obj_vecs = torch.cat(obj_vecs, dim=-1)  # [B, N, d * A]
        if hasattr(self, 'attribute_fc_gen'):
            obj_vecs = self.attribute_fc_gen(obj_vecs)  # [B, N, d']
        return obj_vecs
