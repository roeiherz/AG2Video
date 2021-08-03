import torch
from torch.tensor import Tensor
import torch.nn.functional as F

from data.args import ALIGN_CORNERS


class LossManager(object):
    def __init__(self):
        self.total_loss = None
        self.all_losses = {}

    def add_loss(self, loss, name, weight=1.0):
        cur_loss = loss * weight
        if self.total_loss is not None:
            self.total_loss += cur_loss
        else:
            self.total_loss = cur_loss

        self.all_losses[name] = cur_loss.data.cpu().item()

    def items(self):
        return self.all_losses.items()


def batch_to(batch, device='cuda'):
    output_batch = []
    for obj in batch:

        if obj is None:
            obj = None
        elif isinstance(obj, Tensor):
            if device == 'cuda':
                obj = obj.cuda()
            elif isinstance(device, torch.device):
                obj = obj.to(device)
            else:
                obj = obj.cpu().numpy()
        elif isinstance(obj, list):
            if len(obj) == 0:
                obj = None
        else:
            for k, v in obj.items():
                if device == 'cuda':
                    obj[k] = obj[k].cuda()
                elif isinstance(device, torch.device):
                    obj[k] = obj[k].to(device)
                else:
                    obj[k] = obj[k].cpu().numpy()
        output_batch.append(obj)
    return output_batch


def remove_small_boxes(bbox_pred, min_size=0):
    """
    Only keep boxes with both sides >= min_size
    Arguments:
        boxlist (Boxlist)
        min_size (int)
    :return: indices to keep
    """
    x0, y0, ww, hh = bbox_pred.unbind(dim=1)
    return ((ww > min_size) & (hh > min_size)).nonzero().squeeze(1)


def compute_new_lr(lr, decay):
    new_lr = lr - lr / decay
    new_lr_G = new_lr / 2
    new_lr_D = new_lr * 2
    return new_lr_G, new_lr_D


def update_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def log_scalar_dict(writer, d, tag, itr, every=500):
    if itr%every==0:
        for k, v in d.items():
            writer.add_scalar('%s/%s' % (tag, k), v, itr)


def remove_dummy_objects_old(objs):

    if objs.size(1) < 1:
        # Objs contain only single attribute (VG/COCO), dummy is 0
        mask = (objs != 0).squeeze(1)
    else:
        # Objs contain multiple attributes (CLEVR), dummy are [0, 0...0]
        mask = torch.sum(objs, dim=1) != 0
    return mask


def remove_dummy_objects(objs, vocab):
    # boxes = boxes.reshape((-1, 4))
    # isnotpadding_mask = (boxes != -1).any(dim=-1)
    isnotpadding_mask = (objs != 0)[:, 0]
    __image__ = vocab['object_name_to_idx']["__image__"]
    dummies_objs_mask = (objs != __image__)[:, 0]
    mask = dummies_objs_mask & isnotpadding_mask
    return mask


def remove_dummies_and_padding(boxes, objs, vocab, items_lst):
    isnotpadding_mask = (boxes != -1).any(dim=-1)
    __image__ = vocab['object_name_to_idx']["__image__"]
    dummies_objs_mask = (objs != __image__)[:, 0]
    new_mask = dummies_objs_mask & isnotpadding_mask
    return [item[new_mask] for item in items_lst]


def resample(image, flow=None):
    if flow is None:
        b, c, h, w = image.size()
        grid = get_grid(b, h, w)
        final_grid = (grid).permute(0, 2, 3, 1).cuda(image.get_device())
        return F.grid_sample(image, final_grid, mode='bilinear', padding_mode='border', align_corners=ALIGN_CORNERS)

    b, c, h, w = image.size()
    grid = get_grid(b, h, w, gpu_id=flow.get_device())
    flow = torch.cat([flow[:, 0:1, :, :] / ((w - 1.0) / 2.0), flow[:, 1:2, :, :] / ((h - 1.0) / 2.0)], dim=1)
    final_grid = (grid + flow).permute(0, 2, 3, 1).cuda(image.get_device())
    return torch.nn.functional.grid_sample(image, final_grid, mode='bilinear', padding_mode='border', align_corners=ALIGN_CORNERS)


def get_grid(batchsize, rows, cols, gpu_id=0):
    hor = torch.linspace(-1.0, 1.0, cols)
    hor.requires_grad = False
    hor = hor.view(1, 1, 1, cols)
    hor = hor.expand(batchsize, 1, rows, cols)
    ver = torch.linspace(-1.0, 1.0, rows)
    ver.requires_grad = False
    ver = ver.view(1, 1, rows, 1)
    ver = ver.expand(batchsize, 1, rows, cols)

    t_grid = torch.cat([hor, ver], 1)
    t_grid.requires_grad = False

    return t_grid.cuda(gpu_id)
