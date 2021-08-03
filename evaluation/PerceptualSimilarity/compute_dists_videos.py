from os.path import join
from os.path import exists
import torch
import argparse
import os
import torch.nn.functional as F
import models
from evaluation.PerceptualSimilarity.models import PerceptualLoss
from evaluation.PerceptualSimilarity.util import util
import glob
import pickle
import numpy as np


def plot_vid(vids, boxes_gt=None, boxes_pred=None):
    vids = vids.cpu().numpy()
    vids = np.transpose(vids, [0, 2, 3, 1])
    output_imgs = []
    for i in range(0, vids.shape[0], 1):
        img = np.clip((vids[i] * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255, 0,
                      255).astype('uint8').copy()
        normalized_img = util.im2tensor(img)  # RGB image from [-1,1]
        # normalized_img = F.interpolate(normalized_img, size=64)
        output_imgs.append(normalized_img)

    return torch.cat(output_imgs)


def get_video_from_pkl(ff):
    video_tensor = ff['image']
    # Remove the first batch dim if exits
    if len(video_tensor.size()) == 5:
        video_tensor = video_tensor.squeeze()
    video = plot_vid(video_tensor)
    return video


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d0', '--dir0', type=str, default='./imgs/ex_dir0')
    parser.add_argument('-d1', '--dir1', type=str, default='./imgs/ex_dir1')
    parser.add_argument('-o', '--out', type=str, default='./imgs/example_dists.txt')
    parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')
    opt = parser.parse_args()

    ## Initializing the model
    model = PerceptualLoss(model='net-lin', net='alex', use_gpu=opt.use_gpu)

    # crawl directories
    files = glob.glob(opt.dir0 + '/*.pkl')
    videos = [os.path.basename(fl) for fl in files]
    res = []
    for vid in videos:
        if exists(join(opt.dir1, vid)):

            # Load pickles
            f0 = pickle.load(open(join(opt.dir0, vid), 'rb'))
            f1 = pickle.load(open(join(opt.dir1, vid), 'rb'))

            img0 = get_video_from_pkl(f0)  # RGB images from [-1,1]
            img1 = get_video_from_pkl(f1)

            # Load images
            # img0 = util.im2tensor(util.load_image(os.path.join(opt.dir0, folder, file)))
            # img1 = util.im2tensor(util.load_image(os.path.join(opt.dir1, folder, file)))

            if (opt.use_gpu):
                img0 = img0.cuda()
                img1 = img1.cuda()

            # Compute distance
            dist01 = model.forward(img0, img1)
            # print('%s: %.3f' % (file, dist01))
            res.append(dist01.mean())

    # Save
    np.save(opt.out, torch.stack(res).data.cpu().numpy())
    mean = torch.mean(torch.stack(res))
    std = torch.std(torch.stack(res))
    print("Diversity: {}±{}".format(mean, std))

