import tempfile, os
import torch
import numpy as np
import cv2
import imageio

"""
Utilities for making visualizations.
"""


def save_video(images, output_fn):
    writer = imageio.get_writer(output_fn, fps=20)
    for im in images:
        writer.append_data(im)
    writer.close()


def draw_boxes(img, boxes, grid_size, color):
    img = img.copy()
    for j in range(boxes.shape[0]):
        x, y, w, h = boxes[j] * grid_size
        if x == 0 and y == 0 and w == grid_size and h == grid_size:
            continue
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
    return img


def plot_vid(vids, boxes_gt=None, boxes_pred=None):
    vids = vids.cpu().numpy()
    vids = np.transpose(vids, [0, 2, 3, 1])
    output_imgs = []
    for i in range(0, vids.shape[0], 1):
        img = np.clip((vids[i] * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255, 0,
                      255).astype('uint8').copy()
        grid_size = img.shape[0] - 1
        if boxes_gt is not None:
            img = draw_boxes(img, boxes_gt[i], grid_size, color=(255, 0, 0))
        if boxes_pred is not None:
            img = draw_boxes(img, boxes_pred[i], grid_size, color=(0, 0, 255))

        output_imgs.append(img)

    return output_imgs


def save_images(args, t, val_samples, dir_name='val'):
    path = os.path.join(args.output_dir, dir_name, str(t))
    if not os.path.exists(path):
        os.makedirs(path)

    vids = val_samples['vids']
    boxes = val_samples['gt_boxes']
    pred_boxes = val_samples['pred_boxes']
    # Video generation and boxes prediction
    pred_vid_gtbox = val_samples.get('pred_vids_gt_boxes', None)
    pred_vid_gtbox_boxes = val_samples.get('pred_vids_gt_boxes_boxes', None)
    pred_vid_predbox = val_samples.get('pred_vids_pred_boxes', None)
    pred_vid_predbox_boxes = val_samples.get('pred_vids_pred_boxes_boxes', None)
    pred_vid_gtflows = val_samples.get('pred_vids_gt_flows', None)
    pred_vid_predflows = val_samples.get('pred_vids_pred_flows', None)

    ind = -1
    for b in range(len(vids)):
        for i in range(vids[b].shape[0]):
            try:
                # print("Save video id: {}".format(val_samples['video_id'][b][i]))
                ind += 1
                vids_i = vids[b][i]
                boxes_i = boxes[b][i]
                pred_boxes_i = pred_boxes[b][i]
                boxes_mask = ~((boxes_i == torch.LongTensor([-1, -1, -1, -1]).to(boxes_i)).all(dim=-1) +
                               (boxes_i == torch.LongTensor([0, 0, 1, 1]).to(boxes_i)).all(dim=-1))

                # save gt
                boxes_i = [boxes_i[j][boxes_mask[j]] for j in range(boxes_mask.shape[0])]
                output_imgs = plot_vid(vids_i.clone(), boxes_i.copy(), None)
                save_video(output_imgs, os.path.join(path, f"gt_box_{ind}.mp4"))

                # save pred
                pred_boxes_i = [pred_boxes_i[j][boxes_mask[j]] for j in range(boxes_mask.shape[0])]
                output_imgs = plot_vid(vids_i.clone(), None, pred_boxes_i.copy())
                save_video(output_imgs, os.path.join(path, f"pred_box_{ind}.mp4"))

                # save video gt
                if pred_vid_gtbox is not None and pred_vid_gtbox_boxes is not None:
                    pred_vid_gtbox_vid_i = pred_vid_gtbox[b][i]
                    pred_vid_gtbox_boxes_i = pred_vid_gtbox_boxes[b][i]
                    pred_vid_gtbox_boxes_i = [pred_vid_gtbox_boxes_i[j][boxes_mask[j]] for j in range(boxes_mask.shape[0])]
                    output_imgs = plot_vid(pred_vid_gtbox_vid_i.clone(), pred_vid_gtbox_boxes_i.copy(), None)
                    save_video(output_imgs, os.path.join(path, f"pred_vid_gt_box_{ind}.mp4"))

                # save gt flows video gt
                if pred_vid_gtflows is not None and pred_vid_gtbox_boxes is not None:
                    pred_vid_gtbox_vid_i = pred_vid_gtflows[b][i]
                    pred_vid_gtbox_boxes_i = pred_vid_gtbox_boxes[b][i]
                    pred_vid_gtbox_boxes_i = [pred_vid_gtbox_boxes_i[j][boxes_mask[j]] for j in range(boxes_mask.shape[0])]
                    output_imgs = plot_vid(pred_vid_gtbox_vid_i.clone(), pred_vid_gtbox_boxes_i.copy(), None)
                    save_video(output_imgs, os.path.join(path, f"pred_vid_gt_flows_gtbox_{ind}.mp4"))

                # save pred flows video gt
                if pred_vid_predflows is not None and pred_vid_gtbox_boxes is not None:
                    pred_vid_gtbox_vid_i = pred_vid_predflows[b][i]
                    pred_vid_gtbox_boxes_i = pred_vid_gtbox_boxes[b][i]
                    pred_vid_gtbox_boxes_i = [pred_vid_gtbox_boxes_i[j][boxes_mask[j]] for j in range(boxes_mask.shape[0])]
                    output_imgs = plot_vid(pred_vid_gtbox_vid_i.clone(), pred_vid_gtbox_boxes_i.copy(), None)
                    save_video(output_imgs, os.path.join(path, f"pred_vid_pred_flows_gtbox_{ind}.mp4"))

                # save video pred
                if pred_vid_predbox is not None and pred_vid_predbox_boxes is not None:
                    pred_vid_predbox_vid_i = pred_vid_predbox[b][i]
                    pred_vid_predbox_boxes_i = pred_vid_predbox_boxes[b][i]
                    pred_vid_predbox_boxes_i = [pred_vid_predbox_boxes_i[j][boxes_mask[j]] for j in range(boxes_mask.shape[0])]
                    output_imgs = plot_vid(pred_vid_predbox_vid_i.clone(), None, pred_vid_predbox_boxes_i.copy())
                    save_video(output_imgs, os.path.join(path, f"pred_vid_pred_box_{ind}.mp4"))

            except Exception as e:
                print("error in saving video: {}".format(e))


