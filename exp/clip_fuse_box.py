import PIL.Image as I
import numpy as np
import pandas as pd
import json
import os

__SPLIT__ = 'test'

def merge_bbox_to_original_image(pred_bboxes, scale, patch_bbox):
    """
    Args:
        pred_bboxes: X,Y,W,H
        scale:
        patch_bbox:
    Returns:
    """
    import torch
    pred_bboxes_center = pred_bboxes[:, :2] + pred_bboxes[:, 2:] / 2
    pred_bboxes_center = pred_bboxes_center / scale
    w_h_pred_bboxes = pred_bboxes[:, 2:] / scale

    scaled_bboxes = torch.zeros_like(pred_bboxes)
    scaled_bboxes[:, :2] = pred_bboxes_center - w_h_pred_bboxes / 2
    scaled_bboxes[:, 2:] = w_h_pred_bboxes

    # add original bbox
    scaled_bboxes[:, :2] += patch_bbox[:2]
    return scaled_bboxes


def resize_bbox_according_to_scale(bboxes, scale, patch_box):
    """
    Args:
        bboxes: x,y,w,h
        scale: float
        patch_box:
    Returns: bboxes x,y,w,h
    """
    # get relative position
    bboxes[:, :2] = bboxes[:, :2] - patch_box[:2]

    # get boxes center
    bboxes_center = bboxes[:, :2] + bboxes[:, 2:] / 2
    bboxes_center = bboxes_center * scale

    w_h = bboxes[:, 2:] * scale
    scaled_bboxes = np.zeros_like(bboxes).astype(np.float)
    scaled_bboxes[:, :2] = bboxes_center - w_h / 2
    scaled_bboxes[:, 2:] = w_h
    return scaled_bboxes


def get_labels_according_to_bbox(txt_file, bbox, min_scale):
    root = '/home/haida_sunxin/lqx/data/DronesDET/'+__SPLIT__+'/annotations'
    sample_txt = os.path.join(root, txt_file)
    data = pd.read_table(sample_txt, header=None)
    data = np.array(data[0])
    bboxes = []
    for box in data:
        temp = box.split(',')
        x, y, w, h = float(temp[0]), float(temp[1]), float(temp[2]), float(temp[3])
        temp_box = [x, y, w+x, h+y] # x1y1 x2y2
        bboxes.append(temp_box)
    bboxes = np.array(bboxes).astype(np.int)

    # crop bbox
    x1,y1,x2,y2 = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
    condition = np.zeros_like(bboxes)
    condition[:, 0] = bboxes[:, 0] > x1
    condition[:, 1] = bboxes[:, 1] > y1
    condition[:, 2] = bboxes[:, 2] < x2
    condition[:, 3] = bboxes[:, 3] < y2
    inds = condition.all(axis=1)
    annos = data[inds]

    inside_bboxes = []
    for box in annos:
        temp = box.split(',')
        x, y, w, h = float(temp[0]), float(temp[1]), float(temp[2]), float(temp[3])
        temp_box = [x, y, w, h] # x1y1 x2y2
        inside_bboxes.append(temp_box)
    inside_bboxes = np.array(inside_bboxes).astype(np.int)
    # Resize bbox
    if len(inside_bboxes) == 0:
        return []
    inside_scaled_bboxes = resize_bbox_according_to_scale(inside_bboxes, min_scale, bbox)

    new_annos = []
    for id, data in enumerate(annos):
        temp_bbox = inside_scaled_bboxes[id]
        temp = data.split(',')
        if int(temp[5]) == 11:
            continue
        line = '%d,%d,%d,%d,%d,%d,0,1' % (
            round(float(temp_bbox[0])), round(float(temp_bbox[1])), round(float(temp_bbox[2])), round(float(temp_bbox[3])),
            int(temp[4]), int(temp[5])
        )
        new_annos.append(line)

    return new_annos


def crop_image_patch_and_convert_json_to_txt(json_file, output_dir='./patches_'+__SPLIT__):
    import os
    # make directory

    txt_output = os.path.join(output_dir, 'annotations')
    img_output = os.path.join(output_dir, 'images')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(txt_output, exist_ok=True)
    os.makedirs(img_output, exist_ok=True)

    root = "/home/haida_sunxin/lqx/data/DronesDET"
    img_root = os.path.join(root, __SPLIT__, 'images')

    from pycocotools.coco import COCO
    # TODO  make there more beautiful
    # coco_gt = COCO('/home/haida_sunxin/lqx/data/DronesDET/val/annotations/val.json')
    # coco_dt = coco_gt.loadRes(json_file)
    coco_dt = COCO(json_file)

    filename_scale_bboxes = {}
    filename_scale_bboxes['data'] = []

    imgs = coco_dt.imgs
    for id, im in imgs.items():

        if (id+1) % 100 == 0:
            print("Step : %d / %d" %(id, len(imgs.keys())))

        img_id = im['id']
        img_file_name = im['file_name']
        img_file_name = os.path.basename(img_file_name)
        txt_file_name = img_file_name[:-3] + 'txt'
        detections = coco_dt.loadAnns(coco_dt.getAnnIds(img_id))
        # Crop image patches according to detections

        for id, item in enumerate(detections):
            p_name = 'patch_'+str(id)+'_'+img_file_name
            p_txt_name = 'patch_'+str(id)+'_'+txt_file_name

            x, y, w, h = item['bbox']

            # read image # H, W, C
            im = np.array(I.open(os.path.join(img_root, img_file_name)))
            im_patch = im[int(y):int(y+h), int(x):int(x+w)]
            im_patch = I.fromarray(im_patch)
            H, W, C = im.shape
            min_scale = min(W / w, H / h)
            if min_scale > 10:
                continue
            # min_scale = 1
            # annos = get_labels_according_to_bbox(txt_file_name, item['bbox'], min_scale)
            annos = ['10,10,100,100,1,1,0,1']
            # Annotations need transform

            if len(annos) == 0:
                continue
            new_w, new_h = int(w*min_scale), int(h*min_scale)
            im_patch = im_patch.resize((new_w, new_h), I.BILINEAR)
            im_patch.save(os.path.join(img_output, p_name))

            # get corresponding labels

            with open(os.path.join(txt_output, p_txt_name), 'w') as f:
                for l in annos:
                    f.write(l + '\n')

            patch_info = {
                'file_name': os.path.join("/home/haida_sunxin/lqx/data/DronesDET/" + __SPLIT__ + "/patch_"+__SPLIT__, p_name),
                'scale': min_scale,
                'patch_box': item['bbox']
            }
            filename_scale_bboxes['data'].append(patch_info)

        with open(os.path.join(output_dir, 'patch_info.json'), 'w') as f:
            json.dump(filename_scale_bboxes, f)
    print("Done ...")

if __name__ == '__main__':
    crop_image_patch_and_convert_json_to_txt(json_file='/home/haida_sunxin/lqx/data/DronesDET/Clusters/visdrone_cluster_'+__SPLIT__+'_cluster.json')