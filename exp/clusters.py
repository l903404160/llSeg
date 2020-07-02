import os
import pandas as pd
import numpy as np
import PIL.Image as I
import glob
import cv2


def read_centers(sample_path):
    data = pd.read_table(sample_path, header=None)
    data = np.array(data[0])
    center = []
    bbox = []
    for box in data:
        temp = box.split(',')
        # except ignore
        if temp[5] == 0:
            continue
        x,y,w,h = float(temp[0]), float(temp[1]), float(temp[2]), float(temp[3])
        temp_box = [x,y,w,h]
        bbox.append(temp_box)
        temp_center = [x+w/2, y+h/2]
        center.append(temp_center)
    center = np.array(center)
    bbox = np.array(bbox)
    return center, bbox

def kmeans_predict(centers, k=3):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=k, random_state=9).fit(centers)
    temp_center = centers.copy()
    # temp_center[:, 0] = temp_center[:, 0] / 1.2
    labels = kmeans.predict(temp_center)
    return labels

def generate_cluster_bbox(bbox, labels):
    """
    Args:
        bbox: xywh
        labels:
    Returns:
    """
    bbox[:, 2:] = bbox[:, 2:] + bbox[:, :2]  # TO xyxy
    clusters = np.unique(labels)
    cluster_bboxes = []
    for cluster in clusters:
        idxs = labels == cluster
        cluster_boxes = bbox[idxs]
        cluster_box_tl = cluster_boxes[:, :2].min(axis=0)
        cluster_box_br = cluster_boxes[:, 2:].max(axis=0)
        cluster_box_wh = cluster_box_br - cluster_box_tl
        cluster_bbox = [cluster_box_tl[0], cluster_box_tl[1], cluster_box_wh[0], cluster_box_wh[1]]
        cluster_bboxes.append(cluster_bbox)
    cluster_bboxes = np.array(cluster_bboxes)
    return cluster_bboxes


def generate_cluster_results(input_dir, image_dir, output_dir, n=0):
    txt_files = glob.glob(os.path.join(input_dir, '*.txt'))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for i, label_file in enumerate(txt_files):
        centers, bbox = read_centers(label_file)
        total_centers = len(centers)
        if total_centers <= 1:
            labels = [0]
        elif total_centers > 1 and total_centers < 4: # 20
            labels = kmeans_predict(centers, k=2)
            # continue
        elif total_centers >= 4 and total_centers < 10:  # [20, 60]
            labels = kmeans_predict(centers, k=3)
        elif total_centers >=10 and total_centers < 20:  # [60, 100]
            labels = kmeans_predict(centers, k=4)
        else:
            labels = kmeans_predict(centers, k=5)
        clusters = generate_cluster_bbox(bbox, labels)
        # paint
        with open(os.path.join(output_dir, os.path.basename(label_file)), 'w') as f:
            for c_box in clusters:
                x, y, w, h = c_box[0], c_box[1], c_box[2], c_box[3]
                score, category = 1, 0
                line = '%f,%f,%f,%f,%.4f,%d,-1,-1\n' % (
                    float(x), float(y), float(w), float(h),
                    float(score), int(category)
                )
                f.write(line)
        if (i+1) % 100 == 0:
            print('Step %d/6471' % i)

        # image_file = os.path.basename(label_file)[:-3] + 'jpg'
        # img = cv2.imread(os.path.join(image_dir, image_file))
        # for c_box in clusters:
        #     cv2.rectangle(img, (c_box[0], c_box[1]), (c_box[0] + c_box[2], c_box[1] + c_box[3]), color=(0, 255, 0))
        # cv2.imwrite(os.path.join(output_dir, image_file), img)


if __name__ == '__main__':
    # in_dir = '/home/haida_sunxin/szq/rrnet/data/DronesDET/val/annotations'
    # img_dir = '/home/haida_sunxin/szq/rrnet/data/DronesDET/val/images'

    in_dir = '/home/haida_sunxin/lqx/code/llseg/exp/Cascade_cluster_101_test/test_cluster'
    img_dir = '/home/haida_sunxin/lqx/code/github/RRNet_train/data/DronesDET/test/images'

    output_dir = './cluster_annotations_test'
    generate_cluster_results(in_dir, image_dir=img_dir, output_dir=output_dir, n=0)
    print('done')

    # import os
    # import json
    # from pycocotools.coco import COCO
    #
    # train_json_file = '/home/haida_sunxin/lqx/code/github/RRNet/data/DronesDET/train/annotations/train.json'
    #
    # coco = COCO(train_json_file)
    #
    # import random
    # import numpy as np
    # np.random.choice([1,23,4], p=[0.2, 0.5, 0.3])
    #
    # cats = coco.cats
    # coco.loadImgs()
    # print(cats)
    # print('done')

# sample_path = '/home/haida_sunxin/szq/rrnet/data/DronesDET/train/annotations/9999999_00887_d_0000407.txt'
# image_path = '/home/haida_sunxin/szq/rrnet/data/DronesDET/train/images/9999999_00887_d_0000407.jpg'
#
# centers, bbox = read_centers(sample_path)
# labels = kmeans_predict(centers)
# clusters = generate_cluster_bbox(bbox, labels)
# import matplotlib.pyplot as plt
#
# """
#
#     K-means Clustering
#
# """
#
# from sklearn.cluster import KMeans
# kmeans = KMeans(n_clusters=5, random_state=9).fit(centers)
# temp_center = centers.copy()
# temp_center[:, 0] = temp_center[:, 0] / 1.2
#
# # temp_center[:, 1] = temp_center[:, 1][temp_center[:, 1] < 400]
# temp_center[:, 1][temp_center[:, 1] < 400] -= 200
# temp_center[:, 1][temp_center[:, 1] > 400] += 200
#
# labels = kmeans.predict(temp_center)
# plt.scatter(centers[:, 0], centers[:, 1], c=labels, s=50, cmap='viridis')
#
# print(np.unique(labels))
#
# idxs0 = labels == 0
# center0 = centers[idxs0]
# center0 = center0.mean(axis=0)
# print(center0)
# im = np.array(I.open(image_path))
# plt.imshow(im)
#
# plt.show()
#
#
#
#
