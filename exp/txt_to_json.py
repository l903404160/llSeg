import datetime
import json
import os
import re
import fnmatch
from PIL import Image
import numpy as np
from exp.clusters import read_centers
from pycococreatortools import pycococreatortools

from pycocotools.coco import COCO

def filter_for_jpeg(root, files):
    file_types = ['*.jpeg', '*.jpg']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if 'bg' not in f]

    return files


def filter_for_annotations(root, files, image_filename):
    file_types = ['*.txt']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    file_name_prefix = basename_no_extension + '.*'
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]

    # return ['/home/haida_sunxin/lqx/code/github/RRNet_train/test.txt']
    return files

def create_annotation_info(annotation_id, image_id, category_info, binary_mask,
                           image_size=None, tolerance=2, bounding_box=None):

    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_info["id"],
        "iscrowd": 0,
        "area": 0,
        "bbox": bounding_box.tolist(),
        "segmentation": [],
        "width": image_size[0],
        "height": image_size[1],
    }

    return annotation_info


def generate_whole_davis_json_file_with_only_one_class(root, video_sets, output_dir, output_file):
    info_dict = {
        "description": 'VisDrone',
        "url": "ouc_vision_lab",
        "version": "0.1.0",
        "year": 2020,
        "contributor": "lqx",
        "date_created": datetime.datetime.utcnow().isoformat(' ')
    }

    licenses_dict = [
        {
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License",
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
        }
    ]

    categories_dict = []
    data = {
        'id': 0,
        'name': 'foreground',
        'supercategory': 'cluster'
    }
    categories_dict.append(data)

    coco_output = {
        "info": info_dict,
        "licenses": licenses_dict,
        "categories": categories_dict,
        "images": [],
        "annotations": []
    }

    image_id = 0
    seg_id = 0

    def _append_video_data(root, video, coco_output_data, image_id, seg_id):
        IMAGE_ID = image_id
        SEG_ID = seg_id

        image_dir = os.path.join('/home/haida_sunxin/lqx/data/DronesDET', 'test', 'images')

        annotation_dir = os.path.join('/home/haida_sunxin/lqx/code/llseg/exp', 'cluster_annotations_test')

        count = 0

        # filter for jpeg images
        for root, _, files in os.walk(image_dir):
            image_files = filter_for_jpeg(root, files)
            image_files.sort()
            # go through each image
            for image_filename in image_files:
                image = Image.open(image_filename)
                image_info = pycococreatortools.create_image_info(
                    IMAGE_ID, os.path.join(image_dir, os.path.basename(image_filename)), image.size)
                coco_output_data["images"].append(image_info)

                # filter for associated png annotations
                for root, _, files in os.walk(annotation_dir):
                    annotation_files = filter_for_annotations(root, files, image_filename)
                    if len(annotation_files) == 0:
                        continue
                    anno_file = annotation_files[0]

                    center, bbox = read_centers(anno_file)

                    num_objects = len(bbox)
                    print('Image infos: %s, %d' % (anno_file, num_objects))

                    for i in range(num_objects):
                        lb = None
                        #
                        category_info = {'id': 0, 'is_crowd': False}
                        annotation_info = create_annotation_info(
                            SEG_ID, IMAGE_ID, category_info, lb,
                            image.size, tolerance=2, bounding_box=bbox[i])
                        if annotation_info is not None:
                            coco_output_data["annotations"].append(annotation_info)
                        SEG_ID = SEG_ID + 1

                IMAGE_ID = IMAGE_ID + 1

        return coco_output, IMAGE_ID, SEG_ID

    v = None
    coco_output, image_id, seg_id = _append_video_data(root, v, coco_output, image_id, seg_id)
    print(coco_output)
    with open('{}/{}'.format(output_dir, output_file), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)


def generate_val_set_json(root, output_dir):
    video_set = os.path.join(root, 'ImageSets', '2017', 'val.txt')
    output_file = 'davis_foreground_ins_val2020.json'

    with open(video_set, 'r') as f:
        videos = f.readlines()
    videos = ''.join(videos).split('\n')[:-1]

    generate_whole_davis_json_file_with_only_one_class(root, videos, output_dir, output_file)


def generate_train_set_json(root, output_dir):
    output_file = 'visdrone_cluster_test_cluster.json'
    videos = None
    generate_whole_davis_json_file_with_only_one_class(root, videos, output_dir, output_file)


if __name__ == "__main__":
    """
        Use this file to generate the COCO format Annotations 
        which can be further used to registry the dataset in Detectron2
    """

    root = '/home/haida_sunxin/lqx/code/github/RRNet_train/data/DronesDET'
    output_dir = os.path.join(root, 'Clusters')
    # os.makedirs(output_dir, exist_ok=True)

    # generate_val_set_json(root, output_dir)
    generate_train_set_json(root, output_dir)