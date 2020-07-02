import os
import json
from pycocotools.coco import COCO

# Test dataset infomations
test_images = COCO('/home/haida_sunxin/lqx/code/github/two_aspp/data/DronesDET/Clusters/visdrone_cluster_test.json')
test_images = test_images.imgs
print(test_images[0])

# Cluster predictions
test_clusters_json = '/home/haida_sunxin/lqx/code/llseg/exp/Cascade_cluster_101_test/inference_TTA/coco_instances_results.json'
test_data = json.load(open(test_clusters_json, 'r'))

output_dir = './Cascade_cluster_101_test/test_cluster'
os.makedirs(output_dir, exist_ok=True)

# process image and clusters
for i in range(len(test_images)):
    item = test_images[i]
    name = item['file_name']
    base_name = os.path.basename(name)

    bbox = []
    scores = []
    classes = []
    clusters = [c_it for c_it in test_data if c_it['image_id'] == item['id']]
    for j in range(len(clusters)):
        cluster = clusters[j]
        score = cluster['score']
        if score < 0.5:
            continue
        box = cluster['bbox']
        class_c = cluster['category_id']
        bbox.append(box)
        classes.append(class_c)
        scores.append(score)

    # 判断是否有cluster
    if len(bbox) == 0:
        continue

    # 写入txt
    txt_file_name = base_name[:-3] + 'txt'
    txt_file_name = os.path.join(output_dir, txt_file_name)
    with open(txt_file_name, 'w') as f:
        for k in range(len(bbox)):
            temp_bbox = bbox[k]
            temp_score = scores[k]
            temp_class = classes[k]
            line = '%f,%f,%f,%f,%f,%d,0,1\n' % (
                round(float(temp_bbox[0])), round(float(temp_bbox[1])), round(float(temp_bbox[2])), round(float(temp_bbox[3])),
                int(temp_score), int(temp_class)
            )
            f.write(line)

    if (i+1) % 100 == 0:
        print("Step : %d/%d" % (i, len(test_images)))

