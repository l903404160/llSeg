import glob
import os
import PIL.Image as I
import numpy as np



label_path = '/root/dataset/suim/train_val/masks'
image_path = '/root/dataset/suim/train_val/images'
cvt_path = '/root/dataset/suim/train_val/masks_cvt'
label_path = '/root/dataset/suim/test/masks'
image_path = '/root/dataset/suim/test/images'
cvt_path = '/root/dataset/suim/test/masks_cvt'

# files = glob.glob(os.path.join(label_path, '*.bmp'))
imgs = glob.glob(os.path.join(image_path, '*.jpg'))
files = [os.path.join(label_path, os.path.basename(f)[:-3] + 'bmp') for f in imgs]
cvt_files = [os.path.join(cvt_path, os.path.basename(f)[:-3] + 'png') for f in imgs]

for im_f, lb_f in zip(imgs, cvt_files):
    im = I.open(im_f)
    lb = I.open(lb_f)
    if im.size != lb.size:
        print('Image: {}'.format(im_f),im.size,'============\n')
        print('Label: {}'.format(lb_f),lb.size,'============\n')
        # os.remove(im_f)
        # os.remove(lb_f)

import sys
sys.exit(0)
label_table = np.array([
    [0,0,0],
    [0,0,255],
    [0,255,0],
    [0,255,255],
    [255,0,0],
    [255,0,255],
    [255,255,0],
    [255,255,255],
])
class_table = label_table[:,0] * 1 + label_table[:,1]*2 + label_table[:,2]*4


for f in files:
    lb = np.array(I.open(f)).astype(np.int64)
    h, w, c = lb.shape
    temp_label = np.zeros(shape=(h, w)).astype(np.int64)
    temp_sum = lb[:,:,0] * 1 + lb[:,:,1]*2 + lb[:,:,2]*4
    for i in range(len(class_table)):
        temp_label[temp_sum == class_table[i]] = i
    temp_label = temp_label.astype(np.uint8)
    lb = I.fromarray(temp_label)
    name = os.path.join(cvt_path, os.path.basename(f))
    # lb.save(name[:-3]+'png')
    print(lb.size)



