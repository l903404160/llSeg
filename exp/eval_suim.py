from engine.defaults import DefaultPredictor
from configs.sem_seg_configs.pcnet_config import PCNet_config as config
import cv2
import glob
import os

import torch

files = glob.glob(os.path.join('/root/dataset/suim/test/images', '*.jpg'))



config.MODEL.WEIGHTS = '/root/codes/vot/exp/output_dir/model_0029999.pth'

predictor = DefaultPredictor(config)

import numpy as np
colormap = np.array([
    [0,0,0],
    [0,0,128],
    [0,128,0],
    [0,128,128],
    [128,0,0],
    [128,0,128],
    [128,128,0],
    [128,128,128],
]).astype(np.uint8)

import PIL.Image as I
os.makedirs('./result', exist_ok=True)

for f in files:

    inputs = cv2.imread(f)

    output = predictor(inputs)
    output = torch.max(output, dim=1)[1].detach().cpu().numpy().astype(np.uint8)[0]
    print(output.shape)
    out = colormap[output]
    im = I.fromarray(out)
    im.save(os.path.join('./result', os.path.basename(f)))




print('done')
