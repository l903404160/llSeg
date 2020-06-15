from torchvision.ops import boxes as box_ops
import torch
import numpy as np
from layers import soft_nms


boxes = np.array([
    [100, 100, 200, 200],
    [80, 80, 180, 180],
    [90, 90, 190, 190],
    [110, 110, 210, 210],
    [120, 120, 220, 220],
    [300, 300, 400, 400],
    [350, 350, 450, 450],
    [370, 370, 470, 470],
    [230, 230, 330, 330],
    [280, 280, 380, 380],
    [30, 30, 40, 40],
    [31, 30, 41, 40],
    [37, 37, 47, 47],
    [23, 23, 33, 33],
    [28, 28, 38, 38],
])

scores = np.array([0.9, 0.8, 0.85, 0.88, 0.75, 0.95, 0.7, 0.4, 0.5, 0.85, 0.95, 0.7, 0.4, 0.5, 0.85])
idxs = np.array([1,1,1,1,1,2,2,2,2,2,3,3,3,3,3])


boxes = torch.from_numpy(boxes).float()
scores = torch.from_numpy(scores).float()
idxs = torch.from_numpy(idxs).float()
print(idxs.to(boxes))

max_coordinate = boxes.max()
offsets = idxs.to(boxes) * (max_coordinate + 1)
print("max_coordinate : ----", max_coordinate)
print("Offsets : ----", offsets)
thre = 0.7

keep = box_ops.batched_nms(boxes, scores, idxs, thre)
print("NMS --------------------- ", keep)

keep = soft_nms(boxes, scores, method="linear", sigma=0.5, threshold=thre)
print("Soft NMS --------------------- ", keep)
print('done')
