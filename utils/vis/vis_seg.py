import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as vutils

_COLOR_MAP = np.array([
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
    [0, 0, 0]
], dtype=np.uint8)


def get_sem_seg_image_grid(img, pred, label, ignore_label=255):
    """
    :param img: tensor [C, H, W]
    :param pred:  tensor [K, H, W]
    :param label:  tensor [H, W]
    :return: concated image tensor
    """
    pred = pred.detach().cpu()
    pred = torch.max(F.softmax(pred, dim=0), dim=0)[1].numpy()
    pred = _COLOR_MAP[pred.astype(np.uint8)]
    pred = torch.from_numpy(pred).permute(2,0,1)
    label = label.detach().cpu()
    label[label == ignore_label] = 19
    label = _COLOR_MAP[label.numpy().astype(np.uint8)]
    label = torch.from_numpy(label).permute(2,0,1)
    final_img = torch.stack([img, pred, label])
    return vutils.make_grid(final_img, normalize=False)


if __name__ == '__main__':
    img = torch.randint(0,255, [3, 100, 100]).byte()
    pred = torch.randn(19, 100, 100)
    label = torch.randint(0, 19, [100, 100])

    fi = get_sem_seg_image_grid(img, pred, label)
    print(fi.size())