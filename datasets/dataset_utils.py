import torch
import numpy as np

from PIL import Image


# https://en.wikipedia.org/wiki/YUV#SDTV_with_BT.601
_M_RGB2YUV = [[0.299, 0.587, 0.114], [-0.14713, -0.28886, 0.436], [0.615, -0.51499, -0.10001]]
_M_YUV2RGB = [[1.0, 0.0, 1.13983], [1.0, -0.39465, -0.58060], [1.0, 2.03211, 0.0]]

def read_image(file_name, format=None):
    with open(file_name, 'rb') as f:
        image = Image.open(f)
        if format == 'BGR':
            image = image.convert('RGB')
            image = np.asarray(image)
            image = image[:,:,::-1]
        if format == 'RGB':
            image = np.asarray(image.convert('RGB'))
        if format == 'L':
            image = np.asarray(image)
        return image


def convert_image_to_rgb(image, format):
    """
    Convert an image from given format to RGB.
    Args:
        image (np.ndarray or Tensor): an HWC image
        format (str): the format of input image, also see `read_image`
    Returns:
        (np.ndarray): (H,W,3) RGB image in 0-255 range, can be either float or uint8
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if format == "BGR":
        image = image[:, :, [2, 1, 0]]
    elif format == "YUV-BT.601":
        image = np.dot(image, np.array(_M_YUV2RGB).T)
        image = image * 255.0
    else:
        if format == "L":
            image = image[:, :, 0]
        image = image.astype(np.uint8)
        image = np.asarray(Image.fromarray(image, mode=format).convert("RGB"))
    return image