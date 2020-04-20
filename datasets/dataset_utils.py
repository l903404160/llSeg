import numpy as np
import PIL.Image as I


def read_image(file_name, format=None):
    with open(file_name, 'rb') as f:
        image = I.open(f)
        if format == 'BGR':
            image = image.convert('RGB')
            image = np.asarray(image)
            image = image[:,:,::-1]
        if format == 'RGB':
            image = np.asarray(image.convert('RGB'))
        if format == 'L':
            image = np.asarray(image)
        return image
