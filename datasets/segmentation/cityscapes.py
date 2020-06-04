import os
import glob
import PIL.Image as I


def load_cityscapes_sem_seg_dict(image_dir, gt_dir):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/cityscapes/leftImg8bit/train".
        gt_dir (str): path to the raw annotations. e.g., "~/cityscapes/gtFine/train".
    Returns:
        list[dict]: a list of dict, each has "file_name" and
            "sem_seg_file_name".
    """
    ret = []
    for image_file in glob.glob(os.path.join(image_dir, "**/*.png")):
        suffix = "leftImg8bit.png"
        assert image_file.endswith(suffix)
        prefix = image_dir

        label_file = gt_dir + image_file[len(prefix): -len(suffix)] + "gtFine_labelTrainIds.png"
        assert os.path.isfile(
            label_file
        ), "Please generate labelTrainIds.png with cityscapesscripts/preparation/createTrainIdLabelImgs.py"

        w, h = I.open(label_file).size
        ret.append(
            {
                'file_name': image_file,
                'sem_seg_file_name': label_file,
                'height': h,
                'width': w
            }
        )
    return ret

