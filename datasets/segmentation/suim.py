import os
import glob
import PIL.Image as I


# Now support SUIM
def load_suim_sem_seg_dict(image_dir, gt_dir):
    """
        Args:
            image_dir (str): path to the raw dataset. e.g., "/root/train".
            gt_dir (str): path to the raw annotations. e.g., "/root/train_labels".
        Returns:
            list[dict]: a list of dict, each has "file_name" and
                "sem_seg_file_name".
        """

    ret = []
    for image_file in glob.glob(os.path.join(image_dir, "*.jpg")):
        prefix = image_dir

        label_file = gt_dir + image_file[len(prefix): -4] + ".png"

        assert os.path.isfile(
            label_file
        ), "Please generate labelTrainIds.png with cityscapesscripts/preparation/createTrainIdLabelImgs.py"

        with I.open(label_file) as lb:
            w, h = lb.size
            ret.append(
                {
                    'file_name': image_file,
                    'sem_seg_file_name': label_file,
                    'height': h,
                    'width': w
                }
            )
    return ret
