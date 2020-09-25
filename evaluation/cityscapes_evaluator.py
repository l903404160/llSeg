import glob
import logging
import numpy as np
import os
import tempfile
from collections import OrderedDict
import torch
from PIL import Image
from datasets.metacatalog import MetadataCatalog
import utils.comm as comm
from .evaluator import DatasetEvaluator


class CityscapesEvaluator(DatasetEvaluator):
    """
    Base Class for evaluation using cityscapes API
    """
    def __init__(self, dataset_name):
        """
        Args:
            dataset_name (str): the name of the dataset.
                It must have the following metadata associated with it:
                "thing_classes", "gt_dir".
        """
        self._metadata = MetadataCatalog.get(dataset_name)
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger("OUCWheel." + __name__)

    def reset(self):
        self._working_dir = tempfile.TemporaryDirectory(prefix="cityscapes_eval_")
        self._temp_dir = self._working_dir.name
        # All workers will write to the same results directory
        # TODO this does not work in distributed training
        self._temp_dir = comm.all_gather(self._temp_dir)[0]
        if self._temp_dir != self._working_dir.name:
            self._working_dir.cleanup()
        self._logger.info(
            "Writing cityscapes results to temporary directory {} ...".format(self._temp_dir)
        )


class CityscapesSemSegEvaluator(CityscapesEvaluator):
    """
    Evaluate semantic segmentation results on cityscapes dataset using cityscapes API.
    Note:
        * It does not work in multi-machine distributed training.
        * It contains a synchronization, therefore has to be used on all ranks.
        * Only the main process runs evaluation.
    """
    def process(self, inputs, outputs):
        inputs = [inputs]
        outputs = [outputs]

        from cityscapesscripts.helpers.labels import trainId2label

        for input, output in zip(inputs, outputs):
            file_name = input["file_name"]
            basename = os.path.splitext(os.path.basename(file_name))[0]
            pred_filename = os.path.join(self._temp_dir, basename + "_pred.png")

            output = output['sem_seg'][0].to(self._cpu_device).numpy()

            pred = 255 * np.ones(output.shape, dtype=np.uint8)
            for train_id, label in trainId2label.items():
                if label.ignoreInEval:
                    continue
                pred[output == train_id] = label.id
            Image.fromarray(pred).save(pred_filename)

    def evaluate(self):
        comm.synchronize()
        if comm.get_rank() > 0:
            return
        # Load the Cityscapes eval script *after* setting the required env var,
        # since the script reads CITYSCAPES_DATASET into global variables at load time.
        import cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling as cityscapes_eval

        self._logger.info("Evaluating results under {} ...".format(self._temp_dir))

        # set some global states in cityscapes evaluation API, before evaluating
        cityscapes_eval.args.predictionPath = os.path.abspath(self._temp_dir)
        cityscapes_eval.args.predictionWalk = None
        cityscapes_eval.args.JSONOutput = False
        cityscapes_eval.args.colorized = False

        # These lines are adopted from
        # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/evalPixelLevelSemanticLabeling.py # noqa
        gt_dir = self._metadata.label_root
        groundTruthImgList = glob.glob(os.path.join(gt_dir, "*", "*_gtFine_labelIds.png"))
        assert len(
            groundTruthImgList
        ), "Cannot find any ground truth images to use for evaluation. Searched for: {}".format(
            cityscapes_eval.args.groundTruthSearch
        )
        predictionImgList = []
        for gt in groundTruthImgList:
            predictionImgList.append(cityscapes_eval.getPrediction(cityscapes_eval.args, gt))
        results = cityscapes_eval.evaluateImgLists(
            predictionImgList, groundTruthImgList, cityscapes_eval.args
        )
        ret = OrderedDict()
        ret["sem_seg"] = {
            "IoU": 100.0 * results["averageScoreClasses"],
            "iIoU": 100.0 * results["averageScoreInstClasses"],
            "IoU_sup": 100.0 * results["averageScoreCategories"],
            "iIoU_sup": 100.0 * results["averageScoreInstCategories"],
        }
        self._working_dir.cleanup()
        return ret