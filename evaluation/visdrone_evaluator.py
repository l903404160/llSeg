import os
import torch
import copy
import logging
import itertools
import json
from collections import OrderedDict
from structures import BoxMode
import utils.comm as comm

from utils.visdrone_metric.metric import convert_json_to_txt, evaluate_results
from .evaluator import DatasetEvaluator
from datasets.metacatalog.catalog import MetadataCatalog
from utils.logger import create_small_table


class VisDroneEvaluator(DatasetEvaluator):
    """
    Evaluate VisDrone object detection
    """
    def __init__(self, dataset_name, cfg, distributed, output_dir=None):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            cfg (CfgNode): config instance
            distributed (True): if True, will collect results from all ranks and run evaluation
                in the main process.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:
                1. "instance_predictions.pth" a file in torch serialization
                   format that contains all the raw original predictions.
                2. "coco_instances_results.json" a json file in COCO's result
                   format.
                3. "./result" a folder which contains the whole prediction '.txt' file for further evaluation
        """
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger("OUCWheel."+__name__)
        self._metadata = MetadataCatalog.get(dataset_name)

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            # TODO this is ugly
            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances_to_coco_json(instances, input["image_id"])
            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._cpu_device)
            self._predictions.append(prediction)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions
        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            os.makedirs(self._output_dir, exist_ok=True)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with open(file_path, "wb") as f:
                torch.save(predictions, f)

        self._logger.info("Preparing results for COCO format ...")
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))
        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
            for result in coco_results:
                category_id = result["category_id"]
                assert (
                    category_id in reverse_id_mapping
                ), "A prediction has category_id={}, which is not available in the dataset.".format(
                    category_id
                )
                result["category_id"] = reverse_id_mapping[category_id]
        # save the prediction to json file
        file_path = os.path.join(self._output_dir, "coco_instances_results.json")
        self._logger.info("Saving results to {}".format(file_path))
        with open(file_path, "w") as f:
            f.write(json.dumps(coco_results))
            f.flush()

        self._results = OrderedDict()

        # evaluation logic
        txt_output_dir = os.path.join(self._output_dir, 'result')

        convert_json_to_txt(json_file=file_path, output_dir=txt_output_dir)
        target_dir = '/home/haida_sunxin/lqx/data/DronesDET/val/annotations'
        ap_results = evaluate_results(pred_dir=txt_output_dir, target_dir=target_dir)
        self._logger.info(
            "Evaluation results for {}: \n".format("bbox") + create_small_table(ap_results)
        )
        self._results['bbox'] = ap_results
        return copy.deepcopy(self._results)


def instances_to_coco_json(instances, img_id):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.
    Args:
        instances (Instances):
        img_id (int): the image id
    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()

    results = []
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "category_id": classes[k],
            "bbox": boxes[k],
            "score": scores[k],
        }
        results.append(result)
    return results
