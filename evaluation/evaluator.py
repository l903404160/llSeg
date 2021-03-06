import datetime
import logging
import time
from collections import OrderedDict
from contextlib import contextmanager
import torch
import PIL.Image as Image
import numpy as np
from torch.nn.functional import interpolate

from utils.comm import get_world_size, is_main_process
from utils.logger import log_every_n_seconds


class DatasetEvaluator:
    """
    Base lass for a dataset evaluator

    The function :func: 'inference_on_dataset' runs the model over all samples
    in the dataset, and have a DatasetEvaluator to process the inputs/outputs

    This class will accumulate information of the inouts/outputs (by :meth:`process`),
    and produce evaluation results in the end (by:meth:`evaluate`).

    """
    def reset(self):
        """
        Preparation for a new round of evaluation
        should be called before starting a round of evaluation
        :return:
        """
        pass

    def process(self, inputs, outputs):
        """
        Process an input/output pair
        Args:
            input: the input that's used to call the model.
            otput: the return value of `model(input)`
        :param input:
        :param output:
        :return:
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.
        :return:
            dict:
                A new evaluator class can return a dict of arbitary format
                as long as the user cam [rocess the results
                In out train_net.py, we expect the following format:
                * key: the name of the task (e.g. bbox)
                * value: adict of {metric name: score}, e.g. {"AP50" : 80}
        """
        pass


class DatasetEvaluators(DatasetEvaluator):
    def __init__(self, evaluators):
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, inputs, outputs):
        for evaluator in self._evaluators:
            evaluator.process(inputs, outputs)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if is_main_process() and result is not None:
                for k,v in result.items():
                    assert (
                        k not in results
                    ), "Different evaluators produce results with the same key {}".format(k)

                    results[k] = v
        return results


def inference_on_dataset_with_multi_scale(model, data_loader, evaluator, scales=None):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.forward` accurately
    The model will be used in eval mode.
    Eache item will be scaled first and then compute the final results.

    :param model: the model will be temporarily set to 'eval' model
    :param data_loader: an iterable object with a length
    :param evaluator: the evaluator to run
    :param scales: [0.5, 0.75 ...]
    :return: the return value of 'evaluator.evaluate()'
    """


    if scales is None:
        scales = [1.]
    num_devices = get_world_size()
    logger = logging.getLogger("OUCWheel."+__name__)

    logger.info("Start Multi Scale inference on {} images".format(len(data_loader)))
    total = len(data_loader)
    if evaluator is None:
        evaluator = DatasetEvaluators([])
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            # TODO May need change the inference interface
            # TODO Make multi scale inputs
            h, w = inputs['height'], inputs['width']
            outputs = []
            for scale in scales:
                # Norm
                inputs['scale'] = scale
                inputs['flip'] = False
                norm_out = model(inputs)
                # Flip
                # inputs['flip'] = True
                # flip_out = model(inputs)
                # flip_out['seg_seg'] = torch.flip(flip_out['sem_seg'], dims=(3,))
                # out = 0.5 * (norm_out['sem_seg'] + flip_out['sem_seg'])
                out = norm_out['sem_seg']
                outputs.append(interpolate(out, size=(h, w), mode='bilinear', align_corners=True))
            outputs = torch.cat(outputs, dim=0).mean(dim=0, keepdim=True)
            outputs = torch.max(outputs, dim=1)[1]
            outputs = {'sem_seg': outputs}
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            evaluator.process(inputs, outputs)

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Inference done {}/{}. {:.4f} s / img ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )

    # Mesure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time :{} ({:.6f}s /image per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    results = evaluator.evaluate()
    # An evaluator may return None when not in main process
    # Replace it by an empty dict insted to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


# inference on Segmentation
def inference_on_dataset(model, data_loader, evaluator):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.forward` accurately
    The model will be used in eval mode.

    :param model: the model will be temporarily set to 'eval' model
    :param data_loader: an iterable object with a length
    :param evaluator: the evaluator to run
    :return: the return value of 'evaluator.evaluate()'
    """
    num_devices = get_world_size()
    logger = logging.getLogger("OUCWheel."+__name__)
    logger.info("Start inference on {} images".format(len(data_loader)))

    total = len(data_loader)
    if evaluator is None:
        evaluator = DatasetEvaluators([])
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            # TODO May need change the inference interface
            outputs = model(inputs)
            outputs['sem_seg'] = torch.max(outputs['sem_seg'], dim=1)[1]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            evaluator.process(inputs, outputs)

            iters_after_start = idx + 1 - num_warmup * int(idx >=num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Inference done {}/{}. {:.4f} s / img ETA={}".format(
                        idx+1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )

    # Mesure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time :{} ({:.6f}s /image per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    results = evaluator.evaluate()
    # An evaluator may return None when not in main process
    # Replace it by an empty dict insted to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


# Inference on Detection
def inference_on_detection_dataset(model, data_loader, evaluator):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.forward` accurately.
    The model will be used in eval mode.
    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use `None` if you only want
            to benchmark, but don't want to do any evaluation.
    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger("OUCWheel."+__name__)
    logger.info("Start inference on {} images".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            evaluator.process(inputs, outputs)

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode
    and restored to previous mode afterwards
    :param model:
    :return:
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)