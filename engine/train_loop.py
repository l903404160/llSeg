import logging
import numpy as np
import time
import weakref
import torch
import torch.nn as nn

import utils.comm as comm
from utils.events import EventStorage

__all__ = ["HookBase", "TrainerBase", "SimpleTrainer"]

class HookBase:
    """
    Base class for hooks that can be registered with :class:'TrainerBase'

    Each hook can implement 4 methods. The way they are called is demonstrated of the following snippet:

    .. code-block:: python
        hook.before_train()
        for iter in range(start_iter, max_iter):
            hook.before_step()
            trainer.run_step()
            hook.after_step()
        hook.after_train()

    REF: https://github.com/facebookresearch/detectron2/blob/master/detectron2/engine/train_loop.py
    """
    def before_train(self):
        """
        Called before the first iteration
        :return:
        """
        pass

    def after_train(self):
        """
        Called after the last iteration
        :return:
        """
        pass

    def before_step(self):
        """
        Called before each iteration
        :return:
        """
        pass

    def after_step(self):
        """
        Called after each iteration
        :return:
        """
        pass

class TrainerBase:
    """
    Base class for iterative trainer with books
    """
    def __init__(self):
        self._hooks = []

    def register_hooks(self, hooks):
        """
        Register hooks to the trainer. The hooks are executed in the order they are registered
        :param hooks: list[Optional[HookBase]] : list of hooks
        :return:
        """
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookBase)
            h.trainer = weakref.proxy(self)
        self._hooks.extend(hooks)

    def train(self, start_iter: int, max_iter: int):
        """
        :param start_iter:
        :param max_iter:
        :return:
        """
        logger = logging.getLogger('OUCWheel.'+__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()
            except Exception:
                logger.exception(" Exception during training: ")
                raise

            finally:
                self.after_train()

    def before_train(self):
        for h in self._hooks:
            h.before_train()

    def after_train(self):
        for h in self._hooks:
            h.after_train()

    def before_step(self):
        for h in self._hooks:
            h.before_step()

    def after_step(self):
        for h in self._hooks:
            h.after_step()
        self.storage.step()

    def run_step(self):
        raise NotImplementedError


class SimpleTrainer(TrainerBase):
    """
    A simple trainer for the most common type of task:
    single-cost single-optimizer single-data-source iterative optimization.
    It assumes that every step, you:

    1. Compute the loss with a data from the data_loader.
    2. Compute the gradients with the above loss.
    3. Update the model with the optimizer.

    If you want to do anything fancier than this,
    either subclass TrainerBase and implement your own `run_step`,
    or write your own training loop.
    """
    def __init__(self, model: nn.Module, data_loader, optimizer):
        """

        :param model:  a torch Module
        :param data_loader:  an iterable
        :param optimizer:  a torch optimizer
        """
        super(SimpleTrainer, self).__init__()
        """
        We set the model to training mode in the trainer.
        However it's valid to train a model that's in eval mode.
        If you want your model (or a submodule of it) to behave
        like evaluation during training, you can overwrite its train() method.
        """
        model.train()
        self.model = model
        self.data_loader = data_loader
        self._data_loader_iter = iter(data_loader)
        self.optimizer = optimizer

    def run_step(self):
        """
        Implement the standard training logic described above
        :return:
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval model!"
        start = time.perf_counter()

        """
            If you want to do something with the data, you can wrap the dataloader
        """
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        """
            If you want to do something with the losses, you can wrap the model
        """

        loss_dict = self.model(data)
        losses = sum(loss_dict.values())
        self._detect_anomaly(losses, loss_dict)

        metrics_dict = loss_dict
        metrics_dict["data_time"] = data_time

        self._write_metrics(metrics_dict)

        """
            If you need to accumulate gradients or something similar, you can 
            wrap the optimizer with your custom 'zero_grad()' method
        """
        self.optimizer.zero_grad()
        losses.backward()

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method.
        """
        self.optimizer.step()

    def _detect_anomaly(self, losses, loss_dict):
        if not torch.isfinite(losses).all():
            raise FloatingPointError(
                "Loss became infinite or NaN at iteration={}!\nloss_dict = {}".format(
                    self.iter, loss_dict
                )
            )

    def _write_metrics(self, metrics_dict: dict):
        """
        Args:
            metrics_dict (dict): dict of scalar metrics
        """
        metrics_dict = {
            k:v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)
        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
            }

            total_losses_reduced = sum(loss for loss in metrics_dict.values())
            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)