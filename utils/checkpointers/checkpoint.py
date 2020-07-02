# from fvcore
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import copy
import torch
import logging
import torch.nn as nn
import numpy as np

from typing import Dict, Any, List, Optional
from torch.nn.parallel import DataParallel, DistributedDataParallel
from .functional import get_unexpected_parameters_message, get_missing_parameters_message, _strip_prefix_if_present, _filter_reused_missing_keys

__all__ = ["Checkpointer", "PeriodicCheckpointer"]


class Checkpointer(object):
    """
    A checkpointer that can save or load model as well as extra checkpointable objects.
    """
    def __init__(self, model: nn.Module, save_dir: str="", *,
                 save_to_disk: bool=True, **checkpointables: object) -> None:
        """
        :param model: model
        :param save_dir: a directory to save and find checkpoints
        :param save_to_dist: if True save checkpoint to disk , otherwise disable saving for this checkpoint
        :param checkpointables: any checkpointable objects i.e, objects that have the 'state_dict()' method.
                For example, it can be used like 'Checkpoint(model, "dir", optimizer=optimizer)'.
        """
        if isinstance(model, (DistributedDataParallel, DataParallel)):
            model = model.module
        self.model = model
        self.checkpointables = copy.copy(checkpointables)
        self.logger = logging.getLogger('OUCWheel.'+__name__) # pyre-ignore
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk

    def save(self, name: str, **kwargs: Dict[str, str]) -> None:
        """
        Dump model and checkpointables to a file
        :param name:
        :param kwargs:
        :return:
        """
        if not self.save_dir or not self.save_to_disk:
            return

        data = {}
        data['model'] = self.model.state_dict()
        for key, obj in self.checkpointables.items():
            data[key] = obj.state_dict()
        data.update(kwargs)

        basename = "{}.pth".format(name)
        save_file = os.path.join(self.save_dir, basename)
        assert os.path.basename(save_file) == basename, basename
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)
        self.tag_last_checkpoint(basename)

    def load(self, path: str) -> object:
        """
        Load from the given checkpoint, When path points to network file, this
        function has to be called on all ranks.
        :param path:
        :return:
        """
        if not path:
            # no checkpoint provided
            self.logger.info(
                "No checkpoint dound. Initializing model from scratch---"
            )
            return {}
        self.logger.info("Loading checkpoing from {}".format(path))
        if not os.path.isfile(path):
            self.logger.info("Checkpoint file {} not found.".format(path))

        checkpoint = self._load_file(path)
        self._load_model(checkpoint)

        for key, obj in self.checkpointables.items():
            if key in checkpoint:
                self.logger.info("Loading {} from {}".format(key, path))
                obj.load_state_dict(checkpoint.pop(key))

        return checkpoint

    def has_checkpoint(self) -> bool:
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return os.path.exists(save_file)

    def get_checkpoint_file(self) -> str:
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with open(save_file, 'r') as f:
                last_saved = f.read().strip()
        except IOError:
            return ""
        return os.path.join(self.save_dir, last_saved)

    def _convert_ndarray_to_tensor(self, state_dict: Dict[str, Any]) -> None:
        """
        In-place convert all numpy arrays in the state_dict to torch tensor.
        Args:
            state_dict (dict): a state-dict to be loaded to the model.
                Will be modified.
        """
        # model could be an OrderedDict with _metadata attribute
        # (as returned by Pytorch's state_dict()). We should preserve these
        # properties.
        for k in list(state_dict.keys()):
            v = state_dict[k]
            if not isinstance(v, np.ndarray) and not isinstance(v, torch.Tensor):
                raise ValueError(
                    "Unsupported type found in checkpoint! {}: {}".format(k, type(v))
                )
            if not isinstance(v, torch.Tensor):
                state_dict[k] = torch.from_numpy(v)

    def get_all_checkpoint_files(self) -> List[str]:
        all_model_checkpoints = [
            os.pardir.join(self.save_dir, file)
            for file in os.listdir(self.save_dir)
            if os.path.isfile(os.path.join(self.save_dir, file))
            and file.endswith(".pth")
        ]
        return all_model_checkpoints

    def resume_or_load(self, path: str, *, resume: bool=True) -> object:
        if resume and self.has_checkpoint():
            path = self.get_checkpoint_file()
        return self.load(path)

    def tag_last_checkpoint(self, last_filename_basename: str) -> None:
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with open(save_file, 'w') as f:
            f.write(last_filename_basename)

    def _load_file(self, f: str) -> object:
        return torch.load(f, map_location=torch.device('cpu'))

    def _load_model(self, checkpoint: Any) -> None:
        """
        Load weights from a checkpoint
        :param checkpoint:
        :return:
        """
        checkpoint_state_dict = checkpoint.pop("model")
        # if the state_dict comes from a model that was wrapped in a
        # DataParallel or DistributedDataParallel during serialization,
        # remove the "module" prefix before performing the matching.
        _strip_prefix_if_present(checkpoint_state_dict, 'module.')

        model_state_dict = self.model.state_dict()
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                shape_model = tuple(model_state_dict[k].shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    self.logger.warning(
                        "'{}' has shape {} in the checkpoint but {} in the model!, Skipped.".format(
                            k, shape_checkpoint, shape_model
                        )
                    )
                    checkpoint_state_dict.pop(k)
        incompatible = self.model.load_state_dict(
            checkpoint_state_dict, strict=False
        )
        if incompatible.missing_keys:
            missing_keys = _filter_reused_missing_keys(
                self.model, incompatible.missing_keys
            )
            if missing_keys:
                self.logger.info(get_missing_parameters_message(missing_keys))

        if incompatible.unexpected_keys:
            self.logger.info(
                get_unexpected_parameters_message(incompatible.unexpected_keys)
            )


class PeriodicCheckpointer:
    """
    Save checkpointer periodically. When '.step(iteration)' is called, it will
    execute 'checkpoint.save' on the given checkpointer, if iteration is a multiple
    of period or if 'max_iter' is reached
    """
    def __init__(self, checkpointer: Any, period: int,
                 max_iter: Optional[int] = None,
                 max_to_keep: Optional[int] = None) -> None:
        """
        :param checkpointer: the checkpoint object used to save checkpoints
        :param period: the period to save checkpoint
        :param max_iter: maximum number of iterations. when it reached, a checkpoint named "model_final" will be saved
        :param max_to_keep: maximum number of most current checkpoints to keep previous checkpoints will be deleted
        """
        self.checkpointer = checkpointer
        self.period = period
        self.max_iter = max_iter
        if max_to_keep is not None:
            assert max_to_keep > 0
        self.max_to_keep = max_to_keep
        self.recent_checkpoints = []

    def step(self, iteration: int, **kwargs: Any) -> None:
        """
        Perform the appropriate action at the given iteration
        :param iteration:
        :param kwargs:
        :return:
        """
        iteration = int(iteration)
        additional_state = {"iteration": iteration}
        additional_state.update(kwargs)
        if (iteration + 1) % self.period == 0:
            self.checkpointer.save(
                "model_{:07d}".format(iteration), **additional_state
            )

            if self.max_to_keep is not None:
                self.recent_checkpoints.append(
                    self.checkpointer.get_checkpoint_file()
                )
                if len(self.recent_checkpoints) > self.max_to_keep:
                    file_to_delete = self.recent_checkpoints.pop(0)
                    if os.path.exists(file_to_delete) and not file_to_delete.endswith("model_final.pth"):
                        os.remove(file_to_delete)
        if iteration >= self.max_iter - 1:
            self.checkpointer.save("model_final", **additional_state)

    def save(self, name: str, **kwargs: Any) -> None:
        self.checkpointer.save(name, **kwargs)


