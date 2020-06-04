from utils.checkpointers.checkpoint import Checkopointer
from .functional import align_and_update_state_dicts
from typing import Any
import utils.comm as comm


class GenericCheckpoint(Checkopointer):
    def __init__(self, model, save_dir="", *, save_to_disk=None, **checkpointables):
        # TODO: change the is main process
        is_main_process = comm.is_main_process()
        super().__init__(
            model,
            save_dir,
            save_to_disk=is_main_process if save_to_disk is None else save_to_disk,
            **checkpointables,
        )

    def _load_model(self, checkpoint: Any) -> None:
        model_state_dict = self.model.state_dict()
        align_and_update_state_dicts(model_state_dict, checkpoint['model'])
        checkpoint['model'] = model_state_dict
        super()._load_model(checkpoint)
