import os
import logging
import yaml
from typing import Dict, Any

from yacs.config import CfgNode as _CfgNode

BASE_KEY = "_BASE_"


class CfgNode(_CfgNode):

    @staticmethod
    def load_yaml_with_base(filename: str, allow_unsafe: bool = False):
        with open(filename, 'r') as file:
            try:
                cfg = yaml.safe_load(file)
            except:
                logger = logging.getLogger(__name__)
                logger.warning(
                    "Loading config {} with yaml.unsafe_load. Your machine may "
                    "be at risk if the file contains malicious content.".format(
                        filename
                    )
                )
                cfg = yaml.unsafe_load(file)

        def merge_a_into_b(a: Dict[Any, Any], b: Dict[Any, Any]) -> None:
            # merge dict a into dict b. values in a will overwrite b.
            for k, v in a.items():
                if isinstance(v, dict) and k in b:
                    assert isinstance( b[k], dict ), "Cannot inhert key '{}' from base !".format(k)
                    merge_a_into_b(v, b[k])
                else:
                    b[k] = v

        if BASE_KEY in cfg:
            base_cfg_file = cfg[BASE_KEY]
            if base_cfg_file.startswith("~"):
                base_cfg_file = os.path.expanduser(base_cfg_file)
            if not any(
                map(base_cfg_file.startswith, ["/", "https://", "http://"])
            ):
                # the path to base cfg is relative to the config file itself
                base_cfg_file = os.path.join(os.path.dirname(filename), base_cfg_file)
            base_cfg = CfgNode.load_yaml_with_base(
                base_cfg_file, allow_unsafe=allow_unsafe
            )
            del cfg[BASE_KEY]
            merge_a_into_b(cfg, base_cfg)
            return base_cfg
        return cfg

    def merge_from_file(self, cfg_filename: str, allow_unsave: bool=False) -> None:
        loaded_cfg = CfgNode.load_yaml_with_base(cfg_filename, allow_unsafe=allow_unsave)
        loaded_cfg = type(self)(loaded_cfg)
        self.merge_from_other_cfg(loaded_cfg)

    def merge_from_other_cfg(self, cfg_other):
        assert (
            BASE_KEY not in cfg_other
        ), "The reserved key '{}' can only be used in files!".format(BASE_KEY)
        return super(CfgNode, self).merge_from_other_cfg(cfg_other)

    def merge_from_list(self, cfg_list):
        keys = set(cfg_list[0::2])
        assert (
            BASE_KEY not in keys
        ), "The reserved key '{}' can obly be used in files!".format(BASE_KEY)
        return super(CfgNode, self).merge_from_list(cfg_list)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("COMPUTED_"):
            if name in self:
                old_val = self[name]
                if old_val == value:
                    return
                raise KeyError(
                    "Computed attributed '{}' alread exists"
                    "with a different value! old={}, net={}".format(
                        name, old_val, value
                    )
                )
            self[name] = value
        else:
            super(CfgNode, self).__setattr__(name=name, value=value)


    def dump(self, **kwargs):
        return super(CfgNode, self).dump()