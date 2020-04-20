# from fvcore
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Dict, Optional


class Registry(object):
    """
    The registry that provides name -> object mapping, to support third-party
    users' custom modules.
    To create a registry (e.g. a backbone registry):
    .. code-block:: python
        BACKBONE_REGISTRY = Registry('BACKBONE')
    To register an object:
    .. code-block:: python
        @BACKBONE_REGISTRY.register()
        class MyBackbone():
            ...
    Or:
    .. code-block:: python
        BACKBONE_REGISTRY.register(MyBackbone)
    """
    def __init__(self, name: str) -> None:
        """
        :param name: The name of the Registry
        """
        self._name = name
        self._obj_map: Dict[str, object] = {}

    def _do_registry(self, name: str, obj: object) -> None:
        assert (
            name not in self._obj_map
        ), "An object named '{}' was already registered in '{}' registry!".format(
            name, self._name
        )
        self._obj_map[name] = obj

    def register(self, obj: object=None) -> Optional[object]:
        """
        Register the given object under the name 'obj.__name__'.
        Can be used as eigher a decorator or not. See dosctring of this class for usage
        :param obj:
        :return:
        """
        if obj is None:
            def deco(func_or_class: object) -> object:
                name = func_or_class.__name__ # pyre-ignore
                self._do_registry(name, func_or_class)
                return func_or_class
            return deco

        # used as a function call
        name = obj.__name__
        self._do_registry(name, obj)

    def get(self, name: str) -> object:
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError(
                "No object names '{}' found in '{}' Registry".format(
                    name, self._name
                )
            )
        return ret

    def __contains__(self, name: str) -> bool:
        return name in self._obj_map
