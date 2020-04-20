import torch
import logging
import torch.nn as nn

from termcolor import colored
from collections import defaultdict
from typing import Dict, Any, Iterable, List, Tuple


def get_missing_parameters_message(keys: List[str]) -> str:
    """
    Get a logging-friendly message to report parameter names (keys) that are in
    the model but not found in a checkpoint.
    Args:
        keys (list[str]): List of keys that were not found in the checkpoint.
    Returns:
        str: message.
    """
    groups = _group_checkpoint_keys(keys)
    msg = "Some model parameters are not in the checkpoint:\n"
    msg += "\n".join(
        "  " + colored(k + _group_to_str(v), "blue") for k, v in groups.items()
    )
    return msg


def get_unexpected_parameters_message(keys: List[str]) -> str:
    """
    Get a logging-friendly message to report parameter names (keys) that are in
    the checkpoint but not found in the model.
    Args:
        keys (list[str]): List of keys that were not found in the model.
    Returns:
        str: message.
    """
    groups = _group_checkpoint_keys(keys)
    msg = "The checkpoint contains parameters not used by the model:\n"
    msg += "\n".join(
        "  " + colored(k + _group_to_str(v), "magenta")
        for k, v in groups.items()
    )
    return msg


def _strip_prefix_if_present(state_dict: Dict[str, Any], prefix: str) -> None:
    keys = sorted(state_dict.keys())
    if not all(len(keys) == 0 or key.startswith(prefix) for key in keys):
        return
    for key in keys:
        newkey = key[len(prefix):]
        state_dict[newkey] = state_dict.pop(key)

    try:
        metadata = state_dict._metadata
    except AttributeError:
        pass
    else:
        for key in list(metadata.keys()):
            if len(key) == 0:
                continue
            newkey = key[len(prefix):]
            metadata[newkey] = metadata.pop(key)


def _group_checkpoint_keys(keys: List[str]) -> Dict[str, List[str]]:
    """
    Group keys based on common prefixes. A prefix is the string up to the final
    "." in each key.
    Args:
        keys (list[str]): list of parameter names, i.e. keys in the model
            checkpoint dict.
    Returns:
        dict[list]: keys with common prefixes are grouped into lists.
    """
    groups = defaultdict(list)
    for key in keys:
        pos = key.rfind(".")
        if pos >= 0:
            head, tail = key[:pos], [key[pos + 1 :]]
        else:
            head, tail = key, []
        groups[head].extend(tail)
    return groups


def _group_to_str(group: List[str]) -> str:
    """
    Format a group of parameter name suffixes into a loggable string.
    Args:
        group (list[str]): list of parameter name suffixes.
    Returns:
        str: formated string.
    """
    if len(group) == 0:
        return ""

    if len(group) == 1:
        return "." + group[0]

    return ".{" + ", ".join(group) + "}"


def _filter_reused_missing_keys(model: nn.Module, keys: List[str]) -> List[str]:
    """
    Filter "missing keys" to not include keys that have been loaded with another name.
    :param model:
    :param keys:
    :return:
    """
    keyset = set(keys)
    param_to_names = defaultdict(set) # param -> names that points to it
    for module_prefix, module in _named_modules_with_dup(model):
        for name, param in list(module.named_parameters(recurse=False)) + list(
            module.named_buffers(recurse=False)
        ):
            full_name = (module_prefix + '.' if module_prefix else "") + name
            param_to_names[param].add(full_name)
    for names in param_to_names.values():
        if any(n in keyset for n in names) and not all(n in keyset for n in names):
            [keyset.remove(n) for n in names if n in keyset]

    return list(keyset)


def _named_modules_with_dup(
    model: nn.Module, prefix: str = ""
) -> Iterable[Tuple[str, nn.Module]]:
    """
    The same as `model.named_modules()`, except that it includes
    duplicated modules that have more than one name.
    """
    yield prefix, model
    for name, module in model._modules.items():  # pyre-ignore
        if module is None:
            continue
        submodule_prefix = prefix + ("." if prefix else "") + name
        yield from _named_modules_with_dup(module, submodule_prefix)


def align_and_update_state_dicts(model_state_dict, ckpt_state_dict):
    """
    Match names between the two state-dict, and update the values of model_state_dict in-place with
    copies of the matched tensor in ckpt_state_dict.
    If `c2_conversion==True`, `ckpt_state_dict` is assumed to be a Caffe2
    model and will be renamed at first.
    Strategy: suppose that the models that we will create will have prefixes appended
    to each of its keys, for example due to an extra level of nesting that the original
    pre-trained weights from ImageNet won't contain. For example, model.state_dict()
    might return backbone[0].body.res2.conv1.weight, while the pre-trained model contains
    res2.conv1.weight. We thus want to match both parameters together.
    For that, we look for each model weight, look among all loaded keys if there is one
    that is a suffix of the current weight name, and use it if that's the case.
    If multiple matches exist, take the one with longest size
    of the corresponding name. For example, for the same model as before, the pretrained
    weight file can contain both res2.conv1.weight, as well as conv1.weight. In this case,
    we want to match backbone[0].body.conv1.weight to conv1.weight, and
    backbone[0].body.res2.conv1.weight to res2.conv1.weight.
    """
    model_keys = sorted(model_state_dict.keys())
    original_keys = {x: x for x in ckpt_state_dict.keys()}
    ckpt_keys = sorted(ckpt_state_dict.keys())

    def match(a, b):
        # Matched ckpt_key should be a complete (starts with '.') suffix.
        # For example, roi_heads.mesh_head.whatever_conv1 does not match conv1,
        # but matches whatever_conv1 or mesh_head.whatever_conv1.
        return a == b or a.endswith("." + b)

    # get a matrix of string matches, where each (i, j) entry correspond to the size of the
    # ckpt_key string, if it matches
    match_matrix = [len(j) if match(i, j) else 0 for i in model_keys for j in ckpt_keys]
    match_matrix = torch.as_tensor(match_matrix).view(len(model_keys), len(ckpt_keys))
    # use the matched one with longest size in case of multiple matches
    max_match_size, idxs = match_matrix.max(1)
    # remove indices that correspond to no-match
    idxs[max_match_size == 0] = -1

    # used for logging
    max_len_model = max(len(key) for key in model_keys) if model_keys else 1
    max_len_ckpt = max(len(key) for key in ckpt_keys) if ckpt_keys else 1
    log_str_template = "{: <{}} loaded from {: <{}} of shape {}"
    logger = logging.getLogger('OUCWheel.'+__name__)
    # matched_pairs (matched checkpoint key --> matched model key)
    matched_keys = {}
    for idx_model, idx_ckpt in enumerate(idxs.tolist()):
        if idx_ckpt == -1:
            continue
        key_model = model_keys[idx_model]
        key_ckpt = ckpt_keys[idx_ckpt]
        value_ckpt = ckpt_state_dict[key_ckpt]
        shape_in_model = model_state_dict[key_model].shape

        if shape_in_model != value_ckpt.shape:
            logger.warning(
                "Shape of {} in checkpoint is {}, while shape of {} in model is {}.".format(
                    key_ckpt, value_ckpt.shape, key_model, shape_in_model
                )
            )
            logger.warning(
                "{} will not be loaded. Please double check and see if this is desired.".format(
                    key_ckpt
                )
            )
            continue

        model_state_dict[key_model] = value_ckpt.clone()
        if key_ckpt in matched_keys:  # already added to matched_keys
            logger.error(
                "Ambiguity found for {} in checkpoint!"
                "It matches at least two keys in the model ({} and {}).".format(
                    key_ckpt, key_model, matched_keys[key_ckpt]
                )
            )
            raise ValueError("Cannot match one checkpoint key to multiple keys in the model.")

        matched_keys[key_ckpt] = key_model
        logger.info(
            log_str_template.format(
                key_model,
                max_len_model,
                original_keys[key_ckpt],
                max_len_ckpt,
                tuple(shape_in_model),
            )
        )
    matched_model_keys = matched_keys.values()
    matched_ckpt_keys = matched_keys.keys()
    # print warnings about unmatched keys on both side
    unmatched_model_keys = [k for k in model_keys if k not in matched_model_keys]
    if len(unmatched_model_keys):
        logger.info(get_missing_parameters_message(unmatched_model_keys))

    unmatched_ckpt_keys = [k for k in ckpt_keys if k not in matched_ckpt_keys]
    if len(unmatched_ckpt_keys):
        logger.info(
            get_unexpected_parameters_message(original_keys[x] for x in unmatched_ckpt_keys)
        )