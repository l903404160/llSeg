import os
import sys
import time
import logging
import functools
from tabulate import tabulate
from termcolor import colored
from collections import Counter

class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log


@functools.lru_cache()
def setup_logger(output=None, distributed_rank=0, *, color=True, name='OUCWheel', abbrev_name=None):
    """
    :param output: a file name or a directory to save log
    :param color:
    :param name: the root module name of this logger
    :param abbrev_name:  abbreviate "detectron2" to "d2"
    :return: a logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if abbrev_name is None:
        abbrev_name = "OW" if name == "OUCWheel" else name

    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
    )

    if distributed_rank == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        if color:
            formatter = _ColorfulFormatter(
                colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
                datefmt="%m/%d %H:%M:%S",
                root_name=name,
                abbrev_name=str(abbrev_name)
            )
        else:
            formatter = plain_formatter

        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if output is not None:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, "log.txt")
        if distributed_rank > 0:
            filename = filename + ".rang{}".format(distributed_rank)
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        fh = logging.StreamHandler(_cached_log_stream(filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)
    return logger


@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    return open(filename, 'a')

###  some other convenient logging methods

def _find_caller():
    """
    :return: str: module name of the caller
    tuple: a hashable key to be used to identify different callers
    """
    frame = sys._getframe(2)
    while frame:
        code = frame.f_code
        if os.path.join("utils", "logger.") not in code.co_filename:
            mod_name = frame.f_globals["__name__"]
            if mod_name == "__main__":
                mod_name = "OUCWheel"
            return mod_name, (code.co_filename, frame.f_lineno, code.co_name)
        frame = frame.f_back

_LOG_COUNTER = Counter()
_LOG_TIMER = dict()

def log_first_n(lvl, msg, n=1, *, name=None, key="caller"):
    """
        Log only for the first n times
    :param lvl:
    :param msg:
    :param n:
    :param name:
    :param key:
    :return:
    """
    if isinstance(key, str):
        key = (key, )
    assert len(key) > 0
    caller_module, caller_key = _find_caller()
    hash_key = ()
    if "caller" in key:
        hash_key = hash_key + caller_key
    if "message" in key:
        hash_key = hash_key + (msg, )

    _LOG_COUNTER[hash_key] += 1
    if _LOG_COUNTER[hash_key] <= n:
        logging.getLogger("OUCWheel."+caller_module).log(lvl, msg)


def log_every_n(lvl, msg, n=1, *, name=None):
    """
    Log once per n times.
    Args:
        lvl (int): the logging level
        msg (str):
        n (int):
        name (str): name of the logger to use. Will use the caller's module by default.
    """
    caller_module, key = _find_caller()

    _LOG_COUNTER[key] += 1
    if n == 1 or _LOG_COUNTER[key] % n == 1:
        logging.getLogger("OUCWheel."+caller_module).log(lvl, msg)

def log_every_n_seconds(lvl, msg, n=1, *, name=None):
    """
        Log no more than once per n seconds
    :param lvl: the logging level
    :param msg:
    :param n:
    :param name:  name of the logger to use. Will use the caller's module by default
    :return:
    """
    caller_module, key = _find_caller()
    last_logged = _LOG_TIMER.get(key, None)
    current_time = time.time()
    if last_logged is None or current_time - last_logged >= n:
        logging.getLogger('OUCWheel.'+caller_module).log(lvl, msg)
        _LOG_TIMER[key] = current_time

def create_small_table(small_dict):
    """
        Create a small table using the keys of small_dict as headers.
        This is only suitable for small dictionaries
    :param small_dict:
    :return: str: the table as a string
    """
    keys, values = tuple(zip(*small_dict.items()))
    table = tabulate(
        [values],
        headers=keys,
        tablefmt="pipe",
        floatfmt=".3f",
        stralign="center",
        numalign="center"
    )
    return table
