import logging
from termcolor import colored
import torch.distributed as dist
import os
import atexit
import functools
import sys
import time
from collections import Counter
from iopath.common.file_io import PathManager as PathManagerBase

PathManager = PathManagerBase()
logger_initialized = {}

VERBOSE_LOG_FORMAT = ('%(asctime)s | %(levelname)s | pid-%(process)d | '
                      '%(filename)s:<%(funcName)s>:%(lineno)d | %(message)s')
BRIEF_LOG_FORMAT = '%(asctime)s | %(levelname)s | %(message)s'

LEVEL_DICT = dict(
    NOTSET=logging.NOTSET,  # 0
    DEBUG=logging.DEBUG,  # 10
    INFO=logging.INFO,  # 20
    WARNING=logging.WARNING,  # 30
    ERROR=logging.ERROR,  # 40
    CRITICAL=logging.CRITICAL,  # 50
)


class _ColorfulFormatter(logging.Formatter):
    """
    detectron2: detectron2/utils/logger.py
    """

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
        elif record.levelno == logging.INFO:
            prefix = colored("INFO", "green", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log


VERBOSE_LEVELS = [logging.DEBUG, logging.WARNING, logging.ERROR, logging.CRITICAL]


@functools.lru_cache()  # so that calling setup_logger multiple times won't add many handlers
def get_logger(
        output=None, distributed_rank=0, *, color=True, name="cvalgorithms", abbrev_name=None
):
    """
    Initialize the detectron2 logger and set its verbosity level to "DEBUG".

    Args:
        output (str): a file name or a directory to save log. If None, will not save log file.
            If ends with ".txt" or ".log", assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.
        name (str): the root module name of this logger
        abbrev_name (str): an abbreviation of the module, to avoid long names in logs.
            Set to "" to not log the root module in logs.
            By default, will abbreviate "detectron2" to "d2" and leave other
            modules unchanged.

    Returns:
        logging.Logger: a logger
    """
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if abbrev_name is None:
        abbrev_name = "d2" if name == "cvalgorithms" else name

    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
    )
    if distributed_rank == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        if color:
            formatter = _ColorfulFormatter(
                colored("[%(asctime)s %(filename)s]: ", "green") + "%(message)s",
                datefmt="%m/%d %H:%M:%S",
                root_name=name,
                abbrev_name=str(abbrev_name),
            )
        else:
            formatter = plain_formatter
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # file logging: all workers
    if output is not None:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, "log.txt")

        if distributed_rank > 0:
            filename = filename + ".rank{}".format(distributed_rank)

        PathManager.mkdirs(os.path.dirname(filename))

        fh = logging.StreamHandler(_cached_log_stream(filename))
        # set log level
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)

    logger_initialized[name] = True

    return logger


def print_log(msg, logger=None, level=logging.INFO):
    """Print a log message.

    Parameters
    ----------
    msg : str
        The message to be logged.
    logger : {logging.Logger, str}, optional
        The logger to be used.
        Some special loggers are:
            - "silent": no message will be printed.
            - other str: the logger obtained with `get_root_logger(logger)`.
            - None: The `print()` method will be used to print log messages.
    level : int
        Logging level. Only available when `logger` is a Logger object or "root".
    """
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger == 'silent':
        pass
    elif isinstance(logger, str):
        _logger = get_logger(logger)
        _logger.log(level, msg)
    else:
        raise TypeError(
            'logger should be either a logging.Logger object, str, '
            f'"silent" or None, but got {type(logger)}')


# cache the opened file object, so that different calls to `setup_logger`
# with the same file name can safely write to the same file.
@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    # use 1K buffer if writing to cloud storage
    io = PathManager.open(filename, "a", buffering=1024 if "://" in filename else -1)
    atexit.register(io.close)
    return io


def get_root_logger(log_file=None, log_level=logging.INFO):
    logger = get_logger(log_file)

    return logger


_LOG_COUNTER = Counter()
_LOG_TIMER = {}


def _find_caller():
    """
    Returns:
        str: module name of the caller
        tuple: a hashable key to be used to identify different callers
    """
    frame = sys._getframe(2)
    while frame:
        code = frame.f_code
        if os.path.join("utils", "logger.") not in code.co_filename:
            mod_name = frame.f_globals["__name__"]
            if mod_name == "__main__":
                mod_name = "detectron2"
            return mod_name, (code.co_filename, frame.f_lineno, code.co_name)
        frame = frame.f_back


def log_every_n_seconds(lvl, msg, n=1, *, name=None):
    """
    Log no more than once per n seconds.

    Args:
        lvl (int): the logging level
        msg (str):
        n (int):
        name (str): name of the logger to use. Will use the caller's module by default.
    """
    caller_module, key = _find_caller()
    last_logged = _LOG_TIMER.get(key, None)
    current_time = time.time()
    if last_logged is None or current_time - last_logged >= n:
        logging.getLogger(name or caller_module).log(lvl, msg)
        _LOG_TIMER[key] = current_time
