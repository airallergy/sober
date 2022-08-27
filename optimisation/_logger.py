import time
import logging
from pathlib import Path
from platform import node
from functools import wraps
from collections.abc import Callable
from contextlib import ContextDecorator, AbstractContextManager
from typing import TypeVar, ParamSpec, TypeAlias, SupportsIndex

from ._typing import SubprocessRes

_P = ParamSpec("_P")
_R = TypeVar("_R")

_F: TypeAlias = Callable[_P, _R]  # type: ignore[misc] # python/mypy#11855

HOST_STEM = node().split(".")[0]


def _cwd_to_logger_name(cwd: Path) -> str:
    return str(cwd)


class _Filter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if hasattr(record, "stdout_lines"):
            stdout = "\t" + "\n\t".join(getattr(record, "stdout_lines"))
            if stdout.strip() != "":
                record.msg += ":\n" + stdout
        return super().filter(record)


class _LoggerManager(AbstractContextManager, ContextDecorator):
    _cwd_index: SupportsIndex
    _is_first: bool
    _name: str
    _log_file: Path
    _logger: logging.Logger

    def __init__(self, cwd_index: SupportsIndex, is_first: bool = False) -> None:
        self._cwd_index = cwd_index
        self._is_first = is_first

    def __call__(self, f: _F) -> _F:
        @wraps(f)
        def wrapper(*args, **kwargs) -> _R:
            cwd: Path = args[self._cwd_index]
            cwd.mkdir(parents=True, exist_ok=True)

            self._name = _cwd_to_logger_name(cwd)
            self._log_file = cwd / "console.log"
            if self._is_first:
                self._log_file.unlink(missing_ok=True)

            with self._recreate_cm():  # type: ignore[attr-defined] # NOTE: why?
                return f(*args, **kwargs)

        return wrapper

    def __enter__(self) -> "_LoggerManager":  # TODO: use typing.Self after 3.11
        # create a logger
        self.logger = logging.getLogger(self._name)
        self.logger.setLevel(logging.DEBUG)

        # create a file handler
        fh = logging.FileHandler(self._log_file, "at")
        fh.setLevel(logging.DEBUG)
        fh.addFilter(_Filter())

        # set format for the file handler
        formatter = logging.Formatter(
            f"%(asctime_)s {HOST_STEM}: %(message)s", datefmt="%c", style="%"
        )
        fh.setFormatter(formatter)

        self.logger.addHandler(fh)
        return self

    def __exit__(self, *args) -> None:
        self.logger.handlers.clear()
        logging.shutdown()


def _asctime(secs: float) -> str:
    return time.strftime("%c", time.localtime(secs))


class _SubprocessLogger(AbstractContextManager):
    _logger: logging.Logger
    _begin_time: float
    res: SubprocessRes

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def __enter__(self) -> "_SubprocessLogger":  # TODO: use typing.Self after 3.11
        self._begin_time = time.time()
        return self

    def __exit__(self, *args) -> None:
        res = self.res

        self._logger.info(
            f"running '{' '.join(str(item) for item in res.args)}'",
            extra={
                "asctime_": _asctime(self._begin_time),
                "stdout_lines": res.stdout.strip("\n").splitlines(),
            },
        )
        self._logger.info(
            f"completed with exit code {res.returncode}",
            extra={"asctime_": _asctime(time.time())},
        )


def _log(cwd: Path, msg: str = "") -> _SubprocessLogger:
    name = _cwd_to_logger_name(cwd)
    assert name in logging.Logger.manager.loggerDict, f"unmanaged logger: {name}."

    logger = logging.getLogger(name)
    if msg:
        logger.info(msg, extra={"asctime_": _asctime(time.time())})

    return _SubprocessLogger(logger)
