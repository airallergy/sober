import time
import logging
from pathlib import Path
from platform import node
from warnings import warn
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


class _Logger(AbstractContextManager, ContextDecorator):
    _cwd_index: SupportsIndex
    _name: str
    _log_file: Path
    _logger: logging.Logger

    def __init__(self, cwd_index: SupportsIndex) -> None:
        self._cwd_index = cwd_index

    def __call__(self, f: _F) -> _F:
        @wraps(f)
        def wrapper(*args, **kwargs) -> _R:
            cwd: Path = args[self._cwd_index]
            cwd.mkdir(parents=True, exist_ok=True)

            self._name = _cwd_to_logger_name(cwd)
            self._log_file = cwd / "console.log"
            with self._recreate_cm():  # type: ignore[attr-defined] # NOTE: why?
                return f(*args, **kwargs)

        return wrapper

    def __enter__(self) -> "_Logger":  # TODO: use typing.Self after 3.11
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


class _log_subprocess(AbstractContextManager):
    _logger: logging.Logger
    _begin_time: float
    res: SubprocessRes

    def __init__(self, cwd: Path) -> None:
        name = _cwd_to_logger_name(cwd)
        if name not in logging.Logger.manager.loggerDict:
            warn(f"no '{name}' logger found.")

        self._logger = logging.getLogger(name)

    @staticmethod
    def _asctime(seconds: float) -> str:
        return time.strftime("%c", time.localtime(seconds))

    def __enter__(self) -> "_log_subprocess":  # TODO: use typing.Self after 3.11
        self._begin_time = time.time()
        return self

    def __exit__(self, *args) -> None:
        res = self.res

        self._logger.info(
            f"running '{' '.join(str(item) for item in res.args)}'",
            extra={
                "asctime_": self._asctime(self._begin_time),
                "stdout_lines": res.stdout.strip("\n").splitlines(),
            },
        )
        self._logger.info(
            f"completed with exit code {res.returncode}\n",
            extra={"asctime_": self._asctime(time.time())},
        )
