from os import PathLike
from typing import TypeAlias

# from numpy.typing import _ArrayLikeFloat_co

AnyStrPath: TypeAlias = str | PathLike[str]
# AnyRArray: TypeAlias = _ArrayLikeFloat_co

DATACLASS_PARAMS = {
    "init": True,
    "repr": False,
    "eq": False,  # not work for float("nan")
    "order": False,  # see 'eq'
    "unsafe_hash": False,
    "frozen": False,
    "match_args": False,
    "kw_only": False,
    # "slots": True,  # explicit super(), see https://bugs.python.org/issue46404, https://stackoverflow.com/a/1817840
}
