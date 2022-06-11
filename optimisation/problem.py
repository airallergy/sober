from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from eppy import openidf
from pymoo.core.problem import Problem as _PymooProblem

from .parameters import _Parameter
from ._tools import DATACLASS_PARAMS
from .config import _CONFIG
from ._simulator import _split_model

from typing import Callable, TypeVar
from collections.abc import Sequence
from ._tools import AnyStrPath

AnyReals = TypeVar("AnyReals", bound=float | Sequence[float])


class PymooProblem(_PymooProblem):
    def __init__(
        self,
        evaluator: Callable,
        n_var: int,
        n_obj: int,
        n_constr: int,
        xl: AnyReals,
        xu: AnyReals,
        callback: Callable,
    ):
        self.evaluator = evaluator
        super().__init__(
            n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu, callback=callback
        )

    def _evaluate(self, x, out, *args, **kwargs):
        self.evaluator(x, out, *args, **kwargs)


@dataclass(**DATACLASS_PARAMS)
class _Problem(ABC):
    model_file: Path
    weathers: Sequence[_Parameter]
    parameters: Sequence[_Parameter]
    # collector: Sequence[_Collector]
    callback: Callable = None
    _tagged_model: str = field(init=False)
    # objectives: Sequence[_Objective] = field(init=False)
    # constraints: Sequence[_Constraint] = field(init=False)

    @abstractmethod
    def _tag_model(self) -> None:
        ...

    @abstractmethod
    def _to_pymoo(self):
        ...


@dataclass(**DATACLASS_PARAMS)
class Problem(_Problem):
    model_file: AnyStrPath

    def __post_init__(self) -> None:
        self.model_file = Path(self.model_file)

    def _tag_model(self) -> None:
        macros, regulars = _split_model(self.model_file)
        idf = openidf(StringIO(regulars), _CONFIG["schema.energyplus"])

        for parameter in self.parameters:
            tagger = parameter.tagger
            match tagger._loc:
                case "regular":
                    idf = tagger._tagged(idf)
                case "macro":
                    macros = tagger._tagged(macros)

        self._tagged_model = macros + idf.idfstr()

    def _to_pymoo(self) -> PymooProblem:
        return PymooProblem(
            self.evaluator,
            len(self.parameters),
            len(self.objectives),
            len(self.constraints),
            tuple(parameter.low for parameter in self.parameters),
            tuple(parameter.high for parameter in self.parameters),
            self.callback,
        )
