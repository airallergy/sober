from io import StringIO
from pathlib import Path
from collections.abc import Sequence
from typing import TypeVar, Callable, Iterable, Optional

from eppy import openidf
from pymoo.core.problem import Problem as _PymooProblem

from .config import _CONFIG
from ._tools import AnyStrPath
from .parameters import _Parameter
from ._simulator import _split_model

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
    ) -> None:
        self.evaluator = evaluator
        super().__init__(
            n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu, callback=callback
        )

    def _evaluate(self, x, out, *args, **kwargs) -> None:
        ...


#############################################################################
#######                        PROBLEM CLASSES                        #######
#############################################################################
class Problem:
    model_file: Path
    weather: _Parameter
    parameters: tuple[_Parameter, ...]
    # collector: Sequence[_Collector]
    callback: Optional[Callable] = None
    _model_type: str
    _tagged_model: str
    # objectives: Sequence[_Objective] = field(init=False)
    # constraints: Sequence[_Constraint] = field(init=False)

    def __init__(
        self,
        model_file: AnyStrPath,
        weather: _Parameter,
        parameters: Iterable[_Parameter],
        # collector: Iterable[_Collector],
        callback: Optional[Callable] = None,
        # objectives: Sequence[_Objective] = field(init=False),
        # constraints: Sequence[_Constraint] = field(init=False),
    ) -> None:
        self.model_file = Path(model_file)
        self.weather = weather
        self.parameters = tuple(parameters)

        self._model_type = self.model_file.suffix
        if self._model_type not in ("idf", "imf"):
            raise NotImplementedError(f"a '{self._model_type}' model is not supported.")

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

    # def _to_pymoo(self) -> PymooProblem:
    #     return PymooProblem(
    #         self.evaluator,
    #         len(self.parameters),
    #         len(self.objectives),
    #         len(self.constraints),
    #         tuple(parameter.low for parameter in self.parameters),
    #         tuple(parameter.high for parameter in self.parameters),
    #         self.callback,
    #     )
