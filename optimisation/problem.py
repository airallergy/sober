from io import StringIO
from pathlib import Path
from typing import Iterable

import numpy as np
from eppy import openidf
from numpy.typing import NDArray
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem as _PymooProblem

from .config import _CONFIG
from ._tools import AnyStrPath
from ._evaluator import _evaluate
from .collector import _Collector
from ._simulator import _split_model
from .parameters import _Parameter, _IntParameter


class PymooProblem(_PymooProblem):
    def __init__(
        self,
        n_var: int,
        n_obj: int,
        n_constr: int,
        xl: NDArray[np.float_],
        xu: NDArray[np.float_],
        callback: Callback | None,
    ) -> None:
        super().__init__(
            n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu, callback=callback
        )

    def _evaluate(self, x, out, *args, **kwargs) -> None:
        _evaluate


#############################################################################
#######                        PROBLEM CLASSES                        #######
#############################################################################
class Problem:
    model_file: Path
    weather: _IntParameter
    parameters: tuple[_Parameter, ...]
    objectives: tuple[_Collector, ...]
    constraints: tuple[_Collector, ...]
    extra_outputs: tuple[_Collector, ...]
    callback: Callback | None
    outputs_directory: Path
    _model_type: str
    _tagged_model: str

    def __init__(
        self,
        model_file: AnyStrPath,
        weather: _IntParameter,
        parameters: Iterable[_Parameter],
        objectives: Iterable[_Collector] = (),
        constraints: Iterable[_Collector] = (),
        extra_outputs: Iterable[_Collector] = (),
        callback: Callback | None = None,
        outputs_directory: AnyStrPath | None = None,
    ) -> None:
        self.model_file = Path(model_file)
        self.weather = weather
        self.parameters = tuple(parameters)
        self.objectives = tuple(objectives)
        self.constraints = tuple(constraints)
        self.extra_outputs = tuple(extra_outputs)
        self.callback = callback
        self.outputs_directory = (
            self.model_file.parent / "outputs"
            if outputs_directory is None
            else Path(outputs_directory)
        )

        self._model_type = self.model_file.suffix
        if self._model_type not in ("idf", "imf"):
            raise NotImplementedError(f"a '{self._model_type}' model is not supported.")

    def _tag_model(self) -> None:
        macros, regulars = _split_model(self.model_file)
        idf = openidf(StringIO(regulars), _CONFIG["schema.energyplus"])

        for parameter in self.parameters:
            tagger = parameter.tagger
            match tagger._LOCATION:
                case "regular":
                    idf = tagger._tagged(idf)
                case "macro":
                    macros = tagger._tagged(macros)

        self._tagged_model = macros + idf.idfstr()

    def _to_pymoo(self) -> PymooProblem:
        if self.objectives == ():
            raise ValueError("Optimisation needs at least one objective")

        return PymooProblem(
            len(self.parameters),
            len(self.objectives),
            len(self.constraints),
            np.fromiter(
                (parameter.low for parameter in self.parameters), dtype=np.float_
            ),
            np.fromiter(
                (parameter.high for parameter in self.parameters), dtype=np.float_
            ),
            self.callback,
        )
