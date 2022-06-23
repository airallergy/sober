import itertools as it
from pathlib import Path

from .collector import _Collector
from ._evaluator import _product_evaluate, _parallel_evaluate
from .parameters import WeatherParameter, AnyIntModelParameter


def _product(
    tagged_model: str,
    weather: WeatherParameter,
    parameters: tuple[AnyIntModelParameter, ...],
    outputs: tuple[_Collector, ...],
    outputs_directory: Path,
    model_type: str,
) -> None:
    variation_idxs_iter = it.product(
        range(len(weather.variations)),
        *(range(len(parameter.variations)) for parameter in parameters),
    )

    _parallel_evaluate(
        _product_evaluate,
        variation_idxs_iter,
        tagged_model,
        weather,
        parameters,
        outputs,
        outputs_directory,
        model_type,
    )
