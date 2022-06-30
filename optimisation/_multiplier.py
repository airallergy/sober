from pathlib import Path
from itertools import product

from .results import _Collector
from ._evaluator import _product_evaluate, _parallel_evaluate
from .parameters import WeatherParameter, AnyIntModelParameter


def _multiply(
    tagged_model: str,
    weather: WeatherParameter,
    parameters: tuple[AnyIntModelParameter, ...],
    results: tuple[_Collector, ...],
    results_directory: Path,
    model_type: str,
) -> None:
    variation_idxs_iter = product(
        range(weather._n_variations),
        *(range(parameter._n_variations) for parameter in parameters),
    )

    _parallel_evaluate(
        _product_evaluate,
        variation_idxs_iter,
        tagged_model=tagged_model,
        weather=weather,
        parameters=parameters,
        results=results,
        results_directory=results_directory,
        model_type=model_type,
    )
