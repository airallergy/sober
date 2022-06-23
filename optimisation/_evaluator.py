import itertools as it
from pathlib import Path
from shutil import copyfile
from multiprocessing import get_context
from typing import Any, Callable, Iterable

from .collector import _Collector
from .config import _CONFIG, _update_config
from ._simulator import _run_epmacro, _run_energyplus
from .parameters import WeatherParameter, AnyIntModelParameter


def _product_evaluate(
    variation_idxs: tuple[int, ...],
    tagged_model: str,
    weather: WeatherParameter,
    parameters: tuple[AnyIntModelParameter, ...],
    outputs: tuple[_Collector, ...],
    outputs_directory: Path,
    model_type: str,
) -> None:
    weather_variation_idx = variation_idxs[0]
    parameter_variation_idxs = variation_idxs[1:]

    # generate job uid
    job_uid = f"EP_M-0_W-{weather_variation_idx}_" + "_".join(
        (
            f"P{idx}-{variation_idx}"
            for idx, variation_idx in enumerate(parameter_variation_idxs)
        )
    )

    # create job folder
    job_directory = outputs_directory / job_uid
    job_directory.mkdir(exist_ok=True)

    # copy job weather files
    job_epw_file = job_directory / "in.epw"
    copyfile(weather.variations[weather_variation_idx], job_epw_file)

    model = tagged_model
    # insert parameter value
    for variation_idx, parameter in zip(parameter_variation_idxs, parameters):
        model = model.replace(
            parameter.tagger._tag, str(parameter.variations[variation_idx])
        )

    # write job model file
    job_model_file = job_directory / ("in" + model_type)
    with open(job_model_file, "wt") as f:
        f.write(model)

    # run epmacro if needed
    if model_type == ".imf":
        job_idf_file = _run_epmacro(job_model_file)
    elif model_type == ".idf":
        job_idf_file = job_model_file

    # run energyplus
    _run_energyplus(job_idf_file, job_epw_file, job_directory, False)

    # collect outputs per job


def _pymoo_evaluate():
    ...


def _parallel_evaluate(
    func: Callable,
    params: Iterable[Iterable[Any]],
    *meta_params,
    processess: int | None = None,
) -> None:
    ctx = get_context("forkserver")
    with ctx.Pool(processess, initializer=_update_config, initargs=(_CONFIG,)) as pool:
        pool.starmap(func, zip(params, *map(it.repeat, meta_params)))
