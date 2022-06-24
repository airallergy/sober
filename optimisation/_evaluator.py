from pathlib import Path
from platform import system
from shutil import copyfile
from itertools import repeat
from multiprocessing import get_context
from typing import Any, Callable, Iterable
from multiprocessing.context import BaseContext

from .collector import _Collector
from .config import _config, _update_config
from ._simulator import _run_epmacro, _run_energyplus
from .parameters import WeatherParameter, AnyIntModelParameter


def _product_evaluate(variation_idxs: tuple[int, ...]) -> None:
    tagged_model = _meta_params["tagged_model"]
    weather = _meta_params["weather"]
    parameters = _meta_params["parameters"]
    outputs = _meta_params["outputs"]
    outputs_directory = _meta_params["outputs_directory"]
    model_type = _meta_params["model_type"]

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


def _multiprocessing_context() -> BaseContext:
    match system():
        case "Linux" | "Darwin":
            return get_context("forkserver")
        case "Windows":
            return get_context("spawn")
        case _ as system_name:
            raise NotImplementedError(f"unsupported system: '{system_name}'.")


def _initialise(config, meta_params) -> None:
    _update_config(config)
    global _meta_params
    _meta_params = meta_params


def _parallel_evaluate(
    func: Callable,
    params: Iterable[Iterable[Any]],
    processess: int | None = None,
    **meta_params,
) -> None:
    ctx = _multiprocessing_context()
    with ctx.Manager() as manager:
        _meta_params = manager.dict(meta_params)
        with ctx.Pool(
            processess, initializer=_initialise, initargs=(_config, _meta_params)
        ) as pool:
            pool.map(func, params)
