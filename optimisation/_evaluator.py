from pathlib import Path
from shutil import copyfile
from typing import Any, TypedDict
from itertools import chain, product
from collections.abc import Callable, Iterable

from . import config as cf
from .results import _ResultsManager
from ._tools import _multiprocessing_context
from ._simulator import _run_epmacro, _run_energyplus
from .parameters import WeatherParameter, AnyIntModelParameter

MetaParams = TypedDict(
    "MetaParams",
    {
        "tagged_model": str,
        "weather": WeatherParameter,
        "parameters": tuple[AnyIntModelParameter, ...],
        "results_manager": _ResultsManager,
        "evaluation_directory": Path,
        "model_type": cf.AnyModelType,
    },
)

_meta_params: MetaParams


def _product_evaluate(variation_idxs: tuple[int, ...]) -> cf.AnyUIDsPair:
    model = _meta_params["tagged_model"]
    weather = _meta_params["weather"]
    parameters = _meta_params["parameters"]
    results_manager = _meta_params["results_manager"]
    evaluation_directory = _meta_params["evaluation_directory"]
    model_type = _meta_params["model_type"]

    weather_variation_idx = variation_idxs[0]
    parameter_variation_idxs = variation_idxs[1:]

    # generate job uid
    job_uid = f"V_W-{weather_variation_idx}_" + "_".join(
        (
            f"P{idx}-{variation_idx}"
            for idx, variation_idx in enumerate(parameter_variation_idxs)
        )
    )

    # create job folder
    job_directory = evaluation_directory / job_uid
    job_directory.mkdir(exist_ok=True)

    # handle uncertain parameters
    task_uids = []
    for uncertainty_idxs in product(
        range(weather._ns_uncertainty[weather_variation_idx]),
        *map(
            range,
            (
                parameter._ns_uncertainty[variation_idx]
                for variation_idx, parameter in zip(
                    parameter_variation_idxs, parameters
                )
            ),
        ),
    ):
        weather_uncertainty_idx = uncertainty_idxs[0]
        parameter_uncertainty_idxs = uncertainty_idxs[1:]

        # generate task uid
        task_uid = "_".join(
            chain(
                ("U", f"W-{weather_uncertainty_idx}")
                if weather._is_uncertain
                else ("U",),
                (
                    f"P{idx}-{uncertainty_idx}"
                    for idx, uncertainty_idx in enumerate(parameter_uncertainty_idxs)
                    if parameters[idx]._is_uncertain
                ),
            )
        )
        task_uids.append(task_uid)

        # create task folder
        task_directory = job_directory / task_uid
        task_directory.mkdir(exist_ok=True)

        # copy task weather files
        task_epw_file = task_directory / "in.epw"
        copyfile(weather[weather_variation_idx, weather_uncertainty_idx], task_epw_file)

        # insert parameter value
        for variation_idx, uncertainty_idx, parameter in zip(
            parameter_variation_idxs, parameter_uncertainty_idxs, parameters
        ):
            model = model.replace(
                parameter._tagger._tag, str(parameter[variation_idx, uncertainty_idx])
            )

        # write task model file
        task_model_file = task_directory / ("in" + model_type)
        with open(task_model_file, "wt") as f:
            f.write(model)

        # run epmacro if needed
        if model_type == ".imf":
            task_idf_file = _run_epmacro(task_model_file)
        elif model_type == ".idf":
            task_idf_file = task_model_file
        else:
            raise

        # run energyplus
        _run_energyplus(task_idf_file, task_epw_file, task_directory, False)

    return job_uid, tuple(task_uids)

    # # collect results per task
    # for result in results_manager._task_collectors:
    #     result._collect(task_directory)


def _pymoo_evaluate():
    ...


def _initialise(config: cf.Config, meta_params: MetaParams) -> None:
    cf._update_config(config)
    global _meta_params
    _meta_params = meta_params


def _parallel_evaluate(
    func: Callable[..., cf.AnyUIDsPair],
    params: Iterable[Iterable[Any]],
    **meta_params,  # TODO: **MetaParams after PEP 692/3.12
) -> None:
    ctx = _multiprocessing_context()
    with ctx.Manager() as manager:
        _meta_params = manager.dict(meta_params)
        with ctx.Pool(
            cf._config["n.processes"],
            initializer=_initialise,
            initargs=(cf._config, _meta_params),
        ) as pool:
            job_uids = tuple(pool.map(func, params))
