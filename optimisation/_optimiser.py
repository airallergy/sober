from pathlib import Path
from typing import TypedDict

from ._logger import _LoggerManager
from . import _pymoo_namespace as pm
from .parameters import AnyParameter, ContinuousParameter, _ParametersManager


class Operators(TypedDict):
    sampling: pm.MixedVariableSampling
    crossover: pm.MixedVariableCrossover
    mutation: pm.MixedVariableMutation


def _operators(parameters_manager: _ParametersManager[AnyParameter]) -> Operators:
    mask = tuple(
        "real" if isinstance(parameter, ContinuousParameter) else "int"
        for parameter in parameters_manager
    )
    return {
        "sampling": pm.MixedVariableSampling(
            mask,
            {"real": pm.get_sampling("real_lhs"), "int": pm.get_sampling("int_lhs")},
        ),
        "crossover": pm.MixedVariableCrossover(
            mask,
            {"real": pm.get_crossover("real_sbx"), "int": pm.get_crossover("int_sbx")},
        ),
        "mutation": pm.MixedVariableMutation(
            mask,
            {"real": pm.get_mutation("real_pm"), "int": pm.get_mutation("int_pm")},
        ),
    }


def _algorithm(
    population_size: int, parameters_manager: _ParametersManager[AnyParameter]
) -> pm.NSGA2:
    return pm.NSGA2(
        pop_size=population_size,
        eliminate_duplicates=True,
        **_operators(parameters_manager)
    )


@_LoggerManager(cwd_index=0, is_first=True)
def _optimise_epoch(
    cwd: Path,
    problem: pm.Problem,
    population_size: int,
    termination: pm.Termination,
    seed: int,
) -> pm.Result:
    return pm.minimize(
        problem=problem,
        algorithm=_algorithm(population_size, problem._parameters_manager),
        termination=termination,
        seed=seed,
        save_history=False,
    )
