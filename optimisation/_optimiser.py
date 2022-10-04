from pathlib import Path
from typing import TypedDict

from ._logger import _LoggerManager
from . import _pymoo_namespace as pm
from .parameters import AnyParameter, ContinuousParameter, _ParametersManager


class Operators(TypedDict):
    sampling: pm.Sampling
    crossover: pm.Crossover
    mutation: pm.Mutation


def _operators(
    parameters_manager: _ParametersManager[AnyParameter],
    p_crossover: float,
    p_mutation: float,
) -> Operators:
    mask = tuple(
        "real" if isinstance(parameter, ContinuousParameter) else "int"
        for parameter in parameters_manager
    )
    return {
        "sampling": pm.MixedVariableSampling(
            mask,
            {
                "real": pm.LatinHypercubeSampling(),
                "int": pm.IntegerFromFloatSampling(pm.LatinHypercubeSampling),
            },
        ),
        "crossover": pm.MixedVariableCrossover(
            mask,
            {
                "real": pm.SimulatedBinaryCrossover(
                    prob=p_crossover, eta=15  # TODO: eta has defaults from 0.60
                ),
                "int": pm.IntegerFromFloatCrossover(
                    pm.SimulatedBinaryCrossover, prob=p_crossover, eta=15
                ),
            },
        ),
        "mutation": pm.MixedVariableMutation(
            mask,
            {
                "real": pm.PolynomialMutation(prob=p_mutation, eta=20),
                "int": pm.IntegerFromFloatMutation(
                    pm.PolynomialMutation, prob=p_mutation, eta=20
                ),
            },
        ),
    }


def _algorithm(
    population_size: int,
    parameters_manager: _ParametersManager[AnyParameter],
    p_crossover: float,
    p_mutation: float,
) -> pm.NSGA2:
    return pm.NSGA2(
        pop_size=population_size,
        eliminate_duplicates=True,
        **_operators(parameters_manager, p_crossover, p_mutation)
    )


@_LoggerManager(cwd_index=0, is_first=True)
def _optimise_epoch(
    cwd: Path,
    problem: pm.Problem,
    population_size: int,
    termination: pm.Termination,
    p_crossover: float,
    p_mutation: float,
    save_history: bool,
    seed: int,
) -> pm.Result:
    return pm.minimize(
        problem=problem,
        algorithm=_algorithm(
            population_size, problem._parameters_manager, p_crossover, p_mutation
        ),
        termination=termination,
        seed=seed,
        save_history=save_history,
    )
