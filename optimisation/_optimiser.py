from typing import TypedDict

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
