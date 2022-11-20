from typing import Literal, TypedDict, overload

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


@overload
def _algorithm(
    algorithm: Literal["nsga2"],
    population_size: int,
    parameters_manager: _ParametersManager[AnyParameter],
    p_crossover: float,
    p_mutation: float,
) -> pm.Algorithm:
    ...


@overload
def _algorithm(
    algorithm: Literal["nsga3"],
    population_size: int,
    parameters_manager: _ParametersManager[AnyParameter],
    p_crossover: float,
    p_mutation: float,
    reference_directions: pm.ReferenceDirectionFactory,
) -> pm.Algorithm:
    ...


def _algorithm(
    algorithm,
    population_size,
    parameters_manager,
    p_crossover,
    p_mutation,
    reference_directions=None,
) -> pm.Algorithm:
    # python/mypy#7213, reference_directions cannot be inferred correctly
    if algorithm == "nsga2":
        return pm.NSGA2(
            pop_size=population_size,
            eliminate_duplicates=True,
            **_operators(parameters_manager, p_crossover, p_mutation),
        )
    else:
        return pm.NSGA3(
            reference_directions,
            pop_size=population_size,
            eliminate_duplicates=True,
            **_operators(parameters_manager, p_crossover, p_mutation),
        )
