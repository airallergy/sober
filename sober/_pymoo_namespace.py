from typing import TYPE_CHECKING

from pymoo.algorithms.moo.nsga2 import NSGA2, binary_tournament
from pymoo.algorithms.moo.nsga3 import NSGA3, comp_by_cv_then_random
from pymoo.core.mixed import (
    MixedVariableDuplicateElimination,
    MixedVariableMating,
    MixedVariableSampling,
)
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.core.variable import Integer as Integral  # follow the numbers stdlib
from pymoo.core.variable import Real
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.optimize import minimize
from pymoo.termination.max_gen import MaximumGenerationTermination
from pymoo.util.ref_dirs.energy import RieszEnergyReferenceDirectionFactory

if TYPE_CHECKING:
    # ruff: noqa: PLC0414  # astral-sh/ruff#3711
    from pymoo.algorithms.base.genetic import GeneticAlgorithm as GeneticAlgorithm
    from pymoo.core.algorithm import Algorithm as Algorithm
    from pymoo.core.callback import Callback as Callback
    from pymoo.core.result import Result as Result
    from pymoo.core.termination import Termination as Termination
    from pymoo.util.reference_direction import (
        ReferenceDirectionFactory as ReferenceDirectionFactory,
    )


__all__ = (
    "NSGA2",
    "NSGA3",
    "Integral",
    "MaximumGenerationTermination",
    "MixedVariableDuplicateElimination",
    "MixedVariableMating",
    "MixedVariableSampling",
    "PolynomialMutation",
    "Population",
    "Problem",
    "Real",
    "RieszEnergyReferenceDirectionFactory",
    "RoundingRepair",
    "SimulatedBinaryCrossover",
    "TournamentSelection",
    "binary_tournament",
    "comp_by_cv_then_random",
    "minimize",
)
