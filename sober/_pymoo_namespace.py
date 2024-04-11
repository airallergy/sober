# ruff: noqa: F401
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

if TYPE_CHECKING:
    from pymoo.algorithms.base.genetic import GeneticAlgorithm
    from pymoo.core.algorithm import Algorithm
    from pymoo.core.callback import Callback
    from pymoo.core.result import Result
    from pymoo.core.termination import Termination
    from pymoo.util.ref_dirs.energy import RieszEnergyReferenceDirectionFactory
    from pymoo.util.reference_direction import ReferenceDirectionFactory
