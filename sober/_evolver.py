from __future__ import annotations

import itertools as it
import pickle
import shutil
from typing import TYPE_CHECKING, cast, overload

import numpy as np

import sober._pymoo_namespace as pm
import sober.config as cf
from sober._evaluator import _evaluate
from sober._logger import _log, _LoggerManager
from sober._tools import _natural_width, _write_records
from sober._typing import AnyCtrlKeyVec  # cast
from sober.input import _RealModifier

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path
    from typing import Any, Literal, TypeAlias, TypedDict

    from numpy.typing import NDArray

    from sober._io_managers import _InputManager, _OutputManager
    from sober._typing import AnyPymooCallback

    _AnyPymooX: TypeAlias = dict[str, np.integer[Any] | np.floating[Any]]

    class _PymooOut(TypedDict):
        F: NDArray[np.float_] | None
        G: NDArray[np.float_] | None

    class _PymooOperators(TypedDict):
        sampling: pm.Population
        mating: pm.MixedVariableMating
        eliminate_duplicates: pm.MixedVariableDuplicateElimination


#############################################################################
#######                   PYMOO PROBLEM CHILD CLASS                   #######
#############################################################################
class _PymooProblem(pm.Problem):  # type: ignore[misc]  # pymoo
    """interfaces the pymoo problem"""

    _input_manager: _InputManager
    _output_manager: _OutputManager
    _evaluation_dir: Path
    _saves_batches: bool
    _i_batch_width: int

    def __init__(
        self,
        input_manager: _InputManager,
        output_manager: _OutputManager,
        evaluation_dir: Path,
        callback: AnyPymooCallback,
        saves_batches: bool,
        expected_n_generations: int,
    ) -> None:
        self._input_manager = input_manager
        self._output_manager = output_manager
        self._evaluation_dir = evaluation_dir
        self._saves_batches = saves_batches
        self._i_batch_width = _natural_width(expected_n_generations)

        # NOTE: pymoo0.6 asks for a dict from input uids to pymoo variable types
        # only control variables are passed
        ctrl_vars = {
            item._label: (
                (pm.Real if isinstance(item, _RealModifier) else pm.Integral)(
                    bounds=item._bounds
                )
            )
            for item in (item for item in input_manager if item._is_ctrl)
        }

        super().__init__(
            n_obj=len(output_manager._objectives),
            n_ieq_constr=len(output_manager._constraints),
            vars=ctrl_vars,
            callback=callback,
            requires_kwargs=True,
        )

    def _evaluate(
        self,
        x: Iterable[_AnyPymooX],
        out: _PymooOut,
        *args: object,
        algorithm: pm.Algorithm,
        **kwargs: object,
    ) -> None:
        # NOTE: in pymoo0.6
        #           n_gen follows 1, 2, 3, ...
        #           x is a numpy array of dicts, each dict is a control key map
        #               whose keys are input labels
        #                     values are control key vectors
        #           out has to be a dict of numpy arrays

        i_batch = algorithm.n_gen - 1
        batch_uid = f"B{i_batch:0{self._i_batch_width}}"

        # convert pymoo x to ctrl key vecs
        ctrl_key_vecs = tuple(
            tuple(
                ctrl_key_map[item._label].item()
                if item._is_ctrl
                else item._hype_ctrl_key()
                for item in self._input_manager
            )
            for ctrl_key_map in x
        )

        ctrl_key_vecs = cast(tuple[AnyCtrlKeyVec, ...], ctrl_key_vecs)  # mypy

        # evaluate and get objectives and constraints
        batch_dir = self._evaluation_dir / batch_uid
        _evaluate(
            *ctrl_key_vecs,
            input_manager=self._input_manager,
            output_manager=self._output_manager,
            batch_dir=batch_dir,
        )
        objectives = self._output_manager._recorded_objectives(batch_dir)
        constraints = self._output_manager._recorded_constraints(batch_dir)

        out["F"] = np.asarray(objectives, dtype=np.float_)
        if self._output_manager._constraints:
            out["G"] = np.asarray(constraints, dtype=np.float_)

        if not self._saves_batches:
            shutil.rmtree(self._evaluation_dir / batch_uid)

        _log(self._evaluation_dir, f"evaluated {batch_uid}")

    def _record_survival(
        self, level: Literal["batch"], record_dir: Path, result: pm.Result
    ) -> None:
        # get evaluated individuals
        if result.algorithm.save_history:
            # from all generations
            _individuals = tuple(
                it.chain.from_iterable(item.pop for item in result.history)
            )
        else:
            # from the last generation
            _individuals = tuple(result.pop)

        # re-evaluate survival of individuals
        individuals = pm.Population.create(
            *sorted(_individuals, key=lambda x: tuple(x.X.values()))
        )
        individuals = _survival(individuals, result.algorithm)

        # create header row
        # TODO: consider prepending retrieved UIDs
        header_row = (
            tuple(item._label for item in self._input_manager)
            + tuple(self._output_manager._objectives)
            + tuple(self._output_manager._constraints)
            + ("is_pareto", "is_feasible")
        )

        # convert pymoo x to ctrl val vecs
        ctrl_key_vecs = tuple(
            tuple(
                individual.X[item._label].item()
                if item._is_ctrl
                else item._hype_ctrl_key()
                for item in self._input_manager
            )
            for individual in individuals
        )
        ctrl_val_vecs = tuple(
            tuple(
                item(key) if item._is_ctrl else item._hype_ctrl_val()
                for item, key in zip(self._input_manager, ctrl_key_vec, strict=True)
            )
            for ctrl_key_vec in ctrl_key_vecs
        )

        # create record rows
        # NOTE: pymoo sums cvs of all constraints
        #       hence all(individual.FEAS) == individual.FEAS[0]
        record_rows = [
            ctrl_val_vec
            + tuple(item.F)
            + tuple(item.G)
            + (item.get("rank") == 0, all(item.FEAS))
            for item, ctrl_val_vec in zip(individuals, ctrl_val_vecs, strict=True)
        ]

        # write records
        _write_records(
            record_dir / cf._RECORDS_FILENAMES[level], header_row, *record_rows
        )

    @_LoggerManager(cwd_index=1, is_first=True)
    def _evolve_epoch(
        self,
        epoch_dir: Path,
        algorithm: pm.Algorithm,
        termination: pm.Termination,
        save_history: bool,
        checkpoint_interval: int,
        seed: int | None,
    ) -> pm.Result:
        if checkpoint_interval <= 0:
            result = pm.minimize(
                self, algorithm, termination, save_history=save_history, seed=seed
            )
        else:
            if not isinstance(termination, pm.MaximumGenerationTermination):
                # TODO: add support for other termination criteria
                #       possibly with a while loop
                raise ValueError(
                    "checkpoints only work with max generation termination."
                )

            n_loops = termination.n_max_gen // checkpoint_interval + 1
            for i in range(n_loops):
                current_termination = (
                    termination
                    if i + 1 == n_loops
                    else pm.MaximumGenerationTermination((i + 1) * checkpoint_interval)
                )

                # NOTE: in pymoo0.6
                #           algorithm.n_gen will increase by one at the end of optimisation
                #           at least when using a MaximumGenerationTermination
                if algorithm.n_gen and (
                    algorithm.n_gen - 1 >= current_termination.n_max_gen
                ):
                    # this should only be invoked by resume
                    continue

                # NOTE: in pymoo0.6
                #           algorithm.setup() is only triggered when algorithm.problem is None
                #           but seed will be reset too
                #           manually change algorithm.termination to avoid resetting seed
                algorithm.termination = current_termination

                result = pm.minimize(
                    self,
                    algorithm,
                    current_termination,
                    save_history=save_history,
                    seed=seed,
                )

                # update algorithm
                algorithm = result.algorithm

                i_checkpoint = (i + 1) * checkpoint_interval - 1
                if algorithm.n_gen - 1 == i_checkpoint + 1:
                    with (epoch_dir / "checkpoint.pickle").open("wb") as fp:
                        pickle.dump((epoch_dir, result), fp)

                    _log(epoch_dir, f"created checkpoint at generation {i_checkpoint}")

        self._record_survival("batch", epoch_dir, result)

        return result


#############################################################################
#######                      OPERATOR FUNCTIONS                       #######
#############################################################################
def _sampling(problem: pm.Problem, init_population_size: int) -> pm.Population:
    """samples the initial generation"""

    return pm.MixedVariableSampling()(problem, init_population_size)


def _operators(
    algorithm_name: Literal["nsga2", "nsga3"],
    p_crossover: float,
    p_mutation: float,
    sampling: pm.Population,
) -> _PymooOperators:
    """a pymoo operators constructor"""

    # defaults from respective algorithm classes in pymoo
    selections = {
        "nsga2": pm.TournamentSelection(func_comp=pm.binary_tournament),
        "nsga3": pm.TournamentSelection(func_comp=pm.comp_by_cv_then_random),
    }
    etas = {
        "nsga2": {"crossover": 15, "mutation": 20},
        "nsga3": {"crossover": 30, "mutation": 20},
    }

    crossover_kwargs = {"prob": p_crossover, "eta": etas[algorithm_name]["crossover"]}

    # NOTE: in pymoo0.6 mutation
    #           prob (0.5) -> prob_var, controlling mutation for each gene/variable
    #           prob controls the whole mutation operation
    #           see https://github.com/anyoptimization/pymoo/discussions/360
    #           though the answer is not entirely correct
    mutation_kwargs = {
        "prob": 1.0,
        "prob_var": p_mutation,
        "eta": etas[algorithm_name]["mutation"],
    }

    return {
        "sampling": sampling,
        "mating": pm.MixedVariableMating(
            selection=selections[algorithm_name],
            crossover={
                pm.Real: pm.SimulatedBinaryCrossover(**crossover_kwargs),
                pm.Integral: pm.SimulatedBinaryCrossover(
                    **crossover_kwargs, vtype=float, repair=pm.RoundingRepair()
                ),
            },
            mutation={
                pm.Real: pm.PolynomialMutation(**mutation_kwargs),
                pm.Integral: pm.PolynomialMutation(
                    **mutation_kwargs, vtype=float, repair=pm.RoundingRepair()
                ),
            },
            eliminate_duplicates=pm.MixedVariableDuplicateElimination(),
        ),
        "eliminate_duplicates": pm.MixedVariableDuplicateElimination(),
    }


#############################################################################
#######                      ALGORITHM FUNCTIONS                      #######
#############################################################################
@overload
def _algorithm(
    algorithm_name: Literal["nsga2"],
    population_size: int,
    p_crossover: float,
    p_mutation: float,
    sampling: pm.Population,
) -> pm.NSGA2: ...
@overload
def _algorithm(
    algorithm_name: Literal["nsga3"],
    population_size: int,
    p_crossover: float,
    p_mutation: float,
    sampling: pm.Population,
    reference_directions: pm.ReferenceDirectionFactory,
) -> pm.NSGA3: ...
def _algorithm(
    algorithm_name: Literal["nsga2", "nsga3"],
    population_size: int,
    p_crossover: float,
    p_mutation: float,
    sampling: pm.Population,
    reference_directions: None | pm.ReferenceDirectionFactory = None,
) -> pm.NSGA2 | pm.NSGA3:
    """a pymoo algorithm constructor"""

    if algorithm_name == "nsga2":
        return pm.NSGA2(
            population_size,
            **_operators(algorithm_name, p_crossover, p_mutation, sampling),
        )
    else:
        return pm.NSGA3(
            reference_directions,
            population_size,
            **_operators(algorithm_name, p_crossover, p_mutation, sampling),
        )


#############################################################################
#######                      SURVIVAL FUNCTIONS                       #######
#############################################################################
def _survival(
    individuals: pm.Population, algorithm: pm.GeneticAlgorithm
) -> pm.Population:
    """evaluates survival of individuals"""
    # remove duplicates
    individuals = pm.MixedVariableDuplicateElimination().do(individuals)

    # runs survival
    # this resets the value of Individual().data["rank"] for each individual
    algorithm.survival.do(algorithm.problem, individuals, algorithm=algorithm)

    return individuals
