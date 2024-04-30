from __future__ import annotations

import itertools as it
import math
import pickle
from abc import ABC
from typing import TYPE_CHECKING

import sober._evolver_pymoo as pm
import sober.config as cf
from sober._logger import _log, _LoggerManager
from sober._tools import _parsed_path, _write_records

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Literal

    from sober._io_managers import _InputManager, _OutputManager
    from sober._typing import AnyReferenceDirections, AnyStrPath


#############################################################################
#######                     ABSTRACT BASE CLASSES                     #######
#############################################################################
class _Evolver(ABC):
    """an abstract base class for evolvers"""

    __slots__ = ("_input_manager", "_output_manager", "_evaluation_dir")

    _input_manager: _InputManager
    _output_manager: _OutputManager
    _evaluation_dir: Path

    def __init__(
        self,
        input_manager: _InputManager,
        output_manager: _OutputManager,
        evaluation_dir: Path,
    ) -> None:
        self._input_manager = input_manager
        self._output_manager = output_manager
        self._evaluation_dir = evaluation_dir

        self._prepare()

    def _check_args(self) -> None:
        if not self._output_manager._objectives:
            raise ValueError("optimisation needs at least one objective.")

    def _prepare(self) -> None:
        self._check_args()

        # global variables
        cf._has_batches = True


#############################################################################
#######                        EVOLVER CLASSES                        #######
#############################################################################
class _PymooEvolver(_Evolver):
    """evolves via pymoo"""

    __slots__ = ("_problem",)

    _problem: pm._Problem

    def __init__(
        self,
        input_manager: _InputManager,
        output_manager: _OutputManager,
        evaluation_dir: Path,
    ) -> None:
        self._problem = pm._Problem(input_manager, output_manager, evaluation_dir)

        super().__init__(input_manager, output_manager, evaluation_dir)

    def _record_survival(
        self, level: Literal["batch"], record_dir: Path, result: pm.Result
    ) -> None:
        # get evaluated individuals
        if result.algorithm.save_history:
            # from all generations
            individuals = tuple(
                it.chain.from_iterable(item.pop for item in result.history)
            )
        else:
            # from the last generation
            individuals = tuple(result.pop)

        # re-evaluate survival of individuals
        population = pm._survival(individuals, result.algorithm)

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
                item.X[input._label].item()
                if input._is_ctrl
                else input._hype_ctrl_key()
                for input in self._input_manager
            )
            for item in population
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
            for item, ctrl_val_vec in zip(population, ctrl_val_vecs, strict=True)
        ]

        # write records
        _write_records(
            record_dir / cf._RECORDS_FILENAMES[level], header_row, *record_rows
        )

    @_LoggerManager(is_first=True)
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
                self._problem,
                algorithm,
                termination,
                save_history=save_history,
                seed=seed,
            )
        else:
            if not (
                isinstance(termination, pm.MaximumGenerationTermination)
                and isinstance(termination.n_max_gen, int)
            ):
                # TODO: add support for other termination criteria
                #       possibly with a while loop
                raise NotImplementedError(
                    "checkpoints only work with finite max generation termination."
                )

            n_loops = math.ceil(termination.n_max_gen / checkpoint_interval)
            for i in range(n_loops):
                current_termination = (
                    termination
                    if i + 1 == n_loops
                    else pm.MaximumGenerationTermination((i + 1) * checkpoint_interval)
                )

                # NOTE: [1] in pymoo0.6
                #               algorithm.n_gen += 1 for next gen at the end of the current gen
                #               that is, in each gen, n_gen matches the current gen for most of the time
                if algorithm.is_initialized and (
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
                    self._problem,
                    algorithm,
                    current_termination,
                    save_history=save_history,
                    seed=seed,
                )

                # update algorithm
                algorithm = result.algorithm

                i_gen_checkpoint = (i + 1) * checkpoint_interval - 1
                # as per [1], algorithm.n_gen has been increased for next gen at this point, hence -2
                if algorithm.n_gen - 2 == i_gen_checkpoint:
                    with (epoch_dir / "checkpoint.pickle").open("wb") as fp:
                        pickle.dump((cf._global_vars(), self, result), fp)

                    _log(
                        epoch_dir,
                        f"created checkpoint at generation {i_gen_checkpoint}",
                    )

        self._record_survival("batch", epoch_dir, result)

        return result

    def _nsga2(
        self,
        population_size: int,
        termination: pm.Termination,
        p_crossover: float,
        p_mutation: float,
        init_population_size: int,
        saves_history: bool,
        checkpoint_interval: int,
        seed: int | None,
    ) -> pm.Result:
        """runs optimisation via the NSGA2 algorithm"""

        if init_population_size <= 0:
            init_population_size = population_size
        sampling = pm._sampling(self._problem, init_population_size)

        algorithm = pm._algorithm(
            "nsga2", population_size, p_crossover, p_mutation, sampling
        )

        return self._evolve_epoch(
            self._evaluation_dir,
            algorithm,
            termination,
            saves_history,
            checkpoint_interval,
            seed,
        )

    def _nsga3(
        self,
        population_size: int,
        termination: pm.Termination,
        reference_directions: AnyReferenceDirections | None,
        p_crossover: float,
        p_mutation: float,
        init_population_size: int,
        saves_history: bool,
        checkpoint_interval: int,
        seed: int | None,
    ) -> pm.Result:
        """runs optimisation via the NSGA3 algorithm"""

        if init_population_size <= 0:
            init_population_size = population_size
        sampling = pm._sampling(self._problem, init_population_size)

        if not reference_directions:
            reference_directions = pm._default_reference_directions(
                self._problem.n_obj, population_size, seed=seed
            )

        algorithm = pm._algorithm(
            "nsga3",
            population_size,
            p_crossover,
            p_mutation,
            sampling,
            reference_directions,
        )

        return self._evolve_epoch(
            self._evaluation_dir,
            algorithm,
            termination,
            saves_history,
            checkpoint_interval,
            seed,
        )

    @classmethod
    def resume(
        cls,
        checkpoint_file: AnyStrPath,
        termination: pm.Termination,
        checkpoint_interval: int = 0,
    ) -> pm.Result:
        """resumes optimisation using checkpoint files"""
        # NOTE: although seed will not be reset
        #       randomness is not reproducible when resuming for some unknown reason
        # TODO: explore implementing custom serialisation for Problem via TOML/YAML
        # TODO: termination may not be necessary, as the old one may be reused

        checkpoint_file = _parsed_path(checkpoint_file, "checkpoint file")

        with checkpoint_file.open("rb") as fp:
            global_vars, self, result = pickle.load(fp)

        cf._update_global_vars(**global_vars)

        # checks validity of the checkpoint file
        # currently only checks the object type, but there might be better checks
        if not (isinstance(self, cls) and isinstance(result, pm.Result)):
            raise TypeError(f"invalid checkpoint file: {checkpoint_file}.")

        result.algorithm.termination = termination

        return self._evolve_epoch(
            self._evaluation_dir,
            result.algorithm,
            result.algorithm.termination,  # void
            result.algorithm.save_history,  # void
            checkpoint_interval,
            result.algorithm.seed,  # void
        )  # void: not updated into the algorithm by pymoo once the algorithm has a problem
