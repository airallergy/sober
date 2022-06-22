import itertools as it
from pathlib import Path
from shutil import copyfile
from itertools import product
from multiprocessing import Pool

from .collector import _Collector
from ._simulator import _run_epmacro, _run_energyplus
from .parameters import WeatherParameter, AnyIntModelParameter


def _product_evaluate(
    tagged_model: str,
    weather: WeatherParameter,
    parameters: tuple[AnyIntModelParameter, ...],
    outputs: tuple[_Collector, ...],
    outputs_directory: Path,
    model_type: str,
) -> None:
    # create outputs folder
    outputs_directory.mkdir(exist_ok=True)

    for variation_idxs in it.product(
        range(len(weather.variations)),
        *(range(len(parameter.variations)) for parameter in parameters),
    ):
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
                parameter.tagger._tag, parameter.variations[variation_idx]
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


# def _parallel_evaluate(processess: int) -> None:
#     with Pool(processess) as pool:
#         # setup parallel processes
#         processes = tuple(
#             pool.apply_async(self.run, (job_idx, parameter_set_idxs))
#             for job_idx, parameter_set_idxs in enumerate(
#                 product(*self.parameter_values_idxs)
#             )
#         )

#         # run parallel processes
#         batch_logger.begin("calling batch simulator...", stream=True)
#         try:
#             for process in processes:
#                 process.get()
#         except:
#             # log runned jobs
#             if self.job_tracks:
#                 job_tracks = sorted(
#                     self.job_tracks,
#                     key=lambda x: int(x.split("running #")[-1].split(" ")[0]),
#                 )
#                 batch_logger.info("".join(job_tracks).rstrip(), stream=False)

#             # log exception
#             batch_logger.critical("".join(format_exc()).rstrip(), stream=False)
#             exit_code = 1
#             raise
#         else:
#             # log runned jobs
#             job_tracks = sorted(
#                 self.job_tracks,
#                 key=lambda x: int(x.split("running #")[-1].split(" ")[0]),
#             )
#             batch_logger.info("".join(job_tracks).rstrip(), stream=False)

#             exit_code = 0
#         finally:
#             batch_logger.end(f"batch simulator finished with exit code {exit_code}")

#     # process outputs per batch
#     batch_logger.begin("calling batch results collector...", stream=True)
#     try:
#         self.collect_batch_results(
#             self.job_idx_uid_pairs, self.runtime_infos, batch_logger
#         )
#     except:
#         batch_logger.critical("".join(format_exc()).rstrip(), stream=False)
#         exit_code = 1
#         raise
#     else:
#         exit_code = 0
#     finally:
#         batch_logger.end(f"batch simulator finished with exit code {exit_code}")
