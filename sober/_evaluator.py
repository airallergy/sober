from __future__ import annotations

from typing import TYPE_CHECKING

import sober.config as cf
from sober._tools import _Parallel

if TYPE_CHECKING:
    from pathlib import Path

    from sober._io_managers import _InputManager, _OutputManager
    from sober._typing import AnyBatch, AnyCtrlKeyVec


def _evaluate(
    *ctrl_key_vecs: AnyCtrlKeyVec,
    input_manager: _InputManager,
    output_manager: _OutputManager,
    batch_dir: Path,
) -> AnyBatch:
    batch = input_manager._job_items(*ctrl_key_vecs)

    with _Parallel(
        cf._config["n_processes"], cf._set_config, (cf._config,)
    ) as parallel:
        input_manager._make_batch(batch_dir, batch, parallel)

        input_manager._simulate_batch(batch_dir, batch, parallel)

        output_manager._scan_batch(batch_dir, batch, parallel)

        if not output_manager._removes_subdirs:
            output_manager._clean_batch(batch_dir, batch, parallel)

    return batch
