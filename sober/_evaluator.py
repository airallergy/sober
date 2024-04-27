from __future__ import annotations

from typing import TYPE_CHECKING

import sober.config as cf
from sober._tools import _Parallel

if TYPE_CHECKING:
    from pathlib import Path

    from sober._io_managers import _InputManager, _OutputManager
    from sober._typing import AnyCtrlKeyVec


def _evaluate(
    *ctrl_key_vecs: AnyCtrlKeyVec,
    input_manager: _InputManager,
    output_manager: _OutputManager,
    batch_dir: Path,
) -> None:
    batch = input_manager._job_items(*ctrl_key_vecs)

    with _Parallel(
        cf._config["n.processes"], cf._update_config, (cf._config,)
    ) as parallel:
        input_manager._make_batch(batch_dir, batch, parallel)

        input_manager._simulate_batch(batch_dir, batch, parallel)

        output_manager._collect_batch(batch_dir, batch, parallel)

        if not cf._removes_subdirs:  # no need to clean if subdirs are to be removed
            output_manager._clean_batch(batch_dir, batch, parallel)
