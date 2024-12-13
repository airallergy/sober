from __future__ import annotations

import itertools as it
import os.path
import platform
from typing import TYPE_CHECKING

import pytest

import sober.config as cf

if TYPE_CHECKING:
    from pathlib import Path

    from sober.config import _Config

EXEC_SUFFIX = ".exe" if platform.system() == "Windows" else ""


@pytest.fixture(scope="module")
def ep_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("ep_folder")
    (root / "Energy+.idd").touch()
    (root / f"energyplus{EXEC_SUFFIX}").touch()
    (root / f"EPMacro{EXEC_SUFFIX}").touch()
    (root / f"ExpandObjects{EXEC_SUFFIX}").touch()
    (root / "PostProcess").mkdir()
    (root / "PostProcess" / f"ReadVarsESO{EXEC_SUFFIX}").touch()
    return root


@pytest.fixture(scope="module")
def ep_files(ep_dir: Path) -> _Config:
    return {
        "schema_energyplus": os.path.join(ep_dir, "Energy+.idd"),
        "exec_energyplus": os.path.join(ep_dir, f"energyplus{EXEC_SUFFIX}"),
        "exec_epmacro": os.path.join(ep_dir, f"EPMacro{EXEC_SUFFIX}"),
        "exec_expandobjects": os.path.join(ep_dir, f"ExpandObjects{EXEC_SUFFIX}"),
        "exec_readvars": os.path.join(
            ep_dir, "PostProcess", f"ReadVarsESO{EXEC_SUFFIX}"
        ),
    }


@pytest.fixture(scope="module")
def config_energyplus_kwargs(
    request: pytest.FixtureRequest, ep_dir: Path
) -> dict[str, str]:
    return {
        key: os.path.join(*(ep_dir if item == "ep_dir" else item for item in value))
        for key, value in request.param.items()
    }


@pytest.mark.parametrize(
    ("config_energyplus_kwargs", "expected_result"),
    [
        ({"root": ("ep_dir",)}, [slice(None)]),  # via root
        (
            {
                "schema_energyplus": ("ep_dir", "Energy+.idd"),
                "exec_energyplus": ("ep_dir", f"energyplus{EXEC_SUFFIX}"),
            },
            [slice(2)],  # via schema_energyplus & exec_energyplus
        ),
        (
            {
                "schema_energyplus": ("ep_dir", "Energy+.idd"),
                "exec_energyplus": ("ep_dir", f"energyplus{EXEC_SUFFIX}"),
                "exec_epmacro": ("ep_dir", f"EPMacro{EXEC_SUFFIX}"),
                "exec_readvars": ("ep_dir", "PostProcess", f"ReadVarsESO{EXEC_SUFFIX}"),
            },
            [
                slice(3),
                slice(4, None),
            ],  # via schema_energyplus & exec_energyplus, with optionals
        ),
        (
            {
                "root": ("ep_dir",),
                "schema_energyplus": ("this", "is", "ignored"),
                "exec_expandobjects": ("this", "is", "also", "ignored"),
            },
            [slice(None)],  # via root, ignoring the rest
        ),
    ],
    indirect=["config_energyplus_kwargs"],
)
def test_config_energyplus_pass(
    config_energyplus_kwargs: dict[str, str],
    expected_result: list[slice],
    ep_files: _Config,
) -> None:
    # execute
    cf.config_energyplus(**config_energyplus_kwargs)

    # test
    assert cf._config == dict(
        it.chain.from_iterable(
            tuple(ep_files.items())[item] for item in expected_result
        )
    )

    del cf._config


@pytest.mark.parametrize(
    ("config_energyplus_kwargs", "expected_result"),
    [
        *it.product(
            [
                {"root": ("this", "is", "non-existent")},
                {
                    "schema_energyplus": ("this", "is", "non-existent"),
                    "exec_energyplus": ("this", "is", "unreachable"),
                },
                {
                    "schema_energyplus": ("ep_dir", "Energy+.idd"),
                    "exec_energyplus": ("ep_dir", f"energyplus{EXEC_SUFFIX}"),
                    "exec_epmacro": ("this", "is", "non-existent"),
                },
            ],
            [
                (
                    FileNotFoundError,  # from _tools._parsed_path
                    "endswith",
                    f"not found: '{os.path.join(os.getcwd(), 'this','is','non-existent')}'.",
                )
            ],
        ),
        *it.product(
            [
                {},
                {"schema_energyplus": ("ep_dir", "Energy+.idd")},
                {
                    "exec_readvars": (
                        "ep_dir",
                        "PostProcess",
                        f"ReadVarsESO{EXEC_SUFFIX}",
                    )
                },
            ],
            [
                (
                    ValueError,  # from config.config_energyplus
                    "equals",
                    "one of 'version', 'root' or 'schema_energyplus & exec_energyplus' needs to be provided.",
                )
            ],
        ),
    ],
    indirect=["config_energyplus_kwargs"],
)
def test_config_energyplus_fail(
    config_energyplus_kwargs: dict[str, str],
    expected_result: tuple[type[BaseException], str, str],
) -> None:
    # execute
    exc, op, expected_msg = expected_result
    with pytest.raises(exc) as exc_info:
        cf.config_energyplus(**config_energyplus_kwargs)
    msg = exc_info.value.args[0]

    # test
    if op == "equals":
        assert msg == expected_msg
    elif op == "endswith":
        assert msg.endswith(expected_msg)
    else:
        raise pytest.UsageError
