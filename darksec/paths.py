"""Data roots for the analyses in tests/ and for config.yaml.

Nothing in this repository hard-codes a machine path. The roots are read from environment variables, or from an
untracked `paths.env` file in the repository root (KEY=VALUE lines, `~` allowed), with the defaults below:

    DARKSEC_RAW_ROOT       folder holding the Leica project folders (<name>--Project001/...)   default ~/Leica_rawdata
    DARKSEC_DS_ROOT        darksec output root (config.yaml `output_root`)                    default ~/Leica_DS
    DARKSEC_DS_TELO_ROOT   output root of the sgTelo (IDT) live-movie run                     default ~/Leica_DS_sgTelo_IDT
    DARKSEC_PROJECTS_ROOT  folder holding sibling analysis projects             default ~/projects

config.yaml may use `${DARKSEC_RAW_ROOT}` etc.; darksec.batch.load_config expands them.
"""
from __future__ import annotations
import os
from pathlib import Path

_DEFAULTS = {"DARKSEC_RAW_ROOT": "~/Leica_rawdata", "DARKSEC_DS_ROOT": "~/Leica_DS", "DARKSEC_DS_TELO_ROOT": "~/Leica_DS_sgTelo_IDT", "DARKSEC_PROJECTS_ROOT": "~/projects"}


def _load_env_file() -> None:
    p = Path(__file__).resolve().parent.parent / "paths.env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env_file()


def root(var: str) -> Path:
    return Path(os.path.expandvars(os.environ.get(var, _DEFAULTS[var]))).expanduser()


def expand(s: str) -> str:
    """Expand ${DARKSEC_*} (and any other) environment variables and `~` in a path string."""
    for k, d in _DEFAULTS.items():
        os.environ.setdefault(k, d)
    return os.path.expanduser(os.path.expandvars(str(s)))


RAW_ROOT = root("DARKSEC_RAW_ROOT")
DS_ROOT = root("DARKSEC_DS_ROOT")
DS_TELO_ROOT = root("DARKSEC_DS_TELO_ROOT")
PROJECTS_ROOT = root("DARKSEC_PROJECTS_ROOT")
