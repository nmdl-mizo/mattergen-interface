"""Environment-variable based roots for the gb_eval / figures scripts (no default values: a missing variable raises KeyError).

GBREL_DATA   directory with the data bundle (the extracted Zenodo record, contains training/ generation/ relaxation/ evaluation/ dft/ weights/)
GBREL_CODE   root of this repository
GBREL_WORK   writable work directory; `gb_eval/make_compat_tree.py` builds `$GBREL_WORK/workspace/...` (a symlink view that
             mirrors the directory layout the scripts were written for) and new outputs are written there
MACE_MODEL_PATH  path of the MACE-MH-1 model file (not distributed here)
MACE_PY      python interpreter of the evaluation environment (environment-eval.yml)
GEN_PY       python interpreter of the generation environment (environment-gen.txt)
GEN_VENV     root of the generation virtualenv (optional; used by a few sbatch scripts)
MACE_ENV, CONDA_ROOT   optional (only used by sbatch scripts that activate conda)
"""
import os
from pathlib import Path

_ENV = {
    "DATA": "GBREL_DATA", "CODE": "GBREL_CODE", "WORK": "GBREL_WORK", "MACE_MODEL": "MACE_MODEL_PATH",
    "MACE_PY": "MACE_PY", "GEN_PY": "GEN_PY", "GEN_VENV": "GEN_VENV", "MACE_ENV": "MACE_ENV", "CONDA_ROOT": "CONDA_ROOT",
}


def __getattr__(name):
    if name in _ENV:
        var = _ENV[name]
        try:
            return os.environ[var]
        except KeyError:
            raise KeyError("environment variable %s is not set (see docs/reproduce.md)" % var) from None
    if name in ("DATA_ROOT", "CODE_ROOT", "WORK_ROOT"):
        return Path(__getattr__(name[:-5]))
    raise AttributeError(name)
