# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Note that importing this module has two side effects:
1. It sets the environment variable `PROJECT_ROOT` to the root of the explorers project.
2. It registers a new resolver for OmegaConf, `eval`, which allows us to use `eval` in our config files.
"""
import os
from functools import lru_cache
from pathlib import Path

import torch
from omegaconf import OmegaConf


@lru_cache
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@lru_cache
def get_pyg_device() -> torch.device:
    """
    Some operations of pyg don't work on MPS, so fall back to CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


MODELS_PROJECT_ROOT = Path(__file__).resolve().parents[2]
print(f"MODELS_PROJECT_ROOT: {MODELS_PROJECT_ROOT}")

# Set environment variable PROJECT_ROOT so that hydra / OmegaConf can access it.
os.environ["PROJECT_ROOT"] = str(MODELS_PROJECT_ROOT)  # for hydra

DEFAULT_SAMPLING_CONFIG_PATH = Path(__file__).resolve().parents[3] / "sampling_conf"
PROPERTY_SOURCE_IDS = [
    "total_energy_eV",                  # total energy
    "trimmed_total_energy_eV",        # trimmed total energy (total energy of a two-layer structure including the surface; depending on the dataset this one is selected)
    "formation_energy_per_atom",        # formation energy
    "chemical_system",                  # chemical composition
    "host_chemical_system",             # chemical composition (host compound)
    "intercalant_chemical_system",      # chemical composition (intercalant atoms)
    "space_group",                      # space group
    "dft_band_gap",                     # DFT band gap
    "dft_mag_density",                  # magnetic density
    "dft_bulk_modulus",                 # bulk modulus
    "dft_shear_modulus",                # shear modulus
    "energy_above_hull",                # energy above hull
    "hhi_score",                        # HHI score
    "ml_bulk_modulus",                  # machine-learned bulk modulus
    "intercalation_energy_per_atom",    # intercalation energy
    "surface_energy_J_m2",             # surface energy
    "interface_energy_J_m2",            # interface energy
    "energy_per_atom_eV",              # energy per atom
    "cnid_x",                          # CNID x coordinate (grain-boundary RBT grid)
    "cnid_y",                          # CNID y coordinate (grain-boundary RBT grid)
    "rbt_x",                           # RBT x displacement (A)
    "rbt_y",                           # RBT y displacement (A)
    "dx_val",                          # GB spacing displacement (A)
    "dx_frac",                         # GB spacing displacement (fraction of dx_min: 0, 0.25, 0.5, 0.75)
    "gamma_site",                      # gamma_site: GB energy from the node_energy near the GB in a 20-atom window (J/m^2)
    "cell_a",                          # lattice a-axis length (A): conditioning directly on cell_a (derived from the dataset, from the structure)
    "facet",                           # facet category (0=111/1=100/2=110): facet token (derived mechanically from the cell convention (a, b, gamma angle))
    "gb_gamma_J_m2",                   # grain-boundary energy (J/m^2): gamma conditioning of the symmetry-constrained model (the same key as GB_GAMMA_PROPERTY_KEY of mattergen.symmetry_constrained.layout is kept)
]
SITE_PROPERTY_SOURCE_IDS = [
    "intercalant_origin",
    "surface",
    "interface",
    "gb_role",                         # grain-boundary role: fixed/primary/replica
    "surface_role",                    # bulk-fixed surface role: FIXED(0)/PRIMARY(1)
    "grain_label",                     # GB grain membership: grain-1(0)/grain-2(1) per-atom
]

SELECTED_ATOMIC_NUMBERS = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    62,
    63,
    64,
    65,
    66,
    67,
    68,
    69,
    70,
    71,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    83,
    84,
    85,
    86,
    87,
    88,
]
MAX_ATOMIC_NUM = 100


# Set `eval` resolver
def try_eval(s):
    """This is a custom resolver for OmegaConf that allows us to use `eval` in our config files
    with the syntax `${eval:'${foo} + ${bar}'}

    See:
    https://omegaconf.readthedocs.io/en/2.3_branch/how_to_guides.html#id1
    """
    try:
        return eval(s)
    except Exception as e:
        print(f"Calling eval on string {s} raised exception {e}")
        raise


OmegaConf.register_new_resolver("eval", try_eval)
