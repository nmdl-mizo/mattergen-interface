# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Protocol, Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from pymatgen.core import Composition

from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.utils.data_utils import (
    compute_lattice_polar_decomposition,
    get_element_symbol,
)
from mattergen.common.utils.globals import MAX_ATOMIC_NUM


class Transform(Protocol):
    def __call__(self, sample: ChemGraph) -> ChemGraph:
        ...

class SitePropertyTransform(Protocol):
    def get_site_property_entry(self, samples: list[dict[str, float]], num_samples: int) -> tuple[str, NDArray]:
        ...

def symmetrize_lattice(sample: ChemGraph) -> ChemGraph:
    return sample.replace(cell=compute_lattice_polar_decomposition(sample.cell))


def set_chemical_system(sample: ChemGraph) -> ChemGraph:
    chemsys = (
        torch.eye(MAX_ATOMIC_NUM + 1, device=sample.atomic_numbers.device)[
            sample.atomic_numbers
        ].sum(0)
        > 0
    ).float()[None]
    return sample.replace(chemical_system=chemsys)


def set_chemical_system_string(sample: ChemGraph) -> ChemGraph:
    return sample.replace(
        chemical_system=Composition(
            {get_element_symbol(Z=i.item()): 1 for i in sample.atomic_numbers}
        ).chemical_system
    )


class SetProperty:
    def __init__(self, property_name: str, value: float | Sequence[str]):
        self.property_name = property_name
        self.value = (
            torch.tensor(value, dtype=torch.float)
            if isinstance(value, float) or isinstance(value, int)
            else value
        )

    def __call__(self, sample: ChemGraph) -> ChemGraph:
        return sample.replace(**{self.property_name: self.value})

class SetIntercalantOriginProperty:
    def __init__(self):
        self.property_name = "intercalant_origin"

    def get_site_property_entry(self, samples: list[dict[str, float]], num_samples: int) -> tuple[str, NDArray]:
        assert len(samples) == num_samples, "Number of samples must match the number of samples in the distribution"
        assert all(("intercalant" in sample) and (sample["intercalant"] <= sample["n"]) for sample in samples), "intercalant must be less than or equal to n"
        intercalant_origin = np.concatenate([[1] * sample["intercalant"] + [0] * (sample["n"] - sample["intercalant"]) for sample in samples], axis=-1)
        return self.property_name, intercalant_origin


class SetSurfaceProperty:
    def __init__(self):
        self.property_name = "surface"

    def get_site_property_entry(
        self, samples: list[dict[str, float]], num_samples: int
    ) -> tuple[str, NDArray]:
        assert (
            len(samples) == num_samples
        ), "Number of samples must match the number of samples in the distribution"
        assert all(
            ("surface" in sample) and (sample["surface"] <= sample["n"]) for sample in samples
        ), "surface must be less than or equal to n"
        surface = np.concatenate(
            [
                [1] * sample["surface"] + [0] * (sample["n"] - sample["surface"])
                for sample in samples
            ],
            axis=-1,
        )
        return self.property_name, surface

class SetInterfaceProperty:
    def __init__(self):
        self.property_name = "interface"

    def get_site_property_entry(
        self, samples: list[dict[str, float]], num_samples: int
    ) -> tuple[str, NDArray]:
        assert (
            len(samples) == num_samples
        ), "Number of samples must match the number of samples in the distribution"
        assert all(
            ("interface" in sample) and (sample["interface"] <= sample["n"]) for sample in samples
        ), "interface must be less than or equal to n"
        interface = np.concatenate(
            [
                [1] * sample["interface"] + [0] * (sample["n"] - sample["interface"])
                for sample in samples
            ],
            axis=-1,
        )
        return self.property_name, interface