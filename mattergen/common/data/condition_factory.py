# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from functools import partial
from typing import Callable, Iterable, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.data.collate import collate
from mattergen.common.data.dataset import NumAtomsCrystalDataset
from mattergen.common.data.num_atoms_distribution import NUM_ATOMS_DISTRIBUTIONS
from mattergen.common.data.transform import SetProperty, Transform, SitePropertyTransform
from mattergen.common.data.types import TargetProperty
from mattergen.common.utils.data_utils import create_chem_graph_from_composition
from mattergen.diffusion.data.batched_data import BatchedData

ConditionLoader = Iterable[tuple[BatchedData, dict[str, torch.Tensor]] | None]


def _collate_fn(
    batch: Sequence[ChemGraph],
    collate_fn: Callable[[Sequence[ChemGraph]], BatchedData],
) -> tuple[BatchedData, None]:
    return collate_fn(batch), None


def get_number_of_atoms_condition_loader(
    num_atoms_distribution: str,
    num_samples: int,
    batch_size: int,
    shuffle: bool = True,
    transforms: list[Transform] | None = None,
    properties: TargetProperty | None = None,
    set_site_properties: list[SitePropertyTransform] | None = None,
) -> ConditionLoader:
    transforms = transforms or []
    if properties is not None:
        for k, v in properties.items():
            transforms.append(SetProperty(k, v))
    assert (
        num_atoms_distribution in NUM_ATOMS_DISTRIBUTIONS
    ), f"Invalid num_atoms_distribution: {num_atoms_distribution}"
    dataset = NumAtomsCrystalDataset.from_num_atoms_distribution(
        num_atoms_distribution=NUM_ATOMS_DISTRIBUTIONS[num_atoms_distribution],
        num_samples=num_samples,
        transforms=transforms,
        set_site_properties=set_site_properties,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=partial(_collate_fn, collate_fn=collate),
        shuffle=shuffle,
    )


def get_composition_data_loader(
    target_compositions_dict: list[dict[str, float]],
    num_structures_to_generate_per_composition: int,
    batch_size: int,
) -> ConditionLoader:
    """
    Given a list of target compositions, generate a dataset of chemgraphs
    where each chemgraph contains atoms corresponding to the target composition
    without positions or cell information.
    Returns a torch dataloader equipped with the correct collate function containing such dataset.
    """

    dataset_ = []
    for compostion in target_compositions_dict:
        chemgraphs = [
            create_chem_graph_from_composition(compostion)
        ] * num_structures_to_generate_per_composition
        dataset_.extend(chemgraphs)

    dataset = ChemGraphlistDataset(dataset_)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=partial(_collate_fn, collate_fn=collate),
        shuffle=False,
    )


def get_fixed_long_axis_condition_loader(
    num_atoms_distribution: str,
    num_samples: int,
    batch_size: int,
    long_axis_length: float = 30.0,
    long_axis_row: int = 2,
    shuffle: bool = True,
    transforms: list[Transform] | None = None,
    properties: TargetProperty | None = None,
    set_site_properties: list[SitePropertyTransform] | None = None,
) -> ConditionLoader:
    """Like get_number_of_atoms_condition_loader, but fixes one axis to long_axis_length Å.

    long_axis_row: which row of the cell matrix to fix (0=a, 1=b, 2=c).
    After symmetrize_lattice (polar decomposition), training structures have
    the long axis (30 Å) in row 2, so long_axis_row=2 (default).

    The mask fixes cell[:, long_axis_row, :] (the long-axis row) and
    cell[:, other_rows, long_axis_row] (off-diagonal column, for orthogonality).
    The polar decomposition preserves this block-diagonal structure, so the
    mask and corrector are mutually consistent.
    """
    transforms = transforms or []
    if properties is not None:
        for k, v in properties.items():
            transforms.append(SetProperty(k, v))
    assert (
        num_atoms_distribution in NUM_ATOMS_DISTRIBUTIONS
    ), f"Invalid num_atoms_distribution: {num_atoms_distribution}"
    dataset = NumAtomsCrystalDataset.from_num_atoms_distribution(
        num_atoms_distribution=NUM_ATOMS_DISTRIBUTIONS[num_atoms_distribution],
        num_samples=num_samples,
        transforms=transforms,
        set_site_properties=set_site_properties,
    )

    r = long_axis_row

    def collate_with_fixed_axis(batch: Sequence[ChemGraph]) -> tuple[BatchedData, dict[str, torch.Tensor]]:
        data = collate(batch)
        bs = data.get_batch_size()
        # Conditioning cell: long-axis row = [..., long_axis_length, ...] (only diagonal component)
        # others = 0; all will be overridden by mask or denoised freely
        cell = torch.zeros(bs, 3, 3)
        cell[:, r, r] = long_axis_length  # diagonal component of the long axis
        data = data.replace(cell=cell)

        # Mask: 1 = keep fixed, 0 = denoise
        cell_mask = torch.zeros(bs, 3, 3)
        cell_mask[:, r, :] = 1.0                        # fix entire long-axis row
        other_rows = [i for i in range(3) if i != r]
        cell_mask[:, other_rows, r] = 1.0               # fix off-diagonal column (orthogonality)
        mask = {"cell": cell_mask}
        return data, mask

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_with_fixed_axis,
        shuffle=shuffle,
    )


class ChemGraphlistDataset(Dataset):
    def __init__(self, data: list[ChemGraph]) -> None:
        super().__init__()
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> ChemGraph:
        return self.data[index]
