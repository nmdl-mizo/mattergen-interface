"""Training-time projection mirror for the variable-length, padding-free proposed tilt-STGB diffusion
loop (internal notes
follow-up; created outside the scope of the original design, which stopped at "the actual start of training, generation and evaluation are outside the scope of this design" -- see this module's history note below).
(see the above).

**Why this module exists (measured)**: a training job of the proposed for another system (GPU partition, sbatch
a batch script) FAILED
37 seconds into the sanity-check validation batch with::

    File ".../mattergen/symmetry_constrained/diffusion_module.py", line 50, in _corrupt_batch
        noisy = self._scm_projector(noisy)
    File ".../mattergen/symmetry_constrained/projection.py", line 197, in __call__
        raise ValueError(
    ValueError: FixedSlotProjector: batch atom count 476 is not a multiple of N_PAD=128

(full log: ``workspace/experiments/earlier_experiment/train_stdout_proposed.log``). Root
cause: the proposed ``diffusion_module`` config
(``mattergen/conf/lightning_module/diffusion_module/scm_proposed.yaml``, as authored
Step 35) pointed ``_target_`` at ``mattergen.symmetry_constrained.diffusion_module.FixedSlotDiffusionModule``, whose
``__init__`` unconditionally constructs the FIXED N_PAD=128 slot projector
(``mattergen.symmetry_constrained.projection.FixedSlotProjector``) -- own header comment already
recorded this as a known, deliberately deferred gap ("a later design ... needs to create a training-time
projector wiring that supports variable length"). This module is that follow-up: it is the
``variable_length`` counterpart of ``mattergen.symmetry_constrained.diffusion_module.FixedSlotDiffusionModule``,
mirroring its shape (same ``DiffusionModule`` subclass pattern, only the projector
class differs) but wired to the variable-length, padding-free
``mattergen.symmetry_constrained.variable_length.projection.SymmetryProjector`` instead of the fixed-N_PAD one.

**Transplant source (/C22, read then copied)**:
``mattergen.symmetry_constrained.diffusion_module.FixedSlotDiffusionModule`` (51 lines, read in full before writing
this file).

**Intentional differences from the transplant source (record the reason, do not just
silently drop code)**:

1. The projector is ``mattergen.symmetry_constrained.variable_length.projection.SymmetryProjector`` instead of
   ``mattergen.symmetry_constrained.projection.FixedSlotProjector`` -- this swap is the entire point of this module.
2. ``freeze_atomic_numbers`` (the transplant source's constructor kwarg, added by
   W-1 for the E2 fixed-N padding baseline) is **NOT** carried over --
   this class's ``__init__`` takes no such kwarg and ``_corrupt_batch`` has no clean-label-restore
   branch. Reason: the plan states that the config of that proposed restores "the original
   MatterGen default" (the original, unmodified MatterGen default), and the proposed conf
   (``scm_proposed.yaml``) never sets this key. check-6 additionally requires that the
   resolved Hydra config for the proposed run contain the literal string ``"freeze_atomic_numbers"``
   nowhere at all -- a constructor kwarg that merely defaults to ``False`` would still satisfy that
   AC numerically (the resolved config would show the default), but keeping the parameter around
   would misrepresent the proposed architecture's design intent (a fixed-N padding-only feature revived
   on a padding-free variable-length model), so it is dropped entirely rather than defaulted.
"""

from __future__ import annotations

from mattergen.diffusion.diffusion_module import DiffusionModule
from mattergen.symmetry_constrained.variable_length.projection import SymmetryProjector


class SymmetryConstrainedDiffusionModule(DiffusionModule):
    """``DiffusionModule`` + a ``SymmetryProjector`` mirror applied immediately after
    ``_corrupt_batch`` (mirrors ``mattergen.symmetry_constrained.diffusion_module.FixedSlotDiffusionModule``'s role
    for the fixed N_PAD=128 layout, applied here to the variable-length,
    padding-free proposed layout instead -- follow-up). Restores
    FIXED_A/FIXED_B (corrupted by the pos SDE) and re-derives REPLICA deterministically from the
    noised PRIMARY + noised ``gb_t``, keeping the score model's training-time input distribution
    consistent with what ``pc_sampler.py``'s generation-time projector call sees.

    Unlike the fixed-length transplant source, this class does **not** accept a
    ``freeze_atomic_numbers`` kwarg (see module docstring "intentional differences" #2).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._scm_projector = SymmetryProjector()

    def _corrupt_batch(self, batch):
        noisy, t = super()._corrupt_batch(batch)
        if "gb_role" not in noisy:
            raise ValueError(
                "SymmetryConstrainedDiffusionModule training batch is missing 'gb_role'"
            )
        noisy = self._scm_projector(noisy)
        return noisy, t
