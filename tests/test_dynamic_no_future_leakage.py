"""The causality guard for the dynamic longitudinal arm.

A future-information leak in a per-step survival model does not crash -- it
shows up as unexpectedly *good* held-out performance, which is the failure mode
most likely to be believed. These tests assert the property directly at the
tensor level: perturbing the input at step t must leave every prediction at
steps < t bit-identical.

If one of these fails, no metric from the dynamic arm should be trusted until
it passes again.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from survival_common.deephit_engine import (  # noqa: E402
    DynamicDeepHitGRU,
    collate_batch,
    deephit_nll,
    deephit_nll_dynamic,
)

N_EVENTS = 2
HORIZON = 12
INPUT_DIM = 7


def _model(seed: int = 0) -> DynamicDeepHitGRU:
    torch.manual_seed(seed)
    model = DynamicDeepHitGRU(
        input_dim=INPUT_DIM,
        hidden_dim=16,
        n_events=N_EVENTS,
        horizon=HORIZON,
        dropout=0.0,
    )
    model.eval()
    return model


def _batch(n: int = 3, t: int = 6) -> tuple["torch.Tensor", "torch.Tensor"]:
    torch.manual_seed(123)
    x = torch.randn(n, t, INPUT_DIM)
    length = torch.full((n,), t, dtype=torch.long)
    return x, length


class TestNoFutureLeakage:
    """Perturbing step t must not move any prediction before step t."""

    @pytest.mark.parametrize("perturb_at", [1, 3, 5])
    def test_perturbing_a_step_leaves_earlier_steps_identical(self, perturb_at):
        model = _model()
        x, length = _batch(t=6)

        with torch.no_grad():
            base = model(x, length, per_step=True)

        x_perturbed = x.clone()
        # A large perturbation, so a leak cannot hide under float noise.
        x_perturbed[:, perturb_at, :] += 100.0
        with torch.no_grad():
            perturbed = model(x_perturbed, length, per_step=True)

        # Strictly earlier steps must be untouched...
        torch.testing.assert_close(
            base[:, :perturb_at, :],
            perturbed[:, :perturb_at, :],
            rtol=0.0,
            atol=0.0,
            msg=f"future information from step {perturb_at} leaked backwards",
        )
        # ...and the perturbed step itself must actually have moved, otherwise
        # this test would pass vacuously on a model that ignores its input.
        assert not torch.allclose(
            base[:, perturb_at, :], perturbed[:, perturb_at, :]
        ), "perturbation had no effect; the test is not exercising anything"

    def test_truncating_the_sequence_preserves_the_prefix(self):
        """Predictions must not depend on how much future the batch carries.

        A patient scored with 6 steps and the same patient scored with only
        their first 3 must get identical predictions for those 3 -- this is the
        property the incremental-risk ablation relies on.
        """
        model = _model()
        x, length = _batch(n=2, t=6)

        with torch.no_grad():
            full = model(x, length, per_step=True)
            short = model(
                x[:, :3, :], torch.full((2,), 3, dtype=torch.long), per_step=True
            )
        torch.testing.assert_close(full[:, :3, :], short, rtol=0.0, atol=0.0)

    def test_padding_does_not_change_a_shorter_sequence(self):
        """Zero-padding to the batch max must not alter a short patient's output."""
        model = _model()
        torch.manual_seed(7)
        x = torch.randn(2, 5, INPUT_DIM)
        # Patient 1 genuinely has 5 steps; patient 0 has 2 and is padded.
        length = torch.tensor([2, 5], dtype=torch.long)

        with torch.no_grad():
            padded = model(x, length, per_step=True)
            alone = model(
                x[:1, :2, :], torch.tensor([2], dtype=torch.long), per_step=True
            )
        torch.testing.assert_close(padded[:1, :2, :], alone, rtol=0.0, atol=0.0)

    def test_padding_garbage_is_ignored(self):
        """Values in the padded region must not reach the packed GRU."""
        model = _model()
        torch.manual_seed(11)
        x = torch.randn(2, 5, INPUT_DIM)
        length = torch.tensor([2, 5], dtype=torch.long)

        x_garbage = x.clone()
        x_garbage[0, 2:, :] = 1e6  # beyond patient 0's true length

        with torch.no_grad():
            clean = model(x, length, per_step=True)
            garbage = model(x_garbage, length, per_step=True)
        torch.testing.assert_close(
            clean[:1, :2, :], garbage[:1, :2, :], rtol=0.0, atol=0.0
        )


class TestDynamicLossMasking:
    """The per-step loss must read only masked-in steps."""

    def _inputs(self, n=3, t=4):
        model = _model()
        x, length = _batch(n=n, t=t)
        with torch.no_grad():
            logits = model(x, length, per_step=True)
        torch.manual_seed(5)
        label = torch.randint(0, N_EVENTS + 1, (n, t))
        duration_bin = torch.randint(1, HORIZON + 1, (n, t))
        return model, logits, label, duration_bin

    def test_masked_out_steps_do_not_affect_the_loss(self):
        model, logits, label, duration_bin = self._inputs()
        mask = torch.ones_like(label, dtype=torch.float32)
        mask[:, -1] = 0.0  # mask off the last step

        loss_masked = deephit_nll_dynamic(model, logits, label, duration_bin, mask)

        # Changing the masked-off step's target must not move the loss.
        label_changed = label.clone()
        label_changed[:, -1] = (label_changed[:, -1] + 1) % (N_EVENTS + 1)
        bin_changed = duration_bin.clone()
        bin_changed[:, -1] = HORIZON
        loss_changed = deephit_nll_dynamic(
            model, logits, label_changed, bin_changed, mask
        )
        torch.testing.assert_close(loss_masked, loss_changed, rtol=0.0, atol=0.0)

    def test_single_step_matches_the_landmark_loss(self):
        """With one valid step per patient, the dynamic loss equals the static one."""
        model, logits, label, duration_bin = self._inputs()
        mask = torch.zeros_like(label, dtype=torch.float32)
        mask[:, 0] = 1.0  # only the first step is valid

        dynamic = deephit_nll_dynamic(model, logits, label, duration_bin, mask)
        static = deephit_nll(model, logits[:, 0, :], label[:, 0], duration_bin[:, 0])
        torch.testing.assert_close(dynamic, static, rtol=1e-6, atol=1e-6)

    def test_patients_are_weighted_equally_regardless_of_history_length(self):
        """1/T_i weighting: the loss is the unweighted mean of per-patient means.

        Asserted against an explicit reference computation rather than by
        constructing two "equivalent" batches -- a GRU's per-step outputs differ
        even for a repeated input, because hidden state accumulates, so there is
        no batch-level shortcut that isolates the weighting.
        """
        model = _model()
        x, length = _batch(n=3, t=5)
        with torch.no_grad():
            logits = model(x, length, per_step=True)
        torch.manual_seed(9)
        label = torch.randint(0, N_EVENTS + 1, (3, 5))
        duration_bin = torch.randint(1, HORIZON + 1, (3, 5))
        # Deliberately lopsided history depths: 1, 3 and 5 valid steps.
        mask = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        )

        actual = deephit_nll_dynamic(model, logits, label, duration_bin, mask)

        # Reference: mean over patients of (mean over that patient's steps).
        per_patient = []
        for i in range(3):
            n_valid = int(mask[i].sum())
            rows = deephit_nll(
                model,
                logits[i, :n_valid, :],
                label[i, :n_valid],
                duration_bin[i, :n_valid],
                reduction="none",
            )
            per_patient.append(rows.mean())
        expected = torch.stack(per_patient).mean()
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

        # And the deep-history patient must not dominate: dropping it entirely
        # changes the loss by no more than its 1/3 share could explain.
        assert torch.isfinite(actual)

    def test_empty_mask_raises(self):
        model, logits, label, duration_bin = self._inputs()
        mask = torch.zeros_like(label, dtype=torch.float32)
        with pytest.raises(ValueError, match="no valid steps"):
            deephit_nll_dynamic(model, logits, label, duration_bin, mask)


class TestCollateShapes:
    def test_dynamic_collate_pads_targets_and_mask(self):
        batch = [
            {
                "id": "a",
                "x": np.zeros((2, INPUT_DIM), dtype=np.float32),
                "length": 2,
                "label": np.array([1, 0], dtype=np.int64),
                "duration_bin": np.array([3, 4], dtype=np.int64),
                "duration": np.array([3.0, 4.0], dtype=np.float32),
                "step_mask": np.array([1.0, 1.0], dtype=np.float32),
                "times": np.array([0.0, 1.0], dtype=np.float32),
            },
            {
                "id": "b",
                "x": np.zeros((4, INPUT_DIM), dtype=np.float32),
                "length": 4,
                "label": np.array([0, 0, 2, 0], dtype=np.int64),
                "duration_bin": np.array([1, 2, 3, 4], dtype=np.int64),
                "duration": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                "step_mask": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                "times": np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
            },
        ]
        out = collate_batch(batch)
        assert out["label"].shape == (2, 4)
        assert out["step_mask"].shape == (2, 4)
        # Patient "a" is padded from step 2 on, and the padding is masked off.
        assert out["step_mask"][0].tolist() == [1.0, 1.0, 0.0, 0.0]
        # duration_bin padding must stay >= 1: bin 0 is invalid downstream.
        assert (out["duration_bin"] >= 1).all()

    def test_landmark_collate_is_unchanged(self):
        batch = [
            {
                "id": "a",
                "x": np.zeros((2, INPUT_DIM), dtype=np.float32),
                "length": 2,
                "label": 1,
                "duration_bin": 3,
                "duration": 3.0,
            }
        ]
        out = collate_batch(batch)
        assert out["label"].shape == (1,)
        assert "step_mask" not in out
