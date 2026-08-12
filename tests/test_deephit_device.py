"""Device-selection regression tests for Dynamic-DeepHit."""

from types import SimpleNamespace

from survival_common import deephit_engine


def _fake_torch(cuda_available: bool) -> SimpleNamespace:
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda_available),
    )


def test_select_device_automatically_uses_cuda(monkeypatch):
    monkeypatch.setattr(deephit_engine, "torch", _fake_torch(cuda_available=True))

    assert deephit_engine.select_device() == "cuda"


def test_select_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(deephit_engine, "torch", _fake_torch(cuda_available=False))

    assert deephit_engine.select_device() == "cpu"
