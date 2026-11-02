"""Missing proprietary fonts must not stop remote figure generation."""
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
import pytest

from survival_common import plotting


@pytest.fixture(autouse=True)
def restore_font_settings():
    original = plotting.RCPARAMS.copy()
    plotting.require_figure_font.cache_clear()
    yield
    plotting.RCPARAMS.clear()
    plotting.RCPARAMS.update(original)
    plotting.require_figure_font.cache_clear()
    plt.close("all")


@pytest.mark.parametrize("available,expected", [
    ("Arial", "Arial"),
    ("Liberation Sans", "Liberation Sans"),
    ("DejaVu Sans", "DejaVu Sans"),
    (None, "DejaVu Sans"),
])
def test_preference_and_one_substitution_notice(monkeypatch, available, expected):
    default_path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans"))
    attempts = []

    def findfont(prop, *, fallback_to_default=True):
        if prop.get_file():
            return prop.get_file()
        family = prop.get_family()[0]
        attempts.append(family)
        if fallback_to_default:
            return default_path
        if family == available:
            return "/fonts/available.ttf"
        raise ValueError("Font not installed")

    monkeypatch.setattr(font_manager, "findfont", findfont)
    imported_params = plotting.RCPARAMS
    original_sizes = {key: value for key, value in imported_params.items() if "size" in key}
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter("always")
        assert plotting.require_figure_font() == expected
        assert plotting.require_figure_font() == expected
    assert attempts[0] == "Arial"
    assert len(notices) == (0 if expected == "Arial" else 1)
    assert imported_params is plotting.RCPARAMS
    assert imported_params["font.family"] == expected
    assert imported_params["mathtext.rm"] == expected
    assert imported_params["mathtext.it"] == expected + ":italic"
    assert imported_params["mathtext.bf"] == expected + ":bold"
    assert {key: value for key, value in imported_params.items() if "size" in key} == original_sizes


def test_clinical_comparison_renders_without_arial(monkeypatch, tmp_path):
    from COMPASS.survival_analysis import federated_clinical_comparison as fcc

    original_findfont = font_manager.findfont

    def no_arial(prop, **kwargs):
        if prop.get_family()[0] in ("Arial", "Liberation Sans"):
            raise ValueError("Font not installed on remote host")
        return original_findfont(prop, **kwargs)

    monkeypatch.setattr(font_manager, "findfont", no_arial)
    monkeypatch.setattr(plt, "show", lambda: None)
    comparison = pd.DataFrame([
        dict(model=model, subpopulation="gleason", endpoint="platinum",
             landmark_days=180, bundle="Cox", mean_auc=.72,
             mean_auc_lo=.65, mean_auc_hi=.8, n_patients=100, n_events=15)
        for model in (fcc.TRANSFERRED_LABS, fcc.CLINICAL_COX, fcc.COMBINED_COX)
    ])
    with pytest.warns(RuntimeWarning, match="using DejaVu Sans"):
        fcc.plot_clinical_comparison(comparison, tmp_path / "comparison.png", ("platinum",))
    assert (tmp_path / "comparison.png").read_bytes().startswith(b"\x89PNG")
    ax = plt.gcf().axes[0]
    assert ax.yaxis.label.get_family() == ["DejaVu Sans"]
    assert ax.yaxis.label.get_size() == plotting.FONT_SIZES["axis_title"]
    ax.set_ylabel(r"$-\log_{10}(q)$")
    with plt.rc_context(plotting.RCPARAMS):
        plt.gcf().savefig(tmp_path / "comparison.pdf")
    fonts = re.findall(rb"/BaseFont\s*/([^\s/]+)", (tmp_path / "comparison.pdf").read_bytes())
    assert fonts and all(b"DejaVuSans" in font for font in fonts)
