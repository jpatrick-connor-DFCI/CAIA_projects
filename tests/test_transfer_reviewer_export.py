"""Notebook 04 must retain broad NEPC for notebook 06 without changing its figures."""
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from survival_common import federated_endpoint_transfer as fet


@pytest.mark.parametrize("include_broad", [False, True])
@pytest.mark.parametrize("include_checklist", [False, True])
def test_reviewer_export_preserves_broad_nepc(tmp_path, include_broad, include_checklist):
    notebook = Path(__file__).resolve().parents[1] / "COMPASS/notebooks/manuscript/04_federated_transfer.ipynb"
    cells = ["".join(c["source"]) for c in json.loads(notebook.read_text())["cells"]
             if c["cell_type"] == "code"]
    endpoints = fet.DEFAULT_ENDPOINTS + (("nepc",) if include_broad else ())
    bundles = {"xgb": {}, **({"checklist": {}} if include_checklist else {})}
    calls = []

    def scores(selected_bundles, eps):
        return pd.DataFrame([dict(endpoint=ep, bundle=b, person_id=pid, risk_score=pid / 10)
                             for ep in eps for b in selected_bundles for pid in (1, 2)])

    def read_local(path, eps):
        assert set(eps) == set(fet.DEFAULT_ENDPOINTS) | {"nepc"}
        row = {fet.LOCAL_ID_COL: 1}
        for ep in eps:
            flag, date = fet.ENDPOINT_SOURCES[ep]
            row.update({flag: 1, date: "2020-01-01"})
        return pd.DataFrame([row])

    def evaluate(local, selected_bundles, **kwargs):
        calls.append((set(selected_bundles), kwargs))
        return {"scores": scores(selected_bundles, kwargs["endpoints"])}

    env = dict(pd=pd, fet=SimpleNamespace(read_local_frame=read_local, evaluate_transfer=evaluate,
                                         load_lab_mapping=lambda _: {}, LOCAL_ID_COL=fet.LOCAL_ID_COL,
                                         ENDPOINT_SOURCES=fet.ENDPOINT_SOURCES),
               ENDPOINTS=endpoints, DATA_PATH="unused", LAB_MAPPING="unused", bundles=bundles,
               PRIMARY=["xgb"], BASELINE_FALLBACK={}, ALL_MODELS_DIR=tmp_path,
               TRANSFER_KW=dict(endpoints=endpoints, n_bootstrap=200),
               res={"scores": scores(["xgb"], endpoints)},
               combine=lambda *results: {"scores": pd.concat([r["scores"] for r in results])})
    exec(next(c for c in cells if "local = fet.read_local_frame" in c), env)
    exec(next(c for c in cells if "EXTRA = [s for s in bundles" in c), env)
    exported = pd.read_csv(tmp_path / "transfer_scores.csv")
    assert set(exported.endpoint) == set(fet.DEFAULT_ENDPOINTS) | {"nepc"}
    assert set(exported.bundle) == set(bundles)
    assert not exported.duplicated(["endpoint", "bundle", "person_id"]).any()
    assert set(env["all_res"]["scores"].endpoint) == set(endpoints)
    broad_calls = [kwargs for selected, kwargs in calls if kwargs["endpoints"] == ("nepc",)]
    assert len(broad_calls) == (0 if include_broad else 1)
    if broad_calls:
        assert calls[-1][0] == set(bundles)
        assert broad_calls[0]["n_bootstrap"] == 0
    # The reviewer composite must still have shared AVPC / broad-NEPC patients.
    xgb = exported[exported.bundle.eq("xgb")]
    assert set(xgb.loc[xgb.endpoint.eq("avpc"), "person_id"]) == set(
        xgb.loc[xgb.endpoint.eq("nepc"), "person_id"])
