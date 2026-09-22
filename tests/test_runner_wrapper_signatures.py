"""Project wrappers must accept every kwarg the shared runners pass them.

The shared runners in survival_common/cox_runners.py call `cox.<fn>(...)`,
where `cox` is a project's cox_aggregated module. Each project wraps the
shared implementation to bind its own ENDPOINTS/ID_COL/AGE_COL. When a new
parameter is threaded through the shared runner but not added to those
wrappers, nothing fails until the arm is actually run -- on the cluster, an
hour into a fit, with a TypeError about an unexpected keyword argument.

That happened with `restrict_to_canonical_labs`. These tests compare the
runner's call sites against the wrappers' signatures statically, so the next
one is caught at test time instead.
"""

import ast
import inspect
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "survival_common" / "cox_runners.py"

# The project modules the shared runner is invoked with, as `cox`.
WRAPPER_PATHS = {
    "COMPASS": REPO_ROOT / "COMPASS" / "survival_analysis" / "cox_aggregated.py",
    "IPIO": REPO_ROOT / "IPIO" / "survival_analysis" / "cox_aggregated.py",
}

# Functions the runner calls on the `cox` module that each project wraps.
WRAPPED_FUNCTIONS = (
    "tune_multivariable_model",
    "fit_final_multivariable_model",
    "compute_out_of_fold_risk_scores",
)


def _module_functions(path: Path) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(path.read_text())
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _accepted_kwargs(func: ast.FunctionDef) -> set[str]:
    args = func.args
    if args.kwarg is not None:
        # **kwargs forwards anything; nothing to check.
        return set()
    return {a.arg for a in (*args.args, *args.kwonlyargs)}


def _runner_call_kwargs(fn_name: str) -> set[str]:
    """Keyword names the runner passes to `cox.<fn_name>(...)`, across call sites."""
    tree = ast.parse(RUNNER_PATH.read_text())
    passed: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == fn_name
            and isinstance(func.value, ast.Name)
            and func.value.id == "cox"
        ):
            passed.update(kw.arg for kw in node.keywords if kw.arg is not None)
    return passed


@pytest.mark.parametrize("fn_name", WRAPPED_FUNCTIONS)
def test_runner_calls_are_found(fn_name):
    """Guard the guard: if the runner stops calling these, this file is stale."""
    assert _runner_call_kwargs(fn_name), (
        f"No `cox.{fn_name}(...)` call found in {RUNNER_PATH.name}; "
        f"update WRAPPED_FUNCTIONS."
    )


@pytest.mark.parametrize("project,path", WRAPPER_PATHS.items())
@pytest.mark.parametrize("fn_name", WRAPPED_FUNCTIONS)
def test_wrapper_accepts_every_kwarg_the_runner_passes(project, path, fn_name):
    functions = _module_functions(path)
    assert fn_name in functions, f"{project} has no {fn_name} wrapper"
    accepted = _accepted_kwargs(functions[fn_name])
    if not accepted:
        return  # **kwargs
    missing = _runner_call_kwargs(fn_name) - accepted
    assert not missing, (
        f"{project}'s {fn_name} does not accept {sorted(missing)}, which "
        f"survival_common/cox_runners.py passes. The arm will raise TypeError "
        f"at run time."
    )


@pytest.mark.parametrize("project,path", WRAPPER_PATHS.items())
@pytest.mark.parametrize("fn_name", WRAPPED_FUNCTIONS)
def test_wrapper_forwards_every_kwarg_it_accepts(project, path, fn_name):
    """A parameter accepted but not forwarded is silently ignored.

    That is worse than a TypeError: the fit runs and quietly uses the shared
    default, so a caller asking to disable the canonical-lab gate would get it
    left on with no error at all.
    """
    functions = _module_functions(path)
    wrapper = functions[fn_name]
    accepted = _accepted_kwargs(wrapper)
    if not accepted:
        return
    forwarded: set[str] = set()
    for node in ast.walk(wrapper):
        if isinstance(node, ast.Call):
            forwarded.update(kw.arg for kw in node.keywords if kw.arg is not None)
    # Positional-only pass-through (the frame itself) and self-evident names.
    checkable = {a for a in accepted if a not in {"train_val", "test", "cohort"}}
    missing = checkable - forwarded
    assert not missing, (
        f"{project}'s {fn_name} accepts {sorted(missing)} but never forwards "
        f"them to the shared implementation; the value would be silently dropped."
    )


@pytest.mark.parametrize("project", sorted(WRAPPER_PATHS))
def test_multivariate_cli_registers_the_out_of_fold_flags(project):
    """Each project builds its own parser, then calls the shared runner.

    The runner reads args.out_of_fold_risks / args.oof_outer_folds /
    args.oof_inner_folds. A project whose parser never registers them would
    raise AttributeError, or -- worse, given the getattr default -- silently
    skip OOF scoring while appearing to succeed.
    """
    path = REPO_ROOT / project / "survival_analysis" / "multivariate_analysis.py"
    tree = ast.parse(path.read_text())
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "add_out_of_fold_risk_args" in called, (
        f"{project}'s multivariate_analysis.py never calls "
        f"add_out_of_fold_risk_args, so --out-of-fold-risks is unavailable "
        f"there even though the shared runner reads those args."
    )


def test_runner_reads_only_out_of_fold_args_the_helper_registers():
    """The args the runner reads and the flags the helper adds must agree."""
    import argparse

    from survival_common.cox_runners import add_out_of_fold_risk_args

    parser = argparse.ArgumentParser()
    add_out_of_fold_risk_args(parser)
    registered = set(vars(parser.parse_args([])))

    runner_src = RUNNER_PATH.read_text()
    tree = ast.parse(runner_src)
    read: set[str] = set()
    for node in ast.walk(tree):
        # args.oof_outer_folds
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "args"
            and node.attr.startswith("oof_")
        ):
            read.add(node.attr)
        # getattr(args, "out_of_fold_risks", False)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "args"
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
            and node.args[1].value.startswith(("oof_", "out_of_fold"))
        ):
            read.add(node.args[1].value)

    assert read, "runner no longer reads any out-of-fold args; this test is stale"
    missing = read - registered
    assert not missing, (
        f"cox_runners.py reads {sorted(missing)} but add_out_of_fold_risk_args "
        f"does not register them."
    )


def test_restrict_to_canonical_labs_is_wired_end_to_end():
    """The specific regression: the flag must reach the shared implementation."""
    from survival_common import cox_models

    shared = inspect.signature(cox_models.tune_multivariable_model)
    assert "restrict_to_canonical_labs" in shared.parameters
    assert shared.parameters["restrict_to_canonical_labs"].default is True

    for project, path in WRAPPER_PATHS.items():
        wrapper = _module_functions(path)["tune_multivariable_model"]
        assert "restrict_to_canonical_labs" in _accepted_kwargs(wrapper), project
