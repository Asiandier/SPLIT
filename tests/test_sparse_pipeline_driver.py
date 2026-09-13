import argparse
import importlib
import json
import os
from pathlib import Path
import sys

import pytest
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(REPO_ROOT)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)
PKG = importlib.import_module(os.path.basename(REPO_ROOT))
RUN_SPARSE = importlib.import_module(f"{PKG.__name__}.run_sparse_pipeline")
_prepare_combined_fit_inputs = RUN_SPARSE._prepare_combined_fit_inputs


def score_args():
    return ["--mode", "adaptive", "--bed-prefix", "geno", "--pheno-txt", "train.pheno",
            "--validation-pheno-txt", "validation.pheno", "--keep-path", "train.keep",
            "--validation-keep-path", "validation.keep", "--out-prefix", "result"]


def test_trace_score_cli_defaults():
    args = RUN_SPARSE.parse_args(score_args())
    assert args.score_core_rank == 64
    assert args.score_reference_samples == 16383
    assert args.score_trace_probes == 512
    assert args.score_trace_max_probes == 4096
    assert args.score_trace_tol == 0.05
    assert args.score_trace_seed >= 0
    assert not any("bootstrap" in name for name in vars(args))


@pytest.mark.parametrize("accept_first", [False, True])
def test_adaptive_null_refit_precedes_score_and_keeps_joint_endpoint(monkeypatch, tmp_path, accept_first):
    args = RUN_SPARSE.parse_args(score_args())
    args.work_dir = tmp_path
    args.genotype_prefix = tmp_path / "geno"
    args.genotype_format = "bed"
    calls, validation_calls = [], []
    def summary(k, theta, h2):
        return dict(n_grms=k, h2=h2, var_components_lasso_ml=theta,
                    q_chive=.2, genetic_trace_atoms=[1.]*k, support_size=2,
                    lasso_selected_lam_ratio=.1, lasso_selected_validation_r2=.3,
                    outer_iterations=3, outer_stop_reason="converged")
    original = summary(1, [.6,.2], .8)
    def validation(args, *, directory, **kwargs):
        validation_calls.append((directory.name, kwargs))
        directory.mkdir(parents=True, exist_ok=True)
        result = original if directory.name == "k1_validation" else summary(2,[.1,.3,.4],.6)
        prefix, state = directory/"coherit", directory/"sparse_state.npz"
        RUN_SPARSE.atomic_json(Path(str(prefix)+".summary.json"), result)
        return prefix, state, result
    def rank(args):
        path = args.work_dir/"adaptive"/"ld_rank.npz"
        RUN_SPARSE.write_ld_rank_artifact(path, np.arange(8), bins=2)
        return path, {}
    def stage(args, kind, *, component_spec, state_path, extra):
        calls.append((kind, component_spec, state_path, list(extra)))
        return [kind, *extra]
    def run(command, log):
        def value(flag):
            return command[command.index(flag)+1]
        output = Path(value("--out"))
        if command[0] == "frozen-fit":
            initial = json.loads(value("--theta-init-json"))
            k = len(initial)-1
            theta = [.3,.5] if k == 1 else [.1,.3,.4]
            fit = summary(k,theta,.6)
            fit.update(schema_version=1, method="fixed_sparse_mean_covariance_reml",
                       parent_q_sparse_held_fixed=.2)
            RUN_SPARSE.atomic_json(output,fit)
        else:
            parent = Path(value("--summary-path"))
            assert parent.name.endswith(".frozen_fit.json")
            fit = RUN_SPARSE.read_json(parent)
            accepted = accept_first and fit["n_grms"] == 1
            RUN_SPARSE.atomic_json(output, dict(schema_version=3,
                method="joint_quadratic_reml_ld_cusum", accepted=accepted,
                diagnostics=dict(global_p_value=.01 if accepted else .8),
                selected_candidate=dict(boundary_position=4,score_statistic=9.),
                stop_reason="split_signal" if accepted else "global_score_not_significant"))
    monkeypatch.setattr(RUN_SPARSE,"load_variant_records",lambda *args: list(range(8)))
    monkeypatch.setattr(RUN_SPARSE,"_run_validation_fit",validation)
    monkeypatch.setattr(RUN_SPARSE,"_prepare_ld_rank",rank)
    monkeypatch.setattr(RUN_SPARSE,"_adaptive_stage_command",stage)
    monkeypatch.setattr(RUN_SPARSE,"_run_command",run)
    spec,state,endpoint,result = RUN_SPARSE._run_adaptive_selection(args)
    assert [c[0] for c in calls] == (["frozen-fit","score","frozen-fit","score"]
                                    if accept_first else ["frozen-fit","score"])
    assert all(c[2].parent.name == "k1_validation" for c in calls)
    root_args = calls[0][3]
    assert json.loads(root_args[root_args.index("--theta-init-json")+1]) == [.6,.2]
    path = result["fixed_alpha_path"]["path"]
    assert path[0]["stage"] == "fixed_k1_sparse_mean_null_refit"
    assert path[0]["theta"] == [.3,.5]
    assert result["k1_validation"]["theta"] == [.6,.2]
    if accept_first:
        child_args = calls[2][3]
        assert json.loads(child_args[child_args.index("--theta-init-json")+1]) == [.15,.15,.5]
        assert validation_calls[-1][1]["theta_init"] == [.1,.3,.4]
        assert validation_calls[-1][1]["warm_state_in"].parent.name == "k1_validation"
        assert endpoint["n_grms"] == 2
    else:
        assert endpoint is original
        assert len(validation_calls) == 1
    # Completed covariance/score stages must be reused, including the new null.
    monkeypatch.setattr(RUN_SPARSE,"_run_command",lambda *a: pytest.fail("Unexpected repeat fit"))
    RUN_SPARSE._run_adaptive_selection(args)


@pytest.mark.parametrize("option,value", [
    ("--score-trace-probes", "1"), ("--score-trace-max-probes", "64"),
    ("--score-trace-tol", "0"), ("--score-trace-tol", "nan"),
    ("--score-trace-seed", "-1"), ("--bootstrap-draws", "199"),
    ("--score-core-rank", "0"), ("--score-reference-samples", "1"),
    ("--split-alpha", "0.00001"),
])
def test_invalid_trace_configuration_is_rejected(option, value):
    with pytest.raises(SystemExit) as error:
        RUN_SPARSE.parse_args(score_args()+[option, value])
    assert error.value.code == 2


def test_resume_digest_detects_source_changes(monkeypatch, tmp_path: Path) -> None:
    script = tmp_path / "run_sparse_pipeline.py"
    script.write_text("# runner\n")
    algorithm = tmp_path / "adaptive_ld.py"
    algorithm.write_text("# first version\n")
    monkeypatch.setattr(RUN_SPARSE, "__file__", str(script))
    initial = RUN_SPARSE._algorithm_source_digest()
    assert RUN_SPARSE._algorithm_source_digest() == initial
    algorithm.write_text("# changed algorithm\n")
    assert RUN_SPARSE._algorithm_source_digest() != initial


def test_combined_fit_inputs_follow_source_sample_order(tmp_path: Path) -> None:
    prefix = tmp_path / "geno"
    (tmp_path / "geno.fam").write_text(
        "f b 0 0 0 -9\n"
        "f a 0 0 0 -9\n"
        "f d 0 0 0 -9\n"
        "f c 0 0 0 -9\n",
        encoding="utf-8",
    )
    train_keep = tmp_path / "train.keep"
    train_keep.write_text("f a\nf c\n", encoding="utf-8")
    validation_keep = tmp_path / "validation.keep"
    validation_keep.write_text("f b\nf d\n", encoding="utf-8")
    train_pheno = tmp_path / "train.pheno"
    train_pheno.write_text("f a 1\nf c 3\n", encoding="utf-8")
    validation_pheno = tmp_path / "validation.pheno"
    validation_pheno.write_text("f b 2\nf d 4\n", encoding="utf-8")
    args = argparse.Namespace(
        fit_pheno_txt=None,
        fit_keep_path=None,
        keep_path=train_keep,
        validation_keep_path=validation_keep,
        genotype_prefix=prefix,
        genotype_format="bed",
        pheno_txt=train_pheno,
        validation_pheno_txt=validation_pheno,
        work_dir=tmp_path / "work",
    )
    phenotype, keep, constructed = _prepare_combined_fit_inputs(args)
    assert constructed is True
    assert keep.read_text(encoding="utf-8").splitlines() == [
        "b\tb",
        "a\ta",
        "d\td",
        "c\tc",
    ]
    assert phenotype.read_text(encoding="utf-8").splitlines() == [
        "b\tb\t2",
        "a\ta\t1",
        "d\td\t4",
        "c\tc\t3",
    ]


@pytest.fixture
def cohort_inputs(tmp_path):
    prefix = tmp_path / "geno"
    prefix.with_suffix(".fam").write_text("".join(f"f {iid} 0 0 0 -9\n" for iid in "abcde"))
    contents = {
        "train.keep": "f a\nf c\n", "validation.keep": "f b\nf d\n",
        "fit.keep": "f d\nf c\nf b\nf a\n", "test.keep": "f e\n",
        "train.pheno": "f a 1\nf c 3\n", "validation.pheno": "f b 2\nf d 4\n",
        "fit.pheno": "f a 1\nf b 2\nf c 3\nf d 4\n",
    }
    for name, content in contents.items():
        (tmp_path/name).write_text(content)
    return argparse.Namespace(
        fit_pheno_txt=tmp_path/"fit.pheno", fit_keep_path=tmp_path/"fit.keep",
        keep_path=tmp_path/"train.keep", validation_keep_path=tmp_path/"validation.keep",
        genotype_prefix=prefix, genotype_format="bed", pheno_txt=tmp_path/"train.pheno",
        validation_pheno_txt=tmp_path/"validation.pheno", work_dir=tmp_path/"work",
        prediction_prefix=prefix, prediction_format="bed", prediction_keep_path=tmp_path/"test.keep",
    )


def test_explicit_final_inputs_are_validated_without_rewriting(cohort_inputs):
    args = cohort_inputs
    modified = args.fit_pheno_txt.stat().st_mtime_ns
    assert _prepare_combined_fit_inputs(args) == (args.fit_pheno_txt, args.fit_keep_path, False)
    assert args.fit_pheno_txt.stat().st_mtime_ns == modified
    assert not args.work_dir.exists()


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("bad", ["selection_overlap", "prediction_overlap", "prediction_missing",
                                 "own_phenotype_missing"])
def test_cohort_isolation_is_required_with_both_input_modes(cohort_inputs, explicit, bad):
    args = cohort_inputs
    if not explicit:
        args.fit_pheno_txt = args.fit_keep_path = None
    if bad == "selection_overlap":
        args.validation_keep_path.write_text("f a\nf d\n")
        match = "Training and validation.*disjoint"
    elif bad == "prediction_overlap":
        args.prediction_keep_path.write_text("f a\nf e\n")
        match = "Prediction and training.*disjoint"
    elif bad == "prediction_missing":
        args.prediction_keep_path.write_text("f unknown\n")
        match = "Prediction keep ID is absent"
    else:
        args.pheno_txt.write_text("f a 1\n")
        args.validation_pheno_txt.write_text("f b 2\nf c 3\nf d 4\n")
        match = "Training phenotype is missing"
    with pytest.raises(ValueError, match=match):
        _prepare_combined_fit_inputs(args)
    assert not args.work_dir.exists()


@pytest.mark.parametrize("bad", ["missing_id", "extra_id", "missing_value", "changed_value"])
def test_explicit_final_inputs_must_match_selection(cohort_inputs, bad):
    args = cohort_inputs
    if bad == "missing_id":
        args.fit_keep_path.write_text("f a\nf b\nf c\n")
    elif bad == "extra_id":
        args.fit_keep_path.write_text("f a\nf b\nf c\nf d\nf e\n")
    elif bad == "missing_value":
        args.fit_pheno_txt.write_text("f a 1\nf b 2\nf c 3\n")
    else:
        args.fit_pheno_txt.write_text("f a 1\nf b 2\nf c 3\nf d 40\n")
    with pytest.raises(ValueError, match="Final-fit"):
        _prepare_combined_fit_inputs(args)


def test_prediction_without_keep_still_checks_the_actual_sample_set(cohort_inputs):
    cohort_inputs.prediction_keep_path = None
    with pytest.raises(ValueError, match="Prediction and training.*disjoint"):
        _prepare_combined_fit_inputs(cohort_inputs)
