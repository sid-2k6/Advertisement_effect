"""
Smoke test for DeepBudgetVis_Benchmarks_*.ipynb

Purpose: prove the notebook's CODE PATHS execute correctly end-to-end
(discovery -> validation -> models -> training -> checkpoint -> resume ->
best-checkpoint test eval -> CSVs -> plots -> manifest).

This uses RANDOM SYNTHETIC data. It produces NO research results and its
numbers are meaningless by construction. It exists only to verify that the
notebook runs without error and emits every expected artifact.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import nbformat

NB = Path("../DeepBudgetVis_Benchmarks_CNNLSTM_TCNLSTM_LSTMmTransMLP.ipynb")
SMOKE = Path("/tmp/smoke_bench")
DATA = SMOKE / "preprocessed_data"


def make_synthetic_csv_dataset(n_train=400, n_val=140, n_test=140, n_feat=12, seed=0):
    """Synthetic multivariate series with the two required target names."""
    rng = np.random.default_rng(seed)
    if DATA.exists():
        shutil.rmtree(DATA)
    DATA.mkdir(parents=True)

    total = n_train + n_val + n_test
    t = np.arange(total)
    feats = rng.normal(0, 1, size=(total, n_feat)).astype(np.float64)
    feats[:, 0] = np.sin(t / 11.0)
    feats[:, 1] = np.cos(t / 7.0)

    # targets loosely driven by features + trend (values in "financial" magnitude)
    rev = 150_000 + 20_000 * feats[:, 0] + 8_000 * feats[:, 2] + 40 * t + rng.normal(0, 3_000, total)
    exp = 100_000 + 9_000 * feats[:, 1] + 5_000 * feats[:, 3] + 25 * t + rng.normal(0, 2_000, total)

    cols = {f"feat_{i}": feats[:, i] for i in range(n_feat)}
    cols["NET_PATIENT_REVENUE"] = rev
    cols["TOT_OVERALL_EXP"] = exp
    df = pd.DataFrame(cols)

    df.iloc[:n_train].to_csv(DATA / "train.csv", index=False)
    df.iloc[n_train:n_train + n_val].to_csv(DATA / "val.csv", index=False)
    df.iloc[n_train + n_val:].to_csv(DATA / "test.csv", index=False)
    print(f"[smoke] synthetic CSV dataset written to {DATA} "
          f"({n_train}/{n_val}/{n_test} rows, {n_feat} features)")


def make_synthetic_npz_dataset(seq_len=16, n_feat=9, seed=1):
    """Pre-sequenced .npz layout, to exercise Layout A."""
    rng = np.random.default_rng(seed)
    if DATA.exists():
        shutil.rmtree(DATA)
    DATA.mkdir(parents=True)
    def blk(n):
        X = rng.normal(0, 1, (n, seq_len, n_feat)).astype(np.float32)
        y = np.stack([
            150_000 + 10_000 * X[:, -1, 0] + rng.normal(0, 2_000, n),
            100_000 + 6_000 * X[:, -1, 1] + rng.normal(0, 1_500, n),
        ], axis=1).astype(np.float32)
        return X, y
    Xtr, ytr = blk(300); Xva, yva = blk(100); Xte, yte = blk(100)
    np.savez(DATA / "sequences.npz",
             X_train=Xtr, y_train=ytr, X_val=Xva, y_val=yva, X_test=Xte, y_test=yte)
    print(f"[smoke] synthetic NPZ dataset written to {DATA} (seq_len={seq_len}, F={n_feat})")


OVERRIDES = """
BASE_DIR   = "/tmp/smoke_bench"
DATA_DIR   = f"{BASE_DIR}/preprocessed_data"
OUTPUT_DIR = f"{BASE_DIR}/baseline_outputs_v1"
SEQUENCE_LENGTH = 16
BATCH_SIZE = 32
EPOCHS = %(epochs)d
PATIENCE = 99
NUM_WORKERS = 0
USE_AMP = False
RESUME = %(resume)s
SCHED_PATIENCE = 1
print("[smoke] configuration overridden for smoke test")
"""


def run_notebook(epochs, resume, tag):
    nb = nbformat.read(NB, as_version=4)
    code_cells = [c.source for c in nb.cells if c.cell_type == "code"]
    ns = {"__name__": "__smoke__"}
    applied = False
    for i, src in enumerate(code_cells):
        try:
            exec(compile(src, f"<cell {i}>", "exec"), ns)
        except Exception as e:
            print(f"\n[smoke][{tag}] FAILED in code cell {i}:\n{'-'*60}")
            print(src[:1500])
            print("-" * 60)
            raise
        if not applied and "END OF USER CONFIGURATION" in src:
            exec(compile(OVERRIDES % {"epochs": epochs, "resume": resume},
                         "<overrides>", "exec"), ns)
            applied = True
    if not applied:
        raise RuntimeError("Never found the configuration cell to override.")
    return ns


def assert_artifacts(out_dir):
    out = Path(out_dir)
    models = ["cnn_lstm", "tcn_lstm", "lstm_mtrans_mlp"]
    required = [
        ("checkpoints/latest/latest.pt",),
        ("checkpoints/best/best.pt",),
        ("checkpoints/best/best_model_info.json",),
        ("metrics/history.csv",),
        ("metrics/test_metrics.json",),
        ("predictions/test_predictions.csv",),
        ("config/config.json",),
        ("config/model_summary.json",),
    ]
    problems = []
    for m in models:
        for (rel,) in required:
            p = out / m / rel
            if not p.exists():
                problems.append(str(p))
        pngs = list((out / m / "plots").glob("*.png"))
        if not pngs:
            problems.append(f"{out/m/'plots'} has no PNG")

    for rel in ["comparison/test_metrics.csv",
                "comparison/model_comparison.csv",
                "comparison/model_summary.json"]:
        if not (out / rel).exists():
            problems.append(str(out / rel))
    if not list((out / "comparison" / "plots").glob("*.png")):
        problems.append("comparison/plots has no PNG")

    if problems:
        raise AssertionError("Missing artifacts:\n  " + "\n  ".join(problems))
    print("[smoke] all expected artifacts present")


def assert_history_schema(out_dir):
    """Verify 10 metrics x 2 targets x train/val are all present per epoch."""
    metrics = ["MAE", "RMSE", "MAPE", "sMAPE", "R2",
               "ExplainedVar", "MedianAE", "MaxError", "MAE_scaled", "RMSE_scaled"]
    targets = ["NET_PATIENT_REVENUE", "TOT_OVERALL_EXP"]
    h = pd.read_csv(Path(out_dir) / "cnn_lstm" / "metrics" / "history.csv")
    missing = []
    for split in ("train", "val"):
        for t in targets:
            for m in metrics:
                col = f"{split}_{t}_{m}"
                if col not in h.columns:
                    missing.append(col)
    for col in ("epoch", "learning_rate", "train_loss", "val_loss", "epoch_time"):
        if col not in h.columns:
            missing.append(col)
    if missing:
        raise AssertionError(f"history.csv missing columns: {missing}")
    expected_metric_cols = 2 * 2 * 10
    print(f"[smoke] history.csv schema OK "
          f"({expected_metric_cols} metric cols + loss/lr/epoch/time), "
          f"{len(h)} epoch row(s), no duplicate epochs: "
          f"{h['epoch'].duplicated().sum() == 0}")
    return h


def assert_predictions_schema(out_dir):
    p = pd.read_csv(Path(out_dir) / "cnn_lstm" / "predictions" / "test_predictions.csv")
    need = ["index",
            "actual_NET_PATIENT_REVENUE", "predicted_NET_PATIENT_REVENUE",
            "actual_TOT_OVERALL_EXP", "predicted_TOT_OVERALL_EXP"]
    missing = [c for c in need if c not in p.columns]
    if missing:
        raise AssertionError(f"test_predictions.csv missing: {missing}")
    print(f"[smoke] test_predictions.csv schema OK ({len(p)} rows, columns exact)")


def main():
    out_dir = SMOKE / "baseline_outputs_v1"
    if SMOKE.exists():
        shutil.rmtree(SMOKE)

    print("\n" + "=" * 70)
    print("SMOKE TEST 1/3 - Layout C (CSV), fresh run, 2 epochs")
    print("=" * 70)
    make_synthetic_csv_dataset()
    run_notebook(epochs=2, resume="False", tag="csv-fresh")
    assert_artifacts(out_dir)
    h1 = assert_history_schema(out_dir)
    assert_predictions_schema(out_dir)
    assert len(h1) == 2, f"expected 2 epochs, got {len(h1)}"

    print("\n" + "=" * 70)
    print("SMOKE TEST 2/3 - RESUME=True, extend to 4 epochs")
    print("=" * 70)
    run_notebook(epochs=4, resume="True", tag="csv-resume")
    h2 = assert_history_schema(out_dir)
    assert len(h2) == 4, f"expected 4 epochs after resume, got {len(h2)}"
    assert sorted(h2["epoch"].tolist()) == [1, 2, 3, 4], h2["epoch"].tolist()
    assert h2["epoch"].duplicated().sum() == 0, "duplicate epoch rows after resume"
    print("[smoke] resume verified: epochs 1..4, continued numbering, no duplicates")

    print("\n" + "=" * 70)
    print("SMOKE TEST 3/3 - Layout A (NPZ), fresh run, 2 epochs")
    print("=" * 70)
    shutil.rmtree(out_dir, ignore_errors=True)
    make_synthetic_npz_dataset()
    run_notebook(epochs=2, resume="False", tag="npz-fresh")
    assert_artifacts(out_dir)
    print("[smoke] NPZ layout verified")

    print("\n" + "=" * 70)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 70)
    print("NOTE: synthetic data only. No research result is implied by any number "
          "produced during these tests.")
    shutil.rmtree(SMOKE, ignore_errors=True)
    print("[smoke] scratch directory removed")


if __name__ == "__main__":
    main()
