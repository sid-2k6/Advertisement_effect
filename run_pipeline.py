"""Staged runner (each stage fits in one sandbox call).

    python run_pipeline.py <outcome> <stage>
    outcome in {visit, conversion}
    stage   in {baselines, forests, neural, dranet, evaluate}

Artifacts are checkpointed to artifacts/<outcome>/ so stages run independently.
"""
import sys
from dranet import config as C
from dranet import run_experiment as RE


def main():
    outcome = sys.argv[1] if len(sys.argv) > 1 else "visit"
    stage = sys.argv[2] if len(sys.argv) > 2 else "evaluate"
    cfg = C.RunConfig()
    if stage == "evaluate":
        RE.evaluate_all(outcome, cfg, save_models=(outcome == "visit"), do_shap=True)
    else:
        RE.fit_group(outcome, stage, cfg)


if __name__ == "__main__":
    main()
