#!/usr/bin/env python3
"""
Task Intent Understanding — Pipeline Runner

用法示例：
  # 完整跑全流程（默认 5 次评估）
  python run_pipeline.py

  # 只跑数据处理，跳过评估和优化
  python run_pipeline.py --skip-eval --skip-opt

  # 只跑评估（假设数据已处理好）
  python run_pipeline.py --skip-data --skip-opt

  # 自定义并发数和评估轮次
  python run_pipeline.py --num-pool 4 --num-eval-runs 3
"""

import os
import sys
import argparse
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PROCESSED = os.path.join(BASE_DIR, "Data_processing", "Processed_data")
EVAL_DATA = os.path.join(BASE_DIR, "Evaluation_processing", "evaluation_data")
EVAL_PRECISION = os.path.join(BASE_DIR, "Evaluation_processing", "evaluation_precision")
OPT_DATA = os.path.join(BASE_DIR, "Optimization", "data_precision")


def run(label, script_rel, extra_args=None):
    """Run a script from BASE_DIR, print clear progress, exit on failure."""
    script = os.path.join(BASE_DIR, script_rel)
    cmd = [sys.executable, script] + (extra_args or [])
    print(f"\n{'='*60}")
    print(f"[{label}]")
    print(f"  {' '.join(cmd)}")
    print('='*60)
    result = subprocess.run(cmd, cwd=BASE_DIR)
    if result.returncode != 0:
        print(f"\n[FAILED] {label} — exit code {result.returncode}")
        sys.exit(result.returncode)
    print(f"[OK] {label}")


def ensure_dirs():
    for d in [DATA_PROCESSED, EVAL_DATA, EVAL_PRECISION, OPT_DATA]:
        os.makedirs(d, exist_ok=True)


def run_data_processing(num_pool):
    run(
        "Step 1/4 — Query interpretation",
        os.path.join("Data_processing", "01_interpretation.py"),
        ["--dir", DATA_PROCESSED],
    )
    run(
        "Step 2/4 — Data reformatting",
        os.path.join("Data_processing", "02_data_change.py"),
    )
    run(
        "Step 3/4 — Scoring question classification",
        os.path.join("Data_processing", "03_classification.py"),
        ["--dir", DATA_PROCESSED, "--output_path", DATA_PROCESSED,
         "--num_pool", str(num_pool)],
    )
    run(
        "Step 4/4 — LLM response generation",
        os.path.join("Data_processing", "04_generation.py"),
        ["--dir", DATA_PROCESSED, "--output_dir", DATA_PROCESSED,
         "--worker", str(num_pool)],
    )


def run_evaluation(num_pool, num_eval_runs):
    # Step 1: run the point-level judge N times (for p_true estimation)
    for i in range(1, num_eval_runs + 1):
        run(
            f"Eval Step 1/{num_eval_runs+4} — Point judge run {i}/{num_eval_runs}",
            os.path.join("Evaluation_processing", "step1.py"),
            ["--dir", DATA_PROCESSED,
             "--llm_output_path", DATA_PROCESSED,
             "--output_path", EVAL_DATA,
             "--num_pool", str(num_pool),
             "--run_id", str(i)],
        )

    # Step 2: aggregate p_true across runs
    run(
        f"Eval Step {num_eval_runs+1}/{num_eval_runs+4} — Aggregate p_true",
        os.path.join("Evaluation_processing", "step1_p_true.py"),
        ["--eval_dir", EVAL_DATA, "--num_runs", str(num_eval_runs)],
    )

    # Step 3: semantic clustering via entailment
    run(
        f"Eval Step {num_eval_runs+2}/{num_eval_runs+4} — Semantic clustering",
        os.path.join("Evaluation_processing", "step2_entailment.py"),
        ["--dir", EVAL_DATA, "--output_dir", EVAL_DATA],
    )

    # Step 4: compute semantic entropy
    run(
        f"Eval Step {num_eval_runs+3}/{num_eval_runs+4} — Entropy calculation",
        os.path.join("Evaluation_processing", "step3_bincount.py"),
    )

    # Step 5: merge everything into final result
    run(
        f"Eval Step {num_eval_runs+4}/{num_eval_runs+4} — Final merge",
        os.path.join("Evaluation_processing", "step4.py"),
    )


def run_optimization(num_pool):
    run(
        "Opt Step 1/2 — Constraint generation",
        os.path.join("Optimization", "01_contraints_generate.py"),
        ["--inputdir", DATA_PROCESSED, "--outputdir", OPT_DATA],
    )
    run(
        "Opt Step 2/2 — Constraint recombination",
        os.path.join("Optimization", "02_recombination.py"),
        ["--dir", OPT_DATA, "--worker", str(num_pool)],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Task Intent Understanding — automated pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--skip-data", action="store_true",
                        help="Skip data processing steps (01–04)")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip evaluation steps (step1–step4)")
    parser.add_argument("--skip-opt", action="store_true",
                        help="Skip optimization steps")
    parser.add_argument("--num-eval-runs", type=int, default=5,
                        help="Number of independent evaluator runs for p_true (default: 5)")
    parser.add_argument("--num-pool", type=int, default=4,
                        help="Concurrent worker threads (default: 4)")
    args = parser.parse_args()

    ensure_dirs()

    if not args.skip_data:
        print("\n### Data Processing ###")
        run_data_processing(args.num_pool)

    if not args.skip_eval:
        print("\n### Evaluation ###")
        run_evaluation(args.num_pool, args.num_eval_runs)

    if not args.skip_opt:
        print("\n### Optimization ###")
        run_optimization(args.num_pool)

    print(f"\n{'='*60}")
    print("Pipeline complete!")
    print(f"  Processed data : {DATA_PROCESSED}")
    print(f"  Evaluation data: {EVAL_DATA}")
    print(f"  Final result   : {os.path.join(EVAL_DATA, 'step4_gemini.json')}")
    print('='*60)


if __name__ == "__main__":
    main()
