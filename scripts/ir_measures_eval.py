import os
import argparse
import csv

import ir_measures
from ir_measures import nDCG, MRR, P, RR, R


# ----------------------------
# Metrics
# ----------------------------
METRICS = [
    R@1,
    R@5,
    R@10,
    MRR@10,
    nDCG@5,
    nDCG@10,
    P@5
]


def load_runs_for_dataset(ds_path):
    """
    Return (t2v_runs, v2t_runs) dicts for a dataset folder.
    Keys: run label (derived from filename)
    Values: list(...) of run entries (so we can iterate multiple times).
    """
    t2v_runs = {}
    v2t_runs = {}

    for fname in os.listdir(ds_path):
        fpath = os.path.join(ds_path, fname)
        if not os.path.isfile(fpath):
            continue
        if not fname.lower().endswith(".txt"):
            continue
        if fname.lower().startswith("qrels"):
            continue  # skip qrels files

        upper = fname.upper()

        # Text-to-video runs (T2V)
        if upper.endswith("T2V.TXT"):
            label = fname[:-4]  # strip .txt
            label = label.replace("_T2V", "").replace("T2V", "")
            label = label or "run"
            t2v_runs[label] = list(ir_measures.read_trec_run(fpath))

        # Video-to-text runs (V2T)
        elif upper.endswith("V2T.TXT"):
            label = fname[:-4]
            label = label.replace("_V2T", "").replace("V2T", "")
            label = label or "run"
            v2t_runs[label] = list(ir_measures.read_trec_run(fpath))

    return t2v_runs, v2t_runs


def evaluate_dataset(ds_path, dataset_name, results_rows):
    """
    Evaluate all runs in a dataset folder and append rows to results_rows.
    """
    print("\n" + "=" * 80)
    print(f"DATASET: {dataset_name}")
    print("=" * 80)

    t2v_runs, v2t_runs = load_runs_for_dataset(ds_path)

    # ----- T2V -----
    qrels_t2v_path = os.path.join(ds_path, "qrels_t2v.txt")
    if os.path.exists(qrels_t2v_path) and t2v_runs:
        qrels_t2v = list(ir_measures.read_trec_qrels(qrels_t2v_path))
        print("\nT2V Results:")

        for run_name, run in t2v_runs.items():
            results = ir_measures.calc_aggregate(METRICS, qrels_t2v, run)
            print(f"\n  {run_name}")
            print(" ", results)
            print("  " + "------" * 10)

            # Add CSV rows
            for metric in METRICS:
                metric_str = str(metric) 
                value = results.get(metric, float("nan"))
                results_rows.append({
                    "dataset": dataset_name,
                    "task": "T2V",
                    "run": run_name,
                    "metric": metric_str,
                    "value": value
                })

    # ----- V2T -----
    qrels_v2t_path = os.path.join(ds_path, "qrels_v2t.txt")
    if os.path.exists(qrels_v2t_path) and v2t_runs:
        qrels_v2t = list(ir_measures.read_trec_qrels(qrels_v2t_path))
        print("\nV2T Results:")

        for run_name, run in v2t_runs.items():
            results = ir_measures.calc_aggregate(METRICS, qrels_v2t, run)
            print(f"\n  {run_name}")
            print(" ", results)
            print("  " + "------" * 10)

            # Add CSV rows
            for metric in METRICS:
                metric_str = str(metric)
                value = results.get(metric, float("nan"))
                results_rows.append({
                    "dataset": dataset_name,
                    "task": "V2T",
                    "run": run_name,
                    "metric": metric_str,
                    "value": value
                })


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate TREC run files with ir_measures and export to CSV."
    )
    parser.add_argument(
        "base_dir",
        nargs="?",
        default="./trec_runs",
        help="Base directory containing dataset subfolders (default: ./trec_runs)",
    )
    parser.add_argument(
        "-o", "--output",
        default="./trec_runs/trec_eval_results.csv",
        help="Path to output CSV file (default: trec_eval_results.csv)",
    )

    args = parser.parse_args()
    base_dir = args.base_dir
    output_dir = args.output

    os.makedirs(output_dir, exist_ok=True)

    output_filename = "triangle_trec_eval_results.csv"
    output_csv = os.path.join(output_dir, output_filename)

    print(f"Using base directory: {base_dir}")
    results_rows = []

    # Treat subdirectories of base_dir as datasets
    if os.path.isdir(base_dir):
        for dataset in os.listdir(base_dir):
            ds_path = os.path.join(base_dir, dataset)
            if not os.path.isdir(ds_path):
                continue
            evaluate_dataset(ds_path, dataset, results_rows)

        evaluate_dataset(base_dir, os.path.basename(os.path.abspath(base_dir)) or "root", results_rows)
    else:
        print(f"ERROR: {base_dir} is not a directory.")
        return

    # Write results to CSV
    if results_rows:
        fieldnames = ["dataset", "task", "run", "metric", "value"]
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results_rows)

        print(f"\nSaved results to: {output_csv}")
    else:
        print("\nNo results to write (no runs/qrels found?).")


if __name__ == "__main__":
    main()
