#!/usr/bin/env python3
"""
Compute missing-point estimation errors from saved rollout NPZ logs.

Each NPZ is expected to contain:
- observed_xy: (T, N, 2)  observed (imputed/corrected) FP xy
- true_xy:     (T, N, 2)  ground-truth FP xy
- missing_mask:(T, N)     bool mask where True means FP was missing/occluded
- frames:      (T,)       frame indices (optional for reporting)

Metrics computed (using only entries where missing_mask == True):
- Overall missing-only L2: mean, std, count, median, p90
- Per-frame missing-only mean L2 (NaN when no missing in a frame)
- Per-point missing-only mean L2 across all frames (NaN when point never missing)

Optionally, distances can be scaled (e.g., pixels->cm) via --scale.
"""
import argparse
import glob
import os
from typing import List, Dict, Any

import numpy as np
import pandas as pd


def load_npz(path: str) -> Dict[str, np.ndarray]:
    d = np.load(path)
    # Required keys
    required = ["observed_xy", "true_xy", "missing_mask"]
    for k in required:
        if k not in d:
            raise ValueError(f"{path} is missing key '{k}'")
    observed_xy = d["observed_xy"]
    true_xy = d["true_xy"]
    missing_mask = d["missing_mask"].astype(bool)
    frames = d["frames"] if "frames" in d else np.arange(observed_xy.shape[0], dtype=int)

    if observed_xy.shape != true_xy.shape:
        raise ValueError(f"{path}: observed_xy shape {observed_xy.shape} != true_xy shape {true_xy.shape}")
    if observed_xy.shape[:2] != missing_mask.shape:
        raise ValueError(
            f"{path}: leading dims of observed/true {observed_xy.shape[:2]} != missing_mask {missing_mask.shape}"
        )
    if frames.shape[0] != observed_xy.shape[0]:
        raise ValueError(f"{path}: frames length {frames.shape[0]} != T {observed_xy.shape[0]}")

    return dict(observed_xy=observed_xy, true_xy=true_xy, missing_mask=missing_mask, frames=frames)


def compute_metrics(
    observed_xy: np.ndarray, true_xy: np.ndarray, missing_mask: np.ndarray, scale: float = 1.0
) -> Dict[str, Any]:
    """
    observed_xy, true_xy: (T, N, 2)
    missing_mask: (T, N) bool
    scale: multiply distances by this factor (e.g., cm_per_pixel)
    """
    # L2 per (T,N)
    l2 = np.linalg.norm(true_xy - observed_xy, axis=-1) * scale  # (T,N)

    # --- Metric (1): overall mean/std across all missing entries ---
    missing_values = l2[missing_mask]
    total_missing_count = int(missing_mask.sum())
    if total_missing_count == 0:
        overall_missing_mean = np.nan
        overall_missing_std = np.nan
    else:
        overall_missing_mean = float(np.mean(missing_values))
        overall_missing_std = float(np.std(missing_values, ddof=0))

    # --- Metric (2): framewise mean over missing points, then mean over frames ---
    per_frame_den = missing_mask.sum(axis=1).astype(float)     # (T,)
    per_frame_num = (l2 * missing_mask).sum(axis=1)            # (T,)
    with np.errstate(invalid="ignore", divide="ignore"):
        per_frame_mean_missing = np.where(per_frame_den > 0, per_frame_num / per_frame_den, np.nan)  # (T,)

    # Mean over frames that had missing points
    frames_with_missing = np.isfinite(per_frame_mean_missing)
    if frames_with_missing.any():
        #devide by total num of frames
        framewise_mean_over_missing = float(np.nanmean(per_frame_mean_missing))
    else:
        framewise_mean_over_missing = np.nan

    return dict(
        overall_missing_mean=overall_missing_mean,
        overall_missing_std=overall_missing_std,
        total_missing_count=total_missing_count,
        framewise_mean_over_missing=framewise_mean_over_missing,
        per_frame_mean_missing=per_frame_mean_missing,  # kept for CSV if needed, a column data
    )


def process_files(files: List[str], outdir: str, scale: float) -> pd.DataFrame:
    os.makedirs(outdir, exist_ok=True)
    rows = []

    for f in files:
        try:
            data = load_npz(f)
            res = compute_metrics(data["observed_xy"], data["true_xy"], data["missing_mask"], scale=scale)
        except Exception as e:
            print(f"[SKIP] {f}: {e}")
            continue

        base = os.path.splitext(os.path.basename(f))[0]
        rows.append(dict(
            file=base,
            overall_missing_mean=res["overall_missing_mean"],
            overall_missing_std=res["overall_missing_std"],
            total_missing_count=res["total_missing_count"],
            framewise_mean_over_missing=res["framewise_mean_over_missing"],
        ))

        # # Optional: save per-frame values
        # per_frame = pd.DataFrame({
        #     "frame_idx": np.arange(len(res["per_frame_mean_missing"]), dtype=int),
        #     "mean_missing_l2": res["per_frame_mean_missing"],
        # })
        # per_frame.to_csv(os.path.join(outdir, f"{base}_per_frame_missing_l2.csv"), index=False)

    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        print("[WARN] No valid results found.")
        return summary_df

    # --- Simple averages across rollouts ---
    avg_overall_mean = summary_df["overall_missing_mean"].mean()
    avg_overall_std = summary_df["overall_missing_std"].mean()
    avg_framewise_mean = summary_df["framewise_mean_over_missing"].mean()
    total_missing = int(summary_df["total_missing_count"].sum())

    # --- Append average row ---
    avg_row = dict(
        file="__AVERAGE__",
        overall_missing_mean=avg_overall_mean,
        overall_missing_std=avg_overall_std,
        total_missing_count=total_missing,
        framewise_mean_over_missing=avg_framewise_mean,
    )

    summary_df = pd.concat([summary_df, pd.DataFrame([avg_row])], ignore_index=True)

    # # --- Add backward-compatible aliases ---
    # summary_df["mean"] = summary_df["overall_missing_mean"]
    # summary_df["std"] = summary_df["overall_missing_std"]
    # summary_df["count"] = summary_df["total_missing_count"]

    # --- Save CSV and print results ---
    summary_path = os.path.join(outdir, "summary_missing_only.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"[WROTE] {summary_path}")

    print("\n===== Final Averages Across All Rollouts =====")
    print(f"Mean of overall_missing_mean:        {avg_overall_mean:.6f}")
    print(f"Mean of overall_missing_std:         {avg_overall_std:.6f}")
    print(f"Mean of framewise_mean_over_missing: {avg_framewise_mean:.6f}")
    print(f"Total missing points (all rollouts): {total_missing}")

    return summary_df

def parse_args():
    p = argparse.ArgumentParser(description="Compute missing-only FP errors from NPZ logs.")
    p.add_argument("--path", default="../saved_DLO_prediction/online_trial",
                   help="Path to a single .npz file or a directory containing .npz files")
    p.add_argument("--pattern", default="*.npz",
                   help="Glob pattern (used when 'path' is a directory)")
    p.add_argument("--outdir", default="error_reports",
                   help="Where to write CSV outputs")
    p.add_argument("--scale", type=float, default=1.0,
                   help="Multiply distances by this factor (e.g., cm_per_pixel)")
    return p.parse_args()

def main():
    args = parse_args()

    setting = 'Drop_246'
    tag = "Adam_0.1"

    args.path = f"./{setting}/saved_DLO_prediction/{tag}"
    args.outdir = f"./{setting}/saved_DLO_prediction/{tag}"

    if os.path.isdir(args.path):
        files = sorted(glob.glob(os.path.join(args.path, args.pattern)))
    else:
        files = [args.path] if args.path.endswith(".npz") else []
    if not files:
        raise SystemExit(f"No NPZ files found for path={args.path} (pattern={args.pattern})")

    print(f"[INFO] Found {len(files)} file(s). Computing errors with scale={args.scale}...")
    summary_df = process_files(files, args.outdir, scale=args.scale)



if __name__ == "__main__":
    main()