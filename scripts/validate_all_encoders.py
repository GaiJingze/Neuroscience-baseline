#!/usr/bin/env python
"""
Quick smoke test: fit + encode every baseline encoder on a tiny dataset
and verify the output contract (shapes, dtypes, value ranges).

Usage:
    python scripts/validate_all_encoders.py          # all encoders
    python scripts/validate_all_encoders.py --skip-snn  # skip slow SNN baselines
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

# ---------------------------------------------------------------------------
# Tiny synthetic MNIST-like data (50 train, 10 test, 784-dim, values in [0,1])
# ---------------------------------------------------------------------------
np.random.seed(42)
N_TRAIN, N_TEST, INPUT_DIM = 50, 10, 784
TRAIN_DATA = np.random.rand(N_TRAIN, INPUT_DIM).astype(np.float32)
TRAIN_LABELS = np.random.randint(0, 10, size=N_TRAIN)
TEST_DATA = np.random.rand(N_TEST, INPUT_DIM).astype(np.float32)

# ---------------------------------------------------------------------------
# Registry: (name, import_path, class_name, config, is_snn)
# ---------------------------------------------------------------------------
ENCODERS = [
    # --- instant / fast baselines ---
    ("FlyHash", "baselines.flyhash.encoder", "FlyHashEncoder",
     {"input_dim": INPUT_DIM, "projection_dim": 1000, "hash_length": 64, "sampling_ratio": 0.1},
     False),
    ("SimHash (LSH)", "baselines.lsh.encoder", "SimHashEncoder",
     {"input_dim": INPUT_DIM, "hash_dim": 128},
     False),
    ("WTA Hash", "baselines.wta_hash.encoder", "WTAHashEncoder",
     {"input_dim": INPUT_DIM, "n_hashes": 64, "window_size": 8},
     False),
    # --- Hebbian / competitive ---
    ("BioHash", "baselines.biohash.encoder", "BioHashEncoder",
     {"input_dim": INPUT_DIM, "hash_dim": 128, "n_epochs": 2, "batch_size": 32},
     False),
    ("Krotov", "baselines.krotov.encoder", "KrotovEncoder",
     {"input_dim": INPUT_DIM, "n_neurons": 50, "n_epochs": 3, "batch_size": 50, "verbose": False},
     False),
    ("SoftHebb", "baselines.softhebb.encoder", "SoftHebbEncoder",
     {"input_dim": INPUT_DIM, "hidden_dims": [128], "output_dim": 64, "n_epochs": 2, "batch_size": 32, "device": "cpu"},
     False),
    ("SOM", "baselines.som.encoder", "SOMEncoder",
     {"input_dim": INPUT_DIM, "map_height": 5, "map_width": 5, "n_epochs": 2, "verbose": False},
     False),
    # --- SNN baselines (slower, need BindsNET / PyTorch) ---
    ("Diehl & Cook", "baselines.diehl_cook.encoder", "DiehlCookEncoder",
     {"input_dim": INPUT_DIM, "n_neurons": 25, "simulation_time": 50, "n_train_samples": 30, "device": "cpu"},
     True),
    ("Deep STDP", "baselines.deep_stdp.encoder", "DeepSTDPEncoder",
     {"input_dim": INPUT_DIM, "layer_sizes": [25], "n_rounds": 1, "simulation_time": 50,
      "n_train_samples": 30, "n_clusters": 5, "device": "cpu"},
     True),
    ("LC-SNN", "baselines.lc_snn.encoder", "LCSNNEncoder",
     {"input_dim": INPUT_DIM, "image_side": 28, "patch_size": 14, "patch_stride": 14,
      "neurons_per_patch": 10, "simulation_time": 50, "n_train_samples": 30, "device": "cpu"},
     True),
    ("LM-SNN", "baselines.lm_snn.encoder", "LMSNNEncoder",
     {"input_dim": INPUT_DIM, "n_neurons": 25, "simulation_time": 50,
      "n_train_samples": 30, "device": "cpu"},
     True),
    ("CSDP", "baselines.csdp.encoder", "CSDPEncoder",
     {"input_dim": INPUT_DIM, "img_size": 28, "hidden_dims": [64], "output_dim": 32,
      "n_steps": 20, "n_epochs": 2, "batch_size": 32, "device": "cpu"},
     True),
]


def validate_output(name: str, result: dict, n_samples: int) -> list[str]:
    """Check encode() output against the interface spec. Returns list of issues."""
    issues = []

    # Must be a dict with the right keys
    if not isinstance(result, dict):
        return [f"encode() returned {type(result).__name__}, expected dict"]
    for key in ("pre_code", "code"):
        if key not in result:
            issues.append(f"missing key '{key}'")

    if issues:
        return issues

    pre_code = result["pre_code"]
    code = result["code"]

    # Shapes
    if pre_code.ndim != 2:
        issues.append(f"pre_code.ndim={pre_code.ndim}, expected 2")
    if code.ndim != 2:
        issues.append(f"code.ndim={code.ndim}, expected 2")
    if pre_code.shape[0] != n_samples:
        issues.append(f"pre_code.shape[0]={pre_code.shape[0]}, expected {n_samples}")
    if code.shape[0] != n_samples:
        issues.append(f"code.shape[0]={code.shape[0]}, expected {n_samples}")
    if pre_code.shape != code.shape:
        issues.append(f"shape mismatch: pre_code {pre_code.shape} != code {code.shape}")

    # Dtypes
    if pre_code.dtype not in (np.float32, np.float64):
        issues.append(f"pre_code.dtype={pre_code.dtype}, expected float32/64")
    if code.dtype not in (np.float32, np.float64):
        issues.append(f"code.dtype={code.dtype}, expected float32/64")

    # Code should be binary {0, 1}
    unique_vals = set(np.unique(code))
    if not unique_vals.issubset({0.0, 1.0}):
        issues.append(f"code has values outside {{0,1}}: {sorted(unique_vals)[:5]}")

    # NaN / Inf
    if np.any(np.isnan(pre_code)):
        issues.append("pre_code contains NaN")
    if np.any(np.isnan(code)):
        issues.append("code contains NaN")
    if np.any(np.isinf(pre_code)):
        issues.append("pre_code contains Inf")

    # Sparsity sanity (code should not be all-zero or all-one)
    sparsity = 1 - np.mean(code)
    if np.mean(code) == 0:
        issues.append("code is all zeros (no activations)")
    if np.mean(code) == 1:
        issues.append("code is all ones (no sparsity)")

    return issues


def run_one(name, module_path, class_name, config, skip_snn, is_snn):
    """Returns (status_str, elapsed_seconds)."""
    if is_snn and skip_snn:
        return "SKIP", 0.0

    # Dynamic import
    try:
        mod = __import__(module_path, fromlist=[class_name])
        cls = getattr(mod, class_name)
    except Exception as e:
        return f"IMPORT ERROR: {e}", 0.0

    t0 = time.time()
    try:
        encoder = cls(config)

        # fit
        encoder.fit(TRAIN_DATA, TRAIN_LABELS)
        if not encoder.is_trained:
            return "FAIL: is_trained still False after fit()", time.time() - t0

        # encode
        result = encoder.encode(TEST_DATA)
        elapsed = time.time() - t0

        issues = validate_output(name, result, N_TEST)
        if issues:
            return "FAIL: " + "; ".join(issues), elapsed

        # Summary stats
        code = result["code"]
        sparsity = 1 - np.mean(code)
        return (
            f"OK  shape={code.shape}  sparsity={sparsity:.2f}  "
            f"code_mean={np.mean(code):.3f}",
            elapsed,
        )
    except Exception:
        return f"ERROR:\n{traceback.format_exc()}", time.time() - t0


def main():
    parser = argparse.ArgumentParser(description="Validate all encoders")
    parser.add_argument("--skip-snn", action="store_true",
                        help="Skip SNN baselines (Diehl&Cook, DeepSTDP, LC-SNN, LM-SNN, CSDP)")
    args = parser.parse_args()

    print("=" * 72)
    print(f"Encoder Smoke Test  —  train={N_TRAIN}  test={N_TEST}  dim={INPUT_DIM}")
    print("=" * 72)

    results = []
    for name, mod_path, cls_name, cfg, is_snn in ENCODERS:
        tag = " [SNN]" if is_snn else ""
        print(f"\n>>> {name}{tag}")
        status, elapsed = run_one(name, mod_path, cls_name, cfg, args.skip_snn, is_snn)
        results.append((name, status, elapsed))
        print(f"    {status}  ({elapsed:.1f}s)")

    # Summary
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    n_ok = sum(1 for _, s, _ in results if s.startswith("OK"))
    n_skip = sum(1 for _, s, _ in results if s == "SKIP")
    n_fail = len(results) - n_ok - n_skip
    for name, status, elapsed in results:
        short = status.split("\n")[0][:60]
        mark = "✓" if status.startswith("OK") else ("—" if status == "SKIP" else "✗")
        print(f"  {mark} {name:20s}  {short}")
    print(f"\nTotal: {n_ok} passed, {n_fail} failed, {n_skip} skipped out of {len(results)}")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
