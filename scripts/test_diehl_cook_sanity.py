#!/usr/bin/env python3
"""
Quick sanity check for Diehl & Cook STDP encoder.

Trains on a small subset (500 samples), encodes 200 test samples, and
prints diagnostic statistics at every stage so you can verify the network
is healthy BEFORE committing to a full 60k-sample training run (~6-10 h).

What to look for in the output
──────────────────────────────
  ✅ HEALTHY signs:
    • Input spikes per sample > 1000         (Poisson encoding works)
    • Exc spikes per sample ≫ 0              (neurons are firing)
    • Theta mean > 0 after training          (adaptive threshold active)
    • Active neurons/sample ≈ 5-50           (WTA competition works)
    • Unique codes > 50% of n_samples        (codes are discriminative)
    • Weight std > 0.05                      (STDP changed weights)

  ❌ BROKEN signs:
    • Input spikes ≈ 0                       → intensity scaling missing
    • Exc spikes = 0 for all samples         → neurons silenced
    • Theta = 0 everywhere                   → wrong node type (LIFNodes)
    • Active neurons = 0 or = n_neurons      → WTA broken
    • Unique codes = 1                       → all codes identical
    • Weight std ≈ 0 (same as init)          → STDP not learning

Usage:
    python scripts/test_diehl_cook_sanity.py               # GPU
    python scripts/test_diehl_cook_sanity.py --device cpu   # CPU
    python scripts/test_diehl_cook_sanity.py --n_train 1000 # more training
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Repo root on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Diehl & Cook sanity check")
    parser.add_argument("--n_train", type=int, default=500,
                        help="Training samples (default 500)")
    parser.add_argument("--n_test", type=int, default=200,
                        help="Test samples to encode (default 200)")
    parser.add_argument("--n_neurons", type=int, default=100,
                        help="Excitatory neurons (default 100, faster than 400)")
    parser.add_argument("--device", type=str, default=None,
                        help="cpu or cuda (auto-detect if omitted)")
    args = parser.parse_args()

    import torch
    from bindsnet.encoding import poisson

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # ── 1. Load MNIST ────────────────────────────────────────────────────
    print("\n[1/6] Loading MNIST...")
    from pipeline.datasets import load_dataset
    ds = load_dataset("mnist", root=str(REPO_ROOT / "data"))
    train_data = ds["train_data"][:args.n_train].astype(np.float32)
    train_labels = ds["train_labels"][:args.n_train]
    test_data = ds["test_data"][:args.n_test].astype(np.float32)
    test_labels = ds["test_labels"][:args.n_test]
    print(f"  Train: {train_data.shape}, Test: {test_data.shape}")
    print(f"  Data range: [{train_data.min():.3f}, {train_data.max():.3f}]")

    # ── 2. Verify Poisson encoding ───────────────────────────────────────
    print("\n[2/6] Testing Poisson encoding...")
    intensity = 128.0
    sample = torch.from_numpy(train_data[0] * intensity).float()
    encoded = poisson(datum=sample, time=350, dt=1.0)
    n_input_spikes = encoded.sum().item()
    print(f"  Sample 0 (label={train_labels[0]}): "
          f"{n_input_spikes:.0f} input spikes over 350 ms")
    if n_input_spikes < 100:
        print("  ❌ WARNING: Very few input spikes! Check intensity scaling.")
    else:
        print(f"  ✅ Input encoding looks healthy "
              f"({n_input_spikes/784:.1f} spikes/neuron avg)")

    # ── 3. Build network ─────────────────────────────────────────────────
    print("\n[3/6] Building network...")
    from baselines.diehl_cook.encoder import DiehlCookEncoder
    config = {
        "input_dim": 784,
        "n_neurons": args.n_neurons,
        "simulation_time": 350,
        "dt": 1.0,
        "nu": (1e-4, 1e-2),
        "intensity": intensity,
        "binarization_percent": 0.05,
        "device": device,
    }
    encoder = DiehlCookEncoder(config)
    encoder.network = encoder._build_network()
    encoder.network.to(device)

    exc = encoder.network.layers["Excitatory"]
    print(f"  Excitatory layer type: {type(exc).__name__}")
    has_theta = hasattr(exc, "theta")
    print(f"  Has adaptive threshold (theta): {has_theta}")
    if not has_theta:
        print("  ❌ CRITICAL: No theta! Must use DiehlAndCookNodes, not LIFNodes.")
        sys.exit(1)
    print(f"  Theta initial: mean={exc.theta.mean():.4f}, "
          f"max={exc.theta.max():.4f}")

    # Grab initial weights for comparison
    w0 = encoder.network.connections[("Input", "Excitatory")].w.clone()
    print(f"  Initial weights: mean={w0.mean():.4f}, std={w0.std():.4f}")

    # ── 4. Train ─────────────────────────────────────────────────────────
    print(f"\n[4/6] Training on {args.n_train} samples...")
    t0 = time.time()

    # Normalise + intensity
    td = train_data.copy()
    if td.max() > 1.0:
        td = td / 255.0
    td = np.clip(td, 0.0, 1.0) * intensity

    train_exc_spikes = []
    for i, image in enumerate(td):
        img_t = torch.from_numpy(image).float()
        enc = poisson(datum=img_t, time=350, dt=1.0)
        if isinstance(enc, torch.Tensor):
            enc = enc.to(device)
        encoder.network.run(inputs={"Input": enc}, time=350)

        # Record excitatory spikes for this sample
        spikes = encoder.network.monitors["ExcitatoryMonitor"].get("s")
        n_exc = spikes.sum().item()
        train_exc_spikes.append(n_exc)

        encoder.network.reset_state_variables()

        if (i + 1) % 100 == 0 or (i + 1) == len(td):
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(td)}  "
                  f"exc_spikes(last)={n_exc:.0f}  "
                  f"theta_mean={exc.theta.mean():.2f}  "
                  f"({elapsed:.1f}s)")

    encoder.is_trained = True
    train_time = time.time() - t0

    # ── 5. Post-training diagnostics ─────────────────────────────────────
    print(f"\n[5/6] Post-training diagnostics ({train_time:.1f}s)...")
    train_exc_spikes = np.array(train_exc_spikes)
    print(f"  Training exc spikes/sample: "
          f"mean={train_exc_spikes.mean():.1f}, "
          f"median={np.median(train_exc_spikes):.1f}, "
          f"zero={np.sum(train_exc_spikes == 0)}/{len(train_exc_spikes)}")

    print(f"  Theta after training: "
          f"mean={exc.theta.mean():.2f}, "
          f"min={exc.theta.min():.2f}, "
          f"max={exc.theta.max():.2f}")

    w1 = encoder.network.connections[("Input", "Excitatory")].w
    dw = (w1 - w0.to(w1.device)).abs()
    print(f"  Weights after training: mean={w1.mean():.4f}, std={w1.std():.4f}")
    print(f"  Weight change |Δw|: mean={dw.mean():.6f}, max={dw.max():.6f}")
    if dw.mean() < 1e-6:
        print("  ❌ WARNING: Weights barely changed — STDP may not be learning!")
    else:
        print("  ✅ Weights changed, STDP is active.")

    # Neuron utilisation: how many neurons have theta > 0?
    theta_active = (exc.theta > 0.01).sum().item()
    print(f"  Neurons with theta > 0.01: {theta_active}/{args.n_neurons} "
          f"({100*theta_active/args.n_neurons:.0f}%)")
    if theta_active < args.n_neurons * 0.3:
        print("  ⚠️  Less than 30% of neurons have been active. "
              "With more training samples, this should improve.")

    # ── 6. Encode test data ──────────────────────────────────────────────
    print(f"\n[6/6] Encoding {args.n_test} test samples...")
    result = encoder.encode(test_data)
    pre_code = result["pre_code"]
    code = result["code"]

    # Summary
    print(f"\n{'='*70}")
    print("SANITY CHECK SUMMARY")
    print(f"{'='*70}")
    print(f"  Node type:              {type(exc).__name__}")
    print(f"  Intensity:              {intensity}")
    print(f"  Training samples:       {args.n_train}")
    print(f"  Training time:          {train_time:.1f}s")
    print(f"  Exc spikes/sample(train): {train_exc_spikes.mean():.1f}")
    total_enc = pre_code.sum(axis=1)
    print(f"  Exc spikes/sample(test):  {total_enc.mean():.1f}")
    active_enc = (pre_code > 0).sum(axis=1)
    print(f"  Active neurons/sample:  {active_enc.mean():.1f}")
    unique = len(np.unique(code, axis=0))
    print(f"  Unique codes:           {unique}/{args.n_test}")

    # Rough per-class check
    if test_labels is not None:
        from collections import Counter
        label_counts = Counter(test_labels.tolist())
        # For each pair of same-label samples, compute code overlap
        overlaps_same, overlaps_diff = [], []
        rng = np.random.RandomState(42)
        for _ in range(500):
            i, j = rng.randint(0, args.n_test, 2)
            overlap = (code[i] * code[j]).sum()
            if test_labels[i] == test_labels[j]:
                overlaps_same.append(overlap)
            else:
                overlaps_diff.append(overlap)
        if overlaps_same and overlaps_diff:
            print(f"  Code overlap (same class):  {np.mean(overlaps_same):.2f}")
            print(f"  Code overlap (diff class):  {np.mean(overlaps_diff):.2f}")
            if np.mean(overlaps_same) > np.mean(overlaps_diff) + 0.5:
                print("  ✅ Same-class codes more similar than diff-class — "
                      "network is learning!")
            else:
                print("  ⚠️  Same/diff overlap similar — may need more training.")

    # Verdict
    print(f"\n{'='*70}")
    healthy = (
        train_exc_spikes.mean() > 10
        and total_enc.mean() > 0
        and unique > args.n_test * 0.3
        and has_theta
    )
    if healthy:
        print("✅ PASS — Network looks healthy. Safe to run full training.")
    else:
        print("❌ FAIL — Network has issues. Check the warnings above.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
