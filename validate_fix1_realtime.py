"""
Fix 1.5 — Real-time CPU / block wall-clock benchmark

Purpose
-------
Measure the steady-state CPU wall-clock time of the Fix 1 streaming
rational resampler and compare it with the physical time available for
each incoming 25 MS/s RX block.

IMPORTANT:
- This file does NOT modify waveform.py or any project file.
- Run it from D:\\phase2_fix1.
- It benchmarks only the resampler process() path, not UHD I/O.
- Results are a readiness check, not a claim that the full B210 pipeline
  is real-time safe.

For a block of N samples arriving at 25 MS/s:

    arrival_time = N / 25e6 seconds

The main metric is:

    load_ratio = median_process_time / arrival_time

A ratio below 1 means the measured resampler processing fits inside the
block arrival interval. We also report p95/p99/max to expose jitter.
"""

from __future__ import annotations

import gc
import statistics
import time
from pathlib import Path
import sys

import numpy as np

# Always import the Fix-1 copy of the project, regardless of cwd.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config as cfg
import waveform


FS_IN = 25e6
FS_OUT = 20e6

# Test sizes deliberately include small and large streaming blocks.
BLOCK_SIZES = [256, 512, 1024, 2048, 4096, 8192]

# Warm-up calls are excluded from timing.
WARMUP_BLOCKS = 100

# Timed calls per block size.
TIMED_BLOCKS = 2000

# Fixed RNG for reproducibility.
RNG = np.random.default_rng(20260930)


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolated percentile in microseconds."""
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def make_input(n: int) -> np.ndarray:
    """Create realistic complex-valued RX-like input."""
    x = (
        RNG.standard_normal(n).astype(np.float32)
        + 1j * RNG.standard_normal(n).astype(np.float32)
    )
    return x.astype(np.complex64)


def benchmark_block_size(n: int) -> dict:
    """
    Benchmark one persistent 25->20 streaming resampler.

    The same resampler instance is retained across all timed blocks so
    this measures the stateful streaming path rather than repeatedly
    paying initialization costs.
    """
    rs = waveform.make_streaming_25to20()

    x = make_input(n)

    # Warm-up: fill caches and establish normal steady-state behavior.
    for _ in range(WARMUP_BLOCKS):
        rs.process(x)

    # Prevent the interpreter from doing garbage collection during the
    # timed section as much as possible. This is restored afterwards.
    old_gc = gc.isenabled()
    gc.disable()

    times_us: list[float] = []

    try:
        for _ in range(TIMED_BLOCKS):
            t0 = time.perf_counter_ns()
            y = rs.process(x)
            t1 = time.perf_counter_ns()

            # Keep the result live until timing is recorded. The resampler
            # output is intentionally not inspected here because this is
            # a timing benchmark, not a numerical correctness test.
            _ = y

            times_us.append((t1 - t0) / 1000.0)
    finally:
        if old_gc:
            gc.enable()

    arrival_us = n / FS_IN * 1e6

    median_us = float(statistics.median(times_us))
    mean_us = float(statistics.mean(times_us))
    p95_us = percentile(times_us, 95)
    p99_us = percentile(times_us, 99)
    max_us = float(max(times_us))

    return {
        "n": n,
        "arrival_us": arrival_us,
        "median_us": median_us,
        "mean_us": mean_us,
        "p95_us": p95_us,
        "p99_us": p99_us,
        "max_us": max_us,
        "median_ratio": median_us / arrival_us,
        "p95_ratio": p95_us / arrival_us,
        "p99_ratio": p99_us / arrival_us,
        "max_ratio": max_us / arrival_us,
    }


def print_result(r: dict) -> None:
    print(
        f"  N={r['n']:5d} | "
        f"arrival={r['arrival_us']:9.2f} us | "
        f"median={r['median_us']:9.2f} us "
        f"({r['median_ratio']:.3f}x) | "
        f"p95={r['p95_us']:9.2f} us "
        f"({r['p95_ratio']:.3f}x) | "
        f"p99={r['p99_us']:9.2f} us "
        f"({r['p99_ratio']:.3f}x) | "
        f"max={r['max_us']:9.2f} us "
        f"({r['max_ratio']:.3f}x)"
    )


def main() -> int:
    print("=" * 88)
    print("FIX 1.5 — REAL-TIME CPU / BLOCK WALL-CLOCK VALIDATION")
    print("=" * 88)
    print(f"Project root : {ROOT}")
    print(f"Input rate   : {FS_IN/1e6:.1f} MS/s")
    print(f"Output rate  : {FS_OUT/1e6:.1f} MS/s")
    print(f"Resampler    : {waveform.make_streaming_25to20().__class__.__name__}")
    print(f"Warm-up      : {WARMUP_BLOCKS} blocks/size")
    print(f"Timed blocks : {TIMED_BLOCKS} blocks/size")
    print()

    print("Engineering interpretation used by this benchmark:")
    print("  median ratio < 0.50  -> excellent headroom")
    print("  median ratio < 0.75  -> good headroom")
    print("  median ratio < 1.00  -> fits median timing, limited headroom")
    print("  median ratio >= 1.00 -> median processing exceeds arrival time")
    print()
    print("NOTE: p95/p99/max are also reported because median alone can hide")
    print("      scheduling jitter. Full UHD I/O + demodulation is NOT measured.")
    print()

    results = []

    for n in BLOCK_SIZES:
        print(f"[TEST] {n} samples")
        r = benchmark_block_size(n)
        results.append(r)
        print_result(r)
        print()

    # Find the 4096-sample result explicitly because that was the previous
    # benchmark size and corresponds to 163.84 us of input arrival time.
    r4096 = next(r for r in results if r["n"] == 4096)

    print("-" * 88)
    print("4096-SAMPLE REFERENCE")
    print("-" * 88)
    print("  Physical arrival interval:")
    print(f"    4096 / 25e6 = {r4096['arrival_us']:.2f} us")
    print("  This is the number to compare against resampler processing time.")
    print()

    # Do NOT call this a hardware pass/fail. We classify the measured
    # resampler-only workload and explicitly flag cases needing investigation.
    median_safe = all(r["median_ratio"] < 1.0 for r in results)
    p95_safe = all(r["p95_ratio"] < 1.0 for r in results)

    print("-" * 88)
    print("SUMMARY")
    print("-" * 88)

    if median_safe:
        print("[PASS] Median resampler processing time is below block arrival time")
        print("       for every tested block size.")
    else:
        print("[WARN] At least one block size has median processing time >= arrival time.")
        print("       Do NOT optimize blindly; inspect that block size first.")

    if p95_safe:
        print("[PASS] p95 resampler processing time is below block arrival time")
    else:
        print("[WARN] p95 processing exceeds arrival time for at least one block size.")
        print("       Real-time headroom may be insufficient under scheduler jitter.")

    print()
    print("This benchmark does not modify any source file.")
    print("Send the COMPLETE output before making any Fix 1.5 code change.")
    print("=" * 88)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
