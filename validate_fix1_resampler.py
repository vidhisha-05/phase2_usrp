"""Fix 1 validation: stateful 25<->20 MS/s streaming resampler.

The oracle is scipy.signal.resample_poly() on the complete signal.  The
streaming implementation must be invariant to input chunk boundaries and must
match the one-shot oracle to numerical precision after flush().
"""
import sys
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent))
import waveform

ATOL = 3e-6


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  <- {detail}" if detail else ""))
    if not cond:
        raise AssertionError(name + (": " + detail if detail else ""))


def run_direction(name, up, down, factory, one_shot):
    rng = np.random.default_rng(20261001 + up * 10 + down)
    lengths = [1, 7, 31, 128, 160, 511, 1024, 4096, 8192]
    chunk_sets = [
        [1],
        [7],
        [128],
        [256],
        [512],
        [1024],
        [2048],
        [4096],
        [8192],
    ]

    for n in lengths:
        x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
        ref = resample_poly(x, up, down, window=waveform.RESAMP_FILTER)
        check(f"{name}: one-shot length", len(one_shot(x)) == len(ref),
              f"got={len(one_shot(x))} ref={len(ref)}")

        for nominal in chunk_sets:
            chunks = []
            left = n
            while left:
                c = min(nominal[0], left)
                chunks.append(c)
                left -= c

            r = factory()
            pieces = []
            i = 0
            for c in chunks:
                pieces.append(r.process(x[i:i+c]))
                i += c
            pieces.append(r.flush())
            got = np.concatenate(pieces) if pieces else np.empty(0, np.complex64)

            err = float(np.max(np.abs(got - ref))) if len(ref) else 0.0
            check(f"{name}: n={n}, chunk={nominal[0]}",
                  len(got) == len(ref) and err <= ATOL,
                  f"out={len(got)}/{len(ref)}, max_err={err:.3e}")

    # A deliberately awkward chunk sequence, including UHD-like 4096 blocks.
    n = 25000
    x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
    ref = resample_poly(x, up, down, window=waveform.RESAMP_FILTER)
    pattern = [256, 4096, 512, 1024, 4096, 1600, 8192, 3000, 777]
    r = factory()
    pieces = []
    i = 0
    j = 0
    while i < n:
        c = min(pattern[j % len(pattern)], n - i)
        pieces.append(r.process(x[i:i+c]))
        i += c
        j += 1
    pieces.append(r.flush())
    got = np.concatenate(pieces)
    err = float(np.max(np.abs(got - ref)))
    check(f"{name}: mixed UHD chunk sizes", len(got) == len(ref) and err <= ATOL,
          f"out={len(got)}/{len(ref)}, max_err={err:.3e}")


if __name__ == "__main__":
    print("=" * 72)
    print("FIX 1 — STATEFUL RATIONAL RESAMPLER VALIDATION")
    print("=" * 72)
    print(f"FIR taps={len(waveform.RESAMP_FILTER)} | ATOL={ATOL:g}")
    print(f"25->20 latency={waveform.make_streaming_25to20().latency_samples} output samples")
    print(f"20->25 latency={waveform.make_streaming_20to25().latency_samples} output samples")

    run_direction("25->20", 4, 5,
                  waveform.make_streaming_25to20,
                  waveform.resample_25to20)
    run_direction("20->25", 5, 4,
                  waveform.make_streaming_20to25,
                  waveform.resample_20to25)

    # Continuous-stream property: before flush, only the final unstable tail
    # may be absent. There must be no dependence on the 4096-sample boundary.
    rng = np.random.default_rng(77)
    x = (rng.standard_normal(20000) + 1j * rng.standard_normal(20000)).astype(np.complex64)
    ref = resample_poly(x, 4, 5, window=waveform.RESAMP_FILTER)
    r = waveform.make_streaming_25to20()
    pieces = []
    for i in range(0, len(x), 4096):
        pieces.append(r.process(x[i:i+4096]))
    live = np.concatenate(pieces)
    check("25->20 continuous output has no internal loss",
          len(ref) - len(live) <= r.latency_samples + 1,
          f"reference={len(ref)}, live={len(live)}, tail={len(ref)-len(live)}")

    print("\nFIX 1 RESAMPLER: ALL TESTS PASS")
