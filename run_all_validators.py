"""Authoritative software regression runner for the current Phase-2 PHY.

This runner covers the validated software baseline only. It does not open a
USRP and does not perform hardware validation.
"""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "evaluation" / "validation_report.txt"

STAGES = [
    ("Waveform unit tests", [sys.executable, "-m", "pytest", "test_waveform.py", "-q"]),
    ("Fix 1 streaming resampler", [sys.executable, "validate_fix1_resampler.py"]),
    ("Fix 1 realtime benchmark", [sys.executable, "validate_fix1_realtime.py"]),
    ("Fix 2 detector indexing", [sys.executable, "validate_fix2_detector_index.py"]),
    ("Fix 4D TX integrity", [sys.executable, "validate_fix4D_tx_integrity.py"]),
    ("Stage 1 PHY primitives", [sys.executable, "validate_stage1.py"]),
    ("Stage 2 configurable impairments", [sys.executable, "validate_stage2.py"]),
    ("Stage 3 end-to-end + HDF5", [sys.executable, "validate_stage3.py", "--n_packets", "30"]),
    ("Stage 4 resampler + alignment", [sys.executable, "validate_stage4.py"]),
]


def run_one(label: str, cmd: list[str]):
    print("\n" + "=" * 72)
    print(f"RUNNING: {label}")
    print("COMMAND:", " ".join(cmd))
    print("=" * 72)
    t0 = time.monotonic()
    result = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.monotonic() - t0
    return result.returncode == 0, elapsed


def main() -> int:
    print("\n" + "#" * 72)
    print("CUSTOM 128-POINT OFDM PHY — SOFTWARE REGRESSION")
    print("Started:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("Root:", ROOT)
    print("#" * 72)

    results = []
    for label, cmd in STAGES:
        ok, elapsed = run_one(label, cmd)
        results.append((label, ok, elapsed))
        if not ok:
            print(f"\nSTOP: {label} FAILED.")
            break

    all_pass = len(results) == len(STAGES) and all(ok for _, ok, _ in results)

    print("\n" + "=" * 72)
    print("SOFTWARE REGRESSION SCORECARD")
    print("=" * 72)
    for label, ok, elapsed in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {label:<40} {elapsed:7.2f}s")
    print("=" * 72)
    print("OVERALL:", "PASS" if all_pass else "FAIL")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open("w", encoding="utf-8") as f:
        f.write("CUSTOM 128-POINT OFDM PHY — SOFTWARE REGRESSION\n")
        f.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"Root: {ROOT}\n")
        f.write("=" * 72 + "\n")
        for label, ok, elapsed in results:
            f.write(f"[{'PASS' if ok else 'FAIL'}] {label} {elapsed:.2f}s\n")
        f.write("=" * 72 + "\n")
        f.write(f"Overall: {'PASS' if all_pass else 'FAIL'}\n")

    print("Report:", REPORT)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
