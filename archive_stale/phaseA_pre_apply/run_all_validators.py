"""
run_all_validators.py  --  Master validation runner
Runs all simulation-mode stages sequentially, reports a final scorecard.

Usage:
    python run_all_validators.py               # run stages 1-4 (sim only)
    python run_all_validators.py --hardware    # also print hw stage instructions

Each stage exits with code 0 (pass) or 1 (fail).
Results are saved to validation_report.txt.
"""

import subprocess
import sys
import time
import os
from datetime import datetime

STAGES_SIM = [
    ("Stage 1 - PHY Primitives",        "validate_stage1.py",        []),
    ("Stage 2 - Configurable Impair.",   "validate_stage2.py",        []),
    ("Stage 3 - ZMQ E2E Pipeline",       "validate_stage3.py",
                                          ["--n_packets", "30"]),
    ("Stage 4 - Resampler + 2-Channel",  "validate_stage4.py",        []),
]

PASS_SYM = "PASS"
FAIL_SYM = "FAIL"


def run_stage(label, script, extra_args):
    print(f"\n{'='*64}")
    print(f"  RUNNING: {label}")
    print(f"{'='*64}")
    t0  = time.monotonic()
    cmd = [sys.executable, os.path.join(r"d:\phase2", script)] + extra_args
    ret = subprocess.run(cmd, cwd=r"d:\phase2")
    dur = time.monotonic() - t0
    ok  = (ret.returncode == 0)
    return ok, dur


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--hardware", action="store_true",
                    help="Also print hardware stage instructions")
    args = ap.parse_args()

    print("\n" + "#"*64)
    print("  CUSTOM 128-pt OFDM PHY -- FULL VALIDATION SUITE")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#"*64)

    results = []
    for label, script, extra in STAGES_SIM:
        ok, dur = run_stage(label, script, extra)
        results.append((label, ok, dur))

    # -- Scorecard ---------------------------------------------------------------
    print("\n" + "="*64)
    print("  VALIDATION SCORECARD")
    print("="*64)
    all_pass = True
    lines    = []
    for label, ok, dur in results:
        tag  = PASS_SYM if ok else FAIL_SYM
        line = f"  [{tag}]  {label:<44}  {dur:>6.1f}s"
        print(line)
        lines.append(line)
        if not ok:
            all_pass = False

    print("="*64)
    final_sym = "[PASS] ALL SIMULATION STAGES PASSED" if all_pass else \
                "[FAIL] ONE OR MORE STAGES FAILED"
    print(f"\n  {final_sym}\n")

    # -- Save report -------------------------------------------------------------
    rpt_path = r"d:\phase2\validation_report.txt"
    with open(rpt_path, 'w', encoding='utf-8') as f:
        f.write("CUSTOM 128-pt OFDM PHY -- VALIDATION REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*64 + "\n")
        for line in lines:
            f.write(line + "\n")
        f.write("="*64 + "\n")
        f.write(f"Overall: {'PASS' if all_pass else 'FAIL'}\n")
    print(f"  Report saved: {rpt_path}\n")

    # -- Hardware instructions ---------------------------------------------------
    if args.hardware:
        print("-"*64)
        print("  NEXT: HARDWARE STAGE (Stage 5)")
        print("-"*64)
        print("""
  Prerequisites:
    1. USRP B210 connected via USB3 (blue SuperSpeed port, no hub)
    2. TX antenna on Port A (TX/RX), RX antenna on Port B (RX2)
       CRITICAL: Do NOT swap A and B — T/R switch has ~10 dB isolation only.
    3. Antenna separation 0.5-1.0 m at chest height 0.9 m

  Step A -- Spectrum scan (pick frequency):
    python spectrum_scan.py --band 2.4
    -> Update USRP_CENTER_FREQ in config.py

  Step B -- Hardware bring-up + CRC check:
    Terminal 1:  python main_tx.py --mode hardware --mod BPSK
    Terminal 2:  python validate_stage5_hardware.py --n_packets 50

  Step C -- Full session dry run:
    Terminal 1:  python main_tx.py --mode hardware
    Terminal 2:  python main_rx.py --mode hardware --hdf5 dry_run.h5 --session_id dry_run

  Expected: CRC > 90%, dropped = 0, H5 file grows steadily
""")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
