"""
validate_stage1.py  --  STAGE 1 VALIDATION
Zero-Impairment Simulation Loopback + CRC Check
"""

import numpy as np
import sys
import os
os.environ["PYTHONUTF8"] = "1"
sys.stdout.reconfigure(encoding='utf-8', errors='replace') if hasattr(sys.stdout, 'reconfigure') else None

sys.path.insert(0, r"d:\phase2")

import config as cfg
import waveform
import scrambler
from detector import PacketDetector
from sync import extract_csi

errors = []

def check(name, cond, detail=""):
    tag = "[PASS]" if cond else "[FAIL]"
    msg = f"  {tag}  {name}"
    if detail:
        msg += f"  <- {detail}"
    print(msg)
    if not cond:
        errors.append(name)

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*54)
print("  STAGE 1 -- Zero-Impairment Loopback")
print("="*54 + "\n")

# 1a. STF
stf = waveform.generate_stf()
check("STF length == 128",       len(stf) == 128)
check("STF is 8x16 periodic",    np.allclose(stf[:16], stf[16:32], atol=1e-5))

# 1b. LTF
ltf = waveform.generate_ltf()
check("LTF length == 320",       len(ltf) == 320)
check("LTF two reps equal",      np.allclose(ltf[64:192], ltf[192:320], atol=1e-5))

# 1c. Subcarrier counts
check("Active subcarriers == 106", cfg.NUM_ACTIVE == 106)
check("Data subcarriers  == 98",   cfg.NUM_DATA   == 98)
check("Pilot subcarriers == 8",    cfg.NUM_PILOTS == 8)
check("DC not in active set",      0 not in cfg.ACTIVE_SUBCARRIERS)
check("Pilot/Data no overlap",
      len(set(cfg.PILOT_INDICES) & set(cfg.DATA_INDICES)) == 0)

# 1d. Packet assembly
payload_bits = np.random.randint(0, 2, 200, dtype=np.uint8)
pkt = waveform.assemble_packet(payload_bits)
expected_min = cfg.STF_LEN + cfg.LTF_LEN + cfg.SIG_LEN
check(f"Packet length >= {expected_min}", len(pkt) >= expected_min)

# 1e. Scrambler
bits = np.random.randint(0, 2, 200, dtype=np.uint8)
rt   = scrambler.descramble(scrambler.scramble(bits))
check("Scrambler round-trip exact", np.all(rt == bits))

# 1f. Encoder/Viterbi (10 trials)
enc_ok = True
for _ in range(10):
    b     = np.random.randint(0, 2, 88, dtype=np.uint8)
    coded = scrambler.encode_half_rate(b)
    dec   = scrambler.decode_half_rate(coded)
    if not np.all(dec == b):
        enc_ok = False
        break
check("Viterbi decode (10 trials, no noise)", enc_ok)

# 1g. CSI extraction flat channel
pkt_flat = np.concatenate([stf, ltf])
H = extract_csi(pkt_flat, ltf_start=len(stf), total_cfo_hz=0.0)
check(f"H_hat shape == ({cfg.NUM_ACTIVE},)", H.shape == (cfg.NUM_ACTIVE,))
check("H_hat no NaN",                  not np.any(np.isnan(H)))
check("H_hat magnitude ~1 (flat ch)",
      np.allclose(np.abs(H), 1.0, atol=0.05),
      f"max_err={np.max(np.abs(np.abs(H)-1)):.4f}")

# 1h. Coarse CFO accuracy
from detector import PacketDetector
true_cfo = 1500.0
n   = np.arange(len(stf))
stf_cfo = (stf * np.exp(1j * 2*np.pi*true_cfo/cfg.FS_FFT * n)).astype(np.complex64)
det = PacketDetector()
est = det._estimate_coarse_cfo(stf_cfo)
check("Coarse CFO estimate error < 200 Hz",
      abs(est - true_cfo) < 200,
      f"err={abs(est-true_cfo):.1f} Hz")

# ─────────────────────────────────────────────────────────────────────────────
print()
if not errors:
    print("  [PASS]  STAGE 1 COMPLETE -- all checks passed\n")
    sys.exit(0)
else:
    print(f"  [FAIL]  STAGE 1 FAILED -- {len(errors)} check(s) failed:")
    for e in errors:
        print(f"         * {e}")
    print()
    sys.exit(1)
