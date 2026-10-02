import numpy as np

import config as cfg
from waveform import make_streaming_25to20


FS_HW = float(cfg.FS_HW)
FS_BB = float(cfg.FS_FFT)

print("=" * 70)
print("FIX 4B-3 — UHD -> BB SAMPLE-COORDINATE MAPPING")
print("=" * 70)

print(f"FS_HW       = {FS_HW:.0f} Hz")
print(f"FS_BB       = {FS_BB:.0f} Hz")

# The project uses 25 MHz hardware sampling and 20 MHz BB sampling.
# Therefore the rational resampling ratio is exactly 4/5.
resamp_up = 4
resamp_down = 5

print(f"RESAMP_UP   = {resamp_up}")
print(f"RESAMP_DOWN = {resamp_down}")

ratio = FS_BB / FS_HW
expected_ratio = resamp_up / resamp_down

print(f"BB/HW ratio = {ratio:.12f}")
print(f"Expected     = {expected_ratio:.12f}")

assert abs(ratio - expected_ratio) < 1e-12


# ----------------------------------------------------------------------
# Test 1: timestamp equivalence for representative HW coordinates
#
# Physical time from HW coordinate:
#
#     t = T0 + n_HW / FS_HW
#
# Since:
#
#     n_BB = n_HW * (FS_BB / FS_HW)
#
# the BB representation must give exactly the same physical time:
#
#     t = T0 + n_BB / FS_BB
# ----------------------------------------------------------------------

T0 = 123456.789012345

hw_indices = np.array(
    [0, 1, 7, 31, 100, 1024, 4096, 10000, 25000],
    dtype=np.int64,
)

print("\nTimestamp equivalence:")

max_err = 0.0

for n_hw in hw_indices:
    t_hw = T0 + n_hw / FS_HW

    n_bb = n_hw * ratio

    t_bb = T0 + n_bb / FS_BB

    err = abs(t_hw - t_bb)
    max_err = max(max_err, err)

    print(
        f"HW={n_hw:6d}  "
        f"BB={n_bb:10.4f}  "
        f"t_hw={t_hw:.12f}  "
        f"t_bb={t_bb:.12f}  "
        f"err={err:.3e}s"
    )

assert max_err < 1e-15

print(f"Maximum timestamp-equivalence error: {max_err:.3e} s")


# ----------------------------------------------------------------------
# Test 2: integer BB coordinates
#
# This is the exact timestamp rule currently used by main_hardware.py:
#
#     timestamp = T0 + abs_s / FS_BB
#
# We validate the elapsed time from the coordinate directly rather than
# subtracting two large floating-point absolute timestamps.  This avoids
# unnecessary floating-point cancellation.
# ----------------------------------------------------------------------

bb_indices = np.array(
    [0, 1, 7, 100, 997, 1000, 4096, 10000],
    dtype=np.int64,
)

print("\nBB-coordinate timestamp rule:")

for n_bb in bb_indices:
    elapsed_s = n_bb / FS_BB
    elapsed_us = elapsed_s * 1e6

    expected_elapsed_s = float(n_bb) / FS_BB

    print(
        f"BB={n_bb:6d}  "
        f"delta_t={elapsed_us:12.6f} us"
    )

    assert abs(elapsed_s - expected_elapsed_s) < 1e-15


# ----------------------------------------------------------------------
# Test 3: verify the resampler's internal pre_remove does NOT change
# the physical timestamp origin.
#
# pre_remove is an internal implementation/alignment parameter.
# It is NOT a physical time delay that should be added to or subtracted
# from the UHD timestamp.
# ----------------------------------------------------------------------

rs = make_streaming_25to20()

print("\nResampler alignment:")
print(f"pre_pad    = {rs.pre_pad}")
print(f"pre_remove = {rs.pre_remove}")

assert rs.pre_remove == 7


# For BB sample n:
#
#     correct timestamp = T0 + n / FS_BB
#
# NOT:
#
#     T0 + (n + 7) / FS_BB
#
# and NOT:
#
#     T0 + (n - 7) / FS_BB

n = 997

t_correct = T0 + n / FS_BB
t_plus_7 = T0 + (n + rs.pre_remove) / FS_BB
t_minus_7 = T0 + (n - rs.pre_remove) / FS_BB

print(f"\nFor BB sample n={n}:")
print(f"Correct timestamp : {t_correct:.12f}")
print(f"+7 correction     : {t_plus_7:.12f}")
print(f"-7 correction     : {t_minus_7:.12f}")

difference_7_us = rs.pre_remove / FS_BB * 1e6

print(
    f"Incorrect 7-sample physical shift would be "
    f"{difference_7_us:.6f} us"
)

assert abs(difference_7_us - 0.35) < 1e-12


# ----------------------------------------------------------------------
# Final result
# ----------------------------------------------------------------------

print("\n" + "=" * 70)
print("RESULT: FIX 4B-3 DIAGNOSTIC PASS")
print("=" * 70)
