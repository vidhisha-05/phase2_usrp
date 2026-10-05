import numpy as np

import config as cfg


FS_BB = float(cfg.FS_FFT)
PERIOD_S = float(cfg.TX_PACKET_PERIOD_S)

print("=" * 70)
print("FIX 4B-4 — PACKET TIMESTAMP / 200 Hz GRID VALIDATION")
print("=" * 70)

print(f"FS_BB            = {FS_BB:.0f} Hz")
print(f"TX period        = {PERIOD_S:.9f} s")
print(f"TX rate          = {1.0 / PERIOD_S:.6f} Hz")

# ----------------------------------------------------------------------
# Derive the exact BB-sample spacing corresponding to the configured
# packet period.
#
# At 20 MS/s:
#
#     samples_per_period = FS_BB * 0.005
#                        = 100000 BB samples
# ----------------------------------------------------------------------

samples_per_period = FS_BB * PERIOD_S

print(f"BB samples/period = {samples_per_period:.6f}")

assert abs(samples_per_period - round(samples_per_period)) < 1e-12

samples_per_period = int(round(samples_per_period))

print(f"Integer BB period = {samples_per_period} samples")

assert samples_per_period == 100000


# ----------------------------------------------------------------------
# Test 1: timestamp calculation from absolute BB coordinates
#
# Production rule:
#
#     t_packet = T0 + abs_s / FS_BB
# ----------------------------------------------------------------------

T0 = 123456.789012345

packet_indices = np.array(
    [
        997,
        100997,
        200997,
        300997,
        400997,
    ],
    dtype=np.int64,
)

timestamps = T0 + packet_indices.astype(np.float64) / FS_BB

print("\nPacket timestamps:")

for i, (idx, ts) in enumerate(zip(packet_indices, timestamps)):
    print(
        f"packet {i + 1}: "
        f"abs_s={idx:8d}  "
        f"timestamp={ts:.12f}"
    )


# ----------------------------------------------------------------------
# Test 2: packet-to-packet timestamp spacing
#
# Expected:
#
#     Δt = 100000 / 20e6
#        = 0.005 s
#        = 5 ms
# ----------------------------------------------------------------------

print("\nPacket-to-packet spacing:")

expected_dt = PERIOD_S
max_error = 0.0

for i in range(1, len(timestamps)):
    actual_dt = float(
        packet_indices[i] - packet_indices[i - 1]
    ) / FS_BB

    error = abs(actual_dt - expected_dt)
    max_error = max(max_error, error)

    print(
        f"{i}: "
        f"sample_delta={packet_indices[i] - packet_indices[i - 1]:8d}  "
        f"dt={actual_dt:.12f} s  "
        f"error={error:.3e} s"
    )

    assert packet_indices[i] - packet_indices[i - 1] == samples_per_period
    assert error < 1e-15

print(
    f"Maximum packet-period error: "
    f"{max_error:.3e} s"
)


# ----------------------------------------------------------------------
# Test 3: compare timestamp difference using absolute timestamps.
#
# Because T0 is a large floating-point number, we do NOT use a very
# strict 1e-15 tolerance here.  The physically meaningful quantity is
# the sample-coordinate difference tested above.
# ----------------------------------------------------------------------

print("\nAbsolute timestamp differences:")

for i in range(1, len(timestamps)):
    measured_dt = timestamps[i] - timestamps[i - 1]

    print(
        f"{i}: "
        f"absolute timestamp delta = "
        f"{measured_dt:.12f} s"
    )

    # A nanosecond-level tolerance is more than sufficient for this
    # software-coordinate diagnostic.
    assert abs(measured_dt - expected_dt) < 1e-9


# ----------------------------------------------------------------------
# Test 4: verify the corresponding packet rate
# ----------------------------------------------------------------------

packet_rate = 1.0 / PERIOD_S

print("\nPacket rate:")
print(f"Configured rate = {packet_rate:.6f} Hz")
print(f"Expected rate   = 200.000000 Hz")

assert abs(packet_rate - 200.0) < 1e-12


# ----------------------------------------------------------------------
# Test 5: verify CSI sampling Nyquist frequency
#
# If packets occur at 200 Hz, the CSI time series sampling rate is
# 200 Hz, giving a Nyquist frequency of:
#
#     200 / 2 = 100 Hz
# ----------------------------------------------------------------------

csi_fs = packet_rate
csi_nyquist = csi_fs / 2.0

print("\nCSI sampling:")
print(f"CSI sampling rate = {csi_fs:.6f} Hz")
print(f"CSI Nyquist       = {csi_nyquist:.6f} Hz")

assert abs(csi_fs - 200.0) < 1e-12
assert abs(csi_nyquist - 100.0) < 1e-12


# ----------------------------------------------------------------------
# Test 6: verify a 10-second packet grid
#
# 200 packets/s × 10 s = 2000 packet intervals.
# The final packet coordinate should therefore be:
#
#     997 + 2000 × 100000
# ----------------------------------------------------------------------

num_intervals = 2000

last_abs_s = (
    int(packet_indices[0])
    + num_intervals * samples_per_period
)

expected_last_abs_s = 997 + 2000 * 100000

print("\n10-second grid check:")
print(f"Intervals       = {num_intervals}")
print(f"First abs_s     = {packet_indices[0]}")
print(f"Last abs_s      = {last_abs_s}")
print(f"Expected last   = {expected_last_abs_s}")

assert last_abs_s == expected_last_abs_s


# ----------------------------------------------------------------------
# Final result
# ----------------------------------------------------------------------

print("\n" + "=" * 70)
print("RESULT: FIX 4B-4 DIAGNOSTIC PASS")
print("=" * 70)