import numpy as np
import config as cfg


def validate_interval(packet_interval_s, burst_len_hw, payload_bytes, modulation):
    burst_duration_s = burst_len_hw / float(cfg.FS_HW)

    if not np.isfinite(packet_interval_s) or packet_interval_s <= 0:
        return False

    if packet_interval_s < burst_duration_s:
        return False

    return True


print("=== FIX 4D TX INTERVAL VALIDATION ===")

# Normal deployment case: 100 B BPSK
normal_burst_hw = 5184
normal_duration_us = normal_burst_hw / cfg.FS_HW * 1e6

print(f"[INFO] Normal burst       : {normal_burst_hw} HW")
print(f"[INFO] Normal duration   : {normal_duration_us:.3f} us")
print(f"[INFO] Target interval   : {cfg.TX_PACKET_PERIOD_S * 1e6:.3f} us")

assert validate_interval(
    cfg.TX_PACKET_PERIOD_S,
    normal_burst_hw,
    100,
    "BPSK"
)
print("[PASS] 100 B BPSK fits the configured packet period.")

# Maximum supported burst
max_burst_hw = 8584
max_duration_us = max_burst_hw / cfg.FS_HW * 1e6

print(f"[INFO] Maximum burst      : {max_burst_hw} HW")
print(f"[INFO] Maximum duration  : {max_duration_us:.3f} us")

assert validate_interval(
    cfg.TX_PACKET_PERIOD_S,
    max_burst_hw,
    829,
    "16QAM"
)
print("[PASS] Maximum supported packet fits the configured period.")

# Impossible interval
impossible_interval = 100e-6

assert not validate_interval(
    impossible_interval,
    max_burst_hw,
    829,
    "16QAM"
)
print("[PASS] Impossible 100 us interval is rejected.")

# Zero interval
assert not validate_interval(
    0.0,
    normal_burst_hw,
    100,
    "BPSK"
)
print("[PASS] Zero interval is rejected.")

# Negative interval
assert not validate_interval(
    -1e-3,
    normal_burst_hw,
    100,
    "BPSK"
)
print("[PASS] Negative interval is rejected.")

# NaN
assert not validate_interval(
    float("nan"),
    normal_burst_hw,
    100,
    "BPSK"
)
print("[PASS] NaN interval is rejected.")

print()
print("FIX 4D TX INTERVAL VALIDATION COMPLETE")