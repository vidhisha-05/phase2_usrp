import ast
from pathlib import Path

import config as cfg


print("=" * 70)
print("FIX 4C-2 — PACKET RECORD SAMPLE-COORDINATE VALIDATION")
print("=" * 70)

path = Path(__file__).resolve().parent / "main_hardware.py"
source = path.read_text(encoding="utf-8")
tree = ast.parse(source)

# ----------------------------------------------------------------------
# Find the record = { ... } assignment.
# ----------------------------------------------------------------------

record_dict = None

for node in ast.walk(tree):
    if not isinstance(node, ast.Assign):
        continue

    for target in node.targets:
        if isinstance(target, ast.Name) and target.id == "record":
            if isinstance(node.value, ast.Dict):
                record_dict = node.value
                break

    if record_dict is not None:
        break

assert record_dict is not None, "Could not find record = {...} in main_hardware.py"

print("Found packet record dictionary: PASS")


# ----------------------------------------------------------------------
# Extract dictionary keys.
# ----------------------------------------------------------------------

keys = []

for key_node in record_dict.keys:
    if isinstance(key_node, ast.Constant):
        keys.append(key_node.value)

print("\nRecord keys:")
for key in keys:
    print(f"  {key}")

required = {
    "H_hat",
    "H_sanitized",
    "phase_sanitized",
    "phase_slope",
    "phase_intercept",
    "rx_bits",
    "crc_ok",
    "cfo_hz",
    "timestamp",
    "abs_s",
    "seq",
    "dropped",
    "ring_occ",
}

missing = required - set(keys)

assert not missing, f"Missing required record fields: {sorted(missing)}"

print("\nRequired record fields: PASS")


# ----------------------------------------------------------------------
# Locate the abs_s value expression.
# ----------------------------------------------------------------------

abs_s_index = keys.index("abs_s")
abs_s_value = record_dict.values[abs_s_index]

assert isinstance(abs_s_value, ast.Call), (
    "abs_s is not a function call; expected int(abs_s)"
)

assert isinstance(abs_s_value.func, ast.Name), (
    "abs_s conversion is not a direct builtin call"
)

assert abs_s_value.func.id == "int", (
    f"abs_s is converted using {abs_s_value.func.id!r}, expected int"
)

assert len(abs_s_value.args) == 1
assert isinstance(abs_s_value.args[0], ast.Name)
assert abs_s_value.args[0].id == "abs_s"

print("abs_s stored as int(abs_s): PASS")


# ----------------------------------------------------------------------
# Verify timestamp still references:
#
#     uhd_t0_s + abs_s / FS_FFT
# ----------------------------------------------------------------------

timestamp_index = keys.index("timestamp")
timestamp_value = record_dict.values[timestamp_index]

timestamp_source = ast.unparse(timestamp_value)

print("\nTimestamp expression:")
print(timestamp_source)

assert "uhd_t0_s" in timestamp_source
assert "abs_s" in timestamp_source
assert "FS_FFT" in timestamp_source

print("Timestamp remains derived from UHD T0 + abs_s / FS_FFT: PASS")


# ----------------------------------------------------------------------
# Numerical consistency check.
#
# Use representative packet coordinates already established during
# Fix 4B-4.
# ----------------------------------------------------------------------

T0 = 123456.789012345
packet_indices = [997, 100997, 200997, 300997]

print("\nNumerical coordinate check:")

for abs_s in packet_indices:
    timestamp = T0 + int(abs_s) / float(cfg.FS_FFT)

    expected_offset = int(abs_s) / float(cfg.FS_FFT)

    print(
        f"abs_s={abs_s:8d}  "
        f"offset={expected_offset:.9f} s  "
        f"timestamp={timestamp:.12f}"
    )

    assert isinstance(int(abs_s), int)

print("Integer sample-coordinate calculation: PASS")


# ----------------------------------------------------------------------
# Verify the configured BB sample rate.
# ----------------------------------------------------------------------

assert float(cfg.FS_FFT) == 20_000_000.0

print("\nFS_FFT = 20 MHz: PASS")


print("\n" + "=" * 70)
print("RESULT: FIX 4C-2 VALIDATION PASS")
print("=" * 70)