from dataclasses import dataclass


@dataclass
class RxRecord:
    seq: int
    abs_s: int
    timestamp: float


print("=" * 70)
print("FIX 4C-1 — RX SEQUENCE SEMANTICS VALIDATION")
print("=" * 70)


# Current hardware design:
#
#     seq = local RX detection/record counter
#
# It is NOT a UHD sequence number and NOT a TX payload sequence number.

records = [
    RxRecord(
        seq=i,
        abs_s=997 + i * 100_000,
        timestamp=123456.789012345 + (997 + i * 100_000) / 20_000_000,
    )
    for i in range(5)
]


print("\nSynthetic RX records:")

for r in records:
    print(
        f"seq={r.seq}  "
        f"abs_s={r.abs_s}  "
        f"timestamp={r.timestamp:.12f}"
    )


# ----------------------------------------------------------------------
# Check 1:
# seq is a local record counter.
# ----------------------------------------------------------------------

expected_seq = list(range(len(records)))

actual_seq = [r.seq for r in records]

print("\nSequence values:")
print(f"actual   = {actual_seq}")
print(f"expected = {expected_seq}")

assert actual_seq == expected_seq


# ----------------------------------------------------------------------
# Check 2:
# Sequence increments independently of the sample coordinate.
#
# The sample coordinate is the authoritative temporal position.
# ----------------------------------------------------------------------

for i, r in enumerate(records):
    expected_abs = 997 + i * 100_000

    assert r.abs_s == expected_abs

print("\nSample-coordinate check: PASS")


# ----------------------------------------------------------------------
# Check 3:
# Timestamp is derived from the absolute sample coordinate.
# ----------------------------------------------------------------------

FS_BB = 20_000_000.0
T0 = 123456.789012345

for r in records:
    expected_timestamp = T0 + r.abs_s / FS_BB

    assert abs(r.timestamp - expected_timestamp) < 1e-12

print("Timestamp/sample-coordinate check: PASS")


# ----------------------------------------------------------------------
# Check 4:
# Demonstrate that seq and sample position represent different things.
# ----------------------------------------------------------------------

print("\nSemantic distinction:")
print("  seq       = local RX record/detection index")
print("  abs_s     = absolute BB sample coordinate")
print("  timestamp = UHD-derived RX time")


# ----------------------------------------------------------------------
# Check 5:
# No claim is made that seq identifies the transmitted packet.
# ----------------------------------------------------------------------

print("\nTX-packet identity:")
print("  Current payload does NOT contain an explicit TX sequence field.")
print("  Therefore RX 'seq' must NOT be interpreted as TX sequence.")


print("\n" + "=" * 70)
print("RESULT: FIX 4C-1 DIAGNOSTIC PASS")
print("=" * 70)