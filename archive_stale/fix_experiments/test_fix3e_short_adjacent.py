import numpy as np

import config as cfg
import waveform
import scrambler

from detector import PacketDetector


# ============================================================
# 3E-3
# Short packet immediately followed by another short packet
#
# Packet 1:
#   100 B BPSK
#   17 DATA symbols
#   3328 BB samples
#
# Packet 2:
#   100 B BPSK
#   17 DATA symbols
#   3328 BB samples
#
# There is NO guard between the packets.
#
# This test measures the effect of the detector's
# MAX_DATA_SYMS-based suppression interval.
# ============================================================

PAYLOAD_BYTES = 100
MODULATION = "BPSK"

PREFIX = 1000
SUFFIX = 7000


# ============================================================
# Generate two packets
# ============================================================

packets = []

for packet_number in range(2):

    payload_bits = (
        (np.arange(PAYLOAD_BYTES * 8, dtype=np.uint8)
         + packet_number)
        & 1
    )

    pkt = waveform.assemble_packet(
        payload_bits,
        modulation=MODULATION,
        scrambler_mod=scrambler,
        encoder_mod=scrambler,
        mapper_fn=scrambler.map_bits_to_symbols,
        idle_samples=0,
    )

    pkt = np.asarray(pkt, dtype=np.complex64)

    packets.append(pkt)

    print(
        f"Packet {packet_number + 1}: "
        f"{PAYLOAD_BYTES} B {MODULATION}, "
        f"length={len(pkt)}"
    )


# ============================================================
# Verify actual packet length
# ============================================================

expected_short_packet_len = (
    cfg.STF_LEN
    + cfg.LTF_LEN
    + cfg.SIG_LEN
    + 17 * cfg.SYMBOL_LEN
)

print()
print("SHORT PACKET GEOMETRY")
print("---------------------")
print("Expected 100 B BPSK length:",
      expected_short_packet_len)
print("Actual packet 1 length:", len(packets[0]))
print("Actual packet 2 length:", len(packets[1]))

if len(packets[0]) != expected_short_packet_len:
    raise AssertionError(
        "Packet 1 has unexpected length."
    )

if len(packets[1]) != expected_short_packet_len:
    raise AssertionError(
        "Packet 2 has unexpected length."
    )


# ============================================================
# Construct zero-guard stream
# ============================================================

true_start_1 = PREFIX

true_start_2 = (
    true_start_1
    + len(packets[0])
)

stream = np.concatenate([
    np.zeros(PREFIX, dtype=np.complex64),
    packets[0],
    packets[1],
    np.zeros(SUFFIX, dtype=np.complex64),
]).astype(np.complex64)

print()
print("STREAM GEOMETRY")
print("---------------")
print("True packet 1 start:", true_start_1)
print("True packet 2 start:", true_start_2)
print(
    "Packet spacing:",
    true_start_2 - true_start_1
)
print("Guard between packets:", 0)
print("Total stream length:", len(stream))


# ============================================================
# Calculate suppression boundary
# ============================================================

expected_detector_start_1 = true_start_1 - 3

suppression_end = (
    expected_detector_start_1
    + cfg.STF_LEN
    + cfg.LTF_LEN
    + cfg.SIG_LEN
    + cfg.MAX_DATA_SYMS * cfg.SYMBOL_LEN
)

print()
print("SUPPRESSION GEOMETRY")
print("--------------------")
print("Maximum suppression length:",
      cfg.STF_LEN
      + cfg.LTF_LEN
      + cfg.SIG_LEN
      + cfg.MAX_DATA_SYMS * cfg.SYMBOL_LEN)

print(
    "Expected packet 1 detector start:",
    expected_detector_start_1
)

print(
    "Expected suppress_until:",
    suppression_end
)

print(
    "Packet 2 true start:",
    true_start_2
)

print(
    "Packet 2 start - suppress_until:",
    true_start_2 - suppression_end
)


# ============================================================
# Run detector with irregular chunks
# ============================================================

det = PacketDetector()

chunk_sizes = [
    137,
    509,
    73,
    1000,
    257,
    701,
    333,
    911,
    149,
    1200,
    421,
    683,
    97,
    1500,
    271,
]

detections = []

pos = 0
chunk_index = 0

while pos < len(stream):

    n = chunk_sizes[
        chunk_index % len(chunk_sizes)
    ]

    end = min(
        pos + n,
        len(stream)
    )

    chunk = stream[pos:end]

    chunk_dets = det.process(chunk)

    if chunk_dets:
        print(
            f"chunk {chunk_index:03d}: "
            f"stream [{pos}:{end}] -> "
            f"detections {chunk_dets}"
        )

    detections.extend(chunk_dets)

    pos = end
    chunk_index += 1


# ============================================================
# Results
# ============================================================

detected_starts = [
    int(x[0])
    for x in detections
]

print()
print("DETECTION RESULTS")
print("-----------------")
print("Expected physical starts:", [
    true_start_1,
    true_start_2,
])
print("Detected starts:", detected_starts)
print("Number detected:", len(detections))


# ============================================================
# This is an OBSERVATION test.
#
# We expect the current detector to suppress packet 2 because
# packet 2 starts before suppress_until.
#
# Do NOT modify detector.py based only on this result.
# ============================================================

print()
print("INTERPRETATION")
print("--------------")

if true_start_2 < suppression_end:

    print(
        "Packet 2 begins INSIDE the current "
        "maximum-length suppression interval."
    )

    print(
        "Therefore the current detector is expected "
        "to suppress packet 2."
    )

else:

    print(
        "Packet 2 begins OUTSIDE the suppression interval."
    )


if len(detections) == 1:

    print()
    print(
        "OBSERVED: Packet 1 detected and Packet 2 suppressed."
    )

elif len(detections) == 2:

    print()
    print(
        "OBSERVED: Both packets detected despite "
        "the nominal suppression interval."
    )

else:

    print()
    print(
        "OBSERVED: Unexpected number of detections:",
        len(detections)
    )


# ============================================================
# Verify packet 1 was detected correctly
# ============================================================

if len(detections) < 1:
    raise AssertionError(
        "Packet 1 was not detected."
    )

packet_1_offset = (
    detections[0][0]
    - true_start_1
)

print()
print("PACKET 1 CHECK")
print("--------------")
print("True start:", true_start_1)
print("Detected:", detections[0][0])
print("Offset:", packet_1_offset)

if abs(packet_1_offset) > 8:
    raise AssertionError(
        "Packet 1 detector boundary is too far "
        "from the true packet start."
    )


# ============================================================
# Final detector indexing invariant
# ============================================================

final_index = (
    det._sample_idx
    + len(det._buf)
)

print()
print("FINAL DETECTOR STATE")
print("--------------------")
print("sample_idx:", det._sample_idx)
print("buffer_len:", len(det._buf))
print("sum:", final_index)
print("stream_len:", len(stream))

if final_index != len(stream):
    raise AssertionError(
        "Detector indexing invariant failed: "
        f"{final_index} != {len(stream)}"
    )


# ============================================================
# Final result
# ============================================================

print()
print("RESULT")
print("------")
print("3E-3 short adjacent-packet suppression test: COMPLETE")
print("Packet 1 detection: PASS")
print("Detector indexing invariant: PASS")

if len(detections) == 1:
    print(
        "Current behavior confirms that Packet 2 is "
        "suppressed by the maximum-length suppression rule."
    )
elif len(detections) == 2:
    print(
        "Current behavior confirms that both packets "
        "are detected."
    )
else:
    print(
        "Current behavior produced an unexpected "
        "detection count."
    )