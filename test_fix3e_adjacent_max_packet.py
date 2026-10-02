import numpy as np

import config as cfg
import waveform
import scrambler

from detector import PacketDetector
from sync import sync_packet


# ============================================================
# 3E-1
# Adjacent maximum-size packets
#
# Packet 1:
#   204 B BPSK
#   34 DATA symbols
#   6048 BB samples
#
# Packet 2:
#   204 B BPSK
#   34 DATA symbols
#   6048 BB samples
#
# There is NO guard between the packets.
# ============================================================

PAYLOAD_BYTES = 204
MODULATION = "BPSK"

PREFIX = 1000
SUFFIX = 7000


# ============================================================
# Generate two maximum-size packets
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
# Verify maximum packet length mathematically
# ============================================================

expected_packet_len = (
    cfg.STF_LEN
    + cfg.LTF_LEN
    + cfg.SIG_LEN
    + cfg.MAX_DATA_SYMS * cfg.SYMBOL_LEN
)

print()
print("MAXIMUM PACKET GEOMETRY")
print("-----------------------")
print("MAX_DATA_SYMS:", cfg.MAX_DATA_SYMS)
print("Expected maximum packet length:", expected_packet_len)

if len(packets[0]) != expected_packet_len:
    raise AssertionError(
        "Generated maximum BPSK packet does not occupy "
        "the configured maximum packet length."
    )

if len(packets[1]) != expected_packet_len:
    raise AssertionError(
        "Second generated maximum BPSK packet has "
        "unexpected length."
    )


# ============================================================
# Adjacent stream
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
print("Packet spacing:", true_start_2 - true_start_1)
print("Guard between packets:", 0)
print("Total stream length:", len(stream))


# ============================================================
# Detector with irregular chunks
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

print()
print("DETECTION RESULTS")
print("-----------------")

detected_starts = [
    int(x[0])
    for x in detections
]

print("Expected starts:", [
    true_start_1,
    true_start_2,
])

print("Detected starts:", detected_starts)

if len(detections) != 2:
    raise AssertionError(
        "Adjacent maximum-size packet test failed: "
        f"expected 2 detections, got {len(detections)}"
    )


# ============================================================
# Expected detector references
#
# The detector reference is the first threshold crossing,
# not necessarily the exact physical packet start.
#
# Earlier tests observed -3 samples. This adjacent-packet
# test produced -3 for packet 1 and -2 for packet 2.
# ============================================================

print()
print("DETECTOR OFFSETS")
print("----------------")

for i, (detected_start, true_start) in enumerate(
    zip(
        detected_starts,
        [true_start_1, true_start_2]
    )
):

    offset = detected_start - true_start

    print(
        f"Packet {i + 1} detector offset: "
        f"{offset} samples"
    )

    if abs(offset) > 8:
        raise AssertionError(
            f"Packet {i + 1}: detector boundary is "
            f"too far from true packet start: "
            f"{offset} samples"
        )


# ============================================================
# Suppression arithmetic
# ============================================================

if len(detections) >= 1:

    detected_1 = int(
        detections[0][0]
    )

    suppression_end = (
        detected_1
        + cfg.STF_LEN
        + cfg.LTF_LEN
        + cfg.SIG_LEN
        + cfg.MAX_DATA_SYMS * cfg.SYMBOL_LEN
    )

    print()
    print("SUPPRESSION ARITHMETIC")
    print("----------------------")
    print("Packet 1 detected start:", detected_1)
    print("suppress_until:", suppression_end)
    print("Packet 2 true start:", true_start_2)
    print(
        "Packet 2 start - suppress_until:",
        true_start_2 - suppression_end
    )

    if true_start_2 <= suppression_end:
        raise AssertionError(
            "Packet 2 begins inside the suppression interval."
        )


# ============================================================
# 3E-2: Downstream LTF synchronization
# ============================================================

HEADROOM = (
    cfg.STF_LEN
    + cfg.LTF_LEN
    + cfg.SIG_LEN
    + cfg.MAX_DATA_SYMS * cfg.SYMBOL_LEN
)

print()
print("DOWNSTREAM SYNCHRONIZATION")
print("--------------------------")
print("HEADROOM:", HEADROOM)

true_starts = [
    true_start_1,
    true_start_2,
]

for i, (detected_start, det_cfo) in enumerate(detections):

    true_start = true_starts[i]

    if (
        detected_start < 0
        or detected_start + HEADROOM > len(stream)
    ):
        raise AssertionError(
            f"Packet {i + 1}: insufficient samples "
            f"for HEADROOM extraction."
        )

    win = stream[
        detected_start:
        detected_start + HEADROOM
    ].astype(np.complex64)

    if len(win) != HEADROOM:
        raise AssertionError(
            f"Packet {i + 1}: incorrect extraction "
            f"length {len(win)} != {HEADROOM}"
        )

    true_start_in_window = (
        true_start - detected_start
    )

    if true_start_in_window < 0:
        raise AssertionError(
            f"Packet {i + 1}: detected start occurs "
            f"after true packet start."
        )

    true_ltf_cp_start = (
        true_start_in_window
        + cfg.STF_LEN
    )

    res = sync_packet(
        win,
        coarse_cfo_hz=det_cfo,
        n_data_symbols=0,
    )

    recovered_ltf_start = int(
        res["ltf_start"]
    )

    timing_error = (
        recovered_ltf_start
        - true_ltf_cp_start
    )

    print()
    print(f"Packet {i + 1}")
    print("  true start          :", true_start)
    print("  detected start      :", detected_start)
    print("  detector offset     :", detected_start - true_start)
    print("  true LTF CP         :", true_ltf_cp_start)
    print("  recovered LTF CP    :", recovered_ltf_start)
    print("  timing error        :", timing_error)

    if abs(timing_error) > 4:
        raise AssertionError(
            f"Packet {i + 1}: downstream LTF timing "
            f"error too large: {timing_error} samples"
        )

print()
print("3E-2 downstream synchronization: PASS")


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
print("3E-1 adjacent maximum-packet test: PASS")
print("3E-2 downstream synchronization: PASS")
print("Both packets detected.")
print("Both LTF boundaries recovered.")
print("Detector indexing invariant: PASS")