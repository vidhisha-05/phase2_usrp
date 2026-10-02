import numpy as np

import config as cfg
import waveform
import scrambler
from detector import PacketDetector


# ============================================================
# Test configuration
# ============================================================

PREFIX_LEN = 1000
GUARD_LEN = 7000
SUFFIX_LEN = 1000

CASES = [
    ("BPSK", 100),
    ("QPSK", 100),
    ("16QAM", 100),
]


# ============================================================
# Build deterministic packets
# ============================================================

rng = np.random.default_rng(20261001)

packets = []
expected_starts = []

absolute_pos = PREFIX_LEN

for modulation, n_bytes in CASES:

    payload_bits = rng.integers(
        0,
        2,
        size=n_bytes * 8,
        dtype=np.uint8,
    )

    pkt = waveform.assemble_packet(
        payload_bits,
        modulation=modulation,
        scrambler_mod=scrambler,
        encoder_mod=scrambler,
        mapper_fn=scrambler.map_bits_to_symbols,
        idle_samples=0,
    )

    packets.append(pkt)

    expected_starts.append(absolute_pos)

    absolute_pos += len(pkt) + GUARD_LEN


# ============================================================
# Construct continuous stream
# ============================================================

parts = []

parts.append(
    np.zeros(
        PREFIX_LEN,
        dtype=np.complex64,
    )
)

for i, pkt in enumerate(packets):

    parts.append(pkt.astype(np.complex64))

    if i < len(packets) - 1:
        parts.append(
            np.zeros(
                GUARD_LEN,
                dtype=np.complex64,
            )
        )

parts.append(
    np.zeros(
        SUFFIX_LEN,
        dtype=np.complex64,
    )
)

stream = np.concatenate(parts)


# ============================================================
# Print geometry
# ============================================================

print("STREAM GEOMETRY")
print("----------------")

print("Prefix:", PREFIX_LEN)

for i, ((modulation, n_bytes), pkt, expected) in enumerate(
    zip(CASES, packets, expected_starts),
    start=1,
):
    print(
        f"Packet {i}: "
        f"{modulation}, {n_bytes} B, "
        f"length={len(pkt)}, "
        f"expected_start={expected}"
    )

print("Guard:", GUARD_LEN)
print("Suffix:", SUFFIX_LEN)
print("Total stream length:", len(stream))

print()
print(
    "Detector minimum buffer:",
    cfg.STF_LEN + cfg.LTF_LEN,
)

print(
    "Detector maximum suppression length:",
    cfg.STF_LEN
    + cfg.LTF_LEN
    + cfg.SIG_LEN
    + cfg.MAX_DATA_SYMS * cfg.SYMBOL_LEN,
)

assert GUARD_LEN > (
    cfg.STF_LEN
    + cfg.LTF_LEN
    + cfg.SIG_LEN
    + cfg.MAX_DATA_SYMS * cfg.SYMBOL_LEN
)


# ============================================================
# Feed detector in irregular chunks
#
# This is important:
# packet boundaries are NOT aligned with process() calls.
# ============================================================

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


det = PacketDetector()

detections = []

pos = 0
chunk_index = 0

while pos < len(stream):

    chunk_len = chunk_sizes[
        chunk_index % len(chunk_sizes)
    ]

    chunk = stream[
        pos:min(pos + chunk_len, len(stream))
    ]

    chunk_dets = det.process(chunk)

    if chunk_dets:
        print(
            f"chunk {chunk_index:03d}: "
            f"stream [{pos}:{pos + len(chunk)}] "
            f"-> detections {chunk_dets}"
        )

    detections.extend(chunk_dets)

    pos += len(chunk)
    chunk_index += 1


# ============================================================
# Detection results
# ============================================================

print()
print("DETECTION RESULTS")
print("-----------------")

print("Expected starts:", expected_starts)
print("Detected starts:", [d[0] for d in detections])

print("Number expected:", len(expected_starts))
print("Number detected:", len(detections))

assert len(detections) == len(expected_starts), (
    f"Expected {len(expected_starts)} detections, "
    f"got {len(detections)}"
)


detected_starts = [int(d[0]) for d in detections]

assert detected_starts == expected_starts, (
    f"Detection starts mismatch:\n"
    f"expected={expected_starts}\n"
    f"detected={detected_starts}"
)


# ============================================================
# Final detector-state invariant
# ============================================================

print()
print("FINAL DETECTOR STATE")
print("--------------------")

print("_sample_idx:", det._sample_idx)
print("buffer length:", len(det._buf))
print(
    "_sample_idx + len(_buf):",
    det._sample_idx + len(det._buf),
)
print("total stream length:", len(stream))

assert (
    det._sample_idx + len(det._buf)
    == len(stream)
), (
    "Detector absolute-index invariant failed: "
    f"{det._sample_idx} + {len(det._buf)} "
    f"!= {len(stream)}"
)


print()
print("==============================================")
print("FIX 3D-1 — MULTI-PACKET BOUNDARY TEST: PASS")
print("==============================================")