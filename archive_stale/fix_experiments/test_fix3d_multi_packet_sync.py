import numpy as np

import config as cfg
import waveform
import scrambler

from detector import PacketDetector
from sync import sync_packet


# ============================================================
# TEST PARAMETERS
# ============================================================

PREFIX = 1000
GUARD = 7000
SUFFIX = 6000

PACKETS = [
    (100, "BPSK"),
    (100, "QPSK"),
    (100, "16QAM"),
]


# ============================================================
# 1. Generate packets
# ============================================================

packets = []
expected_starts = []

cursor = PREFIX

for payload_bytes, modulation in PACKETS:

    payload_bits = (
        np.arange(payload_bytes * 8, dtype=np.uint8) & 1
    )

    pkt = waveform.assemble_packet(
        payload_bits,
        modulation=modulation,
        scrambler_mod=scrambler,
        encoder_mod=scrambler,
        mapper_fn=scrambler.map_bits_to_symbols,
        idle_samples=0,
    )

    pkt = np.asarray(pkt, dtype=np.complex64)

    packets.append(pkt)
    expected_starts.append(cursor)

    print(
        f"{modulation:>5} | "
        f"payload={payload_bytes:3d} B | "
        f"packet_len={len(pkt):4d} | "
        f"expected_start={cursor}"
    )

    cursor += len(pkt) + GUARD


# ============================================================
# 2. Construct continuous stream
# ============================================================

parts = [
    np.zeros(PREFIX, dtype=np.complex64)
]

for i, pkt in enumerate(packets):

    parts.append(pkt)

    if i != len(packets) - 1:
        parts.append(
            np.zeros(GUARD, dtype=np.complex64)
        )

parts.append(
    np.zeros(SUFFIX, dtype=np.complex64)
)

stream = np.concatenate(parts).astype(np.complex64)

print()
print("STREAM")
print("------")
print("Expected starts:", expected_starts)
print("Total samples:", len(stream))


# ============================================================
# 3. Run detector using irregular chunks
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

    n = chunk_sizes[chunk_index % len(chunk_sizes)]

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
# 4. Detector result checks
# ============================================================

print()
print("DETECTOR RESULTS")
print("----------------")

detected_starts = [
    int(x[0])
    for x in detections
]

print("Expected starts:", expected_starts)
print("Detected starts:", detected_starts)

if len(detections) != len(PACKETS):
    raise AssertionError(
        f"Expected {len(PACKETS)} detections, "
        f"got {len(detections)}"
    )


# ============================================================
# 5. Validate each packet's downstream synchronization
# ============================================================

HEADROOM = (
    cfg.STF_LEN
    + cfg.LTF_LEN
    + cfg.SIG_LEN
    + cfg.MAX_DATA_SYMS * cfg.SYMBOL_LEN
)

print()
print("DOWNSTREAM SYNC")
print("----------------")
print("HEADROOM:", HEADROOM)

timing_errors = []

for i, ((payload_bytes, modulation), true_start) in enumerate(
    zip(PACKETS, expected_starts)
):

    det_start, det_cfo = detections[i]

    print()
    print(f"PACKET {i + 1}")
    print(f"  modulation      : {modulation}")
    print(f"  payload         : {payload_bytes} B")
    print(f"  true start      : {true_start}")
    print(f"  detected start  : {det_start}")
    print(f"  detector offset : {det_start - true_start}")

    if det_start < 0:
        raise AssertionError(
            f"Packet {i + 1}: negative detector start"
        )

    if det_start + HEADROOM > len(stream):
        raise AssertionError(
            f"Packet {i + 1}: insufficient samples "
            f"for HEADROOM"
        )

    win = stream[
        det_start:
        det_start + HEADROOM
    ].astype(np.complex64)

    true_start_in_window = (
        true_start - det_start
    )

    if true_start_in_window < 0:
        raise AssertionError(
            f"Packet {i + 1}: detector starts after "
            f"true packet"
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

    ltf_start = int(
        res["ltf_start"]
    )

    timing_error = (
        ltf_start
        - true_ltf_cp_start
    )

    timing_errors.append(timing_error)

    print(
        f"  true LTF CP     : {true_ltf_cp_start}"
    )

    print(
        f"  sync LTF CP     : {ltf_start}"
    )

    print(
        f"  timing error    : {timing_error} samples"
    )

    if abs(timing_error) > 4:
        raise AssertionError(
            f"Packet {i + 1}: LTF timing error "
            f"{timing_error} samples"
        )


# ============================================================
# 6. Detector state invariant
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
# 7. Final result
# ============================================================

print()
print("RESULT")
print("------")
print("3D multi-packet downstream timing: PASS")
print("Packets tested:", len(PACKETS))
print("Timing errors:", timing_errors)
print("Detector indexing invariant: PASS")