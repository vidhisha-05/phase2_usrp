import numpy as np

import config as cfg
import waveform
import scrambler

from detector import PacketDetector
from sync import sync_packet


# ============================================================
# 3E-4
# Exact intended 5-ms packet spacing
#
# Baseband sample rate:
#     20 MS/s
#
# Packet start period:
#     5 ms
#
# Therefore:
#     20e6 * 0.005 = 100000 BB samples
# ============================================================

PAYLOAD_BYTES = 100
MODULATION = "BPSK"

PREFIX = 1000
PACKET_PERIOD_BB = int(
    round(cfg.FS_FFT * cfg.TX_PACKET_PERIOD_S)
)

SUFFIX = 7000


# ============================================================
# Verify the configured timing
# ============================================================

print("5-MS PACKET SPACING")
print("===================")

print("FS_FFT:", cfg.FS_FFT)
print("TX_PACKET_PERIOD_S:",
      cfg.TX_PACKET_PERIOD_S)
print("TX_PACKET_RATE_HZ:",
      cfg.TX_PACKET_RATE_HZ)

print(
    "Calculated packet period:",
    PACKET_PERIOD_BB,
    "BB samples"
)

if PACKET_PERIOD_BB != 100000:
    raise AssertionError(
        "Expected the configured 5-ms period to equal "
        "100000 BB samples."
    )


# ============================================================
# Generate two packets
# ============================================================

packets = []

for packet_number in range(2):

    payload_bits = (
        (
            np.arange(
                PAYLOAD_BYTES * 8,
                dtype=np.uint8
            )
            + packet_number
        )
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

    pkt = np.asarray(
        pkt,
        dtype=np.complex64
    )

    packets.append(pkt)

    print(
        f"Packet {packet_number + 1}: "
        f"{PAYLOAD_BYTES} B {MODULATION}, "
        f"length={len(pkt)}"
    )


# ============================================================
# Verify packet length
# ============================================================

expected_packet_len = (
    cfg.STF_LEN
    + cfg.LTF_LEN
    + cfg.SIG_LEN
    + 17 * cfg.SYMBOL_LEN
)

print()
print("PACKET GEOMETRY")
print("---------------")
print(
    "Expected 100 B BPSK length:",
    expected_packet_len
)
print(
    "Actual packet length:",
    len(packets[0])
)

if len(packets[0]) != expected_packet_len:
    raise AssertionError(
        "Unexpected packet length."
    )


# ============================================================
# Place packets exactly 5 ms apart
# ============================================================

true_start_1 = PREFIX
true_start_2 = (
    true_start_1
    + PACKET_PERIOD_BB
)

idle_between = (
    true_start_2
    - true_start_1
    - len(packets[0])
)

stream = np.concatenate([
    np.zeros(
        PREFIX,
        dtype=np.complex64
    ),

    packets[0],

    np.zeros(
        idle_between,
        dtype=np.complex64
    ),

    packets[1],

    np.zeros(
        SUFFIX,
        dtype=np.complex64
    ),
]).astype(np.complex64)


# ============================================================
# Print exact geometry
# ============================================================

print()
print("STREAM GEOMETRY")
print("---------------")
print(
    "True packet 1 start:",
    true_start_1
)
print(
    "True packet 2 start:",
    true_start_2
)
print(
    "Packet start spacing:",
    true_start_2 - true_start_1
)
print(
    "Packet start spacing [ms]:",
    (
        (true_start_2 - true_start_1)
        / cfg.FS_FFT
        * 1000.0
    )
)
print(
    "Packet length:",
    len(packets[0])
)
print(
    "Idle between packets:",
    idle_between,
)
print(
    "Total stream length:",
    len(stream)
)


# ============================================================
# Verify exact 5-ms spacing
# ============================================================

if (
    true_start_2 - true_start_1
    != PACKET_PERIOD_BB
):
    raise AssertionError(
        "Packet starts are not exactly one "
        "configured packet period apart."
    )

if idle_between < 0:
    raise AssertionError(
        "Packet period is shorter than packet duration."
    )


# ============================================================
# Suppression arithmetic
# ============================================================

MAX_PACKET_LEN = (
    cfg.STF_LEN
    + cfg.LTF_LEN
    + cfg.SIG_LEN
    + cfg.MAX_DATA_SYMS * cfg.SYMBOL_LEN
)

expected_detector_start_1 = (
    true_start_1 - 3
)

expected_suppress_until = (
    expected_detector_start_1
    + MAX_PACKET_LEN
)

margin_to_packet_2 = (
    true_start_2
    - expected_suppress_until
)

print()
print("SUPPRESSION ARITHMETIC")
print("----------------------")
print(
    "MAX packet length:",
    MAX_PACKET_LEN
)
print(
    "Expected packet 1 detector start:",
    expected_detector_start_1
)
print(
    "Expected suppress_until:",
    expected_suppress_until
)
print(
    "Packet 2 true start:",
    true_start_2
)
print(
    "Margin after suppression:",
    margin_to_packet_2,
    "BB samples"
)

if true_start_2 <= expected_suppress_until:
    raise AssertionError(
        "Packet 2 begins inside the expected "
        "suppression interval."
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
# Detection results
# ============================================================

detected_starts = [
    int(x[0])
    for x in detections
]

print()
print("DETECTION RESULTS")
print("-----------------")
print(
    "Expected physical starts:",
    [true_start_1, true_start_2]
)
print(
    "Detected starts:",
    detected_starts
)
print(
    "Number detected:",
    len(detections)
)

if len(detections) != 2:
    raise AssertionError(
        "Expected exactly two detections."
    )


# ============================================================
# Check detector offsets
# ============================================================

print()
print("DETECTOR OFFSETS")
print("----------------")

for i, (
    detected_start,
    true_start
) in enumerate(
    zip(
        detected_starts,
        [true_start_1, true_start_2]
    )
):

    offset = (
        detected_start
        - true_start
    )

    print(
        f"Packet {i + 1} detector offset:",
        offset,
        "samples"
    )

    if abs(offset) > 8:
        raise AssertionError(
            f"Packet {i + 1}: detector offset "
            f"is too large."
        )


# ============================================================
# Downstream LTF synchronization
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

for i, (
    detected_start,
    det_cfo
) in enumerate(detections):

    true_start = [
        true_start_1,
        true_start_2
    ][i]

    if (
        detected_start < 0
        or detected_start + HEADROOM
        > len(stream)
    ):
        raise AssertionError(
            f"Packet {i + 1}: insufficient samples "
            "for HEADROOM extraction."
        )

    win = stream[
        detected_start:
        detected_start + HEADROOM
    ].astype(np.complex64)

    true_start_in_window = (
        true_start
        - detected_start
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
    print(
        "  true start       :",
        true_start
    )
    print(
        "  detected start   :",
        detected_start
    )
    print(
        "  detector offset  :",
        detected_start - true_start
    )
    print(
        "  true LTF CP      :",
        true_ltf_cp_start
    )
    print(
        "  recovered LTF CP :",
        recovered_ltf_start
    )
    print(
        "  timing error     :",
        timing_error
    )

    if abs(timing_error) > 4:
        raise AssertionError(
            f"Packet {i + 1}: downstream LTF "
            f"timing error too large."
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
print(
    "sample_idx:",
    det._sample_idx
)
print(
    "buffer_len:",
    len(det._buf)
)
print(
    "sum:",
    final_index
)
print(
    "stream_len:",
    len(stream)
)

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
print(
    "3E-4 exact 5-ms packet-spacing test: PASS"
)
print(
    "Both packets detected."
)
print(
    "Both downstream LTF boundaries recovered."
)
print(
    "5-ms packet spacing verified."
)
print(
    "Detector indexing invariant: PASS"
)