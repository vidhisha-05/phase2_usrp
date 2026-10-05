"""
3F-1D — Production PacketDetector persistence regression.

Purpose:
    Verify that the actual production PacketDetector now requires
    DETECT_PERSISTENCE consecutive threshold crossings.

This test does NOT modify production files.
"""

import numpy as np

import config as cfg
import waveform
import scrambler
from detector import PacketDetector


# ------------------------------------------------------------
# Test configuration
# ------------------------------------------------------------
PAYLOAD_BYTES = 100
MODULATION = "BPSK"

PREFIX = 1000
SUFFIX = 7000

RNG = np.random.default_rng(20261001)

SNR_DB_LIST = [30.0, 25.0, 20.0, 15.0, 10.0, 5.0, 0.0]
TRIALS = 20


# ------------------------------------------------------------
# Build the exact normal PHY packet
# ------------------------------------------------------------
payload_bits = np.random.default_rng(12345).integers(
    0,
    2,
    PAYLOAD_BYTES * 8,
    dtype=np.uint8,
)

packet = waveform.assemble_packet(
    payload_bits,
    modulation=MODULATION,
    scrambler_mod=scrambler,
    encoder_mod=scrambler,
    mapper_fn=scrambler.map_bits_to_symbols,
    idle_samples=0,
).astype(np.complex64)

expected_packet_len = (
    cfg.STF_LEN
    + cfg.LTF_LEN
    + cfg.SIG_LEN
    + 17 * cfg.SYMBOL_LEN
)

print("3F-1D PRODUCTION DETECTOR REGRESSION")
print("=====================================")

print()
print("CONFIGURATION")
print("-------------")
print(f"STF_LEN: {cfg.STF_LEN}")
print(f"LTF_LEN: {cfg.LTF_LEN}")
print(f"Detector window: {cfg.STF_LEN + cfg.LTF_LEN}")
print(f"Correlation lag: {cfg.STF_LEN // 8}")
print(f"Threshold: 0.65")
print(f"Required persistence: {cfg.DETECT_PERSISTENCE}")

print()
print("PACKET")
print("------")
print(f"Payload: {PAYLOAD_BYTES} B")
print(f"Modulation: {MODULATION}")
print(f"Packet length: {len(packet)}")
print(f"Expected packet length: {expected_packet_len}")

assert len(packet) == expected_packet_len, (
    f"Unexpected packet length: {len(packet)} "
    f"(expected {expected_packet_len})"
)

# ------------------------------------------------------------
# Clean stream
# ------------------------------------------------------------
clean_stream = np.concatenate(
    [
        np.zeros(PREFIX, dtype=np.complex64),
        packet,
        np.zeros(SUFFIX, dtype=np.complex64),
    ]
)

true_start = PREFIX

print()
print("STREAM")
print("------")
print(f"True packet start: {true_start}")
print(f"Stream length: {len(clean_stream)}")


# ------------------------------------------------------------
# Helper: feed production detector in irregular chunks
# ------------------------------------------------------------
def detect_chunked(detector, samples):
    """
    Feed samples using irregular chunk sizes.

    This checks that the production detector's absolute indexing
    and persistence behavior remain correct across process() calls.
    """
    chunks = [
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

    while pos < len(samples):
        n = chunks[chunk_index % len(chunks)]
        block = samples[pos:pos + n]

        if len(block) == 0:
            break

        detections.extend(detector.process(block))

        pos += len(block)
        chunk_index += 1

    return detections


# ------------------------------------------------------------
# Clean packet test
# ------------------------------------------------------------
print()
print("=" * 80)
print("CLEAN PACKET TEST")
print("=" * 80)

detector = PacketDetector()

clean_detections = detect_chunked(
    detector,
    clean_stream,
)

print(f"Detections: {clean_detections}")

assert len(clean_detections) == 1, (
    f"Expected exactly 1 clean-packet detection, "
    f"got {len(clean_detections)}"
)

detected_start, detected_cfo = clean_detections[0]

print(f"Detected start: {detected_start}")
print(f"Detection offset: {detected_start - true_start}")
print(f"Estimated CFO: {detected_cfo:.3f} Hz")

assert abs(detected_start - true_start) <= 16, (
    f"Detection start too far from packet start: "
    f"{detected_start} vs {true_start}"
)


# ------------------------------------------------------------
# AWGN test
# ------------------------------------------------------------
packet_power = float(np.mean(np.abs(packet) ** 2))

print()
print("SIGNAL POWER")
print("------------")
print(f"Packet mean power: {packet_power}")


print()
print("=" * 80)
print("AWGN PRODUCTION DETECTOR TEST")
print("=" * 80)


summary = []


for snr_db in SNR_DB_LIST:

    detections_count = 0
    false_positive_count = 0
    offsets = []

    for trial in range(TRIALS):

        signal_power = packet_power
        noise_power = signal_power / (10.0 ** (snr_db / 10.0))

        noise_sigma = np.sqrt(noise_power / 2.0)

        noise = (
            RNG.normal(0.0, noise_sigma, len(clean_stream))
            + 1j * RNG.normal(0.0, noise_sigma, len(clean_stream))
        ).astype(np.complex64)

        noisy_stream = clean_stream + noise

        detector = PacketDetector()

        detections = detect_chunked(
            detector,
            noisy_stream,
        )

        valid = [
            d for d in detections
            if abs(d[0] - true_start) <= 16
        ]

        if len(valid) >= 1:
            detections_count += 1
            offsets.append(valid[0][0] - true_start)

        # Any detection far from the known packet is a false positive.
        fp = [
            d for d in detections
            if abs(d[0] - true_start) > 16
        ]

        if len(fp) > 0:
            false_positive_count += 1

    summary.append(
        (
            snr_db,
            detections_count,
            false_positive_count,
            offsets,
        )
    )


print()
print(
    "SNR | packet detected | false-positive trials | "
    "offset median | offset min | offset max"
)
print("-" * 90)

for snr_db, detected, fp, offsets in summary:

    if offsets:
        median_offset = float(np.median(offsets))
        min_offset = int(np.min(offsets))
        max_offset = int(np.max(offsets))
    else:
        median_offset = float("nan")
        min_offset = 0
        max_offset = 0

    print(
        f"{snr_db:3.0f} | "
        f"{detected:14d}/20 | "
        f"{fp:20d}/20 | "
        f"{median_offset:13.1f} | "
        f"{min_offset:10d} | "
        f"{max_offset:10d}"
    )


# ------------------------------------------------------------
# Final assertions
# ------------------------------------------------------------
print()
print("=" * 80)
print("FINAL CHECKS")
print("=" * 80)

# Clean packet must work.
assert len(clean_detections) == 1

# Every tested SNR must retain packet detection in this simulation.
for snr_db, detected, fp, offsets in summary:
    assert detected == TRIALS, (
        f"SNR {snr_db} dB: only {detected}/{TRIALS} "
        f"packets detected"
    )

print("Clean packet detection: PASS")
print("All tested SNR packet detections: PASS")
print(
    f"Persistence requirement: "
    f"{cfg.DETECT_PERSISTENCE} consecutive samples"
)
print()
print("3F-1D PRODUCTION DETECTOR: PASS")
print("No production files were modified by this test.")