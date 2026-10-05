"""
3F-1D-C — Persistence sweep on the exact failed 15 dB trial.

This reproduces the original RNG sequence exactly and extracts
the 15 dB trial 11 noise realization.

Then it tests that IDENTICAL noisy waveform using the actual
production PacketDetector with persistence values:

8, 16, 32, 64

No production files are modified.
"""

import numpy as np

import config as cfg
import waveform
import scrambler
import detector as detector_mod
from detector import PacketDetector


PAYLOAD_BYTES = 100
MODULATION = "BPSK"

PREFIX = 1000
SUFFIX = 7000

TRIALS = 20
TARGET_SNR = 15.0
TARGET_TRIAL = 11

RNG = np.random.default_rng(20261001)


# ------------------------------------------------------------
# Build exact packet
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

clean_stream = np.concatenate(
    [
        np.zeros(PREFIX, dtype=np.complex64),
        packet,
        np.zeros(SUFFIX, dtype=np.complex64),
    ]
)

true_start = PREFIX

signal_power = float(
    np.mean(np.abs(packet) ** 2)
)


# ------------------------------------------------------------
# Same chunking
# ------------------------------------------------------------

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


def detect_chunked(detector, samples):

    detections = []

    pos = 0
    chunk_index = 0

    while pos < len(samples):

        n = chunks[chunk_index % len(chunks)]

        block = samples[pos:pos + n]

        if len(block) == 0:
            break

        result = detector.process(block)

        if result:
            detections.extend(result)

        pos += len(block)
        chunk_index += 1

    return detections


# ------------------------------------------------------------
# Reproduce the EXACT original RNG sequence
# ------------------------------------------------------------

snr_list = [
    30.0,
    25.0,
    20.0,
    15.0,
]

failed_trial_stream = None

for snr_db in snr_list:

    noise_power = signal_power / (
        10.0 ** (snr_db / 10.0)
    )

    noise_sigma = np.sqrt(
        noise_power / 2.0
    )

    for trial in range(1, TRIALS + 1):

        noise = (
            RNG.normal(
                0.0,
                noise_sigma,
                len(clean_stream),
            )
            + 1j
            * RNG.normal(
                0.0,
                noise_sigma,
                len(clean_stream),
            )
        ).astype(np.complex64)

        if (
            snr_db == TARGET_SNR
            and trial == TARGET_TRIAL
        ):
            failed_trial_stream = (
                clean_stream + noise
            ).astype(np.complex64)

            print(
                "Exact failed trial reproduced:"
            )
            print(
                f"  SNR = {snr_db} dB"
            )
            print(
                f"  Trial = {trial}"
            )
            print(
                f"  Signal power = {signal_power}"
            )
            print(
                f"  Noise power = {noise_power}"
            )
            print(
                f"  Noise sigma = {noise_sigma}"
            )

            break

    if failed_trial_stream is not None:
        break


if failed_trial_stream is None:
    raise RuntimeError(
        "Could not reproduce the exact failed trial."
    )


# ------------------------------------------------------------
# Persistence sweep
# ------------------------------------------------------------

print()
print("=" * 80)
print("EXACT FAILED-TRIAL PERSISTENCE SWEEP")
print("=" * 80)

print(
    f"True packet start: {true_start}"
)

print(
    f"Packet length: {len(packet)}"
)

print(
    f"Stream length: {len(failed_trial_stream)}"
)


original_persistence = detector_mod.DETECT_PERSISTENCE

print()
print(
    f"Production persistence currently: "
    f"{original_persistence}"
)


for persistence in [8, 16, 32, 64]:

    detector_mod.DETECT_PERSISTENCE = persistence

    detector = PacketDetector()

    detections = detect_chunked(
        detector,
        failed_trial_stream,
    )

    valid = [
        d
        for d in detections
        if abs(d[0] - true_start) <= 16
    ]

    false_positives = [
        d
        for d in detections
        if abs(d[0] - true_start) > 16
    ]

    print()
    print("-" * 80)
    print(
        f"PERSISTENCE = {persistence}"
    )
    print("-" * 80)

    print(
        f"Detections: {detections}"
    )

    print(
        f"Valid detections: {valid}"
    )

    print(
        f"False positives: {false_positives}"
    )

    if valid:
        print(
            "Valid offsets: "
            f"{[d[0] - true_start for d in valid]}"
        )

    if false_positives:

        packet_len_max = (
            cfg.STF_LEN
            + cfg.LTF_LEN
            + cfg.SIG_LEN
            + cfg.MAX_DATA_SYMS * cfg.SYMBOL_LEN
        )

        for d in false_positives:

            false_start = d[0]

            print(
                f"False start {false_start}; "
                f"suppression until "
                f"{false_start + packet_len_max}"
            )


# Restore module constant in this process.
detector_mod.DETECT_PERSISTENCE = original_persistence


print()
print("=" * 80)
print("EXACT FAILED-TRIAL PERSISTENCE SWEEP COMPLETE")
print("=" * 80)

print(
    "No production files were modified."
)