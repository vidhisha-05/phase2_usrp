"""
3F-1D-A — Diagnose the single 15 dB production-detector miss.

No production files are modified.
"""

import numpy as np

import config as cfg
import waveform
import scrambler
from detector import PacketDetector


PAYLOAD_BYTES = 100
MODULATION = "BPSK"

PREFIX = 1000
SUFFIX = 7000

SNR_DB = 15.0
TRIALS = 20

RNG = np.random.default_rng(20261001)


# ------------------------------------------------------------
# Build the exact packet used by the production regression
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
packet_power = float(np.mean(np.abs(packet) ** 2))

noise_power = packet_power / (10.0 ** (SNR_DB / 10.0))
noise_sigma = np.sqrt(noise_power / 2.0)


# ------------------------------------------------------------
# Same irregular chunks as production regression
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
    """Feed the production detector using irregular chunks."""

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


print("3F-1D-A 15 dB FAILED-TRIAL DIAGNOSTIC")
print("=======================================")

print()
print("CONFIGURATION")
print("-------------")
print(f"SNR: {SNR_DB} dB")
print(f"Trials: {TRIALS}")
print(f"Threshold: 0.65")
print(f"Persistence: {cfg.DETECT_PERSISTENCE}")
print(f"True packet start: {true_start}")
print(f"Packet length: {len(packet)}")
print(f"Stream length: {len(clean_stream)}")
print(f"Packet power: {packet_power}")
print(f"Noise power: {noise_power}")
print(f"Noise sigma: {noise_sigma}")


for trial in range(TRIALS):

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

    noisy_stream = clean_stream + noise

    detector = PacketDetector()

    detections = detect_chunked(
        detector,
        noisy_stream,
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
    print(f"TRIAL {trial + 1}")
    print("-" * 80)

    print(f"All detections: {detections}")
    print(f"Valid detections: {valid}")
    print(f"False positives: {false_positives}")

    if valid:
        print(
            "Valid offset(s): "
            f"{[d[0] - true_start for d in valid]}"
        )

    if not valid:

        print()
        print("*** FAILED TRIAL FOUND ***")

        if false_positives:

            print(
                "There was at least one false detection "
                "before/away from the true packet."
            )

            packet_len_max = (
                cfg.STF_LEN
                + cfg.LTF_LEN
                + cfg.SIG_LEN
                + cfg.MAX_DATA_SYMS * cfg.SYMBOL_LEN
            )

            for d in false_positives:

                false_start = d[0]
                suppress_until = false_start + packet_len_max

                print(
                    f"False detection absolute start: "
                    f"{false_start}"
                )

                print(
                    f"Distance from true packet: "
                    f"{false_start - true_start}"
                )

                print(
                    f"Detector suppression after this "
                    f"detection: {suppress_until}"
                )

                print(
                    f"True packet start: {true_start}"
                )

                print(
                    f"True packet relative to suppression: "
                    f"{true_start - suppress_until}"
                )

        else:

            print(
                "NO false detection was reported. "
                "The packet itself was missed."
            )

        print()
        print(
            "Detector final absolute index: "
            f"{detector._sample_idx}"
        )

        print(
            "Detector remaining buffer length: "
            f"{len(detector._buf)}"
        )

        print(
            "Detector suppression-until: "
            f"{detector._suppress_until}"
        )

        break


print()
print("=" * 80)
print("3F-1D-A COMPLETE")
print("=" * 80)
print("No production files were modified.")