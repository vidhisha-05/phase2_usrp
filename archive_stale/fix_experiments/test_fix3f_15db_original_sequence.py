"""
3F-1D-B — Reproduce the ORIGINAL 15 dB RNG sequence.

This intentionally consumes the RNG in exactly the same order
as test_fix3f_production_detector.py:

30 dB x 20
25 dB x 20
20 dB x 20
15 dB x 20

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

TRIALS = 20
TARGET_SNR = 15.0

RNG = np.random.default_rng(20261001)


# ------------------------------------------------------------
# Build the exact same packet
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
# Same irregular detector chunks
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


print()
print("=" * 80)
print("3F-1D-B ORIGINAL RNG-SEQUENCE REPRODUCTION")
print("=" * 80)

print()
print("Packet length:", len(packet))
print("True packet start:", true_start)
print("Stream length:", len(clean_stream))
print("Signal power:", signal_power)

print()
print("The RNG sequence is intentionally consumed as:")
print("30 dB x 20")
print("25 dB x 20")
print("20 dB x 20")
print("15 dB x 20")

print()


# ------------------------------------------------------------
# Reproduce the original SNR ordering exactly
# ------------------------------------------------------------

snr_list = [
    30.0,
    25.0,
    20.0,
    15.0,
]


for snr_db in snr_list:

    print()
    print("=" * 80)
    print(f"SNR = {snr_db} dB")
    print("=" * 80)

    noise_power = signal_power / (
        10.0 ** (snr_db / 10.0)
    )

    noise_sigma = np.sqrt(
        noise_power / 2.0
    )

    print("Noise power:", noise_power)
    print("Noise sigma:", noise_sigma)

    detected_count = 0
    false_positive_count = 0

    for trial in range(TRIALS):

        # IMPORTANT:
        # This is deliberately identical to the original
        # production regression's RNG consumption.
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

        if valid:
            detected_count += 1

        if false_positives:
            false_positive_count += 1

        # Only print detailed information for 15 dB.
        if snr_db == TARGET_SNR:

            print()
            print(
                f"15 dB trial {trial + 1:2d}:"
            )

            print(
                f"  detections       = {detections}"
            )

            print(
                f"  valid            = {valid}"
            )

            print(
                f"  false positives  = {false_positives}"
            )

            if valid:

                print(
                    "  offsets          = "
                    f"{[d[0] - true_start for d in valid]}"
                )

            if not valid:

                print()
                print(
                    "  *** THIS IS A FAILED "
                    "15 dB TRIAL ***"
                )

                packet_len_max = (
                    cfg.STF_LEN
                    + cfg.LTF_LEN
                    + cfg.SIG_LEN
                    + cfg.MAX_DATA_SYMS
                    * cfg.SYMBOL_LEN
                )

                if false_positives:

                    for d in false_positives:

                        false_start = d[0]

                        suppress_until = (
                            false_start
                            + packet_len_max
                        )

                        print(
                            f"  false detection start "
                            f"= {false_start}"
                        )

                        print(
                            f"  distance from true start "
                            f"= {false_start - true_start}"
                        )

                        print(
                            f"  suppression until "
                            f"= {suppress_until}"
                        )

                        print(
                            f"  true packet start "
                            f"= {true_start}"
                        )

                        print(
                            f"  true start minus suppression "
                            f"= {true_start - suppress_until}"
                        )

                else:

                    print(
                        "  No false detection occurred."
                    )

                    print(
                        "  The packet itself was missed."
                    )

                print()
                print(
                    "  detector final sample index = "
                    f"{detector._sample_idx}"
                )

                print(
                    "  detector buffer length = "
                    f"{len(detector._buf)}"
                )

                print(
                    "  detector suppress_until = "
                    f"{detector._suppress_until}"
                )

                print()

    print()
    print(
        f"{snr_db:4.1f} dB summary: "
        f"{detected_count}/{TRIALS} detected, "
        f"{false_positive_count}/{TRIALS} "
        f"with false positives"
    )


print()
print("=" * 80)
print("3F-1D-B COMPLETE")
print("=" * 80)
print("No production files were modified.")