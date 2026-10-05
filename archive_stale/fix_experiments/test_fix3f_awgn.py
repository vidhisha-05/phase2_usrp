import numpy as np

import config as cfg
import waveform
import scrambler

from detector import PacketDetector
from sync import sync_packet


# ============================================================
# 3F-1
# AWGN / SNR stress test
#
# One 100 B BPSK packet is transmitted through AWGN.
#
# SNR is defined here as:
#
#     signal_power / noise_power
#
# where complex noise is:
#
#     n = sqrt(noise_power/2) * (nI + j*nQ)
#
# This test measures:
#   1. Packet detection
#   2. Detector timing
#   3. Downstream LTF timing
#   4. Detector indexing invariant
#
# NO production files are modified by this test.
# ============================================================


PAYLOAD_BYTES = 100
MODULATION = "BPSK"

PREFIX = 1000
SUFFIX = 7000

SNR_DB_LIST = [
    30.0,
    25.0,
    20.0,
    15.0,
    10.0,
    5.0,
    0.0,
]

RNG_SEED = 20261001


# ============================================================
# Generate reference packet
# ============================================================

payload_bits = (
    np.arange(
        PAYLOAD_BYTES * 8,
        dtype=np.uint8
    )
    & 1
)

packet = waveform.assemble_packet(
    payload_bits,
    modulation=MODULATION,
    scrambler_mod=scrambler,
    encoder_mod=scrambler,
    mapper_fn=scrambler.map_bits_to_symbols,
    idle_samples=0,
)

packet = np.asarray(
    packet,
    dtype=np.complex64
)

print("3F-1 AWGN / SNR STRESS TEST")
print("============================")

print()
print("PACKET")
print("------")
print("Payload:", PAYLOAD_BYTES, "B")
print("Modulation:", MODULATION)
print("Packet length:", len(packet), "BB samples")

expected_packet_len = (
    cfg.STF_LEN
    + cfg.LTF_LEN
    + cfg.SIG_LEN
    + 17 * cfg.SYMBOL_LEN
)

print(
    "Expected packet length:",
    expected_packet_len
)

if len(packet) != expected_packet_len:
    raise AssertionError(
        "Unexpected reference packet length."
    )


# ============================================================
# Construct clean stream
# ============================================================

clean_stream = np.concatenate([
    np.zeros(
        PREFIX,
        dtype=np.complex64
    ),
    packet,
    np.zeros(
        SUFFIX,
        dtype=np.complex64
    ),
]).astype(np.complex64)

true_packet_start = PREFIX

print()
print("STREAM")
print("------")
print("True packet start:", true_packet_start)
print("Stream length:", len(clean_stream))


# ============================================================
# Signal power definition
# ============================================================

packet_power = float(
    np.mean(
        np.abs(packet.astype(np.complex128)) ** 2
    )
)

print()
print("POWER")
print("-----")
print("Packet mean power:", packet_power)

if not np.isfinite(packet_power):
    raise AssertionError(
        "Packet power is not finite."
    )

if packet_power <= 0:
    raise AssertionError(
        "Packet power must be positive."
    )


# ============================================================
# Run SNR sweep
# ============================================================

rng = np.random.default_rng(RNG_SEED)

results = []


for snr_db in SNR_DB_LIST:

    print()
    print("=" * 60)
    print(f"SNR = {snr_db:.1f} dB")
    print("=" * 60)

    # --------------------------------------------------------
    # Convert SNR to noise power
    # --------------------------------------------------------

    snr_linear = 10.0 ** (
        snr_db / 10.0
    )

    noise_power = (
        packet_power
        / snr_linear
    )

    noise_sigma = np.sqrt(
        noise_power / 2.0
    )

    print(
        "Noise power:",
        noise_power
    )

    print(
        "Complex noise sigma per component:",
        noise_sigma
    )

    # --------------------------------------------------------
    # Generate complex AWGN
    # --------------------------------------------------------

    noise = (
        rng.standard_normal(
            len(clean_stream)
        )
        + 1j
        * rng.standard_normal(
            len(clean_stream)
        )
    )

    noise = (
        noise
        * noise_sigma
    ).astype(np.complex64)

    noisy_stream = (
        clean_stream
        + noise
    ).astype(np.complex64)

    # --------------------------------------------------------
    # Detector
    # --------------------------------------------------------

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

    while pos < len(noisy_stream):

        n = chunk_sizes[
            chunk_index % len(chunk_sizes)
        ]

        end = min(
            pos + n,
            len(noisy_stream)
        )

        chunk = noisy_stream[pos:end]

        chunk_dets = det.process(chunk)

        if chunk_dets:
            print(
                "chunk",
                f"{chunk_index:03d}:",
                f"stream [{pos}:{end}] ->",
                chunk_dets
            )

        detections.extend(chunk_dets)

        pos = end
        chunk_index += 1

    # --------------------------------------------------------
    # Detection result
    # --------------------------------------------------------

    detected_starts = [
        int(x[0])
        for x in detections
    ]

    print()
    print(
        "Detected starts:",
        detected_starts
    )

    print(
        "Detection count:",
        len(detections)
    )

    detection_ok = (
        len(detections) == 1
    )

    timing_ok = False
    sync_ok = False
    index_ok = False

    timing_error = None
    ltf_timing_error = None

    # --------------------------------------------------------
    # Timing check
    # --------------------------------------------------------

    if detection_ok:

        detected_start = (
            detected_starts[0]
        )

        timing_error = (
            detected_start
            - true_packet_start
        )

        print(
            "Detector timing offset:",
            timing_error,
            "samples"
        )

        timing_ok = (
            abs(timing_error) <= 16
        )

        # ----------------------------------------------------
        # Downstream synchronization
        # ----------------------------------------------------

        HEADROOM = (
            cfg.STF_LEN
            + cfg.LTF_LEN
            + cfg.SIG_LEN
            + cfg.MAX_DATA_SYMS
            * cfg.SYMBOL_LEN
        )

        if (
            detected_start >= 0
            and
            detected_start + HEADROOM
            <= len(noisy_stream)
        ):

            win = noisy_stream[
                detected_start:
                detected_start + HEADROOM
            ].astype(
                np.complex64
            )

            true_start_in_window = (
                true_packet_start
                - detected_start
            )

            true_ltf_cp_start = (
                true_start_in_window
                + cfg.STF_LEN
            )

            det_cfo = (
                float(detections[0][1])
            )

            try:

                res = sync_packet(
                    win,
                    coarse_cfo_hz=det_cfo,
                    n_data_symbols=0,
                )

                recovered_ltf_start = int(
                    res["ltf_start"]
                )

                ltf_timing_error = (
                    recovered_ltf_start
                    - true_ltf_cp_start
                )

                print(
                    "True LTF CP:",
                    true_ltf_cp_start
                )

                print(
                    "Recovered LTF CP:",
                    recovered_ltf_start
                )

                print(
                    "LTF timing error:",
                    ltf_timing_error,
                    "samples"
                )

                sync_ok = (
                    abs(ltf_timing_error) <= 8
                )

            except Exception as exc:

                print(
                    "sync_packet exception:",
                    repr(exc)
                )

        else:

            print(
                "Insufficient samples for "
                "downstream synchronization."
            )

    # --------------------------------------------------------
    # Detector indexing invariant
    # --------------------------------------------------------

    final_index = (
        det._sample_idx
        + len(det._buf)
    )

    index_ok = (
        final_index
        == len(noisy_stream)
    )

    print()
    print("Detector final sample_idx:",
          det._sample_idx)

    print("Detector buffer length:",
          len(det._buf))

    print(
        "Index sum:",
        final_index
    )

    print(
        "Stream length:",
        len(noisy_stream)
    )

    print(
        "Index invariant:",
        "PASS" if index_ok else "FAIL"
    )

    # --------------------------------------------------------
    # Record
    # --------------------------------------------------------

    results.append({
        "snr_db": snr_db,
        "detections": len(detections),
        "timing_error": timing_error,
        "ltf_timing_error": ltf_timing_error,
        "detection_ok": detection_ok,
        "timing_ok": timing_ok,
        "sync_ok": sync_ok,
        "index_ok": index_ok,
    })


# ============================================================
# Summary
# ============================================================

print()
print()
print("=" * 70)
print("3F-1 SUMMARY")
print("=" * 70)

print(
    "SNR(dB) | detections | det_err | "
    "ltf_err | detection | timing | sync | index"
)

print("-" * 70)

for r in results:

    print(
        f"{r['snr_db']:7.1f} | "
        f"{r['detections']:10d} | "
        f"{str(r['timing_error']):7} | "
        f"{str(r['ltf_timing_error']):7} | "
        f"{'PASS' if r['detection_ok'] else 'FAIL':9} | "
        f"{'PASS' if r['timing_ok'] else 'FAIL':6} | "
        f"{'PASS' if r['sync_ok'] else 'FAIL':4} | "
        f"{'PASS' if r['index_ok'] else 'FAIL'}"
    )


# ============================================================
# Invariant that must always hold
# ============================================================

for r in results:

    if not r["index_ok"]:
        raise AssertionError(
            f"SNR {r['snr_db']} dB: "
            "detector indexing invariant failed."
        )


# ============================================================
# Final statement
# ============================================================

print()
print("3F-1 TEST COMPLETE")
print("------------------")
print(
    "Indexing invariant passed at every SNR."
)
print(
    "Detection/synchronization performance "
    "is reported above."
)
print(
    "No production files were modified."
)