"""
test_fix3f_2c2_snr_sweep.py

3F-2C-2:
Fine SNR characterization at 0 Hz CFO.

Purpose:
    Determine the transition region between reliable and unreliable
    packet recovery under the same AWGN model used by 3F-2C-1.

No production files are modified.
"""

from pathlib import Path
import sys

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

import config as cfg
import waveform
import scrambler
import demod as demod_mod
from detector import PacketDetector
from sync import sync_packet, apply_cfo_correction


# ---------------------------------------------------------------------
# TEST CONFIGURATION
# ---------------------------------------------------------------------

PAYLOAD_BYTES = 100
MODULATION = "BPSK"

PREFIX = 1000
SUFFIX = 12000

CFO_HZ = 0.0

SNR_DB_LIST = list(range(10, 21))   # 10,11,...,20 dB
N_TRIALS = 20

BASE_SEED = 20261002


# ---------------------------------------------------------------------
# PACKET GENERATION
# ---------------------------------------------------------------------

payload_bits = (
    np.arange(PAYLOAD_BYTES * 8, dtype=np.uint8) & 1
)

clean_packet = waveform.assemble_packet(
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
        clean_packet,
        np.zeros(SUFFIX, dtype=np.complex64),
    ]
)

signal_power = float(
    np.mean(
        np.abs(clean_packet.astype(np.complex128)) ** 2
    )
)


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def apply_cfo(samples: np.ndarray, cfo_hz: float, fs: float):
    n = np.arange(len(samples), dtype=np.float64)
    phase = 2.0 * np.pi * cfo_hz * n / fs
    return samples * np.exp(1j * phase).astype(np.complex64)


def classify_trial(noisy_stream, expected_payload_bits):
    """
    Run the same detector/synchronization/decode chain used by
    the previous diagnostic and return a detailed status.
    """

    detector = PacketDetector()

    detections = detector.process(noisy_stream)

    if len(detections) == 0:
        return {
            "status": "DETECTION_FAILURE",
            "det": None,
            "cfo_est": None,
            "crc": False,
            "bits": 0,
            "detail": "No packet detection",
        }

    if len(detections) > 1:
        return {
            "status": "MULTIPLE_DETECTIONS",
            "det": detections[0][0],
            "cfo_est": detections[0][1],
            "crc": False,
            "bits": 0,
            "detail": f"{len(detections)} detections",
        }

    det_start, det_cfo = detections[0]
    det_start = int(det_start)
    det_cfo = float(det_cfo)

    # -------------------------------------------------------------
    # Use a long window beginning exactly at detector output.
    # -------------------------------------------------------------

    start = det_start
    end = min(
        len(noisy_stream),
        start + len(clean_packet) + SUFFIX,
    )

    win = noisy_stream[start:end]

    # -------------------------------------------------------------
    # Synchronization
    # -------------------------------------------------------------

    try:
        res = sync_packet(
            win,
            coarse_cfo_hz=det_cfo,
            n_data_symbols=0,
        )
    except Exception as exc:
        return {
            "status": "SYNC_EXCEPTION",
            "det": det_start,
            "cfo_est": det_cfo,
            "crc": False,
            "bits": 0,
            "detail": str(exc),
        }

    H_hat = res["H_hat"]
    ltf_start = int(res["ltf_start"])
    cfo_total = float(res["total_cfo"])

    # -------------------------------------------------------------
    # CFO correction
    # -------------------------------------------------------------

    try:
        rx_c = apply_cfo_correction(
            win,
            cfo_total,
            start_n=0,
        )
    except Exception as exc:
        return {
            "status": "CFO_CORRECTION_EXCEPTION",
            "det": det_start,
            "cfo_est": det_cfo,
            "crc": False,
            "bits": 0,
            "detail": str(exc),
        }

    # -------------------------------------------------------------
    # SIGNAL field
    # -------------------------------------------------------------

    sig_body_start = (
        ltf_start
        + cfg.LTF_LEN
        + cfg.CP_LEN
    )

    sig_body_end = (
        sig_body_start
        + cfg.FFT_SIZE
    )

    if sig_body_end > len(rx_c):
        return {
            "status": "SIGNAL_WINDOW_FAILURE",
            "det": det_start,
            "cfo_est": det_cfo,
            "crc": False,
            "bits": 0,
            "detail": (
                f"SIGNAL end={sig_body_end}, "
                f"window={len(rx_c)}"
            ),
        }

    try:
        Y_sig = np.fft.fft(
            rx_c[
                sig_body_start:sig_body_end
            ],
            n=cfg.FFT_SIZE,
        ).astype(np.complex64)

        sig = demod_mod.parse_signal_field(
            Y_sig,
            H_hat,
        )

    except Exception as exc:
        return {
            "status": "SIGNAL_EXCEPTION",
            "det": det_start,
            "cfo_est": det_cfo,
            "crc": False,
            "bits": 0,
            "detail": str(exc),
        }

    if not sig["valid"]:
        return {
            "status": "SIGNAL_PARSE_FAILURE",
            "det": det_start,
            "cfo_est": det_cfo,
            "crc": False,
            "bits": 0,
            "detail": "SIGNAL invalid",
        }

    n_bytes = int(sig["n_payload_bytes"])
    modulation = sig["modulation"]
    n_sym = int(sig["n_data_syms"])

    # -------------------------------------------------------------
    # Production safety limits
    # -------------------------------------------------------------

    max_payload = cfg.MAX_PAYLOAD_BYTES_BY_MODULATION.get(
        modulation
    )

    if (
        max_payload is None
        or n_bytes < 1
        or n_bytes > max_payload
    ):
        return {
            "status": "SIGNAL_INVALID_PARAMETERS",
            "det": det_start,
            "cfo_est": det_cfo,
            "crc": False,
            "bits": 0,
            "detail": (
                f"mod={modulation}, "
                f"bytes={n_bytes}, "
                f"symbols={n_sym}"
            ),
        }

    if (
        n_sym < 1
        or n_sym > cfg.MAX_DATA_SYMS
    ):
        return {
            "status": "SIGNAL_INVALID_PARAMETERS",
            "det": det_start,
            "cfo_est": det_cfo,
            "crc": False,
            "bits": 0,
            "detail": (
                f"mod={modulation}, "
                f"bytes={n_bytes}, "
                f"symbols={n_sym}"
            ),
        }

    # -------------------------------------------------------------
    # DATA extraction
    # -------------------------------------------------------------

    data_start = (
        ltf_start
        + cfg.LTF_LEN
        + cfg.SIG_LEN
    )

    ffts = []

    for m in range(n_sym):

        s = (
            data_start
            + m * cfg.SYMBOL_LEN
            + cfg.CP_LEN
        )

        e = s + cfg.FFT_SIZE

        if e > len(rx_c):
            return {
                "status": "DATA_WINDOW_FAILURE",
                "det": det_start,
                "cfo_est": det_cfo,
                "crc": False,
                "bits": 0,
                "detail": (
                    f"DATA symbol {m}: "
                    f"end={e}, "
                    f"window={len(rx_c)}, "
                    f"ltf_start={ltf_start}, "
                    f"n_sym={n_sym}"
                ),
            }

        X = np.fft.fft(
            rx_c[s:e],
            n=cfg.FFT_SIZE,
        ).astype(np.complex64)

        ffts.append(X)

    # -------------------------------------------------------------
    # DEMODULATION
    # -------------------------------------------------------------

    try:
        rx_bits, crc_ok = demod_mod.demodulate_packet(
            ffts,
            H_hat,
            modulation=modulation,
            n_payload_bytes=n_bytes,
        )
    except Exception as exc:
        return {
            "status": "DEMOD_EXCEPTION",
            "det": det_start,
            "cfo_est": det_cfo,
            "crc": False,
            "bits": 0,
            "detail": str(exc),
        }

    rx_bits = np.asarray(rx_bits, dtype=np.uint8)

    expected = expected_payload_bits[: len(rx_bits)]

    payload_match = (
        len(rx_bits) == len(expected_payload_bits)
        and np.array_equal(
            rx_bits,
            expected_payload_bits,
        )
    )

    if not bool(crc_ok):
        return {
            "status": "CRC_FAILURE",
            "det": det_start,
            "cfo_est": det_cfo,
            "crc": False,
            "bits": len(rx_bits),
            "detail": (
                f"mod={modulation}, "
                f"bytes={n_bytes}, "
                f"symbols={n_sym}, "
                f"ltf_start={ltf_start}, "
                f"payload_match={payload_match}"
            ),
        }

    if not payload_match:
        return {
            "status": "PAYLOAD_MISMATCH",
            "det": det_start,
            "cfo_est": det_cfo,
            "crc": bool(crc_ok),
            "bits": len(rx_bits),
            "detail": (
                f"mod={modulation}, "
                f"bytes={n_bytes}, "
                f"symbols={n_sym}, "
                f"ltf_start={ltf_start}, "
                f"payload_match=False"
            ),
        }

    return {
        "status": "PASS",
        "det": det_start,
        "cfo_est": det_cfo,
        "crc": bool(crc_ok),
        "bits": len(rx_bits),
        "detail": (
            f"mod={modulation}, "
            f"bytes={n_bytes}, "
            f"symbols={n_sym}, "
            f"ltf_start={ltf_start}, "
            f"payload_match=True"
        ),
    }


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

print("=" * 100)
print("3F-2C-2 FINE SNR SWEEP — 0 Hz CFO")
print("=" * 100)

print()
print("PACKET")
print("------")
print(f"Payload bytes:        {PAYLOAD_BYTES}")
print(f"Modulation:           {MODULATION}")
print(f"Packet length:        {len(clean_packet)}")
print(f"Prefix:               {PREFIX}")
print(f"Suffix:               {SUFFIX}")
print(f"True packet start:    {PREFIX}")
print(f"FS_FFT:               {cfg.FS_FFT}")
print(f"Packet mean power:    {signal_power:.12f}")

print()
print("CFO:")
print(f"    {CFO_HZ:.0f} Hz")

print()
print("SNR sweep:")
print("    " + ", ".join(f"{x:.0f} dB" for x in SNR_DB_LIST))

print()
print(f"Trials per condition: {N_TRIALS}")
print("=" * 100)


summary = []

for snr_db in SNR_DB_LIST:

    snr_linear = 10.0 ** (snr_db / 10.0)

    noise_power = signal_power / snr_linear

    sigma = np.sqrt(
        noise_power / 2.0
    )

    print()
    print("=" * 100)
    print(f"SNR = {snr_db:.1f} dB")
    print(f"Noise power = {noise_power:.12e}")
    print(f"Sigma       = {sigma:.12e}")
    print("=" * 100)

    counts = {}

    for trial in range(N_TRIALS):

        seed = (
            BASE_SEED
            + int(round(snr_db * 100))
            + trial
        )

        rng = np.random.default_rng(seed)

        cfo_stream = apply_cfo(
            clean_stream,
            CFO_HZ,
            cfg.FS_FFT,
        )

        noise = (
            rng.standard_normal(len(clean_stream))
            + 1j * rng.standard_normal(len(clean_stream))
        ) * sigma

        noise = noise.astype(np.complex64)

        noisy_stream = (
            cfo_stream + noise
        ).astype(np.complex64)

        result = classify_trial(
            noisy_stream,
            payload_bits,
        )

        status = result["status"]

        counts[status] = counts.get(status, 0) + 1

        print(
            f"  Trial {trial:2d}: "
            f"{status:24s} "
            f"det={str(result['det']):>5s} "
            f"CFO={str(None if result['cfo_est'] is None else round(result['cfo_est'], 2)):>10s} "
            f"CRC={str(result['crc']):5s} "
            f"bits={result['bits']:4d} "
            f"{result['detail']}"
        )

    passed = counts.get("PASS", 0)

    print()
    print(
        f"SNR {snr_db:2d} dB RESULT: "
        f"{passed}/{N_TRIALS} passed "
        f"({100.0 * passed / N_TRIALS:.1f}%)"
    )

    summary.append(
        (
            snr_db,
            passed,
            counts,
        )
    )


# ---------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------

print()
print("=" * 100)
print("3F-2C-2 SUMMARY")
print("=" * 100)

print()
print(
    f"{'SNR':>6} "
    f"{'Pass':>8} "
    f"{'Total':>8} "
    f"{'Rate':>10} "
    f"Failure breakdown"
)
print("-" * 100)

for snr_db, passed, counts in summary:

    failure_parts = []

    for status, count in sorted(counts.items()):

        if status != "PASS":
            failure_parts.append(
                f"{status}={count}"
            )

    failures = (
        ", ".join(failure_parts)
        if failure_parts
        else "none"
    )

    print(
        f"{snr_db:6.0f} "
        f"{passed:8d} "
        f"{N_TRIALS:8d} "
        f"{100.0 * passed / N_TRIALS:9.1f}% "
        f"{failures}"
    )

print()
print("=" * 100)
print("3F-2C-2 COMPLETE")
print("NO PRODUCTION FILES WERE MODIFIED.")
print("=" * 100)