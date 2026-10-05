"""
3F-2C-5
DATA constellation / decision-margin characterization.

Diagnostic only.

Purpose:
    Determine whether the 15 dB CRC failures are caused by
    genuine AWGN-induced BPSK decision errors or by a receiver
    implementation problem in the DATA demapper.

No production files are modified.
"""

from pathlib import Path
import sys

import numpy as np


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))


# ============================================================
# PROJECT IMPORTS
# ============================================================

import config as cfg
import waveform
import scrambler
import demod as demod_mod

from detector import PacketDetector
from sync import sync_packet, apply_cfo_correction


# ============================================================
# TEST CONFIGURATION
# ============================================================

PAYLOAD_BYTES = 100
MODULATION = "BPSK"

PREFIX = 1000
SUFFIX = 12000

CFO_HZ = 0.0
SNR_DB = 15.0

N_TRIALS = 20
BASE_SEED = 20261003


# ============================================================
# PAYLOAD
# ============================================================

payload_bits = (
    np.arange(
        PAYLOAD_BYTES * 8,
        dtype=np.uint8,
    )
    & 1
)


# ============================================================
# CLEAN PACKET
# ============================================================

packet = waveform.assemble_packet(
    payload_bits,
    modulation=MODULATION,
    scrambler_mod=scrambler,
    encoder_mod=scrambler,
    mapper_fn=scrambler.map_bits_to_symbols,
    idle_samples=0,
).astype(np.complex64)


# ============================================================
# CLEAN STREAM
# ============================================================

clean_stream = np.concatenate(
    [
        np.zeros(
            PREFIX,
            dtype=np.complex64,
        ),
        packet,
        np.zeros(
            SUFFIX,
            dtype=np.complex64,
        ),
    ]
)


# ============================================================
# AWGN PARAMETERS
# ============================================================

signal_power = float(
    np.mean(
        np.abs(
            packet.astype(
                np.complex128
            )
        ) ** 2
    )
)

snr_linear = (
    10.0 ** (
        SNR_DB / 10.0
    )
)

noise_power = (
    signal_power
    / snr_linear
)

sigma = np.sqrt(
    noise_power / 2.0
)


# ============================================================
# CFO HELPER
# ============================================================

def apply_cfo(
    samples,
    cfo_hz,
    fs,
):

    n = np.arange(
        len(samples),
        dtype=np.float64,
    )

    phase = (
        2.0
        * np.pi
        * cfo_hz
        * n
        / fs
    )

    return (
        samples
        * np.exp(
            1j * phase
        ).astype(
            np.complex64
        )
    )


# ============================================================
# CLEAN REFERENCE
# ============================================================

def get_clean_reference():

    detector = PacketDetector()

    detections = detector.process(
        clean_stream
    )

    if len(detections) != 1:

        raise RuntimeError(
            "Expected exactly one clean "
            f"detection, got {detections}"
        )

    clean_start, clean_cfo = (
        detections[0]
    )

    clean_start = int(
        clean_start
    )

    # Generous reference window.
    clean_win = clean_stream[
        clean_start:
        clean_start
        + len(packet)
        + SUFFIX
    ]

    sync_res = sync_packet(
        clean_win,
        coarse_cfo_hz=float(
            clean_cfo
        ),
        n_data_symbols=0,
    )

    H_clean = sync_res[
        "H_hat"
    ]

    ltf_start = int(
        sync_res[
            "ltf_start"
        ]
    )

    clean_c = apply_cfo_correction(
        clean_win,
        sync_res[
            "total_cfo"
        ],
        start_n=0,
    )

    # --------------------------------------------------------
    # SIGNAL
    # --------------------------------------------------------

    sig_start = (
        ltf_start
        + cfg.LTF_LEN
        + cfg.CP_LEN
    )

    sig_end = (
        sig_start
        + cfg.FFT_SIZE
    )

    if sig_end > len(clean_c):

        raise RuntimeError(
            "Clean SIGNAL exceeds "
            "reference buffer"
        )

    Y_sig = np.fft.fft(
        clean_c[
            sig_start:
            sig_end
        ],
        n=cfg.FFT_SIZE,
    ).astype(
        np.complex64
    )

    clean_sig = (
        demod_mod.parse_signal_field(
            Y_sig,
            H_clean,
        )
    )

    if not clean_sig["valid"]:

        raise RuntimeError(
            "Clean SIGNAL parse failed"
        )

    n_sym = int(
        clean_sig[
            "n_data_syms"
        ]
    )

    # --------------------------------------------------------
    # DATA FFTS
    # --------------------------------------------------------

    data_start = (
        ltf_start
        + cfg.LTF_LEN
        + cfg.SIG_LEN
    )

    clean_ffts = []

    for m in range(n_sym):

        s = (
            data_start
            + m * cfg.SYMBOL_LEN
            + cfg.CP_LEN
        )

        e = (
            s
            + cfg.FFT_SIZE
        )

        if e > len(clean_c):

            raise RuntimeError(
                f"Clean DATA symbol "
                f"{m} exceeds buffer"
            )

        clean_ffts.append(
            np.fft.fft(
                clean_c[
                    s:e
                ],
                n=cfg.FFT_SIZE,
            ).astype(
                np.complex64
            )
        )

    return (
        clean_ffts,
        H_clean,
        clean_sig,
    )


# ============================================================
# PRINT HEADER
# ============================================================

print("=" * 100)
print(
    "3F-2C-5 DATA CONSTELLATION / DECISION-MARGIN CHARACTERIZATION"
)
print("=" * 100)

print()

print(
    f"SNR:                 {SNR_DB} dB"
)

print(
    f"CFO:                 {CFO_HZ} Hz"
)

print(
    f"Payload:             {PAYLOAD_BYTES} B"
)

print(
    f"Modulation:          {MODULATION}"
)

print(
    f"Packet length:       {len(packet)}"
)

print(
    f"Packet power:        "
    f"{signal_power:.12f}"
)

print(
    f"Noise sigma:         "
    f"{sigma:.12e}"
)


# ============================================================
# CLEAN REFERENCE
# ============================================================

(
    clean_ffts,
    H_clean,
    clean_sig,
) = get_clean_reference()


print()

print(
    "Clean SIGNAL:"
)

print(
    f"  modulation = "
    f"{clean_sig['modulation']}"
)

print(
    f"  bytes      = "
    f"{clean_sig['n_payload_bytes']}"
)

print(
    f"  symbols    = "
    f"{clean_sig['n_data_syms']}"
)


# ============================================================
# ACTIVE SUBCARRIERS
# ============================================================

active = np.asarray(
    cfg.ACTIVE_SUBCARRIERS,
    dtype=int,
)


# ============================================================
# RESULT STORAGE
# ============================================================

results = []


# ============================================================
# TRIAL LOOP
# ============================================================

for trial in range(
    N_TRIALS
):

    # --------------------------------------------------------
    # EXACT SAME SEEDING AS 3F-2C-3 / 3F-2C-4
    # --------------------------------------------------------

    seed = (
        BASE_SEED
        + int(
            round(
                SNR_DB * 100
            )
        )
        + trial
    )

    rng = np.random.default_rng(
        seed
    )

    # --------------------------------------------------------
    # CFO
    # --------------------------------------------------------

    cfo_stream = apply_cfo(
        clean_stream,
        CFO_HZ,
        cfg.FS_FFT,
    )

    # --------------------------------------------------------
    # AWGN
    # --------------------------------------------------------

    noise = (
        rng.standard_normal(
            len(clean_stream)
        )
        + 1j
        * rng.standard_normal(
            len(clean_stream)
        )
    ) * sigma

    noisy_stream = (
        cfo_stream
        + noise.astype(
            np.complex64
        )
    ).astype(
        np.complex64
    )

    # --------------------------------------------------------
    # DETECTION
    # --------------------------------------------------------

    detector = PacketDetector()

    detections = detector.process(
        noisy_stream
    )

    if not detections:

        print(
            f"Trial {trial:2d}: "
            "DETECTION_FAILURE"
        )

        continue

    det_start, det_cfo = (
        detections[0]
    )

    det_start = int(
        det_start
    )

    # --------------------------------------------------------
    # RECEIVER WINDOW
    # --------------------------------------------------------

    start = det_start

    end = min(
        len(noisy_stream),
        start
        + len(packet)
        + SUFFIX,
    )

    win = noisy_stream[
        start:end
    ]

    # --------------------------------------------------------
    # SYNCHRONIZATION
    # --------------------------------------------------------

    try:

        sync_res = sync_packet(
            win,
            coarse_cfo_hz=float(
                det_cfo
            ),
            n_data_symbols=0,
        )

    except Exception as exc:

        print(
            f"Trial {trial:2d}: "
            f"SYNC_EXCEPTION {exc}"
        )

        continue

    H_hat = sync_res[
        "H_hat"
    ]

    ltf_start = int(
        sync_res[
            "ltf_start"
        ]
    )

    total_cfo = float(
        sync_res[
            "total_cfo"
        ]
    )

    # --------------------------------------------------------
    # CFO CORRECTION
    # --------------------------------------------------------

    rx_c = apply_cfo_correction(
        win,
        total_cfo,
        start_n=0,
    )

    # --------------------------------------------------------
    # SIGNAL
    # --------------------------------------------------------

    sig_start = (
        ltf_start
        + cfg.LTF_LEN
        + cfg.CP_LEN
    )

    sig_end = (
        sig_start
        + cfg.FFT_SIZE
    )

    if sig_end > len(rx_c):

        print(
            f"Trial {trial:2d}: "
            "SIGNAL_WINDOW_FAILURE"
        )

        continue

    Y_sig = np.fft.fft(
        rx_c[
            sig_start:
            sig_end
        ],
        n=cfg.FFT_SIZE,
    ).astype(
        np.complex64
    )

    sig = (
        demod_mod.parse_signal_field(
            Y_sig,
            H_hat,
        )
    )

    if not sig["valid"]:

        print(
            f"Trial {trial:2d}: "
            "SIGNAL_PARSE_FAILURE"
        )

        continue

    n_sym = int(
        sig[
            "n_data_syms"
        ]
    )

    # --------------------------------------------------------
    # DATA FFTS
    # --------------------------------------------------------

    data_start = (
        ltf_start
        + cfg.LTF_LEN
        + cfg.SIG_LEN
    )

    noisy_ffts = []

    for m in range(n_sym):

        s = (
            data_start
            + m * cfg.SYMBOL_LEN
            + cfg.CP_LEN
        )

        e = (
            s
            + cfg.FFT_SIZE
        )

        if e > len(rx_c):

            print(
                f"Trial {trial:2d}: "
                "DATA_WINDOW_FAILURE"
            )

            noisy_ffts = []

            break

        noisy_ffts.append(
            np.fft.fft(
                rx_c[
                    s:e
                ],
                n=cfg.FFT_SIZE,
            ).astype(
                np.complex64
            )
        )

    if not noisy_ffts:

        continue

    # ========================================================
    # CURRENT EQUALIZATION
    # ========================================================

    H_active = (
        H_hat[active]
    )

    equalized = []

    clean_equalized = []

    for m in range(n_sym):

        Y = (
            noisy_ffts[m][active]
        )

        X_clean = (
            clean_ffts[m][active]
        )

        Z = (
            Y
            / (
                H_active
                + 1e-12
            )
        )

        Z_clean = (
            X_clean
            / (
                H_active
                + 1e-12
            )
        )

        equalized.append(
            Z
        )

        clean_equalized.append(
            Z_clean
        )

    equalized = np.asarray(
        equalized
    )

    clean_equalized = np.asarray(
        clean_equalized
    )

    # ========================================================
    # BPSK REFERENCE DECISION
    # ========================================================

    reference_sign = (
        np.real(
            clean_equalized
        )
        > 0
    )

    received_real = np.real(
        equalized
    )

    received_imag = np.imag(
        equalized
    )

    received_sign = (
        received_real
        > 0
    )

    # ========================================================
    # SYMBOL ERRORS
    # ========================================================

    error_mask = (
        received_sign
        != reference_sign
    )

    symbol_errors = int(
        np.sum(
            error_mask
        )
    )

    total_symbols = int(
        error_mask.size
    )

    symbol_ber = (
        symbol_errors
        / total_symbols
    )

    # ========================================================
    # DECISION MARGIN
    # ========================================================

    margins = np.abs(
        received_real
    )

    median_margin = float(
        np.median(
            margins
        )
    )

    p10_margin = float(
        np.percentile(
            margins,
            10
        )
    )

    p25_margin = float(
        np.percentile(
            margins,
            25
        )
    )

    min_margin = float(
        np.min(
            margins
        )
    )

    # --------------------------------------------------------
    # Boundary fractions
    # --------------------------------------------------------

    frac_lt_01 = float(
        np.mean(
            margins < 0.1
        )
    )

    frac_lt_02 = float(
        np.mean(
            margins < 0.2
        )
    )

    frac_lt_03 = float(
        np.mean(
            margins < 0.3
        )
    )

    frac_lt_05 = float(
        np.mean(
            margins < 0.5
        )
    )

    # ========================================================
    # EVM
    # ========================================================
    #
    # Compare noisy equalized point to the corresponding
    # clean equalized point.
    #
    # This measures how much the received DATA constellation
    # has moved due to noise/residual effects.
    #
    # ========================================================

    error_vector = (
        equalized
        - clean_equalized
    )

    signal_reference_power = float(
        np.mean(
            np.abs(
                clean_equalized
            ) ** 2
        )
    )

    error_power = float(
        np.mean(
            np.abs(
                error_vector
            ) ** 2
        )
    )

    evm_rms = float(
        np.sqrt(
            error_power
            / (
                signal_reference_power
                + 1e-12
            )
        )
    )

    # ========================================================
    # ACTUAL DEMODULATOR / CRC
    # ========================================================

    try:

        (
            rx_bits,
            crc_ok,
        ) = (
            demod_mod.demodulate_packet(
                noisy_ffts,
                H_hat,
                modulation=sig[
                    "modulation"
                ],
                n_payload_bytes=int(
                    sig[
                        "n_payload_bytes"
                    ]
                ),
            )
        )

        rx_bits = np.asarray(
            rx_bits,
            dtype=np.uint8,
        )

        n_compare = min(
            len(rx_bits),
            len(payload_bits),
        )

        bit_errors = int(
            np.sum(
                rx_bits[
                    :n_compare
                ]
                != payload_bits[
                    :n_compare
                ]
            )
        )

        payload_ber = (
            bit_errors
            / len(payload_bits)
        )

    except Exception as exc:

        print(
            f"Trial {trial:2d}: "
            f"DEMOD_EXCEPTION {exc}"
        )

        continue

    # ========================================================
    # STATUS
    # ========================================================

    status = (
        "PASS"
        if bool(crc_ok)
        else "CRC_FAIL"
    )

    # ========================================================
    # WRONG SYMBOL VALUES
    # ========================================================

    wrong_real = (
        received_real[
            error_mask
        ]
    )

    wrong_imag = (
        received_imag[
            error_mask
        ]
    )

    wrong_margin = (
        margins[
            error_mask
        ]
    )

    # ========================================================
    # STORE
    # ========================================================

    results.append(
        {
            "trial":
                trial,

            "status":
                status,

            "det":
                det_start,

            "cfo":
                total_cfo,

            "ltf":
                ltf_start,

            "symbol_errors":
                symbol_errors,

            "total_symbols":
                total_symbols,

            "symbol_ber":
                symbol_ber,

            "median_margin":
                median_margin,

            "p10_margin":
                p10_margin,

            "p25_margin":
                p25_margin,

            "min_margin":
                min_margin,

            "frac_lt_01":
                frac_lt_01,

            "frac_lt_02":
                frac_lt_02,

            "frac_lt_03":
                frac_lt_03,

            "frac_lt_05":
                frac_lt_05,

            "evm_rms":
                evm_rms,

            "bit_errors":
                bit_errors,

            "payload_ber":
                payload_ber,

            "wrong_real":
                wrong_real,

            "wrong_imag":
                wrong_imag,

            "wrong_margin":
                wrong_margin,
        }
    )

    # ========================================================
    # TRIAL OUTPUT
    # ========================================================

    print(
        f"Trial {trial:2d}: "
        f"{status:8s} "
        f"det={det_start:4d} "
        f"CFO={total_cfo:9.2f} "
        f"LTF={ltf_start:3d} "
        f"| "
        f"symBER={symbol_ber:.6f} "
        f"| "
        f"median|Re|={median_margin:.4f} "
        f"p10={p10_margin:.4f} "
        f"min={min_margin:.4f} "
        f"| "
        f"<.1={frac_lt_01:.3f} "
        f"<.2={frac_lt_02:.3f} "
        f"<.3={frac_lt_03:.3f} "
        f"<.5={frac_lt_05:.3f} "
        f"| "
        f"EVM={evm_rms:.4f} "
        f"| "
        f"payloadBER={payload_ber:.6f}"
    )

    # ========================================================
    # WRONG SYMBOL DETAILS
    # ========================================================

    if symbol_errors > 0:

        print(
            "           wrong-symbol "
            "Re values:"
        )

        print(
            "           "
            + " ".join(
                f"{v:+.3f}"
                for v in wrong_real[:20]
            )
        )

        print(
            "           wrong-symbol "
            "|Re|:"
        )

        print(
            "           "
            + " ".join(
                f"{v:.3f}"
                for v in wrong_margin[:20]
            )
        )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 100)
print(
    "3F-2C-5 SUMMARY"
)
print("=" * 100)


passes = [
    r
    for r in results
    if r["status"] == "PASS"
]

fails = [
    r
    for r in results
    if r["status"] == "CRC_FAIL"
]


print()

print(
    f"PASS trials:        "
    f"{len(passes)}/{len(results)}"
)

print(
    f"CRC failures:       "
    f"{len(fails)}/{len(results)}"
)


# ============================================================
# GROUP STATISTICS
# ============================================================

def print_group_stats(
    name,
    group,
):

    if not group:

        print()
        print(
            f"{name}: no trials"
        )

        return

    print()
    print(name)
    print("-" * len(name))

    print(
        "symbol BER median:",
        np.median(
            [
                r["symbol_ber"]
                for r in group
            ]
        )
    )

    print(
        "symbol BER max:",
        np.max(
            [
                r["symbol_ber"]
                for r in group
            ]
        )
    )

    print(
        "median |Re| median:",
        np.median(
            [
                r["median_margin"]
                for r in group
            ]
        )
    )

    print(
        "10th-percentile |Re| median:",
        np.median(
            [
                r["p10_margin"]
                for r in group
            ]
        )
    )

    print(
        "minimum |Re| median:",
        np.median(
            [
                r["min_margin"]
                for r in group
            ]
        )
    )

    print(
        "fraction |Re| < 0.1 median:",
        np.median(
            [
                r["frac_lt_01"]
                for r in group
            ]
        )
    )

    print(
        "fraction |Re| < 0.2 median:",
        np.median(
            [
                r["frac_lt_02"]
                for r in group
            ]
        )
    )

    print(
        "fraction |Re| < 0.3 median:",
        np.median(
            [
                r["frac_lt_03"]
                for r in group
            ]
        )
    )

    print(
        "fraction |Re| < 0.5 median:",
        np.median(
            [
                r["frac_lt_05"]
                for r in group
            ]
        )
    )

    print(
        "RMS EVM median:",
        np.median(
            [
                r["evm_rms"]
                for r in group
            ]
        )
    )

    print(
        "payload BER median:",
        np.median(
            [
                r["payload_ber"]
                for r in group
            ]
        )
    )


# ============================================================
# PASS GROUP
# ============================================================

print_group_stats(
    "PASS TRIAL STATISTICS",
    passes,
)


# ============================================================
# CRC FAILURE GROUP
# ============================================================

print_group_stats(
    "CRC-FAIL TRIAL STATISTICS",
    fails,
)


# ============================================================
# FAILED TRIAL DETAILS
# ============================================================

if fails:

    print()
    print(
        "CRC-FAIL TRIAL DETAILS"
    )
    print(
        "----------------------"
    )

    for r in fails:

        print(
            f"trial={r['trial']:2d} "
            f"| "
            f"symBER={r['symbol_ber']:.6f} "
            f"| "
            f"median|Re|="
            f"{r['median_margin']:.4f} "
            f"p10="
            f"{r['p10_margin']:.4f} "
            f"min="
            f"{r['min_margin']:.4f} "
            f"| "
            f"<0.1="
            f"{r['frac_lt_01']:.3f} "
            f"<0.2="
            f"{r['frac_lt_02']:.3f} "
            f"<0.3="
            f"{r['frac_lt_03']:.3f} "
            f"<0.5="
            f"{r['frac_lt_05']:.3f} "
            f"| "
            f"EVM="
            f"{r['evm_rms']:.4f} "
            f"| "
            f"payloadBER="
            f"{r['payload_ber']:.6f}"
        )


# ============================================================
# ALL WRONG-SYMBOL MARGINS
# ============================================================

wrong_margins_all = []

for r in results:

    if len(
        r["wrong_margin"]
    ) > 0:

        wrong_margins_all.extend(
            r["wrong_margin"].tolist()
        )


if wrong_margins_all:

    wrong_margins_all = np.asarray(
        wrong_margins_all,
        dtype=np.float64,
    )

    print()
    print(
        "ALL INCORRECT BPSK DECISION MARGINS"
    )
    print(
        "-----------------------------------"
    )

    print(
        "count:",
        len(wrong_margins_all)
    )

    print(
        "median |Re|:",
        np.median(
            wrong_margins_all
        )
    )

    print(
        "mean |Re|:",
        np.mean(
            wrong_margins_all
        )
    )

    print(
        "90th percentile |Re|:",
        np.percentile(
            wrong_margins_all,
            90,
        )
    )

    print(
        "maximum |Re|:",
        np.max(
            wrong_margins_all
        )
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 100)
print(
    "3F-2C-5 COMPLETE"
)
print(
    "NO PRODUCTION FILES WERE MODIFIED."
)
print("=" * 100)