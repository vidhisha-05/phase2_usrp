"""
3F-2C-4
Equalization comparison at 15 dB, 0 Hz CFO.

Diagnostic only.

Compares:

    CURRENT:
        Z = Y / H_hat

    IDEAL:
        Z = Y / 1

The synthetic channel contains AWGN only,
so the ideal physical channel is H = 1.

NO production files are modified.
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
# TEST CONFIG
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
# GENERATE REFERENCE PAYLOAD
# ============================================================

payload_bits = (
    np.arange(
        PAYLOAD_BYTES * 8,
        dtype=np.uint8,
    )
    & 1
)


# ============================================================
# GENERATE CLEAN PACKET
# ============================================================

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
            packet.astype(np.complex128)
        ) ** 2
    )
)

snr_linear = 10.0 ** (
    SNR_DB / 10.0
)

noise_power = (
    signal_power
    / snr_linear
)

sigma = np.sqrt(
    noise_power / 2.0
)


# ============================================================
# HELPER: APPLY CFO
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
        ).astype(np.complex64)
    )


# ============================================================
# CLEAN REFERENCE EXTRACTION
# ============================================================

def get_reference_data_symbols():

    """
    Recover the clean packet DATA FFT symbols
    and clean channel estimate.

    This is diagnostic reference data only.
    """

    detector = PacketDetector()

    clean_det = detector.process(
        clean_stream
    )

    if len(clean_det) != 1:

        raise RuntimeError(
            "Expected exactly one clean "
            f"detection, got {clean_det}"
        )

    clean_start, clean_cfo = (
        clean_det[0]
    )

    clean_start = int(
        clean_start
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Do NOT truncate the clean reference to len(packet).
    #
    # The detector can report the packet start a few samples
    # before the true waveform insertion point, and sync_packet
    # can place ltf_start accordingly.
    #
    # Give the clean reference the same generous window used
    # by the noisy receiver diagnostic.
    # --------------------------------------------------------

    clean_win = clean_stream[
        clean_start:
        clean_start + len(packet) + SUFFIX
    ]

    clean_sync = sync_packet(
        clean_win,
        coarse_cfo_hz=float(
            clean_cfo
        ),
        n_data_symbols=0,
    )

    H_clean = clean_sync[
        "H_hat"
    ]

    ltf_start = int(
        clean_sync[
            "ltf_start"
        ]
    )

    clean_c = apply_cfo_correction(
        clean_win,
        clean_sync[
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
            "Clean SIGNAL window exceeds "
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
    # CLEAN DATA FFTS
    # --------------------------------------------------------

    data_start = (
        ltf_start
        + cfg.LTF_LEN
        + cfg.SIG_LEN
    )

    clean_ffts = []

    for m in range(
        n_sym
    ):

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
                "Clean DATA symbol "
                f"{m} exceeds reference buffer: "
                f"e={e}, "
                f"buffer={len(clean_c)}, "
                f"ltf_start={ltf_start}, "
                f"data_start={data_start}"
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
# HEADER
# ============================================================

print("=" * 100)
print(
    "3F-2C-4 EQUALIZATION COMPARISON"
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
) = get_reference_data_symbols()


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
    # SAME SEEDING AS 3F-2C-3
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
    # APPLY CFO
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
    # SIGNAL FIELD
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
    # DATA FFT EXTRACTION
    # --------------------------------------------------------

    data_start = (
        ltf_start
        + cfg.LTF_LEN
        + cfg.SIG_LEN
    )

    noisy_ffts = []

    for m in range(
        n_sym
    ):

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
    # EQUALIZATION COMPARISON
    # ========================================================

    active = np.asarray(
        cfg.ACTIVE_SUBCARRIERS,
        dtype=int,
    )

    clean_eq = []

    noisy_eq_current = []

    noisy_eq_ideal = []

    # --------------------------------------------------------
    # CURRENT CHANNEL ESTIMATE
    # --------------------------------------------------------

    H_current = (
        H_hat[active]
    )

    # --------------------------------------------------------
    # IDEAL SYNTHETIC CHANNEL
    #
    # AWGN-only channel:
    #
    #       H[k] = 1
    #
    # --------------------------------------------------------

    H_ideal = np.ones_like(
        H_current
    )

    for m in range(
        n_sym
    ):

        X_clean = (
            clean_ffts[m][active]
        )

        X_noisy = (
            noisy_ffts[m][active]
        )

        # ----------------------------------------------------
        # Clean reference through current H estimate
        # ----------------------------------------------------

        clean_eq.append(
            X_clean
            / (
                H_current
                + 1e-12
            )
        )

        # ----------------------------------------------------
        # CURRENT receiver
        # ----------------------------------------------------

        noisy_eq_current.append(
            X_noisy
            / (
                H_current
                + 1e-12
            )
        )

        # ----------------------------------------------------
        # IDEAL receiver
        # ----------------------------------------------------

        noisy_eq_ideal.append(
            X_noisy
            / (
                H_ideal
                + 1e-12
            )
        )

    clean_eq = np.asarray(
        clean_eq
    )

    noisy_eq_current = (
        np.asarray(
            noisy_eq_current
        )
    )

    noisy_eq_ideal = (
        np.asarray(
            noisy_eq_ideal
        )
    )

    # ========================================================
    # SYMBOL DECISION ERRORS
    # ========================================================

    ref = (
        np.real(
            clean_eq
        )
        > 0
    )

    dec_current = (
        np.real(
            noisy_eq_current
        )
        > 0
    )

    dec_ideal = (
        np.real(
            noisy_eq_ideal
        )
        > 0
    )

    symbol_errors_current = int(
        np.sum(
            ref
            != dec_current
        )
    )

    symbol_errors_ideal = int(
        np.sum(
            ref
            != dec_ideal
        )
    )

    total_symbols = int(
        ref.size
    )

    symbol_ber_current = (
        symbol_errors_current
        / total_symbols
    )

    symbol_ber_ideal = (
        symbol_errors_ideal
        / total_symbols
    )

    # ========================================================
    # ACTUAL DEMODULATOR
    # ========================================================

    try:

        # ----------------------------------------------------
        # CURRENT H_hat
        # ----------------------------------------------------

        (
            rx_bits_current,
            crc_current,
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

        # ----------------------------------------------------
        # IDEAL H = 1
        # ----------------------------------------------------

        H_ideal_full = (
            np.ones_like(
                H_hat
            )
        )

        (
            rx_bits_ideal,
            crc_ideal,
        ) = (
            demod_mod.demodulate_packet(
                noisy_ffts,
                H_ideal_full,
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

        rx_bits_current = np.asarray(
            rx_bits_current,
            dtype=np.uint8,
        )

        rx_bits_ideal = np.asarray(
            rx_bits_ideal,
            dtype=np.uint8,
        )

        # ----------------------------------------------------
        # CURRENT PAYLOAD BER
        # ----------------------------------------------------

        n_compare_current = min(
            len(rx_bits_current),
            len(payload_bits),
        )

        bit_errors_current = int(
            np.sum(
                rx_bits_current[
                    :n_compare_current
                ]
                != payload_bits[
                    :n_compare_current
                ]
            )
        )

        payload_ber_current = (
            bit_errors_current
            / len(payload_bits)
        )

        # ----------------------------------------------------
        # IDEAL PAYLOAD BER
        # ----------------------------------------------------

        n_compare_ideal = min(
            len(rx_bits_ideal),
            len(payload_bits),
        )

        bit_errors_ideal = int(
            np.sum(
                rx_bits_ideal[
                    :n_compare_ideal
                ]
                != payload_bits[
                    :n_compare_ideal
                ]
            )
        )

        payload_ber_ideal = (
            bit_errors_ideal
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

    status_current = (
        "PASS"
        if bool(crc_current)
        else "CRC_FAIL"
    )

    status_ideal = (
        "PASS"
        if bool(crc_ideal)
        else "CRC_FAIL"
    )

    # ========================================================
    # STORE
    # ========================================================

    results.append(
        {
            "trial":
                trial,

            "status_current":
                status_current,

            "status_ideal":
                status_ideal,

            "det":
                det_start,

            "cfo":
                total_cfo,

            "ltf":
                ltf_start,

            "symbol_errors_current":
                symbol_errors_current,

            "symbol_errors_ideal":
                symbol_errors_ideal,

            "total_symbols":
                total_symbols,

            "symbol_ber_current":
                symbol_ber_current,

            "symbol_ber_ideal":
                symbol_ber_ideal,

            "bit_errors_current":
                bit_errors_current,

            "bit_errors_ideal":
                bit_errors_ideal,

            "payload_ber_current":
                payload_ber_current,

            "payload_ber_ideal":
                payload_ber_ideal,
        }
    )

    # ========================================================
    # PRINT
    # ========================================================

    print(
        f"Trial {trial:2d}: "
        f"CURRENT={status_current:8s} "
        f"IDEAL={status_ideal:8s} "
        f"det={det_start:4d} "
        f"CFO={total_cfo:9.2f} "
        f"LTF={ltf_start:3d} "
        f"| "
        f"symBER "
        f"current={symbol_ber_current:.6f} "
        f"ideal={symbol_ber_ideal:.6f} "
        f"| "
        f"bitBER "
        f"current={payload_ber_current:.6f} "
        f"ideal={payload_ber_ideal:.6f}"
    )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 100)
print(
    "3F-2C-4 SUMMARY"
)
print("=" * 100)


current_passes = [
    r
    for r in results
    if r["status_current"]
    == "PASS"
]

current_fails = [
    r
    for r in results
    if r["status_current"]
    == "CRC_FAIL"
]

ideal_passes = [
    r
    for r in results
    if r["status_ideal"]
    == "PASS"
]

ideal_fails = [
    r
    for r in results
    if r["status_ideal"]
    == "CRC_FAIL"
]


print()

print(
    "CURRENT H_hat:"
)

print(
    f"  PASS:       "
    f"{len(current_passes)}/{len(results)}"
)

print(
    f"  CRC_FAIL:   "
    f"{len(current_fails)}/{len(results)}"
)

print()

print(
    "IDEAL H = 1:"
)

print(
    f"  PASS:       "
    f"{len(ideal_passes)}/{len(results)}"
)

print(
    f"  CRC_FAIL:   "
    f"{len(ideal_fails)}/{len(results)}"
)


# ============================================================
# SYMBOL BER MEDIANS
# ============================================================

if results:

    print()
    print(
        "SYMBOL BER MEDIANS"
    )
    print(
        "------------------"
    )

    print(
        "Current H_hat:",
        np.median(
            [
                r[
                    "symbol_ber_current"
                ]
                for r in results
            ]
        )
    )

    print(
        "Ideal H=1:",
        np.median(
            [
                r[
                    "symbol_ber_ideal"
                ]
                for r in results
            ]
        )
    )


# ============================================================
# PAYLOAD BER MEDIANS
# ============================================================

if results:

    print()
    print(
        "PAYLOAD BER MEDIANS"
    )
    print(
        "-------------------"
    )

    print(
        "Current H_hat:",
        np.median(
            [
                r[
                    "payload_ber_current"
                ]
                for r in results
            ]
        )
    )

    print(
        "Ideal H=1:",
        np.median(
            [
                r[
                    "payload_ber_ideal"
                ]
                for r in results
            ]
        )
    )


# ============================================================
# CURRENT CRC FAILURES
# ============================================================

if current_fails:

    print()
    print(
        "CURRENT CRC-FAIL TRIALS"
    )
    print(
        "-----------------------"
    )

    for r in current_fails:

        print(
            f"trial={r['trial']:2d} "
            f"| "
            f"current_symBER="
            f"{r['symbol_ber_current']:.6f} "
            f"ideal_symBER="
            f"{r['symbol_ber_ideal']:.6f} "
            f"| "
            f"current_bitBER="
            f"{r['payload_ber_current']:.6f} "
            f"ideal_bitBER="
            f"{r['payload_ber_ideal']:.6f} "
            f"| "
            f"current_bitErr="
            f"{r['bit_errors_current']:3d} "
            f"ideal_bitErr="
            f"{r['bit_errors_ideal']:3d}"
        )


# ============================================================
# TRIAL-BY-TRIAL COMPARISON
# ============================================================

print()
print(
    "TRIAL-BY-TRIAL EQUALIZATION COMPARISON"
)
print(
    "--------------------------------------"
)

for r in results:

    print(
        f"trial={r['trial']:2d} "
        f"| "
        f"CURRENT={r['status_current']:8s} "
        f"IDEAL={r['status_ideal']:8s} "
        f"| "
        f"symBER "
        f"{r['symbol_ber_current']:.6f}"
        f" -> "
        f"{r['symbol_ber_ideal']:.6f} "
        f"| "
        f"bitBER "
        f"{r['payload_ber_current']:.6f}"
        f" -> "
        f"{r['payload_ber_ideal']:.6f}"
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 100)
print(
    "3F-2C-4 COMPLETE"
)
print(
    "NO PRODUCTION FILES WERE MODIFIED."
)
print("=" * 100)