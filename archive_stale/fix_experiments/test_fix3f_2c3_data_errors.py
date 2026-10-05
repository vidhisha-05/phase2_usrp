"""
3F-2C-3
DATA-level error characterization at 15 dB, 0 Hz CFO.

Diagnostic only.
NO production files are modified.
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
# GENERATE REFERENCE PACKET
# ============================================================

payload_bits = (
    np.arange(PAYLOAD_BYTES * 8, dtype=np.uint8) & 1
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

signal_power = float(
    np.mean(
        np.abs(packet.astype(np.complex128)) ** 2
    )
)

snr_linear = 10.0 ** (SNR_DB / 10.0)
noise_power = signal_power / snr_linear
sigma = np.sqrt(noise_power / 2.0)


# ============================================================
# HELPERS
# ============================================================

def apply_cfo(samples, cfo_hz, fs):
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
        * np.exp(1j * phase).astype(np.complex64)
    )


def get_reference_data_symbols():
    """
    Recover the clean packet's DATA FFT symbols and
    clean channel estimate.

    This is used only as a diagnostic reference.
    """

    detector = PacketDetector()

    clean_det = detector.process(
        clean_stream
    )

    if len(clean_det) != 1:
        raise RuntimeError(
            f"Expected 1 clean detection, got {clean_det}"
        )

    clean_start, clean_cfo = clean_det[0]

    start = int(clean_start)

    clean_win = clean_stream[
        start:
        start + len(packet)
    ]

    clean_sync = sync_packet(
        clean_win,
        coarse_cfo_hz=float(clean_cfo),
        n_data_symbols=0,
    )

    H_clean = clean_sync["H_hat"]
    ltf_start = int(clean_sync["ltf_start"])

    clean_c = apply_cfo_correction(
        clean_win,
        clean_sync["total_cfo"],
        start_n=0,
    )

    sig_start = (
        ltf_start
        + cfg.LTF_LEN
        + cfg.CP_LEN
    )

    Y_sig = np.fft.fft(
        clean_c[
            sig_start:
            sig_start + cfg.FFT_SIZE
        ],
        n=cfg.FFT_SIZE,
    ).astype(np.complex64)

    sig = demod_mod.parse_signal_field(
        Y_sig,
        H_clean,
    )

    if not sig["valid"]:
        raise RuntimeError(
            "Clean SIGNAL parse failed"
        )

    n_sym = int(sig["n_data_syms"])

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

        e = s + cfg.FFT_SIZE

        clean_ffts.append(
            np.fft.fft(
                clean_c[s:e],
                n=cfg.FFT_SIZE,
            ).astype(np.complex64)
        )

    return (
        clean_ffts,
        H_clean,
        sig,
    )


# ============================================================
# CLEAN REFERENCE
# ============================================================

print("=" * 100)
print("3F-2C-3 DATA ERROR CHARACTERIZATION")
print("=" * 100)

print()
print(f"SNR:                 {SNR_DB} dB")
print(f"CFO:                 {CFO_HZ} Hz")
print(f"Payload:             {PAYLOAD_BYTES} B")
print(f"Modulation:          {MODULATION}")
print(f"Packet length:       {len(packet)}")
print(f"Packet power:        {signal_power:.12f}")
print(f"Noise sigma:         {sigma:.12e}")

clean_ffts, H_clean, clean_sig = (
    get_reference_data_symbols()
)

print()
print("Clean SIGNAL:")
print(
    f"  modulation = {clean_sig['modulation']}"
)
print(
    f"  bytes      = {clean_sig['n_payload_bytes']}"
)
print(
    f"  symbols    = {clean_sig['n_data_syms']}"
)


# ============================================================
# RUN TRIALS
# ============================================================

results = []


for trial in range(N_TRIALS):

    seed = (
        BASE_SEED
        + int(round(SNR_DB * 100))
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

    noisy_stream = (
        cfo_stream
        + noise.astype(np.complex64)
    ).astype(np.complex64)

    detector = PacketDetector()

    detections = detector.process(
        noisy_stream
    )

    if not detections:

        print(
            f"Trial {trial:2d}: "
            f"DETECTION_FAILURE"
        )

        continue

    det_start, det_cfo = detections[0]

    det_start = int(det_start)

    start = det_start

    end = min(
        len(noisy_stream),
        start + len(packet) + SUFFIX,
    )

    win = noisy_stream[start:end]

    # --------------------------------------------------------
    # SYNCHRONIZATION
    # --------------------------------------------------------

    try:

        sync_res = sync_packet(
            win,
            coarse_cfo_hz=float(det_cfo),
            n_data_symbols=0,
        )

    except Exception as exc:

        print(
            f"Trial {trial:2d}: "
            f"SYNC_EXCEPTION {exc}"
        )

        continue

    H_hat = sync_res["H_hat"]

    ltf_start = int(
        sync_res["ltf_start"]
    )

    total_cfo = float(
        sync_res["total_cfo"]
    )

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
            f"SIGNAL_WINDOW_FAILURE"
        )

        continue

    Y_sig = np.fft.fft(
        rx_c[sig_start:sig_end],
        n=cfg.FFT_SIZE,
    ).astype(np.complex64)

    sig = demod_mod.parse_signal_field(
        Y_sig,
        H_hat,
    )

    if not sig["valid"]:

        print(
            f"Trial {trial:2d}: "
            f"SIGNAL_PARSE_FAILURE"
        )

        continue

    n_sym = int(
        sig["n_data_syms"]
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

        e = s + cfg.FFT_SIZE

        if e > len(rx_c):

            print(
                f"Trial {trial:2d}: "
                f"DATA_WINDOW_FAILURE"
            )

            noisy_ffts = []
            break

        noisy_ffts.append(
            np.fft.fft(
                rx_c[s:e],
                n=cfg.FFT_SIZE,
            ).astype(np.complex64)
        )

    if not noisy_ffts:
        continue

    # --------------------------------------------------------
    # EQUALIZED DATA
    # --------------------------------------------------------

    active = np.asarray(
        cfg.ACTIVE_SUBCARRIERS,
        dtype=int,
    )

    # Clean and noisy equalized symbols.
    #
    # H_hat is the receiver's channel estimate.
    #
    # For this synthetic AWGN channel the true channel is
    # effectively unity, so this comparison helps reveal
    # whether the LTF estimate itself becomes noisy.

    clean_eq = []
    noisy_eq = []

    for m in range(n_sym):

        X_clean = (
            clean_ffts[m][active]
        )

        X_noisy = (
            noisy_ffts[m][active]
        )

        H_use = H_hat[active]

        clean_eq.append(
            X_clean / (
                H_use + 1e-12
            )
        )

        noisy_eq.append(
            X_noisy / (
                H_use + 1e-12
            )
        )

    clean_eq = np.asarray(
        clean_eq
    )

    noisy_eq = np.asarray(
        noisy_eq
    )

    # --------------------------------------------------------
    # SYMBOL ERROR MEASURE
    # --------------------------------------------------------

    # BPSK reference signs from clean symbols.
    #
    # We compare the noisy equalized symbols against the
    # corresponding clean equalized symbols.

    ref = np.real(
        clean_eq
    ) > 0

    dec = np.real(
        noisy_eq
    ) > 0

    symbol_errors = np.sum(
        ref != dec
    )

    total_symbols = ref.size

    symbol_ber = (
        symbol_errors
        / total_symbols
    )

    # --------------------------------------------------------
    # HARD DATA DEMOD
    # --------------------------------------------------------

    try:

        rx_bits, crc_ok = (
            demod_mod.demodulate_packet(
                noisy_ffts,
                H_hat,
                modulation=sig["modulation"],
                n_payload_bytes=int(
                    sig["n_payload_bytes"]
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
                rx_bits[:n_compare]
                != payload_bits[:n_compare]
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

    status = (
        "PASS"
        if bool(crc_ok)
        else "CRC_FAILURE"
    )

    results.append(
        {
            "trial": trial,
            "status": status,
            "det": det_start,
            "cfo": total_cfo,
            "ltf": ltf_start,
            "symbol_errors": int(
                symbol_errors
            ),
            "total_symbols": int(
                total_symbols
            ),
            "symbol_ber": float(
                symbol_ber
            ),
            "bit_errors": bit_errors,
            "payload_ber": float(
                payload_ber
            ),
        }
    )

    print(
        f"Trial {trial:2d}: "
        f"{status:12s} "
        f"det={det_start:4d} "
        f"CFO={total_cfo:9.2f} "
        f"LTF={ltf_start:3d} "
        f"sym_err={symbol_errors:4d}/{total_symbols:<4d} "
        f"symBER={symbol_ber:.6f} "
        f"bit_err={bit_errors:4d}/800 "
        f"bitBER={payload_ber:.6f}"
    )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 100)
print("3F-2C-3 SUMMARY")
print("=" * 100)

passes = [
    r for r in results
    if r["status"] == "PASS"
]

fails = [
    r for r in results
    if r["status"] == "CRC_FAILURE"
]

print()
print(
    f"PASS trials:        {len(passes)}/{len(results)}"
)

print(
    f"CRC failures:       {len(fails)}/{len(results)}"
)

if passes:

    print()
    print("PASS TRIAL ERROR STATISTICS")
    print("----------------------------")

    print(
        "symbol BER median:",
        np.median(
            [r["symbol_ber"] for r in passes]
        )
    )

    print(
        "payload BER median:",
        np.median(
            [r["payload_ber"] for r in passes]
        )
    )

if fails:

    print()
    print("CRC-FAIL TRIAL ERROR STATISTICS")
    print("--------------------------------")

    print(
        "symbol BER median:",
        np.median(
            [r["symbol_ber"] for r in fails]
        )
    )

    print(
        "payload BER median:",
        np.median(
            [r["payload_ber"] for r in fails]
        )
    )

    print()
    print("CRC-FAIL DETAILS")

    for r in fails:

        print(
            f"  trial={r['trial']:2d} "
            f"symBER={r['symbol_ber']:.6f} "
            f"payloadBER={r['payload_ber']:.6f} "
            f"symErr={r['symbol_errors']:4d}/"
            f"{r['total_symbols']} "
            f"bitErr={r['bit_errors']:4d}/800 "
            f"CFO={r['cfo']:.2f} "
            f"LTF={r['ltf']}"
        )

print()
print("=" * 100)
print("3F-2C-3 COMPLETE")
print("NO PRODUCTION FILES WERE MODIFIED.")
print("=" * 100)