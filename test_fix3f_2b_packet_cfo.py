"""
3F-2B — Full packet CFO stress test.

Diagnostic only.
No production files are modified.

Tests the actual PHY chain:

    waveform.assemble_packet()
        -> CFO impairment
        -> PacketDetector
        -> sync_packet()
        -> SIGNAL parsing
        -> DATA demodulation
        -> CRC / payload recovery

Important coordinate convention:
    The synchronization window starts exactly at the detector's
    reported packet start.

This matches the validated 3D downstream synchronization test.
"""

import numpy as np

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

CFO_LIST_HZ = [
    -200_000.0,
    -100_000.0,
    -50_000.0,
    -25_000.0,
    -10_000.0,
    -5_000.0,
    0.0,
    5_000.0,
    10_000.0,
    25_000.0,
    50_000.0,
    100_000.0,
    200_000.0,
]

PREFIX = 1000
SUFFIX = 7000


# ============================================================
# HELPER: APPLY CFO
# ============================================================

def apply_cfo(
    signal: np.ndarray,
    cfo_hz: float,
    fs_hz: float,
) -> np.ndarray:

    n = np.arange(
        len(signal),
        dtype=np.float64,
    )

    phase = (
        2.0
        * np.pi
        * cfo_hz
        * n
        / fs_hz
    )

    return (
        signal
        * np.exp(1j * phase)
    ).astype(np.complex64)


# ============================================================
# GENERATE PAYLOAD
# ============================================================

rng = np.random.default_rng(20260930)

payload_bits = rng.integers(
    0,
    2,
    size=PAYLOAD_BYTES * 8,
    dtype=np.uint8,
)


# ============================================================
# GENERATE ACTUAL PHY PACKET
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
# EXPECTED PACKET LENGTH
# ============================================================

expected_packet_len = (
    cfg.STF_LEN
    + cfg.LTF_LEN
    + cfg.SIG_LEN
    + 17 * cfg.SYMBOL_LEN
)


# ============================================================
# HEADER
# ============================================================

print("=" * 90)
print("3F-2B FULL PACKET CFO STRESS")
print("=" * 90)

print()
print("PACKET")
print("------")

print(f"Payload bytes:        {PAYLOAD_BYTES}")
print(f"Modulation:           {MODULATION}")
print(f"Actual packet length: {len(packet)}")
print(f"Expected length:      {expected_packet_len}")
print(f"Prefix:               {PREFIX}")
print(f"Suffix:               {SUFFIX}")
print(f"FS_FFT:               {cfg.FS_FFT}")

print(
    f"Subcarrier spacing:   "
    f"{cfg.FS_FFT / cfg.FFT_SIZE:.2f} Hz"
)

print()

assert len(packet) == expected_packet_len


# ============================================================
# CFO SWEEP
# ============================================================

print("=" * 90)
print("RESULTS")
print("=" * 90)

print()

print(
    f"{'True CFO':>12} "
    f"{'Detected':>12} "
    f"{'Est. CFO':>14} "
    f"{'CRC':>8} "
    f"{'Bits':>8} "
    f"{'Result':>10}"
)

print("-" * 90)


all_pass = True


# ============================================================
# TEST EACH CFO
# ============================================================

for true_cfo in CFO_LIST_HZ:

    # --------------------------------------------------------
    # Build clean stream
    # --------------------------------------------------------

    prefix = np.zeros(
        PREFIX,
        dtype=np.complex64,
    )

    suffix = np.zeros(
        SUFFIX,
        dtype=np.complex64,
    )

    clean_stream = np.concatenate(
        [
            prefix,
            packet,
            suffix,
        ]
    )


    # --------------------------------------------------------
    # Apply CFO to entire received stream
    # --------------------------------------------------------

    rx = apply_cfo(
        clean_stream,
        true_cfo,
        cfg.FS_FFT,
    )


    # --------------------------------------------------------
    # PACKET DETECTION
    # --------------------------------------------------------

    detector = PacketDetector(
        fs=cfg.FS_FFT,
    )

    detections = detector.process(rx)

    if len(detections) != 1:

        print(
            f"{true_cfo:12.0f} "
            f"{'NONE':>12} "
            f"{'NONE':>14} "
            f"{'ERROR':>8} "
            f"{'0':>8} "
            f"{'DETECT':>10}"
        )

        all_pass = False
        continue


    det_start, det_cfo = detections[0]


    # --------------------------------------------------------
    # SYNCHRONIZATION WINDOW
    #
    # IMPORTANT:
    #
    # Start exactly at detector's reported packet start.
    #
    # Do NOT subtract 64 samples here.
    #
    # For the clean packet:
    #
    #   true packet start = 1000
    #   detector start    = 997
    #
    # therefore true packet begins at sample 3 inside win.
    # --------------------------------------------------------

    start = int(det_start)

    end = min(
        len(rx),
        start + len(packet) + 512,
    )

    win = rx[start:end]


    # --------------------------------------------------------
    # FULL SYNCHRONIZATION
    # --------------------------------------------------------

    try:

        res = sync_packet(
            win,
            coarse_cfo_hz=det_cfo,
            n_data_symbols=0,
        )

        H_hat = res["H_hat"]

        ltf_start = res["ltf_start"]

        total_cfo = res["total_cfo"]


        # ----------------------------------------------------
        # TOTAL CFO CORRECTION
        # ----------------------------------------------------

        rx_c = apply_cfo_correction(
            win,
            total_cfo,
            start_n=0,
        )


        # ----------------------------------------------------
        # SIGNAL FIELD
        # ----------------------------------------------------

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

            raise RuntimeError(
                "SIGNAL window outside received buffer"
            )


        Y_sig = np.fft.fft(
            rx_c[
                sig_body_start:
                sig_body_end
            ],
            n=cfg.FFT_SIZE,
        ).astype(np.complex64)


        sig = demod_mod.parse_signal_field(
            Y_sig,
            H_hat,
        )


        if not sig["valid"]:

            raise RuntimeError(
                "SIGNAL parse failed"
            )


        # ----------------------------------------------------
        # READ SIGNAL PARAMETERS
        # ----------------------------------------------------

        n_bytes = int(
            sig["n_payload_bytes"]
        )

        modulation = sig["modulation"]

        n_sym = int(
            sig["n_data_syms"]
        )


        # ----------------------------------------------------
        # DATA SYMBOL EXTRACTION
        # ----------------------------------------------------

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

            e = (
                s
                + cfg.FFT_SIZE
            )


            if e > len(rx_c):

                raise RuntimeError(
                    "DATA symbol outside received buffer"
                )


            Y = np.fft.fft(
                rx_c[s:e],
                n=cfg.FFT_SIZE,
            ).astype(np.complex64)


            ffts.append(Y)


        # ----------------------------------------------------
        # DATA DEMODULATION
        # ----------------------------------------------------

        rx_bits, crc_ok = (
            demod_mod.demodulate_packet(
                ffts,
                H_hat,
                modulation=modulation,
                n_payload_bytes=n_bytes,
            )
        )


        # ----------------------------------------------------
        # PAYLOAD VALIDATION
        # ----------------------------------------------------

        expected_bits = PAYLOAD_BYTES * 8

        bits_ok = (
            n_bytes == PAYLOAD_BYTES
            and modulation == MODULATION
            and len(rx_bits) == expected_bits
            and np.array_equal(
                np.asarray(
                    rx_bits,
                    dtype=np.uint8,
                ),
                payload_bits,
            )
        )


        passed = bool(
            crc_ok
            and bits_ok
        )


        if not passed:

            all_pass = False


        result_text = (
            "PASS"
            if passed
            else "FAIL"
        )


        print(
            f"{true_cfo:12.0f} "
            f"{det_start:12d} "
            f"{det_cfo:14.2f} "
            f"{str(bool(crc_ok)):>8} "
            f"{len(rx_bits):8d} "
            f"{result_text:>10}"
        )


    # --------------------------------------------------------
    # CATCH PHY FAILURE
    # --------------------------------------------------------

    except Exception as exc:

        all_pass = False

        print(
            f"{true_cfo:12.0f} "
            f"{det_start:12d} "
            f"{det_cfo:14.2f} "
            f"{'ERROR':>8} "
            f"{'0':>8} "
            f"{type(exc).__name__:>10}"
        )

        print(
            f"    {exc}"
        )


# ============================================================
# FINAL RESULT
# ============================================================

print()
print("=" * 90)

if all_pass:

    print("3F-2B RESULT: PASS")

else:

    print("3F-2B RESULT: INVESTIGATE")

print("=" * 90)

print()
print("No production files were modified.")