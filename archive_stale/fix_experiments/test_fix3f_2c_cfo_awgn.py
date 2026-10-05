"""
3F-2C — Full packet CFO + AWGN stress test.

Diagnostic only.
No production files are modified.

Tests the complete PHY chain under simultaneous:

    1. deterministic carrier-frequency offset (CFO)
    2. additive white Gaussian noise (AWGN)

The AWGN definition intentionally matches the previously
validated 3F-1A characterization:

    SNR = signal_power / noise_power

    noise_power = signal_power / 10^(SNR_dB / 10)

    sigma = sqrt(noise_power / 2)

Complex noise:

    noise = sigma * (normal + j*normal)

PHY chain:

    waveform.assemble_packet()
        -> CFO
        -> AWGN
        -> PacketDetector
        -> sync_packet()
        -> total CFO correction
        -> SIGNAL parsing
        -> DATA demodulation
        -> CRC
        -> payload comparison
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
    0.0,
    50_000.0,
    100_000.0,
    200_000.0,
]

SNR_DB_LIST = [
    30.0,
    20.0,
    15.0,
    10.0,
    5.0,
    0.0,
]

N_TRIALS = 5

BASE_SEED = 20261001

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
# GENERATE FIXED PAYLOAD
# ============================================================

payload_bits = (
    np.arange(
        PAYLOAD_BYTES * 8,
        dtype=np.uint8,
    )
    & 1
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
)

packet = np.asarray(
    packet,
    dtype=np.complex64,
)


# ============================================================
# PACKET LENGTH CHECK
# ============================================================

expected_packet_len = (
    cfg.STF_LEN
    + cfg.LTF_LEN
    + cfg.SIG_LEN
    + 17 * cfg.SYMBOL_LEN
)

if len(packet) != expected_packet_len:

    raise AssertionError(
        f"Unexpected packet length: "
        f"{len(packet)} != "
        f"{expected_packet_len}"
    )


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
).astype(
    np.complex64
)

true_start = PREFIX


# ============================================================
# SIGNAL POWER
#
# EXACT SAME DEFINITION AS 3F-1A
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

if not np.isfinite(signal_power):

    raise AssertionError(
        "Packet power is not finite."
    )

if signal_power <= 0:

    raise AssertionError(
        "Packet power must be positive."
    )


# ============================================================
# HEADER
# ============================================================

print("=" * 100)
print("3F-2C FULL PACKET CFO + AWGN STRESS")
print("=" * 100)

print()
print("PACKET")
print("------")

print(
    f"Payload bytes:        {PAYLOAD_BYTES}"
)

print(
    f"Modulation:           {MODULATION}"
)

print(
    f"Packet length:        {len(packet)}"
)

print(
    f"Expected length:      {expected_packet_len}"
)

print(
    f"Prefix:               {PREFIX}"
)

print(
    f"Suffix:               {SUFFIX}"
)

print(
    f"True packet start:    {true_start}"
)

print(
    f"FS_FFT:               {cfg.FS_FFT}"
)

print(
    f"Subcarrier spacing:   "
    f"{cfg.FS_FFT / cfg.FFT_SIZE:.2f} Hz"
)

print(
    f"Packet mean power:    {signal_power:.12f}"
)

print()
print("CFO sweep:")
for cfo in CFO_LIST_HZ:
    print(
        f"  {cfo:10.0f} Hz"
    )

print()
print("SNR sweep:")
for snr in SNR_DB_LIST:
    print(
        f"  {snr:10.1f} dB"
    )

print()
print(
    f"Trials per condition: {N_TRIALS}"
)


# ============================================================
# RESULT STORAGE
# ============================================================

results = []


# ============================================================
# MAIN SWEEP
# ============================================================

for snr_db in SNR_DB_LIST:

    snr_linear = (
        10.0 ** (
            snr_db / 10.0
        )
    )

    noise_power = (
        signal_power
        / snr_linear
    )

    sigma = np.sqrt(
        noise_power / 2.0
    )

    print()
    print("=" * 100)

    print(
        f"SNR = {snr_db:.1f} dB"
    )

    print(
        f"Noise power = {noise_power:.12e}"
    )

    print(
        f"Sigma       = {sigma:.12e}"
    )

    print("=" * 100)

    for cfo_hz in CFO_LIST_HZ:

        condition_passes = 0

        print()
        print(
            f"CFO = {cfo_hz:+.0f} Hz"
        )

        print(
            "-" * 100
        )

        print(
            f"{'Trial':>7} "
            f"{'Det':>7} "
            f"{'CFO est':>14} "
            f"{'CRC':>7} "
            f"{'Bits':>7} "
            f"{'Result':>10}"
        )

        for trial in range(
            N_TRIALS
        ):

            # ------------------------------------------------
            # Match the established 3F-1A deterministic
            # seed construction.
            # ------------------------------------------------

            seed = (
                BASE_SEED
                + int(
                    round(
                        snr_db * 100
                    )
                )
                + int(
                    round(
                        cfo_hz
                    )
                )
                + trial
            )

            rng = np.random.default_rng(
                seed
            )


            # ------------------------------------------------
            # Generate AWGN over the entire received stream.
            # ------------------------------------------------

            noise = (
                rng.standard_normal(
                    len(clean_stream)
                )
                +
                1j
                * rng.standard_normal(
                    len(clean_stream)
                )
            )

            noise = (
                noise * sigma
            ).astype(
                np.complex64
            )


            # ------------------------------------------------
            # Apply CFO to the clean stream first.
            # Then add AWGN.
            # ------------------------------------------------

            cfo_stream = apply_cfo(
                clean_stream,
                cfo_hz,
                cfg.FS_FFT,
            )

            rx = (
                cfo_stream
                + noise
            ).astype(
                np.complex64
            )


            # ------------------------------------------------
            # PACKET DETECTION
            # ------------------------------------------------

            detector = PacketDetector(
                fs=cfg.FS_FFT
            )

            detections = detector.process(
                rx
            )


            # ------------------------------------------------
            # Detection failure
            # ------------------------------------------------

            if len(detections) != 1:

                print(
                    f"{trial:7d} "
                    f"{'NONE':>7} "
                    f"{'NONE':>14} "
                    f"{'ERROR':>7} "
                    f"{'0':>7} "
                    f"{'DETECT':>10}"
                )

                results.append(
                    {
                        "snr_db": snr_db,
                        "cfo_hz": cfo_hz,
                        "trial": trial,
                        "seed": seed,
                        "detected": False,
                        "det_start": None,
                        "det_cfo": None,
                        "crc_ok": False,
                        "bits_ok": False,
                        "passed": False,
                        "failure": "detection",
                    }
                )

                continue


            # ------------------------------------------------
            # Detection succeeded
            # ------------------------------------------------

            det_start, det_cfo = (
                detections[0]
            )


            try:

                # --------------------------------------------
                # Synchronization window
                #
                # IMPORTANT:
                # start exactly at detector start,
                # matching validated 3F-2B geometry.
                # --------------------------------------------

                start = int(
                    det_start
                )

                end = min(
                    len(rx),
                    start
                    + len(packet)
                    + 512,
                )

                win = rx[
                    start:end
                ]


                # --------------------------------------------
                # Full packet synchronization
                # --------------------------------------------

                res = sync_packet(
                    win,
                    coarse_cfo_hz=det_cfo,
                    n_data_symbols=0,
                )

                H_hat = res[
                    "H_hat"
                ]

                ltf_start = res[
                    "ltf_start"
                ]

                total_cfo = res[
                    "total_cfo"
                ]


                # --------------------------------------------
                # Total CFO correction
                # --------------------------------------------

                rx_c = apply_cfo_correction(
                    win,
                    total_cfo,
                    start_n=0,
                )


                # --------------------------------------------
                # SIGNAL extraction
                # --------------------------------------------

                sig_body_start = (
                    ltf_start
                    + cfg.LTF_LEN
                    + cfg.CP_LEN
                )

                sig_body_end = (
                    sig_body_start
                    + cfg.FFT_SIZE
                )

                if (
                    sig_body_end
                    > len(rx_c)
                ):

                    raise RuntimeError(
                        "SIGNAL window outside "
                        "received buffer"
                    )


                Y_sig = np.fft.fft(
                    rx_c[
                        sig_body_start:
                        sig_body_end
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


                if not sig[
                    "valid"
                ]:

                    raise RuntimeError(
                        "SIGNAL parse failed"
                    )


                # --------------------------------------------
                # SIGNAL parameters
                # --------------------------------------------

                n_bytes = int(
                    sig[
                        "n_payload_bytes"
                    ]
                )

                modulation = sig[
                    "modulation"
                ]

                n_sym = int(
                    sig[
                        "n_data_syms"
                    ]
                )


                # --------------------------------------------
                # DATA extraction
                # --------------------------------------------

                data_start = (
                    ltf_start
                    + cfg.LTF_LEN
                    + cfg.SIG_LEN
                )

                ffts = []


                for m in range(
                    n_sym
                ):

                    s = (
                        data_start
                        + m
                        * cfg.SYMBOL_LEN
                        + cfg.CP_LEN
                    )

                    e = (
                        s
                        + cfg.FFT_SIZE
                    )


                    if e > len(
                        rx_c
                    ):

                        raise RuntimeError(
                            "DATA symbol outside "
                            "received buffer"
                        )


                    Y = np.fft.fft(
                        rx_c[
                            s:e
                        ],
                        n=cfg.FFT_SIZE,
                    ).astype(
                        np.complex64
                    )

                    ffts.append(
                        Y
                    )


                # --------------------------------------------
                # DATA demodulation
                # --------------------------------------------

                rx_bits, crc_ok = (
                    demod_mod.demodulate_packet(
                        ffts,
                        H_hat,
                        modulation=modulation,
                        n_payload_bytes=n_bytes,
                    )
                )


                # --------------------------------------------
                # Payload validation
                # --------------------------------------------

                expected_bits = (
                    PAYLOAD_BYTES * 8
                )

                bits_ok = (
                    n_bytes
                    == PAYLOAD_BYTES
                    and
                    modulation
                    == MODULATION
                    and
                    len(rx_bits)
                    == expected_bits
                    and
                    np.array_equal(
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


                if passed:

                    condition_passes += 1


                results.append(
                    {
                        "snr_db": snr_db,
                        "cfo_hz": cfo_hz,
                        "trial": trial,
                        "seed": seed,
                        "detected": True,
                        "det_start": int(
                            det_start
                        ),
                        "det_cfo": float(
                            det_cfo
                        ),
                        "crc_ok": bool(
                            crc_ok
                        ),
                        "bits_ok": bool(
                            bits_ok
                        ),
                        "passed": passed,
                        "failure": None
                        if passed
                        else "payload_or_crc",
                    }
                )


                result_text = (
                    "PASS"
                    if passed
                    else "FAIL"
                )


                print(
                    f"{trial:7d} "
                    f"{det_start:7d} "
                    f"{det_cfo:14.2f} "
                    f"{str(bool(crc_ok)):>7} "
                    f"{len(rx_bits):7d} "
                    f"{result_text:>10}"
                )


            except Exception as exc:

                results.append(
                    {
                        "snr_db": snr_db,
                        "cfo_hz": cfo_hz,
                        "trial": trial,
                        "seed": seed,
                        "detected": True,
                        "det_start": int(
                            det_start
                        ),
                        "det_cfo": float(
                            det_cfo
                        ),
                        "crc_ok": False,
                        "bits_ok": False,
                        "passed": False,
                        "failure": str(
                            exc
                        ),
                    }
                )


                print(
                    f"{trial:7d} "
                    f"{det_start:7d} "
                    f"{det_cfo:14.2f} "
                    f"{'ERROR':>7} "
                    f"{'0':>7} "
                    f"{type(exc).__name__:>10}"
                )

                print(
                    f"        {exc}"
                )


        # ----------------------------------------------------
        # Condition summary
        # ----------------------------------------------------

        print(
            f"Condition result: "
            f"{condition_passes}/{N_TRIALS} "
            f"passed"
        )


# ============================================================
# OVERALL SUMMARY
# ============================================================

print()
print()
print("=" * 110)
print("3F-2C SUMMARY")
print("=" * 110)

print()

print(
    f"{'SNR':>7} "
    f"{'CFO':>12} "
    f"{'Pass':>8} "
    f"{'Total':>8} "
    f"{'Rate':>10}"
)

print("-" * 110)


all_conditions_pass = True


for snr_db in SNR_DB_LIST:

    for cfo_hz in CFO_LIST_HZ:

        rows = [
            r
            for r in results
            if (
                r["snr_db"]
                == snr_db
                and
                r["cfo_hz"]
                == cfo_hz
            )
        ]

        passed_count = sum(
            bool(
                r["passed"]
            )
            for r in rows
        )

        total_count = len(
            rows
        )

        if total_count > 0:

            rate = (
                100.0
                * passed_count
                / total_count
            )

        else:

            rate = 0.0


        if (
            total_count != N_TRIALS
            or
            passed_count
            != total_count
        ):

            all_conditions_pass = False


        print(
            f"{snr_db:7.0f} "
            f"{cfo_hz:12.0f} "
            f"{passed_count:8d} "
            f"{total_count:8d} "
            f"{rate:9.1f}%"
        )


# ============================================================
# FAILURE BREAKDOWN
# ============================================================

print()
print("=" * 110)
print("FAILURE BREAKDOWN")
print("=" * 110)

failure_rows = [
    r
    for r in results
    if not r["passed"]
]

if not failure_rows:

    print(
        "No failures."
    )

else:

    from collections import Counter

    failure_types = Counter()

    for r in failure_rows:

        failure = r[
            "failure"
        ]

        if (
            failure
            == "detection"
        ):

            key = "detection"

        elif (
            failure
            == "payload_or_crc"
        ):

            key = "CRC/payload"

        else:

            key = str(
                failure
            )

        failure_types[
            key
        ] += 1


    for key, count in (
        failure_types.items()
    ):

        print(
            f"{key}: {count}"
        )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 110)

if all_conditions_pass:

    print(
        "3F-2C RESULT: PASS"
    )

else:

    print(
        "3F-2C RESULT: INVESTIGATE"
    )

print("=" * 110)

print()
print(
    "No production files were modified."
)

print(
    "This test uses the same AWGN power definition "
    "as the validated 3F-1A characterization."
)