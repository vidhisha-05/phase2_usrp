import numpy as np

import config as cfg
import waveform
import scrambler


# ============================================================
# 3F-1C
# Exact detector-window persistence characterization
#
# This reproduces the CURRENT detector.py evaluation model:
#
#   min_len = STF_LEN + LTF_LEN = 448
#   L = 16
#   metric length = 448 - 16 = 432
#
# If no detection occurs, detector advances by 432 samples.
#
# NO production files are modified.
# ============================================================


PAYLOAD_BYTES = 100
MODULATION = "BPSK"

PREFIX = 1000
SUFFIX = 7000

DETECT_THRESH = 0.65

PERSISTENCE = 8

SNR_DB_LIST = [
    30.0,
    25.0,
    20.0,
    15.0,
    10.0,
    5.0,
    0.0,
]

N_TRIALS = 20

BASE_SEED = 20261001

L = cfg.STF_LEN // 8

MIN_LEN = (
    cfg.STF_LEN
    + cfg.LTF_LEN
)

EXPECTED_METRIC_LEN = (
    MIN_LEN - L
)


# ============================================================
# Packet
# ============================================================

payload_bits = (
    np.arange(
        PAYLOAD_BYTES * 8,
        dtype=np.uint8,
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
    dtype=np.complex64,
)

expected_packet_len = (
    cfg.STF_LEN
    + cfg.LTF_LEN
    + cfg.SIG_LEN
    + 17 * cfg.SYMBOL_LEN
)

print(
    "3F-1C EXACT DETECTOR-WINDOW PERSISTENCE"
)

print(
    "========================================="
)

print()
print("CONFIGURATION")
print("-------------")

print(
    "STF_LEN:",
    cfg.STF_LEN,
)

print(
    "LTF_LEN:",
    cfg.LTF_LEN,
)

print(
    "Detector window:",
    MIN_LEN,
)

print(
    "Correlation lag:",
    L,
)

print(
    "Metric length:",
    EXPECTED_METRIC_LEN,
)

print(
    "Threshold:",
    DETECT_THRESH,
)

print(
    "Required persistence:",
    PERSISTENCE,
)

if MIN_LEN != 448:
    raise AssertionError(
        f"Expected detector window 448, got {MIN_LEN}"
    )

if L != 16:
    raise AssertionError(
        f"Expected correlation lag 16, got {L}"
    )

if EXPECTED_METRIC_LEN != 432:
    raise AssertionError(
        f"Expected metric length 432, got "
        f"{EXPECTED_METRIC_LEN}"
    )

print()
print("PACKET")
print("------")

print(
    "Payload:",
    PAYLOAD_BYTES,
    "B",
)

print(
    "Modulation:",
    MODULATION,
)

print(
    "Packet length:",
    len(packet),
)

print(
    "Expected packet length:",
    expected_packet_len,
)

if len(packet) != expected_packet_len:
    raise AssertionError(
        f"Packet length mismatch: "
        f"{len(packet)} != "
        f"{expected_packet_len}"
    )


# ============================================================
# Stream
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


print()
print("STREAM")
print("------")

print(
    "True packet start:",
    true_start,
)

print(
    "Stream length:",
    len(clean_stream),
)


# ============================================================
# Power
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

print()
print("SIGNAL POWER")
print("------------")

print(
    "Packet mean power:",
    signal_power,
)


# ============================================================
# EXACT detector.py correlation metric
# ============================================================

def sliding_corr_metric(buf):

    L_local = L

    N = len(buf) - L_local

    if N <= 0:
        return np.array(
            [],
            dtype=np.float64,
        )

    s = buf.strides[0]

    A = np.lib.stride_tricks.as_strided(
        buf,
        shape=(N, L_local),
        strides=(s, s),
    )

    B = np.lib.stride_tricks.as_strided(
        buf[L_local:],
        shape=(N, L_local),
        strides=(s, s),
    )

    P = (
        A * np.conj(B)
    ).sum(
        axis=1
    ).astype(
        np.complex64
    )

    R = (
        np.abs(B).astype(
            np.float64
        ) ** 2
    ).sum(
        axis=1
    )

    metric = (
        np.abs(P).astype(
            np.float64
        ) ** 2
        /
        (
            R ** 2
            + 1e-12
        )
    )

    return metric


# ============================================================
# Find longest consecutive threshold run
# ============================================================

def longest_run(metric):

    above = (
        metric >= DETECT_THRESH
    )

    if not np.any(above):
        return 0, None

    padded = np.concatenate(
        [
            np.array(
                [False],
                dtype=bool,
            ),
            above,
            np.array(
                [False],
                dtype=bool,
            ),
        ]
    )

    transitions = np.diff(
        padded.astype(
            np.int8
        )
    )

    starts = np.where(
        transitions == 1
    )[0]

    ends = (
        np.where(
            transitions == -1
        )[0]
        - 1
    )

    best_length = 0
    best_start = None

    for start, end in zip(
        starts,
        ends,
    ):

        length = (
            int(end)
            - int(start)
            + 1
        )

        if length > best_length:

            best_length = length
            best_start = int(start)

    return (
        best_length,
        best_start,
    )


# ============================================================
# Exact detector-window scan
#
# This mimics the NO-DETECTION path:
#
#   window = buf[:448]
#   metric = ...
#   advance = 432
#
# It does NOT implement persistence in production.
# It merely measures whether a persistence test would
# have enough samples inside the actual detector windows.
# ============================================================

def detector_window_scan(stream):

    stream = np.asarray(
        stream,
        dtype=np.complex64,
    )

    sample_idx = 0

    windows = []

    while (
        sample_idx + MIN_LEN
        <= len(stream)
    ):

        window = stream[
            sample_idx:
            sample_idx + MIN_LEN
        ]

        metric = sliding_corr_metric(
            window
        )

        if len(metric) != EXPECTED_METRIC_LEN:
            raise AssertionError(
                "Unexpected metric length"
            )

        run_length, run_start = (
            longest_run(metric)
        )

        windows.append(
            {
                "window_start":
                    sample_idx,
                "run_length":
                    run_length,
                "run_start":
                    run_start,
            }
        )

        # EXACT current detector no-detection hop.
        sample_idx += EXPECTED_METRIC_LEN

    return windows


# ============================================================
# Main experiment
# ============================================================

results = []


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
    print("=" * 80)
    print(
        f"SNR = {snr_db:.1f} dB"
    )
    print("=" * 80)

    for trial in range(
        N_TRIALS
    ):

        seed = (
            BASE_SEED
            + int(
                round(
                    snr_db * 100
                )
            )
            + trial
        )

        rng = np.random.default_rng(
            seed
        )

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

        noisy_stream = (
            clean_stream
            + noise
        ).astype(
            np.complex64
        )

        windows = detector_window_scan(
            noisy_stream
        )

        # ----------------------------------------------------
        # Noise-only detector windows
        # ----------------------------------------------------

        noise_windows = []

        for w in windows:

            window_start = (
                w["window_start"]
            )

            window_end = (
                window_start
                + MIN_LEN
            )

            if window_end <= true_start:

                noise_windows.append(w)

        # ----------------------------------------------------
        # Packet-overlapping detector windows
        # ----------------------------------------------------

        packet_windows = []

        for w in windows:

            window_start = (
                w["window_start"]
            )

            window_end = (
                window_start
                + MIN_LEN
            )

            if (
                window_end > true_start
                and
                window_start
                < true_start
                + cfg.STF_LEN
            ):

                packet_windows.append(w)

        # ----------------------------------------------------
        # Longest noise run
        # ----------------------------------------------------

        if noise_windows:

            noise_longest = max(
                w["run_length"]
                for w in noise_windows
            )

        else:

            noise_longest = 0

        # ----------------------------------------------------
        # Longest packet-window run
        # ----------------------------------------------------

        if packet_windows:

            packet_longest = max(
                w["run_length"]
                for w in packet_windows
            )

        else:

            packet_longest = 0

        # ----------------------------------------------------
        # Persistence survival
        # ----------------------------------------------------

        packet_survives = (
            packet_longest
            >= PERSISTENCE
        )

        noise_survives = (
            noise_longest
            >= PERSISTENCE
        )

        results.append(
            {
                "snr_db": snr_db,
                "trial": trial,
                "seed": seed,
                "noise_longest":
                    noise_longest,
                "packet_longest":
                    packet_longest,
                "packet_survives":
                    packet_survives,
                "noise_survives":
                    noise_survives,
            }
        )


# ============================================================
# Summary
# ============================================================

print()
print()
print("=" * 110)
print("3F-1C SUMMARY")
print("=" * 110)

print(
    "SNR | "
    "noise longest median | "
    "noise longest max | "
    "packet longest median | "
    "packet longest min | "
    "packet survive | "
    "noise survive"
)

print("-" * 110)


for snr_db in SNR_DB_LIST:

    rows = [
        r
        for r in results
        if r["snr_db"] == snr_db
    ]

    noise_lengths = np.array(
        [
            r["noise_longest"]
            for r in rows
        ],
        dtype=np.int64,
    )

    packet_lengths = np.array(
        [
            r["packet_longest"]
            for r in rows
        ],
        dtype=np.int64,
    )

    packet_survive = sum(
        r["packet_survives"]
        for r in rows
    )

    noise_survive = sum(
        r["noise_survives"]
        for r in rows
    )

    print(
        f"{snr_db:3.0f} | "
        f"{np.median(noise_lengths):20.1f} | "
        f"{np.max(noise_lengths):17d} | "
        f"{np.median(packet_lengths):22.1f} | "
        f"{np.min(packet_lengths):20d} | "
        f"{packet_survive:14d}/20 | "
        f"{noise_survive:12d}/20"
    )


# ============================================================
# Detailed window geometry
# ============================================================

print()
print()
print("=" * 110)
print("PACKET WINDOW GEOMETRY")
print("=" * 110)

clean_windows = detector_window_scan(
    clean_stream
)

for w in clean_windows:

    if (
        w["window_start"]
        <= true_start
        <=
        w["window_start"]
        + MIN_LEN
    ):

        print(
            "Window start:",
            w["window_start"],
        )

        print(
            "Window end:",
            w["window_start"]
            + MIN_LEN
            - 1,
        )

        print(
            "Packet start relative to window:",
            true_start
            - w["window_start"],
        )

        print(
            "Longest threshold run:",
            w["run_length"],
        )

        print(
            "Run start inside window:",
            w["run_start"],
        )


# ============================================================
# Final
# ============================================================

print()
print("=" * 110)
print("3F-1C COMPLETE")
print("=" * 110)

print(
    "No production files were modified."
)

print(
    "This test reproduces the detector's 448-sample "
    "window / 432-sample no-detection advance."
)