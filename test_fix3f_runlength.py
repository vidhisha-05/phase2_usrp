import numpy as np

import config as cfg
import waveform
import scrambler


# ============================================================
# 3F-1B
# Correlation threshold-run characterization
#
# PURPOSE
# -------
# Determine how long the CURRENT detector metric remains
# continuously above DETECT_THRESH.
#
# We are testing:
#
#   1. Noise-only regions
#   2. True packet / STF region
#   3. False-positive threshold crossings
#
# This is a DIAGNOSTIC ONLY.
#
# No production file is modified.
# ============================================================


PAYLOAD_BYTES = 100
MODULATION = "BPSK"

PREFIX = 1000
SUFFIX = 7000

DETECT_THRESH = 0.65

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

L = 16


# ============================================================
# Generate reference packet
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


print(
    "3F-1B CORRELATION RUN-LENGTH CHARACTERIZATION"
)

print(
    "=============================================="
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

expected_packet_len = (
    cfg.STF_LEN
    + cfg.LTF_LEN
    + cfg.SIG_LEN
    + 17 * cfg.SYMBOL_LEN
)

print(
    "Expected packet length:",
    expected_packet_len,
)

if len(packet) != expected_packet_len:
    raise AssertionError(
        f"Unexpected packet length: "
        f"{len(packet)} != "
        f"{expected_packet_len}"
    )


# ============================================================
# Construct clean stream
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
# Signal power
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
# EXACT CURRENT detector.py metric
#
# detector.py currently does:
#
#   L = 16
#   N = len(buf) - L
#
#   A = windows over buf
#   B = windows over buf[L:]
#
#   P = sum(A * conj(B))
#   R = sum(|B|^2)
#
#   metric = |P|^2 / (R^2 + 1e-12)
# ============================================================

def calculate_metric(samples):

    samples = np.asarray(
        samples,
        dtype=np.complex64,
    )

    N = len(samples) - L

    if N <= 0:
        return np.array(
            [],
            dtype=np.float64,
        )

    s = samples.strides[0]

    A = np.lib.stride_tricks.as_strided(
        samples,
        shape=(N, L),
        strides=(s, s),
    )

    B = np.lib.stride_tricks.as_strided(
        samples[L:],
        shape=(N, L),
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
# Find consecutive runs above threshold
# ============================================================

def find_runs_above_threshold(
    metric,
    threshold,
):
    """
    Return all contiguous runs where:

        metric >= threshold

    Each result is:

        {
            "start": first index,
            "end": last index,
            "length": number of samples
        }
    """

    above = (
        metric >= threshold
    )

    if not np.any(above):
        return []

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

    runs = []

    for start, end in zip(
        starts,
        ends,
    ):

        runs.append(
            {
                "start": int(start),
                "end": int(end),
                "length": int(
                    end - start + 1
                ),
            }
        )

    return runs


# ============================================================
# Find packet-region runs
# ============================================================

def packet_runs(
    runs,
    packet_start,
    packet_end,
):
    """
    Keep threshold runs that overlap the packet
    neighborhood.
    """

    result = []

    for run in runs:

        if (
            run["end"] >= packet_start
            and
            run["start"] <= packet_end
        ):
            result.append(run)

    return result


# ============================================================
# Main characterization
# ============================================================

all_results = []


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

        metric = calculate_metric(
            noisy_stream
        )

        if len(metric) == 0:
            raise AssertionError(
                "Metric is empty."
            )

        runs = find_runs_above_threshold(
            metric,
            DETECT_THRESH,
        )

        # ----------------------------------------------------
        # Noise-only runs
        #
        # We define the noise-only region as everything
        # ending 32 samples before the known packet.
        # ----------------------------------------------------

        noise_end = (
            true_start - 32
        )

        noise_runs = [
            r
            for r in runs
            if r["start"] < noise_end
        ]

        # Clip a run if it extends into the protected region.
        clipped_noise_runs = []

        for r in noise_runs:

            start = r["start"]

            end = min(
                r["end"],
                noise_end - 1,
            )

            if end >= start:

                clipped_noise_runs.append(
                    {
                        "start": start,
                        "end": end,
                        "length": end - start + 1,
                    }
                )

        # ----------------------------------------------------
        # Packet neighborhood
        #
        # We inspect:
        #
        #   true_start - 64
        #   through
        #   true_start + 128
        #
        # This covers the detector's observed packet
        # threshold-crossing region.
        # ----------------------------------------------------

        packet_region_start = max(
            0,
            true_start - 64,
        )

        packet_region_end = min(
            len(metric) - 1,
            true_start + 128,
        )

        packet_region_runs = packet_runs(
            runs,
            packet_region_start,
            packet_region_end,
        )

        # ----------------------------------------------------
        # Longest noise run
        # ----------------------------------------------------

        if clipped_noise_runs:

            longest_noise_run = max(
                clipped_noise_runs,
                key=lambda r: r["length"],
            )

            noise_run_length = (
                longest_noise_run["length"]
            )

            noise_run_start = (
                longest_noise_run["start"]
            )

        else:

            noise_run_length = 0
            noise_run_start = None

        # ----------------------------------------------------
        # Longest packet run
        # ----------------------------------------------------

        if packet_region_runs:

            longest_packet_run = max(
                packet_region_runs,
                key=lambda r: r["length"],
            )

            packet_run_length = (
                longest_packet_run["length"]
            )

            packet_run_start = (
                longest_packet_run["start"]
            )

        else:

            packet_run_length = 0
            packet_run_start = None

        # ----------------------------------------------------
        # First threshold crossing
        # ----------------------------------------------------

        above = np.where(
            metric >= DETECT_THRESH
        )[0]

        if len(above):

            first_crossing = int(
                above[0]
            )

        else:

            first_crossing = None

        false_positive = (
            first_crossing is not None
            and
            first_crossing < noise_end
        )

        # ----------------------------------------------------
        # Number of noise runs
        # ----------------------------------------------------

        n_noise_runs = len(
            clipped_noise_runs
        )

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        all_results.append(
            {
                "snr_db": snr_db,
                "trial": trial,
                "seed": seed,
                "noise_run_length":
                    noise_run_length,
                "noise_run_start":
                    noise_run_start,
                "packet_run_length":
                    packet_run_length,
                "packet_run_start":
                    packet_run_start,
                "n_noise_runs":
                    n_noise_runs,
                "first_crossing":
                    first_crossing,
                "false_positive":
                    false_positive,
            }
        )


# ============================================================
# Summary
# ============================================================

print()
print()
print("=" * 120)
print("3F-1B SUMMARY")
print("=" * 120)

print(
    "SNR | "
    "noise longest median | "
    "noise longest max | "
    "packet longest median | "
    "packet longest min | "
    "trials with noise run | "
    "false-positive trials"
)

print("-" * 120)


for snr_db in SNR_DB_LIST:

    rows = [
        r
        for r in all_results
        if r["snr_db"] == snr_db
    ]

    noise_lengths = np.array(
        [
            r["noise_run_length"]
            for r in rows
        ],
        dtype=np.int64,
    )

    packet_lengths = np.array(
        [
            r["packet_run_length"]
            for r in rows
        ],
        dtype=np.int64,
    )

    noise_run_trials = sum(
        r["noise_run_length"] > 0
        for r in rows
    )

    false_positive_trials = sum(
        r["false_positive"]
        for r in rows
    )

    print(
        f"{snr_db:3.0f} | "
        f"{np.median(noise_lengths):20.1f} | "
        f"{np.max(noise_lengths):17d} | "
        f"{np.median(packet_lengths):22.1f} | "
        f"{np.min(packet_lengths):20d} | "
        f"{noise_run_trials:19d} | "
        f"{false_positive_trials:21d}"
    )


# ============================================================
# Detailed low-SNR results
# ============================================================

print()
print()
print("=" * 120)
print("LOW-SNR RUN-LENGTH DETAILS")
print("=" * 120)


for snr_db in [
    15.0,
    10.0,
    5.0,
    0.0,
]:

    print()
    print(
        f"--- {snr_db:.0f} dB ---"
    )

    rows = [
        r
        for r in all_results
        if r["snr_db"] == snr_db
    ]

    for r in rows:

        print(
            f"trial={r['trial']:02d} "
            f"seed={r['seed']} "
            f"noise_run={r['noise_run_length']} "
            f"noise_start={r['noise_run_start']} "
            f"packet_run={r['packet_run_length']} "
            f"packet_start={r['packet_run_start']} "
            f"n_noise_runs={r['n_noise_runs']} "
            f"first_crossing={r['first_crossing']} "
            f"FP={r['false_positive']}"
        )


# ============================================================
# Candidate persistence requirements
#
# We are NOT changing the detector.
#
# We simply calculate how many trials would survive if a
# detector required N consecutive threshold samples.
# ============================================================

print()
print()
print("=" * 120)
print("PERSISTENCE CANDIDATE ANALYSIS")
print("=" * 120)

candidate_lengths = [
    2,
    4,
    8,
    16,
    32,
]


for required in candidate_lengths:

    print()
    print(
        f"Required consecutive threshold samples = {required}"
    )

    print(
        "SNR | "
        "packet trials surviving | "
        "noise trials surviving"
    )

    print(
        "-" * 65
    )

    for snr_db in SNR_DB_LIST:

        rows = [
            r
            for r in all_results
            if r["snr_db"] == snr_db
        ]

        packet_survive = sum(
            r["packet_run_length"] >= required
            for r in rows
        )

        noise_survive = sum(
            r["noise_run_length"] >= required
            for r in rows
        )

        print(
            f"{snr_db:3.0f} | "
            f"{packet_survive:23d} | "
            f"{noise_survive:21d}"
        )


# ============================================================
# Final
# ============================================================

print()
print("=" * 120)
print("3F-1B COMPLETE")
print("=" * 120)

print(
    "No production files were modified."
)

print(
    "This test only characterizes threshold-run persistence."
)