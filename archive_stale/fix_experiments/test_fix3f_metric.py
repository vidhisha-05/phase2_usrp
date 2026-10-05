import numpy as np

import config as cfg
import waveform
import scrambler


# ============================================================
# 3F-1A
# Correlation metric characterization
#
# This test reproduces the CURRENT detector.py implementation
# exactly:
#
#   L = 16
#   N = len(buf) - L
#
#   A = sliding windows over buf
#   B = sliding windows over buf[L:]
#
#   P = sum(A * conj(B))
#   R = sum(|B|^2)
#
#   metric = |P|^2 / (R^2 + 1e-12)
#
# No production files are modified.
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


# ============================================================
# Generate the same reference packet used by the previous
# 3F tests.
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
    "3F-1A CORRELATION METRIC CHARACTERIZATION"
)

print(
    "=========================================="
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
# Packet power
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

if not np.isfinite(signal_power):
    raise AssertionError(
        "Packet power is not finite."
    )

if signal_power <= 0:
    raise AssertionError(
        "Packet power must be positive."
    )


# ============================================================
# EXACT reconstruction of detector._sliding_corr()
#
# Current detector.py:
#
#     L = CORR_WINDOW
#     N = len(buf) - L
#
#     s = buf.strides[0]
#
#     A = as_strided(
#         buf,
#         shape=(N, L),
#         strides=(s, s)
#     )
#
#     B = as_strided(
#         buf[L:],
#         shape=(N, L),
#         strides=(s, s)
#     )
#
#     P = (A * conj(B)).sum(axis=1)
#
#     R = (abs(B)^2).sum(axis=1)
#
#     metric = |P|^2 / (R^2 + 1e-12)
#
# ============================================================

def calculate_metric(samples):

    samples = np.asarray(
        samples,
        dtype=np.complex64,
    )

    L = 16

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
        A
        * np.conj(B)
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
# Run characterization
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
                "Metric calculation returned no samples."
            )

        # ----------------------------------------------------
        # Noise-only region
        #
        # Leave 32 samples of margin before the known packet.
        # ----------------------------------------------------

        noise_region_end = (
            true_start - 32
        )

        noise_metric = metric[
            :noise_region_end
        ]

        if len(noise_metric) == 0:
            raise AssertionError(
                "Noise-only region is empty."
            )

        noise_max = float(
            np.max(
                noise_metric
            )
        )

        noise_argmax = int(
            np.argmax(
                noise_metric
            )
        )

        # ----------------------------------------------------
        # Packet neighborhood
        # ----------------------------------------------------

        packet_lo = max(
            0,
            true_start - 64
        )

        packet_hi = min(
            len(metric),
            true_start + 128
        )

        packet_metric = metric[
            packet_lo:packet_hi
        ]

        if len(packet_metric) == 0:
            raise AssertionError(
                "Packet metric region is empty."
            )

        packet_max = float(
            np.max(
                packet_metric
            )
        )

        packet_argmax = (
            packet_lo
            +
            int(
                np.argmax(
                    packet_metric
                )
            )
        )

        # ----------------------------------------------------
        # First threshold crossing
        # ----------------------------------------------------

        above = np.where(
            metric >= DETECT_THRESH
        )[0]

        if len(above) > 0:

            first_crossing = int(
                above[0]
            )

        else:

            first_crossing = None

        # ----------------------------------------------------
        # False positive
        # ----------------------------------------------------

        false_positive = (
            first_crossing is not None
            and
            first_crossing
            < true_start - 32
        )

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        all_results.append(
            {
                "snr_db": snr_db,
                "trial": trial,
                "seed": seed,
                "noise_max": noise_max,
                "noise_argmax": noise_argmax,
                "packet_max": packet_max,
                "packet_argmax": packet_argmax,
                "first_crossing": first_crossing,
                "false_positive": false_positive,
            }
        )


# ============================================================
# Summary
# ============================================================

print()
print()

print("=" * 110)
print("3F-1A SUMMARY")
print("=" * 110)

print(
    "SNR | "
    "noise max median | "
    "noise max max | "
    "packet max median | "
    "packet max min | "
    "false-positive trials | "
    "threshold-crossing trials"
)

print(
    "-" * 110
)


for snr_db in SNR_DB_LIST:

    rows = [
        r
        for r in all_results
        if r["snr_db"] == snr_db
    ]

    noise_values = np.array(
        [
            r["noise_max"]
            for r in rows
        ]
    )

    packet_values = np.array(
        [
            r["packet_max"]
            for r in rows
        ]
    )

    false_count = sum(
        r["false_positive"]
        for r in rows
    )

    crossing_count = sum(
        r["first_crossing"] is not None
        for r in rows
    )

    print(
        f"{snr_db:3.0f} | "
        f"{np.median(noise_values):16.6f} | "
        f"{np.max(noise_values):13.6f} | "
        f"{np.median(packet_values):17.6f} | "
        f"{np.min(packet_values):15.6f} | "
        f"{false_count:21d} | "
        f"{crossing_count:24d}"
    )


# ============================================================
# Detailed low-SNR trials
# ============================================================

print()
print()

print("=" * 110)
print("LOW-SNR TRIAL DETAILS")
print("=" * 110)


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
            f"noise_max={r['noise_max']:.6f} "
            f"packet_max={r['packet_max']:.6f} "
            f"packet_argmax={r['packet_argmax']} "
            f"first_crossing={r['first_crossing']} "
            f"noise_FP={r['false_positive']}"
        )


# ============================================================
# Threshold separation
# ============================================================

print()
print()

print("=" * 110)
print("THRESHOLD SEPARATION CHECK")
print("=" * 110)


for snr_db in SNR_DB_LIST:

    rows = [
        r
        for r in all_results
        if r["snr_db"] == snr_db
    ]

    maximum_noise = max(
        r["noise_max"]
        for r in rows
    )

    minimum_packet = min(
        r["packet_max"]
        for r in rows
    )

    print()
    print(
        f"SNR {snr_db:.0f} dB:"
    )

    print(
        "  Maximum noise-only metric:",
        maximum_noise
    )

    print(
        "  Minimum packet metric:",
        minimum_packet
    )

    print(
        "  Current threshold:",
        DETECT_THRESH
    )

    print(
        "  Noise-only maximum below threshold:",
        maximum_noise < DETECT_THRESH
    )

    print(
        "  Minimum packet metric above threshold:",
        minimum_packet >= DETECT_THRESH
    )

    if (
        maximum_noise < DETECT_THRESH
        and
        minimum_packet >= DETECT_THRESH
    ):

        print(
            "  --> Threshold 0.65 separates "
            "these measured trials."
        )

    else:

        print(
            "  --> Threshold 0.65 does NOT "
            "cleanly separate these measured trials."
        )


# ============================================================
# Final
# ============================================================

print()
print(
    "=" * 110
)

print(
    "3F-1A COMPLETE"
)

print(
    "=" * 110
)

print(
    "No production files were modified."
)

print(
    "This characterization reproduces "
    "the current detector.py metric."
)