from pathlib import Path
import sys
import csv

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Current project
# ---------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import config as cfg
import waveform
import scrambler
import demod as demod_mod

from sync import (
    sync_packet,
    apply_cfo_correction,
    sco_correct_symbol,
)

from channel_bridge import ChannelModel


# ---------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------
FIG_DIR = PROJECT_DIR / "evaluation" / "figures"
SIM_DIR = PROJECT_DIR / "evaluation" / "simulations"

FIG_DIR.mkdir(parents=True, exist_ok=True)
SIM_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Experiment configuration
#
# We use the same 100-byte BPSK packet used by the existing end-to-end
# simulation validation.
# ---------------------------------------------------------------------
PAYLOAD_BYTES = 100
MODULATION = "BPSK"

SNR_VALUES_DB = [5, 10, 15, 20, 25, 30, 35]
TRIALS_PER_SNR = 30

CFO_HZ = 0.0
SCO_PPM = 0.0

RNG_SEED = 20261003


# ---------------------------------------------------------------------
# Packet generation
#
# This is the exact packet construction used by the current project:
#
# random payload bits
#     -> waveform.assemble_packet()
#     -> CRC
#     -> scrambling
#     -> rate-1/2 convolutional coding
#     -> OFDM waveform
# ---------------------------------------------------------------------
def make_packet(n_bytes=100, modulation="BPSK", seed=0):

    rng = np.random.default_rng(seed)

    bits = rng.integers(
        0,
        2,
        n_bytes * 8,
        dtype=np.uint8,
    )

    packet = waveform.assemble_packet(
        bits,
        modulation,
        scrambler,
        scrambler,
        scrambler.map_bits_to_symbols,
    )

    return bits, packet


# ---------------------------------------------------------------------
# Channel model
# ---------------------------------------------------------------------
def apply_channel(
    packet,
    noise_voltage,
    seed,
):

    channel = ChannelModel(
        noise_voltage=noise_voltage,
        taps=[1.0 + 0j],
        cfo_hz=CFO_HZ,
        sco_ppm=SCO_PPM,
        seed=seed,
    )

    return channel.apply(
        packet.copy()
    )


# ---------------------------------------------------------------------
# Current-project direct packet decoder
#
# This follows the existing project's validated decode chain:
#
# sync_packet()
#     -> H_hat
#     -> ltf_start
#     -> total CFO
#
# apply_cfo_correction()
#
# DATA FFT extraction
#
# sco_correct_symbol()
#
# demodulate_packet()
#     -> payload bits
#     -> CRC status
# ---------------------------------------------------------------------
def decode_packet(
    rx,
    n_bytes=100,
    modulation="BPSK",
):

    try:

        res = sync_packet(
            rx,
            coarse_cfo_hz=0.0,
            n_data_symbols=0,
        )

        H = res["H_hat"]

        if np.any(np.isnan(H)):
            return None, False

        ltf_start = int(
            res["ltf_start"]
        )

        total_cfo = float(
            res["total_cfo"]
        )

        # Apply the CFO estimated by the current synchronizer.
        rx_corrected = apply_cfo_correction(
            rx,
            total_cfo,
            start_n=0,
        )

        # Number of DATA symbols comes from the same shared formula
        # used by waveform.py.
        n_sym = demod_mod.n_data_syms_for_payload(
            n_bytes,
            modulation,
        )

        data_start = (
            ltf_start
            + cfg.LTF_LEN
            + cfg.SIG_LEN
        )

        data_ffts = []

        sco_accum = 0.0

        for symbol_idx in range(n_sym):

            start = (
                data_start
                + symbol_idx * cfg.SYMBOL_LEN
                + cfg.CP_LEN
            )

            end = (
                start
                + cfg.FFT_SIZE
            )

            if end > len(rx_corrected):
                return None, False

            Y = np.fft.fft(
                rx_corrected[start:end],
                n=cfg.FFT_SIZE,
            ).astype(np.complex64)

            Y, sco_accum, _ = sco_correct_symbol(
                Y,
                H,
                symbol_idx=symbol_idx,
                sco_b_accum=sco_accum,
            )

            data_ffts.append(Y)

        if not data_ffts:
            return None, False

        rx_bits, crc_ok = demod_mod.demodulate_packet(
            data_ffts,
            H,
            modulation=modulation,
            n_payload_bytes=n_bytes,
        )

        return rx_bits, bool(crc_ok)

    except Exception:
        return None, False


# ---------------------------------------------------------------------
# SNR calibration
#
# ChannelModel adds:
#
# noise_voltage / sqrt(2) * (N_I + j N_Q)
#
# Therefore:
#
# E[|noise|^2] = noise_voltage^2
#
# so:
#
# SNR = P_signal / noise_voltage^2
#
# and therefore:
#
# noise_voltage = sqrt(P_signal / 10^(SNR/10))
#
# We measure P_signal from the CURRENT generated packet instead of
# hard-coding the older 0.0447 value.
# ---------------------------------------------------------------------
def noise_voltage_for_snr(
    packet,
    snr_db,
):

    signal_power = float(
        np.mean(
            np.abs(packet) ** 2
        )
    )

    snr_linear = (
        10.0 ** (snr_db / 10.0)
    )

    return float(
        np.sqrt(
            signal_power
            / snr_linear
        )
    )


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------
print()
print("=== AWGN END-TO-END CRC EVIDENCE GENERATION ===")
print(
    f"[INFO] Project directory : {PROJECT_DIR}"
)
print(
    f"[INFO] Payload            : "
    f"{PAYLOAD_BYTES} B"
)
print(
    f"[INFO] Modulation         : "
    f"{MODULATION}"
)
print(
    f"[INFO] Trials/SNR         : "
    f"{TRIALS_PER_SNR}"
)
print(
    f"[INFO] CFO                : "
    f"{CFO_HZ} Hz"
)
print(
    f"[INFO] SCO                : "
    f"{SCO_PPM} ppm"
)
print(
    f"[INFO] RNG seed           : "
    f"{RNG_SEED}"
)
print()


# ---------------------------------------------------------------------
# Measure packet power using the current waveform.
# ---------------------------------------------------------------------
_, reference_packet = make_packet(
    PAYLOAD_BYTES,
    MODULATION,
    seed=0,
)

reference_power = float(
    np.mean(
        np.abs(reference_packet) ** 2
    )
)

print(
    f"[INFO] Packet length      : "
    f"{len(reference_packet)} BB samples"
)

print(
    f"[INFO] Measured signal P  : "
    f"{reference_power:.10f}"
)

print()


results = []


# ---------------------------------------------------------------------
# SNR sweep
# ---------------------------------------------------------------------
for snr_db in SNR_VALUES_DB:

    crc_pass = 0

    noise_voltage = None

    for trial in range(
        TRIALS_PER_SNR
    ):

        # Fresh payload and packet for every trial.
        tx_bits, packet = make_packet(
            PAYLOAD_BYTES,
            MODULATION,
            seed=RNG_SEED
            + trial
            + 1000 * SNR_VALUES_DB.index(snr_db),
        )

        # Calibrate noise from THIS packet's actual power.
        noise_voltage = noise_voltage_for_snr(
            packet,
            snr_db,
        )

        rx = apply_channel(
            packet,
            noise_voltage=noise_voltage,
            seed=RNG_SEED
            + trial
            + 1000 * SNR_VALUES_DB.index(snr_db),
        )

        rx_bits, crc_ok = decode_packet(
            rx,
            n_bytes=PAYLOAD_BYTES,
            modulation=MODULATION,
        )

        if crc_ok:
            crc_pass += 1

    crc_fail = (
        TRIALS_PER_SNR
        - crc_pass
    )

    success_rate = (
        crc_pass
        / TRIALS_PER_SNR
    )

    results.append(
        {
            "snr_db": snr_db,
            "trials": TRIALS_PER_SNR,
            "crc_pass": crc_pass,
            "crc_fail": crc_fail,
            "crc_success_rate": success_rate,
            "noise_voltage": noise_voltage,
        }
    )

    print(
        f"SNR {snr_db:>2} dB | "
        f"CRC {crc_pass:>2}/"
        f"{TRIALS_PER_SNR} | "
        f"success="
        f"{100.0 * success_rate:6.2f}% | "
        f"noise="
        f"{noise_voltage:.8f}"
    )


# ---------------------------------------------------------------------
# Save raw evidence
# ---------------------------------------------------------------------
csv_path = (
    SIM_DIR
    / "awgn_crc_sweep.csv"
)

with csv_path.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    fieldnames = [
        "snr_db",
        "trials",
        "crc_pass",
        "crc_fail",
        "crc_success_rate",
        "noise_voltage",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    writer.writerows(results)


# ---------------------------------------------------------------------
# Generate figure
# ---------------------------------------------------------------------
snr = np.array(
    [
        row["snr_db"]
        for row in results
    ],
    dtype=float,
)

success_rate = np.array(
    [
        100.0
        * row["crc_success_rate"]
        for row in results
    ],
    dtype=float,
)


fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.plot(
    snr,
    success_rate,
    marker="o",
    linewidth=1.8,
)

for x, y in zip(
    snr,
    success_rate,
):

    ax.text(
        x,
        y,
        f"{y:.1f}%",
        ha="center",
        va="bottom",
    )


ax.set_xlabel(
    "SNR (dB)"
)

ax.set_ylabel(
    "CRC success rate (%)"
)

ax.set_title(
    "End-to-End Packet Reliability under AWGN"
)

ax.set_ylim(
    -5,
    105,
)

ax.set_xticks(
    SNR_VALUES_DB
)

ax.grid(
    True,
    alpha=0.25,
)

fig.tight_layout()


fig_path = (
    FIG_DIR
    / "fig_awgn_crc_robustness.png"
)

fig.savefig(
    fig_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


print()
print(
    f"[PASS] Raw results : "
    f"{csv_path}"
)

print(
    f"[PASS] Figure      : "
    f"{fig_path}"
)

print()
print(
    "=== AWGN END-TO-END CRC EVIDENCE COMPLETE ==="
)