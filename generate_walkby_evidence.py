from pathlib import Path
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import waveform
import scrambler
from sync import sync_packet


# ============================================================
# Configuration
# ============================================================

FS_FFT = 20e6

PACKET_RATE = 200.0
N_PACKETS = 1400
DURATION = N_PACKETS / PACKET_RATE

SNR_DB = 27.0

FC_HZ = 2.412e9
C = 299_792_458.0
LAMBDA = C / FC_HZ

OUTPUT_DIR = PROJECT_DIR / "evaluation"
SIM_DIR = OUTPUT_DIR / "simulations"
FIG_DIR = OUTPUT_DIR / "figures"

CSV_PATH = SIM_DIR / "walkby_dynamic.csv"
FIG_PATH = FIG_DIR / "fig_walkby_dynamic.png"


# ============================================================
# Packet generation
# ============================================================

def make_packet():

    rng = np.random.default_rng(20261003)

    bits = rng.integers(
        0,
        2,
        100 * 8,
        dtype=np.uint8,
    )

    packet = waveform.assemble_packet(
        bits,
        "BPSK",
        scrambler,
        scrambler,
        scrambler.map_bits_to_symbols,
    )

    return packet.astype(np.complex64)


# ============================================================
# Controlled dynamic walk-by channel
# ============================================================

def walkby_channel(packet, t):

    # Event centered at 3.5 s.
    center = 3.5
    width = 0.60

    # Smooth reflected-path amplitude.
    alpha = 0.45 * np.exp(
        -0.5 * ((t - center) / width) ** 2
    )

    # Slowly changing reflected-path phase.
    phase = 2.0 * np.pi * 1.5 * t

    # Direct path + dynamic reflected path.
    h = 1.0 + alpha * np.exp(1j * phase)

    return (
        packet * np.complex64(h)
    )


# ============================================================
# AWGN
# ============================================================

def add_awgn(x, snr_db, seed):

    signal_power = np.mean(
        np.abs(x) ** 2
    )

    noise_power = signal_power / (
        10.0 ** (snr_db / 10.0)
    )

    rng = np.random.default_rng(seed)

    noise = (
        rng.standard_normal(len(x))
        + 1j * rng.standard_normal(len(x))
    )

    noise *= np.sqrt(
        noise_power / 2.0
    )

    return (
        x + noise
    ).astype(np.complex64)


# ============================================================
# Main
# ============================================================

def main():

    SIM_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 64)
    print("CONTROLLED WALK-BY EVIDENCE GENERATION")
    print("=" * 64)

    print(f"Packets          : {N_PACKETS}")
    print(f"CSI rate         : {PACKET_RATE:.1f} Hz")
    print(f"Duration         : {DURATION:.3f} s")
    print(f"SNR              : {SNR_DB:.1f} dB")
    print(f"Carrier          : {FC_HZ / 1e9:.3f} GHz")
    print(f"Wavelength       : {LAMBDA:.6f} m")
    print()

    packet = make_packet()

    print(
        f"Packet samples   : {len(packet)}"
    )

    print(
        f"Output CSV       : {CSV_PATH}"
    )

    print(
        f"Output figure    : {FIG_PATH}"
    )

    print()
    print("Starting...")
    print()

    times = []
    mean_h = []
    phase_h = []
    cfo_values = []

    valid_count = 0

    start_time = time.perf_counter()

    for i in range(N_PACKETS):

        t = i / PACKET_RATE

        rx = walkby_channel(
            packet,
            t,
        )

        rx = add_awgn(
            rx,
            SNR_DB,
            seed=100000 + i,
        )

        result = sync_packet(
            rx,
            coarse_cfo_hz=0.0,
            n_data_symbols=0,
        )

        H = result["H_hat"]

        valid = (
            H is not None
            and np.all(
                np.isfinite(H)
            )
        )

        if not valid:

            print(
                f"ERROR: invalid CSI "
                f"at packet {i + 1}",
                flush=True,
            )

            raise RuntimeError(
                f"CSI extraction failed "
                f"at packet {i + 1}"
            )

        valid_count += 1

        h_mag = float(
            np.mean(
                np.abs(H)
            )
        )

        h_phase = float(
            np.angle(
                np.mean(H)
            )
        )

        cfo = float(
            result["total_cfo"]
        )

        times.append(t)
        mean_h.append(h_mag)
        phase_h.append(h_phase)
        cfo_values.append(cfo)

        if (i + 1) % 100 == 0:

            elapsed = (
                time.perf_counter()
                - start_time
            )

            print(
                f"Packet "
                f"{i + 1:04d}/{N_PACKETS} | "
                f"t={t:5.2f}s | "
                f"|H|={h_mag:.4f} | "
                f"phase={h_phase:+.4f} rad | "
                f"CFO={cfo:+.1f} Hz | "
                f"elapsed={elapsed:.2f}s",
                flush=True,
            )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    times = np.asarray(times)
    mean_h = np.asarray(mean_h)
    phase_h = np.unwrap(
        np.asarray(phase_h)
    )
    cfo_values = np.asarray(
        cfo_values
    )

    # ========================================================
    # Save CSV
    # ========================================================

    data = np.column_stack(
        (
            times,
            mean_h,
            phase_h,
            cfo_values,
        )
    )

    header = (
        "time_s,"
        "mean_csi_magnitude,"
        "unwrapped_csi_phase_rad,"
        "estimated_cfo_hz"
    )

    np.savetxt(
        CSV_PATH,
        data,
        delimiter=",",
        header=header,
        comments="",
    )

    # ========================================================
    # Summary
    # ========================================================

    event_mask = (
        (times >= 2.0)
        & (times <= 5.0)
    )

    baseline_mask = (
        (times >= 0.0)
        & (times <= 1.5)
    )

    baseline_mag = mean_h[
        baseline_mask
    ]

    event_mag = mean_h[
        event_mask
    ]

    print()
    print("=" * 64)
    print("WALK-BY RESULTS")
    print("=" * 64)

    print(
        f"Valid CSI       : "
        f"{valid_count}/{N_PACKETS}"
    )

    print(
        f"Total runtime   : "
        f"{elapsed:.3f} s"
    )

    print(
        f"Runtime/packet  : "
        f"{elapsed / N_PACKETS:.6f} s"
    )

    print(
        f"Mean |H|        : "
        f"{np.mean(mean_h):.4f}"
    )

    print(
        f"Std |H|         : "
        f"{np.std(mean_h):.4f}"
    )

    print(
        f"Baseline |H|    : "
        f"{np.mean(baseline_mag):.4f}"
    )

    print(
        f"Event |H|       : "
        f"{np.mean(event_mag):.4f}"
    )

    print(
        f"Phase range     : "
        f"{np.min(phase_h):+.4f} to "
        f"{np.max(phase_h):+.4f} rad"
    )

    print(
        f"Mean |CFO|      : "
        f"{np.mean(np.abs(cfo_values)):.2f} Hz"
    )

    print()
    print("CSV saved.")
    print()

    # ========================================================
    # Figure
    # ========================================================

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10, 9),
        sharex=True,
    )

    axes[0].plot(
        times,
        mean_h,
        linewidth=1.2,
    )

    axes[0].axvspan(
        2.0,
        5.0,
        alpha=0.15,
    )

    axes[0].set_ylabel(
        "Mean |H|"
    )

    axes[0].set_title(
        "Controlled Dynamic Walk-by "
        "CSI Simulation"
    )

    axes[0].grid(
        True,
        alpha=0.3,
    )

    axes[1].plot(
        times,
        phase_h,
        linewidth=1.2,
    )

    axes[1].axvspan(
        2.0,
        5.0,
        alpha=0.15,
    )

    axes[1].set_ylabel(
        "Unwrapped phase (rad)"
    )

    axes[1].grid(
        True,
        alpha=0.3,
    )

    axes[2].plot(
        times,
        cfo_values,
        linewidth=1.0,
    )

    axes[2].axhline(
        0.0,
        linewidth=0.8,
    )

    axes[2].set_ylabel(
        "Estimated CFO (Hz)"
    )

    axes[2].set_xlabel(
        "Time (s)"
    )

    axes[2].grid(
        True,
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        FIG_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Figure saved to:\n"
        f"{FIG_PATH}"
    )

    print()
    print("=" * 64)

    if valid_count == N_PACKETS:
        print(
            "PASS: complete walk-by "
            "CSI evidence generated."
        )
    else:
        print(
            "FAIL: incomplete CSI recovery."
        )

    print("=" * 64)


if __name__ == "__main__":
    main()