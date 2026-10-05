"""
generate_micro_gesture_figure.py

Controlled transient micro-gesture CSI simulation.

This is an evidence/simulation script only.
It does NOT modify the production PHY.

Scenario:
    A 2.5 cm transient displacement lasting 250 ms.

The prescribed displacement is converted into propagation phase using:

    phi(t) = 4*pi*d(t)/lambda

The current production synchronization + CSI extraction pipeline is then
used to recover the resulting temporal CSI phase.
"""

from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram

# ---------------------------------------------------------------------
# Project path
# ---------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import config as cfg
import waveform
import scrambler
from sync import sync_packet


# =====================================================================
# PARAMETERS
# =====================================================================

N_PACKETS = 1000

PACKET_INTERVAL_S = 0.005
CSI_RATE_HZ = 1.0 / PACKET_INTERVAL_S

CARRIER_HZ = 2.412e9
C_LIGHT = 299_792_458.0

GESTURE_START_S = 1.50
GESTURE_DURATION_S = 0.250

DISPLACEMENT_M = 0.025

SNR_DB = 27.0

PAYLOAD_BYTES = 100
MODULATION = "BPSK"

BASE_SEED = 20261003


# =====================================================================
# DERIVED VALUES
# =====================================================================

WAVELENGTH_M = C_LIGHT / CARRIER_HZ

EXPECTED_PHASE_AMP = (
    4.0
    * np.pi
    * DISPLACEMENT_M
    / WAVELENGTH_M
)

NYQUIST_HZ = CSI_RATE_HZ / 2.0


# =====================================================================
# PACKET
# =====================================================================

def make_reference_packet():

    rng = np.random.default_rng(BASE_SEED)

    bits = rng.integers(
        0,
        2,
        PAYLOAD_BYTES * 8,
        dtype=np.uint8,
    )

    packet = waveform.assemble_packet(
        bits,
        MODULATION,
        scrambler,
        scrambler,
        scrambler.map_bits_to_symbols,
    )

    return packet.astype(np.complex64)


# =====================================================================
# MAIN
# =====================================================================

def main():

    print("=" * 70)
    print("CONTROLLED TRANSIENT MICRO-GESTURE CSI SIMULATION")
    print("=" * 70)

    print(
        f"[INFO] Packets              : {N_PACKETS}"
    )

    print(
        f"[INFO] Packet interval      : "
        f"{PACKET_INTERVAL_S * 1000:.3f} ms"
    )

    print(
        f"[INFO] CSI sampling rate    : "
        f"{CSI_RATE_HZ:.1f} Hz"
    )

    print(
        f"[INFO] Nyquist frequency    : "
        f"{NYQUIST_HZ:.1f} Hz"
    )

    print(
        f"[INFO] Gesture start        : "
        f"{GESTURE_START_S:.3f} s"
    )

    print(
        f"[INFO] Gesture duration     : "
        f"{GESTURE_DURATION_S * 1000:.1f} ms"
    )

    print(
        f"[INFO] Displacement         : "
        f"{DISPLACEMENT_M * 100:.2f} cm"
    )

    print(
        f"[INFO] Carrier frequency    : "
        f"{CARRIER_HZ / 1e9:.6f} GHz"
    )

    print(
        f"[INFO] Wavelength           : "
        f"{WAVELENGTH_M:.8f} m"
    )

    print(
        f"[INFO] Expected phase amp   : "
        f"{EXPECTED_PHASE_AMP:.6f} rad"
    )

    print(
        f"[INFO] SNR                  : "
        f"{SNR_DB:.1f} dB"
    )

    print("[INFO] CFO                  : 0 Hz")
    print("[INFO] SCO                  : 0 ppm")
    print("[INFO] Timing jitter        : 0 samples")

    # -------------------------------------------------------------
    # Reference packet
    # -------------------------------------------------------------

    packet = make_reference_packet()

    signal_power = float(
        np.mean(np.abs(packet) ** 2)
    )

    noise_voltage = np.sqrt(
        signal_power
        / (10.0 ** (SNR_DB / 10.0))
    )

    print()
    print(
        f"[INFO] Packet length       : "
        f"{len(packet)} BB samples"
    )

    print(
        f"[INFO] Packet duration     : "
        f"{len(packet) / cfg.FS_FFT * 1e6:.3f} us"
    )

    print(
        f"[INFO] Signal power        : "
        f"{signal_power:.10f}"
    )

    print(
        f"[INFO] Noise voltage       : "
        f"{noise_voltage:.10f}"
    )

    # -------------------------------------------------------------
    # Time axis
    # -------------------------------------------------------------

    t = (
        np.arange(N_PACKETS, dtype=np.float64)
        * PACKET_INTERVAL_S
    )

    # -------------------------------------------------------------
    # Transient gesture model
    #
    # Smooth half-sine displacement:
    #
    # d(t) = A sin(pi*t/T)
    #
    # for 0 <= t <= T.
    # -------------------------------------------------------------

    displacement = np.zeros(
        N_PACKETS,
        dtype=np.float64,
    )

    inside = (
        (t >= GESTURE_START_S)
        & (
            t
            <= GESTURE_START_S
            + GESTURE_DURATION_S
        )
    )

    t_local = (
        t[inside]
        - GESTURE_START_S
    )

    displacement[inside] = (
        DISPLACEMENT_M
        * np.sin(
            np.pi
            * t_local
            / GESTURE_DURATION_S
        )
    )

    expected_phase = (
        4.0
        * np.pi
        * displacement
        / WAVELENGTH_M
    )

    # -------------------------------------------------------------
    # Recover CSI
    # -------------------------------------------------------------

    rng = np.random.default_rng(
        BASE_SEED + 1
    )

    recovered_phase = []
    recovered_magnitude = []
    recovered_cfo = []
    recovered_ltf_start = []

    for i in range(N_PACKETS):

        phase_rotation = np.exp(
            1j * expected_phase[i]
        )

        rx = (
            packet
            * phase_rotation
        )

        noise = (
            rng.standard_normal(len(rx))
            + 1j * rng.standard_normal(len(rx))
        )

        noise *= (
            noise_voltage
            / np.sqrt(2.0)
        )

        rx = (
            rx
            + noise.astype(np.complex64)
        ).astype(np.complex64)

        try:

            result = sync_packet(
                rx,
                coarse_cfo_hz=0.0,
                n_data_symbols=0,
            )

            H_hat = np.asarray(
                result["H_hat"],
                dtype=np.complex64,
            )

            if (
                H_hat.shape
                != (cfg.NUM_ACTIVE,)
            ):
                raise RuntimeError(
                    f"Unexpected H_hat shape "
                    f"{H_hat.shape}"
                )

            if np.any(np.isnan(H_hat)):
                raise RuntimeError(
                    "NaN in CSI"
                )

            H_mean = np.mean(H_hat)

            recovered_phase.append(
                float(np.angle(H_mean))
            )

            recovered_magnitude.append(
                float(np.abs(H_mean))
            )

            recovered_cfo.append(
                float(result["total_cfo"])
            )

            recovered_ltf_start.append(
                int(result["ltf_start"])
            )

        except Exception:

            recovered_phase.append(np.nan)
            recovered_magnitude.append(np.nan)
            recovered_cfo.append(np.nan)
            recovered_ltf_start.append(-1)

    recovered_phase = np.asarray(
        recovered_phase,
        dtype=np.float64,
    )

    recovered_magnitude = np.asarray(
        recovered_magnitude,
        dtype=np.float64,
    )

    recovered_cfo = np.asarray(
        recovered_cfo,
        dtype=np.float64,
    )

    recovered_ltf_start = np.asarray(
        recovered_ltf_start,
        dtype=np.int32,
    )

    valid = np.isfinite(
        recovered_phase
    )

    n_valid = int(
        np.sum(valid)
    )

    print()
    print("=" * 70)
    print("MICRO-GESTURE RESULTS")
    print("=" * 70)

    print(
        f"[INFO] CSI samples recovered : "
        f"{n_valid}/{N_PACKETS}"
    )

    if n_valid < N_PACKETS:

        print(
            f"[WARN] CSI failures          : "
            f"{N_PACKETS - n_valid}"
        )

    if n_valid < 10:
        raise RuntimeError(
            "Too few valid CSI samples."
        )

    t_valid = t[valid]
    expected_valid = expected_phase[valid]
    measured_raw = recovered_phase[valid]

    # -------------------------------------------------------------
    # Remove only a constant phase offset.
    #
    # Do NOT apply frequency/subcarrier phase sanitization because
    # temporal common phase is the sensing observable.
    # -------------------------------------------------------------

    phase_offset = np.angle(
        np.mean(
            np.exp(
                1j
                * (
                    measured_raw
                    - expected_valid
                )
            )
        )
    )

    measured_aligned = np.unwrap(
        measured_raw
        - phase_offset
    )

    expected_unwrapped = np.unwrap(
        expected_valid
    )

    # -------------------------------------------------------------
    # Gesture amplitude recovery.
    #
    # Since the gesture is localized in time, compare the peak
    # recovered phase excursion against the prescribed value.
    # -------------------------------------------------------------

    baseline_mask = (
        t_valid < GESTURE_START_S
    )

    event_mask = (
        (
            t_valid
            >= GESTURE_START_S
        )
        &
        (
            t_valid
            <= GESTURE_START_S
            + GESTURE_DURATION_S
        )
    )

    baseline_phase = float(
        np.mean(
            measured_aligned[
                baseline_mask
            ]
        )
    )

    event_phase = (
        measured_aligned[event_mask]
        - baseline_phase
    )

    expected_event = (
        expected_unwrapped[event_mask]
        - np.mean(
            expected_unwrapped[
                baseline_mask
            ]
        )
    )

    recovered_peak_phase = float(
        np.max(
            np.abs(event_phase)
        )
    )

    expected_peak_phase = float(
        np.max(
            np.abs(expected_event)
        )
    )

    # -------------------------------------------------------------
    # Correlation during the event.
    # -------------------------------------------------------------

    if len(event_phase) > 2:

        corr = float(
            np.corrcoef(
                expected_event,
                event_phase,
            )[0, 1]
        )

    else:

        corr = float("nan")

    # -------------------------------------------------------------
    # Magnitude and CFO statistics.
    # -------------------------------------------------------------

    mean_magnitude = float(
        np.nanmean(
            recovered_magnitude
        )
    )

    std_magnitude = float(
        np.nanstd(
            recovered_magnitude
        )
    )

    mean_abs_cfo = float(
        np.nanmean(
            np.abs(recovered_cfo)
        )
    )

    # -------------------------------------------------------------
    # Print results.
    # -------------------------------------------------------------

    print(
        f"[INFO] Gesture interval      : "
        f"{GESTURE_START_S:.3f}–"
        f"{GESTURE_START_S + GESTURE_DURATION_S:.3f} s"
    )

    print(
        f"[INFO] Expected phase peak   : "
        f"{expected_peak_phase:.4f} rad"
    )

    print(
        f"[INFO] Recovered phase peak  : "
        f"{recovered_peak_phase:.4f} rad"
    )

    print(
        f"[INFO] Peak phase error      : "
        f"{abs(recovered_peak_phase - expected_peak_phase):.4f} rad"
    )

    print(
        f"[INFO] Event phase corr.     : "
        f"{corr:.4f}"
    )

    print(
        f"[INFO] Mean |H|              : "
        f"{mean_magnitude:.4f}"
    )

    print(
        f"[INFO] Std |H|               : "
        f"{std_magnitude:.4f}"
    )

    print(
        f"[INFO] Mean |estimated CFO|  : "
        f"{mean_abs_cfo:.4f} Hz"
    )

    # -------------------------------------------------------------
    # Output directories.
    # -------------------------------------------------------------

    simulation_dir = (
        PROJECT_DIR
        / "evaluation"
        / "simulations"
    )

    figure_dir = (
        PROJECT_DIR
        / "evaluation"
        / "figures"
    )

    simulation_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # Save CSV.
    # -------------------------------------------------------------

    csv_path = (
        simulation_dir
        / "micro_gesture.csv"
    )

    header = (
        "time_s,"
        "displacement_m,"
        "expected_phase_rad,"
        "recovered_phase_rad,"
        "csi_magnitude,"
        "estimated_cfo_hz,"
        "ltf_start"
    )

    data = np.column_stack(
        [
            t,
            displacement,
            expected_phase,
            recovered_phase,
            recovered_magnitude,
            recovered_cfo,
            recovered_ltf_start,
        ]
    )

    np.savetxt(
        csv_path,
        data,
        delimiter=",",
        header=header,
        comments="",
    )

    # -------------------------------------------------------------
    # Figure
    # -------------------------------------------------------------

    fig = plt.figure(
        figsize=(12, 10)
    )

    # -------------------------------------------------------------
    # Panel 1: displacement
    # -------------------------------------------------------------

    ax1 = fig.add_subplot(3, 1, 1)

    ax1.plot(
        t,
        displacement * 100.0,
        label="Prescribed displacement",
    )

    ax1.axvspan(
        GESTURE_START_S,
        GESTURE_START_S
        + GESTURE_DURATION_S,
        alpha=0.15,
        label="Gesture interval",
    )

    ax1.set_ylabel(
        "Displacement (cm)"
    )

    ax1.set_title(
        "Controlled 2.5 cm Transient Micro-Gesture"
    )

    ax1.grid(True)
    ax1.legend()

    # -------------------------------------------------------------
    # Panel 2: phase
    # -------------------------------------------------------------

    ax2 = fig.add_subplot(3, 1, 2)

    ax2.plot(
        t_valid,
        expected_unwrapped,
        label="Expected phase",
        linewidth=2,
    )

    ax2.plot(
        t_valid,
        measured_aligned,
        label="Recovered CSI phase",
        alpha=0.8,
    )

    ax2.axvspan(
        GESTURE_START_S,
        GESTURE_START_S
        + GESTURE_DURATION_S,
        alpha=0.15,
    )

    ax2.set_ylabel(
        "Phase (rad)"
    )

    ax2.set_title(
        "Transient Propagation-Phase Recovery"
    )

    ax2.grid(True)
    ax2.legend()

    # -------------------------------------------------------------
    # Panel 3: phase spectrum
    # -------------------------------------------------------------

    centered_phase = (
        measured_aligned
        - np.mean(measured_aligned)
    )

    freqs, psd = periodogram(
        centered_phase,
        fs=CSI_RATE_HZ,
        detrend="linear",
    )

    positive = freqs > 0

    ax3 = fig.add_subplot(3, 1, 3)

    ax3.semilogy(
        freqs[positive],
        psd[positive],
    )

    ax3.set_xlim(
        0,
        20,
    )

    ax3.set_xlabel(
        "Frequency (Hz)"
    )

    ax3.set_ylabel(
        "PSD"
    )

    ax3.set_title(
        "Recovered CSI Phase Spectrum"
    )

    ax3.grid(True)

    fig.tight_layout()

    figure_path = (
        figure_dir
        / "fig_micro_gesture.png"
    )

    fig.savefig(
        figure_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    # -------------------------------------------------------------
    # Evidence threshold.
    # -------------------------------------------------------------

    peak_error = abs(
        recovered_peak_phase
        - expected_peak_phase
    )

    evidence_ok = (
        n_valid == N_PACKETS
        and peak_error < 0.30
        and corr > 0.90
    )

    print()

    if evidence_ok:

        print(
            "[PASS] Controlled transient "
            "motion-to-CSI relationship recovered."
        )

    else:

        print(
            "[WARN] Controlled transient "
            "sensing result did not meet "
            "the evidence thresholds."
        )

    print(
        f"[PASS] Raw data : {csv_path}"
    )

    print(
        f"[PASS] Figure   : {figure_path}"
    )

    print()
    print(
        "CONTROLLED MICRO-GESTURE SIMULATION COMPLETE"
    )


if __name__ == "__main__":
    main()