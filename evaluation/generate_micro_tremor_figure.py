"""
generate_micro_tremor_figure.py

Controlled micro-tremor CSI simulation.

Purpose:
    Demonstrate that the current CSI extraction pipeline can recover a
    prescribed slow propagation-phase modulation corresponding to a
    5 mm, 5.5 Hz micro-motion.

This is an evidence/simulation script only.
It does NOT modify the production PHY.

Model:
    d(t)   = A * sin(2*pi*f*t)
    phi(t) = 4*pi*d(t)/lambda

The entire OFDM packet is given a constant phase rotation corresponding
to the target position during that packet. This is valid because the
5.5 Hz motion changes negligibly during the ~166.4 us baseband packet.

No CFO, SCO, timing jitter, or multipath is introduced in this controlled
experiment.
"""

from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram
from scipy.optimize import minimize_scalar

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
# EXPERIMENT PARAMETERS
# =====================================================================

N_PACKETS = 1000

PACKET_INTERVAL_S = 0.005
CSI_RATE_HZ = 1.0 / PACKET_INTERVAL_S

TREMOR_HZ = 5.5
DISPLACEMENT_M = 0.005

CARRIER_HZ = 2.412e9
C_LIGHT = 299_792_458.0

SNR_DB = 27.0

PAYLOAD_BYTES = 100
MODULATION = "BPSK"

BASE_SEED = 20261003


# =====================================================================
# DERIVED PARAMETERS
# =====================================================================

WAVELENGTH_M = C_LIGHT / CARRIER_HZ

EXPECTED_PHASE_AMP = (
    4.0 * np.pi * DISPLACEMENT_M / WAVELENGTH_M
)

NYQUIST_HZ = CSI_RATE_HZ / 2.0


# =====================================================================
# PACKET GENERATION
# =====================================================================

def make_reference_packet():
    """
    Generate one deterministic 100-byte BPSK packet using the current
    production waveform assembly function.
    """

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
# SINUSOIDAL FREQUENCY FIT
# =====================================================================

def sinusoid_amplitude_at_frequency(t, x, frequency_hz):
    """
    Least-squares fit:

        x(t) = c + a*sin(2*pi*f*t) + b*cos(2*pi*f*t)

    Returns fitted sinusoid amplitude and coefficients.
    """

    omega = 2.0 * np.pi * frequency_hz

    A = np.column_stack(
        [
            np.ones_like(t),
            np.sin(omega * t),
            np.cos(omega * t),
        ]
    )

    coef, _, _, _ = np.linalg.lstsq(
        A,
        x,
        rcond=None,
    )

    offset = float(coef[0])
    a = float(coef[1])
    b = float(coef[2])

    amplitude = float(np.sqrt(a * a + b * b))

    fitted = A @ coef

    return amplitude, offset, fitted


def estimate_frequency(t, x):
    """
    Estimate the dominant frequency by minimizing least-squares
    sinusoidal residual over a narrow search interval around the
    expected 5.5 Hz component.
    """

    def objective(frequency_hz):

        omega = 2.0 * np.pi * frequency_hz

        A = np.column_stack(
            [
                np.ones_like(t),
                np.sin(omega * t),
                np.cos(omega * t),
            ]
        )

        coef, _, _, _ = np.linalg.lstsq(
            A,
            x,
            rcond=None,
        )

        residual = x - A @ coef

        return float(np.mean(residual ** 2))

    result = minimize_scalar(
        objective,
        bounds=(4.0, 7.0),
        method="bounded",
        options={"xatol": 1e-6},
    )

    return float(result.x)


# =====================================================================
# MAIN EXPERIMENT
# =====================================================================

def main():

    print("=" * 70)
    print("CONTROLLED MICRO-TREMOR CSI SIMULATION")
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
        f"[INFO] Tremor frequency     : "
        f"{TREMOR_HZ:.2f} Hz"
    )

    print(
        f"[INFO] Displacement         : "
        f"{DISPLACEMENT_M * 1000:.2f} mm"
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

    print(
        f"[INFO] CFO                  : 0 Hz"
    )

    print(
        f"[INFO] SCO                  : 0 ppm"
    )

    print(
        f"[INFO] Timing jitter        : 0 samples"
    )

    # -------------------------------------------------------------
    # Build one deterministic packet.
    # -------------------------------------------------------------

    packet = make_reference_packet()

    signal_power = float(
        np.mean(np.abs(packet) ** 2)
    )

    noise_voltage = np.sqrt(
        signal_power / (10.0 ** (SNR_DB / 10.0))
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
    # Time axis.
    # -------------------------------------------------------------

    t = (
        np.arange(N_PACKETS, dtype=np.float64)
        * PACKET_INTERVAL_S
    )

    # Prescribed physical displacement.
    displacement = (
        DISPLACEMENT_M
        * np.sin(2.0 * np.pi * TREMOR_HZ * t)
    )

    # Prescribed propagation phase.
    expected_phase = (
        4.0
        * np.pi
        * displacement
        / WAVELENGTH_M
    )

    # -------------------------------------------------------------
    # Recover CSI packet-by-packet.
    # -------------------------------------------------------------

    rng = np.random.default_rng(BASE_SEED + 1)

    recovered_phase = []
    recovered_magnitude = []
    recovered_cfo = []
    recovered_ltf_start = []

    failures = 0

    for i in range(N_PACKETS):

        # ---------------------------------------------------------
        # Controlled motion channel:
        #
        # H_motion(t) = exp(j * phi(t))
        #
        # The packet is rotated by the prescribed propagation phase.
        # ---------------------------------------------------------

        phase_rotation = np.exp(
            1j * expected_phase[i]
        )

        rx = packet * phase_rotation

        # ---------------------------------------------------------
        # Add complex AWGN.
        # ---------------------------------------------------------

        noise = (
            rng.standard_normal(len(rx))
            + 1j * rng.standard_normal(len(rx))
        )

        noise = (
            noise_voltage
            / np.sqrt(2.0)
            * noise
        )

        rx = (
            rx
            + noise.astype(np.complex64)
        ).astype(np.complex64)

        # ---------------------------------------------------------
        # Current production synchronization + CSI extraction.
        # ---------------------------------------------------------

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
                    f"Unexpected H_hat shape: "
                    f"{H_hat.shape}"
                )

            if np.any(np.isnan(H_hat)):
                raise RuntimeError(
                    "H_hat contains NaN"
                )

            # Average the active-subcarrier CSI.
            #
            # The controlled channel is a single zero-delay path,
            # therefore its propagation phase is common across the
            # active subcarriers.
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
            failures += 1

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

    valid = np.isfinite(recovered_phase)

    n_valid = int(np.sum(valid))

    print()
    print("=" * 70)
    print("MICRO-TREMOR RESULTS")
    print("=" * 70)

    print(
        f"[INFO] CSI samples recovered : "
        f"{n_valid}/{N_PACKETS}"
    )

    if n_valid < N_PACKETS:
        print(
            f"[WARN] CSI failures          : "
            f"{failures}"
        )

    if n_valid < 10:
        raise RuntimeError(
            "Too few valid CSI samples for evidence generation."
        )

    t_valid = t[valid]
    expected_valid = expected_phase[valid]
    measured_raw = recovered_phase[valid]

    # -------------------------------------------------------------
    # Remove only a constant phase offset.
    #
    # IMPORTANT:
    # We do NOT perform linear phase sanitization here because the
    # temporal common phase is the sensing quantity we are trying
    # to demonstrate.
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
        measured_raw - phase_offset
    )

    expected_unwrapped = np.unwrap(
        expected_valid
    )

    # -------------------------------------------------------------
    # Fit recovered phase amplitude at the known frequency.
    # -------------------------------------------------------------

    recovered_amp, recovered_offset, fitted_phase = (
        sinusoid_amplitude_at_frequency(
            t_valid,
            measured_aligned,
            TREMOR_HZ,
        )
    )

    # -------------------------------------------------------------
    # Correlation.
    # -------------------------------------------------------------

    expected_centered = (
        expected_unwrapped
        - np.mean(expected_unwrapped)
    )

    measured_centered = (
        measured_aligned
        - np.mean(measured_aligned)
    )

    correlation = float(
        np.corrcoef(
            expected_centered,
            measured_centered,
        )[0, 1]
    )

    # -------------------------------------------------------------
    # Frequency estimate.
    # -------------------------------------------------------------

    detected_frequency = estimate_frequency(
        t_valid,
        measured_aligned,
    )

    # -------------------------------------------------------------
    # PSD.
    # -------------------------------------------------------------

    freqs, psd = periodogram(
        measured_centered,
        fs=CSI_RATE_HZ,
        detrend="linear",
    )

    positive = freqs > 0

    if np.any(positive):
        peak_idx = np.argmax(
            psd[positive]
        )

        positive_freqs = freqs[positive]

        psd_peak_frequency = float(
            positive_freqs[peak_idx]
        )
    else:
        psd_peak_frequency = float("nan")

    # -------------------------------------------------------------
    # Magnitude statistics.
    # -------------------------------------------------------------

    mean_magnitude = float(
        np.nanmean(recovered_magnitude)
    )

    std_magnitude = float(
        np.nanstd(recovered_magnitude)
    )

    mean_abs_cfo = float(
        np.nanmean(np.abs(recovered_cfo))
    )

    # -------------------------------------------------------------
    # Print results.
    # -------------------------------------------------------------

    print(
        f"[INFO] Expected frequency     : "
        f"{TREMOR_HZ:.3f} Hz"
    )

    print(
        f"[INFO] Fitted frequency       : "
        f"{detected_frequency:.3f} Hz"
    )

    print(
        f"[INFO] FFT/PSD peak           : "
        f"{psd_peak_frequency:.3f} Hz"
    )

    print(
        f"[INFO] Expected phase amp     : "
        f"{EXPECTED_PHASE_AMP:.4f} rad"
    )

    print(
        f"[INFO] Recovered phase amp    : "
        f"{recovered_amp:.4f} rad"
    )

    print(
        f"[INFO] Phase amplitude error  : "
        f"{abs(recovered_amp - EXPECTED_PHASE_AMP):.4f} rad"
    )

    print(
        f"[INFO] Motion/phase corr.     : "
        f"{correlation:.4f}"
    )

    print(
        f"[INFO] Mean |H|               : "
        f"{mean_magnitude:.4f}"
    )

    print(
        f"[INFO] Std |H|                : "
        f"{std_magnitude:.4f}"
    )

    print(
        f"[INFO] Mean |estimated CFO|   : "
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
    # Save raw CSV.
    # -------------------------------------------------------------

    csv_path = (
        simulation_dir
        / "micro_tremor.csv"
    )

    header = (
        "time_s,"
        "displacement_m,"
        "expected_phase_rad,"
        "recovered_phase_rad,"
        "fitted_phase_rad,"
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
            np.where(
                valid,
                fitted_phase[
                    np.searchsorted(
                        t_valid,
                        t,
                        side="left",
                    ).clip(
                        0,
                        len(fitted_phase) - 1,
                    )
                ],
                np.nan,
            ),
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
    # Figure.
    # -------------------------------------------------------------

    fig = plt.figure(
        figsize=(12, 10)
    )

    ax1 = fig.add_subplot(3, 1, 1)

    ax1.plot(
        t,
        displacement * 1000.0,
        label="Prescribed displacement",
    )

    ax1.set_ylabel(
        "Displacement (mm)"
    )

    ax1.set_title(
        "Controlled 5.5 Hz Micro-Tremor"
    )

    ax1.grid(True)
    ax1.legend()

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

    ax2.plot(
        t_valid,
        fitted_phase,
        label="5.5 Hz fitted phase",
        linestyle="--",
    )

    ax2.set_ylabel(
        "Phase (rad)"
    )

    ax2.set_title(
        "Propagation Phase Recovery"
    )

    ax2.grid(True)
    ax2.legend()

    ax3 = fig.add_subplot(3, 1, 3)

    ax3.semilogy(
        freqs[positive],
        psd[positive],
    )

    ax3.axvline(
        TREMOR_HZ,
        linestyle="--",
        label=f"Expected {TREMOR_HZ:.1f} Hz",
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

    ax3.set_xlim(
        0,
        20,
    )

    ax3.grid(True)
    ax3.legend()

    fig.tight_layout()

    figure_path = (
        figure_dir
        / "fig_micro_tremor.png"
    )

    fig.savefig(
        figure_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    # -------------------------------------------------------------
    # Final status.
    # -------------------------------------------------------------

    print()

    amplitude_error = abs(
        recovered_amp
        - EXPECTED_PHASE_AMP
    )

    frequency_error = abs(
        detected_frequency
        - TREMOR_HZ
    )

    evidence_ok = (
        n_valid == N_PACKETS
        and amplitude_error < 0.10
        and frequency_error < 0.20
        and correlation > 0.90
    )

    if evidence_ok:

        print(
            "[PASS] Controlled motion-to-CSI "
            "relationship recovered."
        )

    else:

        print(
            "[WARN] Controlled sensing result "
            "did not meet the evidence thresholds."
        )

    print(
        f"[PASS] Raw data : {csv_path}"
    )

    print(
        f"[PASS] Figure   : {figure_path}"
    )

    print()
    print(
        "CONTROLLED MICRO-TREMOR SIMULATION COMPLETE"
    )


if __name__ == "__main__":
    main()