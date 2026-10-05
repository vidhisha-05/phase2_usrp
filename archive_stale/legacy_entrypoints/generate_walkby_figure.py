"""
generate_walkby_figure.py

Controlled dynamic walk-by / gross-motion CSI simulation.

Evidence generator only.
No production PHY files are modified.

Scenario:
    A human walk-by event is represented by:
      - temporary shadow fading,
      - dynamic multipath,
      - a changing propagation phase,
      - a time-varying multipath delay.

The current OFDM synchronization + CSI extraction pipeline is used
to recover CSI snapshots at 200 Hz.
"""

from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

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

N_PACKETS = 1400

PACKET_INTERVAL_S = 0.005
CSI_RATE_HZ = 1.0 / PACKET_INTERVAL_S

CARRIER_HZ = 2.412e9
C_LIGHT = 299_792_458.0

WALK_START_S = 2.0
WALK_END_S = 5.0

SNR_BASE_DB = 27.0
SHADOW_FADE_DB = 8.0

PAYLOAD_BYTES = 100
MODULATION = "BPSK"

BASE_SEED = 20261003


# =====================================================================
# WALK-BY MODEL
# =====================================================================

# Dynamic phase components.
DOPPLER_F1_HZ = 12.0
DOPPLER_F2_HZ = 18.0

# Dynamic path delay.
BASE_DYNAMIC_DELAY = 8
MAX_DYNAMIC_DELAY = 24

# Dynamic reflected-path strength.
DYNAMIC_PATH_GAIN = 10.0 ** (-16.0 / 20.0)


# =====================================================================
# DERIVED VALUES
# =====================================================================

WAVELENGTH_M = C_LIGHT / CARRIER_HZ
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
    print("CONTROLLED WALK-BY / DYNAMIC MULTIPATH CSI SIMULATION")
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
        f"[INFO] Walk-by interval     : "
        f"{WALK_START_S:.2f}–{WALK_END_S:.2f} s"
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
        f"[INFO] Baseline SNR         : "
        f"{SNR_BASE_DB:.1f} dB"
    )

    print(
        f"[INFO] Walk-by SNR          : "
        f"{SNR_BASE_DB - SHADOW_FADE_DB:.1f} dB"
    )

    print(
        f"[INFO] Shadow fading       : "
        f"{SHADOW_FADE_DB:.1f} dB"
    )

    print(
        f"[INFO] Dynamic Doppler     : "
        f"{DOPPLER_F1_HZ:.1f} + {DOPPLER_F2_HZ:.1f} Hz"
    )

    print(
        f"[INFO] Dynamic delay range : "
        f"{BASE_DYNAMIC_DELAY}–{MAX_DYNAMIC_DELAY} BB samples"
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

    # -------------------------------------------------------------
    # Time axis
    # -------------------------------------------------------------

    t = (
        np.arange(N_PACKETS, dtype=np.float64)
        * PACKET_INTERVAL_S
    )

    rng = np.random.default_rng(
        BASE_SEED + 10
    )

    recovered_magnitude = []
    recovered_phase = []
    recovered_cfo = []
    recovered_ltf_start = []

    true_dynamic_gain = []
    true_delay = []
    true_snr = []

    # -------------------------------------------------------------
    # Process packets.
    # -------------------------------------------------------------

    for i, timestamp in enumerate(t):

        in_walkby = (
            WALK_START_S
            <= timestamp
            <= WALK_END_S
        )

        # ---------------------------------------------------------
        # Baseline channel
        # ---------------------------------------------------------

        if not in_walkby:

            effective_snr_db = SNR_BASE_DB

            dynamic_delay = BASE_DYNAMIC_DELAY

            dynamic_phase = (
                0.5
                * np.sin(
                    2.0
                    * np.pi
                    * 1.5
                    * timestamp
                )
            )

            dynamic_gain = (
                10.0 ** (-30.0 / 20.0)
            )

            envelope = 1.0

        # ---------------------------------------------------------
        # Walk-by channel
        # ---------------------------------------------------------

        else:

            # 8 dB shadow fading.
            effective_snr_db = (
                SNR_BASE_DB
                - SHADOW_FADE_DB
            )

            # Dynamic Doppler-like phase variation.
            dynamic_phase = (
                8.0
                * np.sin(
                    2.0
                    * np.pi
                    * DOPPLER_F2_HZ
                    * timestamp
                )
                +
                4.0
                * np.cos(
                    2.0
                    * np.pi
                    * DOPPLER_F1_HZ
                    * timestamp
                )
            )

            # Time-varying delay between 8 and 24 BB samples.
            delay_fraction = (
                0.5
                * (
                    1.0
                    +
                    np.sin(
                        2.0
                        * np.pi
                        * 0.8
                        * timestamp
                    )
                )
            )

            dynamic_delay = int(
                round(
                    BASE_DYNAMIC_DELAY
                    +
                    delay_fraction
                    * (
                        MAX_DYNAMIC_DELAY
                        - BASE_DYNAMIC_DELAY
                    )
                )
            )

            # Stronger human-reflected path.
            dynamic_gain = (
                DYNAMIC_PATH_GAIN
            )

            # Slowly varying shadow envelope.
            envelope = (
                0.50
                +
                0.15
                * np.sin(
                    2.0
                    * np.pi
                    * 2.5
                    * timestamp
                )
            )

        true_dynamic_gain.append(
            dynamic_gain
        )

        true_delay.append(
            dynamic_delay
        )

        true_snr.append(
            effective_snr_db
        )

        # ---------------------------------------------------------
        # Construct dynamic channel.
        #
        # Main path
        # +
        # delayed dynamic reflected path.
        # ---------------------------------------------------------

        max_delay = max(
            dynamic_delay,
            0,
        )

        channel_out = np.zeros(
            len(packet) + max_delay,
            dtype=np.complex128,
        )

        # Direct path.
        channel_out[
            :len(packet)
        ] += packet

        # Dynamic reflected path.
        reflected = (
            dynamic_gain
            * np.exp(
                1j * dynamic_phase
            )
            * packet
        )

        channel_out[
            dynamic_delay:
            dynamic_delay + len(packet)
        ] += reflected

        channel_out = (
            channel_out[:len(packet)]
            * envelope
        )

        # ---------------------------------------------------------
        # AWGN calibrated from actual current packet/channel power.
        # ---------------------------------------------------------

        channel_power = float(
            np.mean(
                np.abs(channel_out) ** 2
            )
        )

        noise_voltage = np.sqrt(
            channel_power
            / (
                10.0
                ** (
                    effective_snr_db
                    / 10.0
                )
            )
        )

        noise = (
            rng.standard_normal(
                len(packet)
            )
            +
            1j
            * rng.standard_normal(
                len(packet)
            )
        )

        noise *= (
            noise_voltage
            / np.sqrt(2.0)
        )

        rx = (
            channel_out
            + noise
        ).astype(np.complex64)

        # ---------------------------------------------------------
        # Current production synchronization / CSI extraction.
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
                    f"Unexpected H_hat shape "
                    f"{H_hat.shape}"
                )

            if np.any(
                np.isnan(H_hat)
            ):
                raise RuntimeError(
                    "NaN in CSI"
                )

            H_mean = np.mean(H_hat)

            recovered_magnitude.append(
                float(np.abs(H_mean))
            )

            recovered_phase.append(
                float(np.angle(H_mean))
            )

            recovered_cfo.append(
                float(result["total_cfo"])
            )

            recovered_ltf_start.append(
                int(result["ltf_start"])
            )

        except Exception:

            recovered_magnitude.append(
                np.nan
            )

            recovered_phase.append(
                np.nan
            )

            recovered_cfo.append(
                np.nan
            )

            recovered_ltf_start.append(
                -1
            )

    # =================================================================
    # Convert results
    # =================================================================

    recovered_magnitude = np.asarray(
        recovered_magnitude,
        dtype=np.float64,
    )

    recovered_phase = np.asarray(
        recovered_phase,
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

    true_dynamic_gain = np.asarray(
        true_dynamic_gain,
        dtype=np.float64,
    )

    true_delay = np.asarray(
        true_delay,
        dtype=np.float64,
    )

    true_snr = np.asarray(
        true_snr,
        dtype=np.float64,
    )

    valid = np.isfinite(
        recovered_magnitude
    )

    n_valid = int(
        np.sum(valid)
    )

    # =================================================================
    # Results
    # =================================================================

    print()
    print("=" * 70)
    print("WALK-BY RESULTS")
    print("=" * 70)

    print(
        f"[INFO] CSI samples recovered : "
        f"{n_valid}/{N_PACKETS}"
    )

    baseline_mask = (
        valid
        &
        (
            t < WALK_START_S
        )
    )

    event_mask = (
        valid
        &
        (
            t >= WALK_START_S
        )
        &
        (
            t <= WALK_END_S
        )
    )

    baseline_mag = float(
        np.mean(
            recovered_magnitude[
                baseline_mask
            ]
        )
    )

    event_mag = float(
        np.mean(
            recovered_magnitude[
                event_mask
            ]
        )
    )

    baseline_phase_std = float(
        np.std(
            recovered_phase[
                baseline_mask
            ]
        )
    )

    event_phase_std = float(
        np.std(
            recovered_phase[
                event_mask
            ]
        )
    )

    baseline_cfo = float(
        np.mean(
            np.abs(
                recovered_cfo[
                    baseline_mask
                ]
            )
        )
    )

    event_cfo = float(
        np.mean(
            np.abs(
                recovered_cfo[
                    event_mask
                ]
            )
        )
    )

    magnitude_change_db = (
        20.0
        * np.log10(
            max(event_mag, 1e-12)
            /
            max(baseline_mag, 1e-12)
        )
    )

    print(
        f"[INFO] Baseline mean |H|   : "
        f"{baseline_mag:.4f}"
    )

    print(
        f"[INFO] Walk-by mean |H|    : "
        f"{event_mag:.4f}"
    )

    print(
        f"[INFO] Mean |H| change     : "
        f"{magnitude_change_db:.3f} dB"
    )

    print(
        f"[INFO] Baseline phase std  : "
        f"{baseline_phase_std:.4f} rad"
    )

    print(
        f"[INFO] Walk-by phase std   : "
        f"{event_phase_std:.4f} rad"
    )

    print(
        f"[INFO] Baseline |CFO|      : "
        f"{baseline_cfo:.3f} Hz"
    )

    print(
        f"[INFO] Walk-by |CFO|       : "
        f"{event_cfo:.3f} Hz"
    )

    print(
        f"[INFO] Delay range         : "
        f"{int(np.min(true_delay))}–"
        f"{int(np.max(true_delay))} BB samples"
    )

    # =================================================================
    # Save CSV
    # =================================================================

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

    csv_path = (
        simulation_dir
        / "walkby_dynamic.csv"
    )

    header = (
        "time_s,"
        "recovered_magnitude,"
        "recovered_phase_rad,"
        "estimated_cfo_hz,"
        "ltf_start,"
        "true_snr_db,"
        "true_dynamic_delay_bb,"
        "true_dynamic_gain"
    )

    data = np.column_stack(
        [
            t,
            recovered_magnitude,
            recovered_phase,
            recovered_cfo,
            recovered_ltf_start,
            true_snr,
            true_delay,
            true_dynamic_gain,
        ]
    )

    np.savetxt(
        csv_path,
        data,
        delimiter=",",
        header=header,
        comments="",
    )

    # =================================================================
    # Figure
    # =================================================================

    fig = plt.figure(
        figsize=(12, 11)
    )

    # -------------------------------------------------------------
    # Panel 1: CSI magnitude
    # -------------------------------------------------------------

    ax1 = fig.add_subplot(3, 1, 1)

    ax1.plot(
        t,
        recovered_magnitude,
        linewidth=1.0,
        label="Recovered mean |H|",
    )

    ax1.axvspan(
        WALK_START_S,
        WALK_END_S,
        alpha=0.15,
        label="Walk-by interval",
    )

    ax1.set_ylabel(
        "Mean |H|"
    )

    ax1.set_title(
        "CSI Magnitude During Dynamic Walk-By"
    )

    ax1.grid(True)
    ax1.legend()

    # -------------------------------------------------------------
    # Panel 2: phase
    # -------------------------------------------------------------

    ax2 = fig.add_subplot(3, 1, 2)

    phase_unwrapped = np.unwrap(
        recovered_phase
    )

    ax2.plot(
        t,
        phase_unwrapped,
        linewidth=1.0,
        label="Recovered CSI phase",
    )

    ax2.axvspan(
        WALK_START_S,
        WALK_END_S,
        alpha=0.15,
    )

    ax2.set_ylabel(
        "Phase (rad)"
    )

    ax2.set_title(
        "CSI Phase Under Dynamic Multipath"
    )

    ax2.grid(True)
    ax2.legend()

    # -------------------------------------------------------------
    # Panel 3: channel-model state
    # -------------------------------------------------------------

    ax3 = fig.add_subplot(3, 1, 3)

    ax3.plot(
        t,
        true_delay,
        label="Dynamic delay (BB samples)",
    )

    ax3_twin = ax3.twinx()

    ax3_twin.plot(
        t,
        true_snr,
        linestyle="--",
        label="Channel SNR",
    )

    ax3.axvspan(
        WALK_START_S,
        WALK_END_S,
        alpha=0.15,
    )

    ax3.set_xlabel(
        "Time (s)"
    )

    ax3.set_ylabel(
        "Delay (BB samples)"
    )

    ax3_twin.set_ylabel(
        "SNR (dB)"
    )

    ax3.set_title(
        "Controlled Dynamic Channel Parameters"
    )

    ax3.grid(True)

    fig.tight_layout()

    figure_path = (
        figure_dir
        / "fig_walkby_dynamic.png"
    )

    fig.savefig(
        figure_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    # =================================================================
    # Evidence criterion
    # =================================================================

    # We require complete CSI recovery and a measurable difference
    # between baseline and the dynamic event. We deliberately do NOT
    # impose a fixed expected |H| value because the dynamic channel
    # is frequency-selective.

    evidence_ok = (
        n_valid == N_PACKETS
        and np.isfinite(
            magnitude_change_db
        )
        and (
            abs(
                event_mag
                - baseline_mag
            )
            > 0.01
        )
    )

    print()

    if evidence_ok:

        print(
            "[PASS] Dynamic walk-by produced "
            "a measurable CSI change."
        )

    else:

        print(
            "[WARN] Dynamic walk-by did not "
            "meet the evidence threshold."
        )

    print(
        f"[PASS] Raw data : {csv_path}"
    )

    print(
        f"[PASS] Figure   : {figure_path}"
    )

    print()
    print(
        "CONTROLLED WALK-BY SIMULATION COMPLETE"
    )


if __name__ == "__main__":
    main()