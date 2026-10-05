"""
generate_basic_tables.py

Generate reproducible evaluation tables directly from the current
D:\phase2_fix1 production configuration.

This script DOES NOT modify production code.
"""

from pathlib import Path
import csv
import sys


# ---------------------------------------------------------------------
# Project import
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg


# ---------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------

EVALUATION_DIR = Path(__file__).resolve().parent
TABLE_DIR = EVALUATION_DIR / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def write_csv(filename, rows, fieldnames):
    path = TABLE_DIR / filename

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return path


def write_markdown(filename, rows, fieldnames, title):
    path = TABLE_DIR / filename

    with path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")

        f.write("| " + " | ".join(fieldnames) + " |\n")
        f.write("| " + " | ".join(["---"] * len(fieldnames)) + " |\n")

        for row in rows:
            f.write(
                "| "
                + " | ".join(str(row[field]) for field in fieldnames)
                + " |\n"
            )

    return path


def bits_per_subcarrier(modulation):
    return {
        "BPSK": 1,
        "QPSK": 2,
        "16QAM": 4,
    }[modulation]


def derived_payload_capacity(modulation, n_symbols):
    """
    Derive maximum payload from the current PHY numerology.

    Each DATA OFDM symbol carries:

        NUM_DATA * bits_per_subcarrier

    coded bits.

    The convolutional code has rate 1/2, therefore:

        information bits/symbol
            = NUM_DATA * bits_per_subcarrier / 2

    The packet reserves 32 bits for CRC.

    Therefore:

        payload_bytes =
            floor(
                (n_symbols * information_bits_per_symbol - 32) / 8
            )
    """

    bpsc = bits_per_subcarrier(modulation)

    coded_bits_per_symbol = cfg.NUM_DATA * bpsc

    information_bits_per_symbol = (
        coded_bits_per_symbol // cfg.CONV_RATE_INV
    )

    total_information_bits = (
        n_symbols * information_bits_per_symbol
    )

    payload_bits = total_information_bits - 32

    return max(0, payload_bits // 8)


def symbols_required(modulation, payload_bytes):
    """
    Find the smallest DATA-symbol count that can carry payload_bytes.
    """

    for n_symbols in range(1, cfg.MAX_DATA_SYMS + 1):

        capacity = derived_payload_capacity(
            modulation,
            n_symbols,
        )

        if capacity >= payload_bytes:
            return n_symbols

    raise ValueError(
        f"{payload_bytes} B cannot fit in "
        f"{cfg.MAX_DATA_SYMS} DATA symbols "
        f"using {modulation}"
    )


# ---------------------------------------------------------------------
# Packet timing
# ---------------------------------------------------------------------

def packet_timing(modulation, payload_bytes):
    """
    Calculate the current packet geometry.

    PHY samples are at FS_FFT = 20 MS/s.

    Hardware samples are at FS_HW = 25 MS/s.

    The TX guard is explicitly defined in config.py as
    GUARD_SAMPLES = 1024 hardware samples.
    """

    n_symbols = symbols_required(
        modulation,
        payload_bytes,
    )

    data_samples_bb = (
        n_symbols * cfg.SYMBOL_LEN
    )

    packet_samples_bb = (
        cfg.STF_LEN
        + cfg.LTF_LEN
        + cfg.SIG_LEN
        + data_samples_bb
    )

    # Convert the complete PHY packet from 20 MS/s
    # to the 25 MS/s hardware sample count.
    packet_samples_hw_exact = (
        packet_samples_bb
        * cfg.FS_HW
        / cfg.FS_FFT
    )

    packet_samples_hw = int(
        round(packet_samples_hw_exact)
    )

    # IMPORTANT:
    # GUARD_SAMPLES is already expressed at FS_HW.
    guard_hw = cfg.GUARD_SAMPLES

    total_burst_hw = (
        packet_samples_hw
        + guard_hw
    )

    burst_duration_s = (
        total_burst_hw / cfg.FS_HW
    )

    packet_period_s = cfg.TX_PACKET_PERIOD_S

    idle_s = (
        packet_period_s
        - burst_duration_s
    )

    occupancy = (
        burst_duration_s / packet_period_s
    )

    return {
        "Modulation": modulation,
        "Payload_B": payload_bytes,
        "DATA_symbols": n_symbols,
        "Packet_BB_samples": packet_samples_bb,
        "Packet_HW_samples": packet_samples_hw,
        "Guard_HW_samples": guard_hw,
        "Total_burst_HW_samples": total_burst_hw,
        "Burst_us": burst_duration_s * 1e6,
        "Packet_period_us": packet_period_s * 1e6,
        "Idle_us": idle_s * 1e6,
        "Occupancy_percent": occupancy * 100.0,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    print("=" * 70)
    print("EVALUATION BASIC TABLE GENERATOR")
    print("=" * 70)

    # ================================================================
    # 1. SYSTEM PARAMETERS
    # ================================================================

    delta_f = cfg.SUBCARRIER_SPACING

    system_rows = [
        {
            "Parameter": "FFT size",
            "Value": cfg.FFT_SIZE,
            "Unit": "samples",
        },
        {
            "Parameter": "Cyclic prefix",
            "Value": cfg.CP_LEN,
            "Unit": "samples",
        },
        {
            "Parameter": "OFDM symbol length",
            "Value": cfg.SYMBOL_LEN,
            "Unit": "samples",
        },
        {
            "Parameter": "FFT sample rate",
            "Value": cfg.FS_FFT / 1e6,
            "Unit": "MS/s",
        },
        {
            "Parameter": "Hardware sample rate",
            "Value": cfg.FS_HW / 1e6,
            "Unit": "MS/s",
        },
        {
            "Parameter": "Subcarrier spacing",
            "Value": delta_f / 1e3,
            "Unit": "kHz",
        },
        {
            "Parameter": "Active subcarriers",
            "Value": cfg.NUM_ACTIVE,
            "Unit": "subcarriers",
        },
        {
            "Parameter": "Pilot subcarriers",
            "Value": cfg.NUM_PILOTS,
            "Unit": "subcarriers",
        },
        {
            "Parameter": "Data subcarriers",
            "Value": cfg.NUM_DATA,
            "Unit": "subcarriers",
        },
        {
            "Parameter": "STF length",
            "Value": cfg.STF_LEN,
            "Unit": "BB samples",
        },
        {
            "Parameter": "LTF length",
            "Value": cfg.LTF_LEN,
            "Unit": "BB samples",
        },
        {
            "Parameter": "SIGNAL length",
            "Value": cfg.SIG_LEN,
            "Unit": "BB samples",
        },
        {
            "Parameter": "Maximum DATA symbols",
            "Value": cfg.MAX_DATA_SYMS,
            "Unit": "symbols",
        },
        {
            "Parameter": "Maximum configured payload",
            "Value": cfg.MAX_PAYLOAD_BYTES,
            "Unit": "bytes",
        },
        {
            "Parameter": "TX packet period",
            "Value": cfg.TX_PACKET_PERIOD_S * 1e3,
            "Unit": "ms",
        },
        {
            "Parameter": "TX packet rate",
            "Value": cfg.TX_PACKET_RATE_HZ,
            "Unit": "Hz",
        },
        {
            "Parameter": "CSI sampling rate",
            "Value": cfg.TX_PACKET_RATE_HZ,
            "Unit": "Hz",
        },
        {
            "Parameter": "CSI Nyquist frequency",
            "Value": cfg.TX_PACKET_RATE_HZ / 2,
            "Unit": "Hz",
        },
        {
            "Parameter": "TX guard",
            "Value": cfg.GUARD_SAMPLES,
            "Unit": "HW samples",
        },
        {
            "Parameter": "TX guard duration",
            "Value": cfg.GUARD_SAMPLES / cfg.FS_HW * 1e6,
            "Unit": "us",
        },
    ]

    system_fields = [
        "Parameter",
        "Value",
        "Unit",
    ]

    write_csv(
        "system_parameters.csv",
        system_rows,
        system_fields,
    )

    write_markdown(
        "system_parameters.md",
        system_rows,
        system_fields,
        "System Parameters",
    )

    # ================================================================
    # 2. MODULATION / PAYLOAD CAPACITY
    # ================================================================

    capacity_rows = []

    for modulation in ["BPSK", "QPSK", "16QAM"]:

        derived_capacity = derived_payload_capacity(
            modulation,
            cfg.MAX_DATA_SYMS,
        )

        configured_capacity = (
            cfg.MAX_PAYLOAD_BYTES_BY_MODULATION[modulation]
        )

        capacity_rows.append(
            {
                "Modulation": modulation,
                "Bits_per_subcarrier": bits_per_subcarrier(
                    modulation
                ),
                "Data_subcarriers": cfg.NUM_DATA,
                "Max_DATA_symbols": cfg.MAX_DATA_SYMS,
                "Derived_max_payload_B": derived_capacity,
                "Configured_max_payload_B": configured_capacity,
                "Capacity_check": (
                    "PASS"
                    if derived_capacity == configured_capacity
                    else "FAIL"
                ),
            }
        )

    capacity_fields = [
        "Modulation",
        "Bits_per_subcarrier",
        "Data_subcarriers",
        "Max_DATA_symbols",
        "Derived_max_payload_B",
        "Configured_max_payload_B",
        "Capacity_check",
    ]

    write_csv(
        "packet_capacity.csv",
        capacity_rows,
        capacity_fields,
    )

    write_markdown(
        "packet_capacity.md",
        capacity_rows,
        capacity_fields,
        "Packet Capacity",
    )

    # ================================================================
    # 3. PACKET TIMING
    # ================================================================

    timing_requests = [
        ("BPSK", 100),
        ("BPSK", 204),
        ("QPSK", 100),
        ("QPSK", 412),
        ("16QAM", 100),
        ("16QAM", 829),
    ]

    timing_rows = []

    for modulation, payload in timing_requests:

        row = packet_timing(
            modulation,
            payload,
        )

        if row["Idle_us"] < 0:
            raise RuntimeError(
                f"{modulation} {payload} B packet "
                f"does not fit into the configured "
                f"{cfg.TX_PACKET_PERIOD_S * 1e3:.3f} ms period."
            )

        timing_rows.append(row)

    timing_fields = [
        "Modulation",
        "Payload_B",
        "DATA_symbols",
        "Packet_BB_samples",
        "Packet_HW_samples",
        "Guard_HW_samples",
        "Total_burst_HW_samples",
        "Burst_us",
        "Packet_period_us",
        "Idle_us",
        "Occupancy_percent",
    ]

    write_csv(
        "packet_timing.csv",
        timing_rows,
        timing_fields,
    )

    write_markdown(
        "packet_timing.md",
        timing_rows,
        timing_fields,
        "Packet Timing and Duty Cycle",
    )

    # ================================================================
    # 4. CONSOLE SUMMARY
    # ================================================================

    print("\nSYSTEM:")
    print(
        f"  FFT                    : {cfg.FFT_SIZE}"
    )
    print(
        f"  CP                     : {cfg.CP_LEN}"
    )
    print(
        f"  Symbol length          : {cfg.SYMBOL_LEN}"
    )
    print(
        f"  FS_FFT                 : {cfg.FS_FFT / 1e6:.3f} MS/s"
    )
    print(
        f"  FS_HW                  : {cfg.FS_HW / 1e6:.3f} MS/s"
    )
    print(
        f"  Subcarrier spacing     : {delta_f / 1e3:.3f} kHz"
    )
    print(
        f"  Active subcarriers     : {cfg.NUM_ACTIVE}"
    )
    print(
        f"  Pilot subcarriers      : {cfg.NUM_PILOTS}"
    )
    print(
        f"  Data subcarriers       : {cfg.NUM_DATA}"
    )
    print(
        f"  Packet period          : "
        f"{cfg.TX_PACKET_PERIOD_S * 1e3:.3f} ms"
    )
    print(
        f"  Packet rate            : "
        f"{cfg.TX_PACKET_RATE_HZ:.3f} Hz"
    )

    print("\nCAPACITY:")

    for row in capacity_rows:

        print(
            f"  {row['Modulation']:6s} -> "
            f"derived={row['Derived_max_payload_B']} B, "
            f"configured={row['Configured_max_payload_B']} B, "
            f"{row['Capacity_check']}"
        )

    print("\nTIMING:")

    for row in timing_rows:

        print(
            f"  {row['Modulation']:6s} "
            f"{row['Payload_B']:3d} B -> "
            f"{row['DATA_symbols']:2d} symbols, "
            f"{row['Packet_BB_samples']:5d} BB, "
            f"{row['Packet_HW_samples']:5d} HW + "
            f"{row['Guard_HW_samples']} guard = "
            f"{row['Total_burst_HW_samples']} HW, "
            f"{row['Burst_us']:.3f} us, "
            f"idle={row['Idle_us']:.3f} us, "
            f"occupancy={row['Occupancy_percent']:.4f}%"
        )

    print("\nGENERATED FILES:")

    for filename in [
        "system_parameters.csv",
        "system_parameters.md",
        "packet_capacity.csv",
        "packet_capacity.md",
        "packet_timing.csv",
        "packet_timing.md",
    ]:
        print(f"  evaluation\\tables\\{filename}")

    print("\nBASIC TABLE GENERATION COMPLETE")


if __name__ == "__main__":
    main()