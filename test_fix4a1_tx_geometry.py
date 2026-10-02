"""
test_fix4a1_tx_geometry.py

Fix 4A-1 diagnostic only.

Verifies the current TX waveform geometry for:
    - 100 B BPSK
    - 100 B QPSK
    - 100 B 16QAM
    - maximum BPSK = 204 B
    - maximum QPSK = 412 B
    - maximum 16QAM = 829 B

Checks:
    1. Actual baseband packet length.
    2. Expected DATA-symbol count.
    3. Actual 20 -> 25 MS/s packet length.
    4. Guard length.
    5. Total burst length.
    6. BB and HW durations.
    7. 5 ms schedule occupancy.
    8. Exact 20->25 duration preservation.

NO production files are modified.
"""

import numpy as np

import config as cfg
import waveform
import scrambler


CASES = [
    ("BPSK", 100),
    ("QPSK", 100),
    ("16QAM", 100),
    ("BPSK", 204),
    ("QPSK", 412),
    ("16QAM", 829),
]


def expected_data_symbols(payload_bytes: int, modulation: str) -> int:
    """
    Independent calculation from the PHY contract.

    wire bits = payload bits + CRC32
              = 8*payload_bytes + 32

    rate-1/2 coding doubles the bit count.

    Each DATA OFDM symbol carries:
        NUM_DATA * bits_per_symbol
    coded bits.
    """
    bps = {
        "BPSK": 1,
        "QPSK": 2,
        "16QAM": 4,
    }[modulation]

    wire_bits = 8 * payload_bytes + 32
    coded_bits = 2 * wire_bits
    capacity_per_symbol = cfg.NUM_DATA * bps

    return int(
        np.ceil(coded_bits / capacity_per_symbol)
    )


def expected_bb_length(n_data_symbols: int) -> int:
    return (
        cfg.STF_LEN
        + cfg.LTF_LEN
        + cfg.SIG_LEN
        + n_data_symbols * cfg.SYMBOL_LEN
    )


def main():
    print("=" * 78)
    print("FIX 4A-1 — TX PACKET GEOMETRY")
    print("=" * 78)

    print()
    print("PHY CONSTANTS")
    print("-" * 78)
    print(f"FFT_SIZE          = {cfg.FFT_SIZE}")
    print(f"CP_LEN            = {cfg.CP_LEN}")
    print(f"SYMBOL_LEN        = {cfg.SYMBOL_LEN}")
    print(f"STF_LEN           = {cfg.STF_LEN}")
    print(f"LTF_LEN           = {cfg.LTF_LEN}")
    print(f"SIG_LEN           = {cfg.SIG_LEN}")
    print(f"NUM_DATA          = {cfg.NUM_DATA}")
    print(f"FS_FFT            = {cfg.FS_FFT}")
    print(f"FS_HW             = {cfg.FS_HW}")
    print(f"GUARD_SAMPLES     = {cfg.GUARD_SAMPLES}")
    print(f"TX_PERIOD         = {cfg.TX_PACKET_PERIOD_S}")
    print(f"TX_RATE           = {cfg.TX_PACKET_RATE_HZ}")

    print()
    print("THEORETICAL SCHEDULE")
    print("-" * 78)

    period_hw_samples = (
        cfg.FS_HW * cfg.TX_PACKET_PERIOD_S
    )

    period_bb_samples = (
        cfg.FS_FFT * cfg.TX_PACKET_PERIOD_S
    )

    print(
        f"5 ms period at HW rate = "
        f"{period_hw_samples:.3f} samples"
    )
    print(
        f"5 ms period at BB rate = "
        f"{period_bb_samples:.3f} samples"
    )

    assert np.isclose(period_hw_samples, 125000.0)
    assert np.isclose(period_bb_samples, 100000.0)

    print("PASS: 5 ms = 125000 HW samples")
    print("PASS: 5 ms = 100000 BB samples")

    print()
    print("PACKET CASES")
    print("-" * 78)

    payload_rng = np.random.default_rng(12345)

    all_pass = True

    for modulation, payload_bytes in CASES:

        payload_bits = payload_rng.integers(
            0,
            2,
            payload_bytes * 8,
            dtype=np.uint8,
        )

        # Build the actual current PHY packet.
        pkt_bb = waveform.assemble_packet(
            payload_bits,
            modulation=modulation,
            scrambler_mod=scrambler,
            encoder_mod=scrambler,
            mapper_fn=scrambler.map_bits_to_symbols,
            idle_samples=0,
        )

        # Current production TX path.
        pkt_hw = waveform.resample_20to25(pkt_bb)

        # The production guard is added AFTER resampling.
        burst_len = cfg.GUARD_SAMPLES + len(pkt_hw)

        # Independent mathematical expectation.
        n_sym_expected = expected_data_symbols(
            payload_bytes,
            modulation,
        )

        bb_len_expected = expected_bb_length(
            n_sym_expected
        )

        # Ideal rational-rate length.
        hw_len_ideal = bb_len_expected * 5 / 4

        # Durations.
        bb_time_us = len(pkt_bb) / cfg.FS_FFT * 1e6
        hw_time_us = len(pkt_hw) / cfg.FS_HW * 1e6
        burst_time_us = burst_len / cfg.FS_HW * 1e6

        idle_time_us = (
            cfg.TX_PACKET_PERIOD_S * 1e6
            - burst_time_us
        )

        occupancy_pct = (
            burst_time_us
            / (cfg.TX_PACKET_PERIOD_S * 1e6)
            * 100.0
        )

        print()
        print(
            f"{modulation:>5} {payload_bytes:>4} B"
        )
        print(
            f"  Expected DATA symbols : "
            f"{n_sym_expected}"
        )
        print(
            f"  Actual BB samples     : "
            f"{len(pkt_bb)}"
        )
        print(
            f"  Expected BB samples   : "
            f"{bb_len_expected}"
        )
        print(
            f"  Ideal HW samples      : "
            f"{hw_len_ideal:.3f}"
        )
        print(
            f"  Actual HW samples     : "
            f"{len(pkt_hw)}"
        )
        print(
            f"  Guard samples         : "
            f"{cfg.GUARD_SAMPLES}"
        )
        print(
            f"  Total burst samples   : "
            f"{burst_len}"
        )
        print(
            f"  BB duration           : "
            f"{bb_time_us:.6f} us"
        )
        print(
            f"  HW packet duration    : "
            f"{hw_time_us:.6f} us"
        )
        print(
            f"  HW burst duration     : "
            f"{burst_time_us:.6f} us"
        )
        print(
            f"  Idle before next pkt  : "
            f"{idle_time_us:.6f} us"
        )
        print(
            f"  Period occupancy      : "
            f"{occupancy_pct:.6f} %"
        )

        # ------------------------------------------------------------
        # Assertions
        # ------------------------------------------------------------

        case_ok = True

        if len(pkt_bb) != bb_len_expected:
            print(
                f"  FAIL: BB length mismatch "
                f"{len(pkt_bb)} != {bb_len_expected}"
            )
            case_ok = False
        else:
            print("  PASS: BB packet length")

        if len(pkt_hw) != round(hw_len_ideal):
            print(
                f"  FAIL: HW length mismatch "
                f"{len(pkt_hw)} != {round(hw_len_ideal)}"
            )
            case_ok = False
        else:
            print("  PASS: HW packet length")

        if burst_len >= period_hw_samples:
            print(
                f"  FAIL: burst does not fit in 5 ms period: "
                f"{burst_len} >= {period_hw_samples}"
            )
            case_ok = False
        else:
            print("  PASS: burst fits in 5 ms period")

        if idle_time_us <= 0:
            print("  FAIL: no idle margin")
            case_ok = False
        else:
            print("  PASS: positive idle margin")

        # Resampling must preserve physical duration.
        duration_error_us = abs(
            bb_time_us - hw_time_us
        )

        print(
            f"  Duration preservation err: "
            f"{duration_error_us:.9f} us"
        )

        if duration_error_us > 0.1:
            print(
                "  FAIL: excessive BB/HW duration mismatch"
            )
            case_ok = False
        else:
            print(
                "  PASS: BB/HW duration preserved"
            )

        if not case_ok:
            all_pass = False

    print()
    print("=" * 78)

    if all_pass:
        print("FIX 4A-1 RESULT: PASS")
        print()
        print(
            "No production files were modified."
        )
        print(
            "TX packet geometry is consistent with the "
            "5 ms / 200 Hz hardware timing contract."
        )
    else:
        print("FIX 4A-1 RESULT: FAIL")
        print()
        print(
            "Do NOT modify production code yet."
        )
        print(
            "Inspect the first failing case before proceeding."
        )

    print("=" * 78)


if __name__ == "__main__":
    main()