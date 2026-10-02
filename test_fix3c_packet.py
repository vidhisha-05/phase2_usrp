import numpy as np

import config as cfg
import waveform
import demod
import scrambler
import sync


CASES = [
    ("BPSK", 100),
    ("BPSK", 204),
    ("QPSK", 100),
    ("QPSK", 412),
    ("16QAM", 100),
    ("16QAM", 829),
]


rng = np.random.default_rng(12345)


for modulation, n_payload_bytes in CASES:

    print("\n" + "=" * 60)
    print(f"CASE: {modulation}, {n_payload_bytes} bytes")
    print("=" * 60)

    # --------------------------------------------------------
    # Deterministic payload
    # --------------------------------------------------------

    payload_bits = rng.integers(
        0,
        2,
        size=n_payload_bytes * 8,
        dtype=np.uint8,
    )

    # --------------------------------------------------------
    # Expected DATA symbol count
    # --------------------------------------------------------

    expected_n_sym = demod.n_data_syms_for_payload(
        n_payload_bytes,
        modulation,
    )

    expected_len = (
        cfg.STF_LEN
        + cfg.LTF_LEN
        + cfg.SIG_LEN
        + expected_n_sym * cfg.SYMBOL_LEN
    )

    print("Expected DATA symbols:", expected_n_sym)
    print("Expected packet length:", expected_len)

    # --------------------------------------------------------
    # TX
    # --------------------------------------------------------

    packet = waveform.assemble_packet(
        payload_bits,
        modulation=modulation,
        scrambler_mod=scrambler,
        encoder_mod=scrambler,
        mapper_fn=scrambler.map_bits_to_symbols,
        idle_samples=0,
    )

    print("Actual packet length:", len(packet))

    assert len(packet) == expected_len

    # --------------------------------------------------------
    # Synchronization
    # --------------------------------------------------------

    result = sync.sync_packet(
        packet,
        coarse_cfo_hz=0.0,
        n_data_symbols=0,
    )

    H_hat = result["H_hat"]
    ltf_start = result["ltf_start"]

    assert H_hat.shape == (cfg.NUM_ACTIVE,)

    # --------------------------------------------------------
    # CFO correction
    # --------------------------------------------------------

    rx_c = sync.apply_cfo_correction(
        packet,
        result["total_cfo"],
        start_n=0,
    )

    # --------------------------------------------------------
    # SIGNAL
    # --------------------------------------------------------

    sig_body_start = (
        ltf_start
        + cfg.LTF_LEN
        + cfg.CP_LEN
    )

    sig_body_end = (
        sig_body_start
        + cfg.FFT_SIZE
    )

    Y_sig = np.fft.fft(
        rx_c[sig_body_start:sig_body_end],
        n=cfg.FFT_SIZE,
    ).astype(np.complex64)

    sig = demod.parse_signal_field(
        Y_sig,
        H_hat,
    )

    print("Decoded SIGNAL:", sig)

    assert sig["valid"] is True
    assert sig["n_payload_bytes"] == n_payload_bytes
    assert sig["modulation"] == modulation
    assert sig["n_data_syms"] == expected_n_sym

    # --------------------------------------------------------
    # DATA FFTs
    # --------------------------------------------------------

    data_start = (
        ltf_start
        + cfg.LTF_LEN
        + cfg.SIG_LEN
    )

    data_ffts = []

    for m in range(sig["n_data_syms"]):

        s = (
            data_start
            + m * cfg.SYMBOL_LEN
            + cfg.CP_LEN
        )

        e = s + cfg.FFT_SIZE

        assert e <= len(rx_c)

        Y = np.fft.fft(
            rx_c[s:e],
            n=cfg.FFT_SIZE,
        ).astype(np.complex64)

        data_ffts.append(Y)

    print("Extracted DATA FFTs:", len(data_ffts))

    assert len(data_ffts) == expected_n_sym

    # --------------------------------------------------------
    # Full demodulation
    # --------------------------------------------------------

    rx_bits, crc_ok = demod.demodulate_packet(
        data_ffts,
        H_hat,
        modulation=sig["modulation"],
        n_payload_bytes=sig["n_payload_bytes"],
    )

    print("CRC OK:", crc_ok)
    print("Recovered bits:", len(rx_bits))
    print("Expected bits:", len(payload_bits))

    # --------------------------------------------------------
    # Final checks
    # --------------------------------------------------------

    assert crc_ok is True

    assert len(rx_bits) == len(payload_bits)

    assert np.array_equal(
        rx_bits.astype(np.uint8),
        payload_bits.astype(np.uint8),
    )

    print("PASS")


print("\n" + "=" * 60)
print("FIX 3C — ALL SIX PACKET ROUND-TRIP CASES: PASS")
print("=" * 60)