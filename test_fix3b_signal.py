import numpy as np
import waveform
import demod
import scrambler
import config as cfg


H = np.ones(cfg.NUM_ACTIVE, dtype=np.complex64)

RATE_BITS = {
    "BPSK": np.array([1, 0, 1, 1], dtype=np.uint8),
    "QPSK": np.array([0, 1, 0, 1], dtype=np.uint8),
    "16QAM": np.array([1, 1, 0, 1], dtype=np.uint8),
}


def length_bits(n_bytes):
    """12-bit MSB-first payload length."""
    return np.array(
        [(n_bytes >> i) & 1 for i in range(11, -1, -1)],
        dtype=np.uint8,
    )


cases = [
    ("BPSK", 100),
    ("BPSK", 204),
    ("QPSK", 100),
    ("QPSK", 412),
    ("16QAM", 100),
    ("16QAM", 829),
]


for modulation, n_bytes in cases:

    rate_bits = RATE_BITS[modulation]
    len_bits = length_bits(n_bytes)

    sig = waveform.generate_signal_symbol(
        rate_bits,
        len_bits,
        scrambler,
        scrambler,
    )

    print(
        f"\nCASE: {modulation}, {n_bytes} bytes"
    )

    print(
        "SIGNAL length:",
        len(sig),
        "expected:",
        cfg.SIG_LEN,
    )

    assert len(sig) == cfg.SIG_LEN

    # Remove cyclic prefix.
    sig_body = sig[
        cfg.CP_LEN:cfg.CP_LEN + cfg.FFT_SIZE
    ]

    # Recover SIGNAL FFT.
    Y_sig = np.fft.fft(
        sig_body,
        n=cfg.FFT_SIZE,
    ).astype(np.complex64)

    result = demod.parse_signal_field(
        Y_sig,
        H,
    )

    print("Decoded SIGNAL:", result)

    assert result["valid"] is True
    assert result["n_payload_bytes"] == n_bytes
    assert result["modulation"] == modulation

    expected_n_sym = demod.n_data_syms_for_payload(
        n_bytes,
        modulation,
    )

    assert result["n_data_syms"] == expected_n_sym

    print(
        "EXPECTED:",
        n_bytes,
        "bytes,",
        modulation,
        ",",
        expected_n_sym,
        "DATA symbols"
    )

    print("PASS")