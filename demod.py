"""
demod.py — Equalization, demapping, Viterbi decode, descramble, CRC check,
            and SIGNAL-field parsing.

Bug 4 fixes applied:
  - n_data_syms_for_payload(): single source-of-truth formula used by BOTH
    TX (waveform.py) and RX to agree on how many DATA symbols a payload needs.
  - parse_signal_field(): ZF-equalizes + Viterbi-decodes the SIGNAL OFDM symbol
    to recover (n_payload_bytes, modulation, n_data_syms) at RX.
  - demodulate_packet(): accepts n_payload_bytes to trim coded bits before
    Viterbi so padding zeros are never decoded as payload bits.

Reference: phase2 (2).md — Sections 5, 6, 12
"""

import math
import numpy as np
import config as cfg
import scrambler

# CRC-32 API — delegate to scrambler.py (canonical single implementation)
crc32_append = scrambler.crc32_append
crc32_check  = scrambler.crc32_check

# Rate indicator 4-bit codes — must match waveform.assemble_packet rate_map
_RATE_MAP = {
    (1, 0, 1, 1): "BPSK",
    (0, 1, 0, 1): "QPSK",
    (1, 1, 0, 1): "16QAM",
}
_BPS = {"BPSK": 1, "QPSK": 2, "16QAM": 4}

# Pre-compute data-SC mask (active minus pilots) — reused in every call
_DATA_SET  = set(cfg.DATA_INDICES)
_DATA_MASK = np.array([k in _DATA_SET for k in cfg.ACTIVE_SUBCARRIERS], dtype=bool)


# ─────────────────────────────────────────────────────────────────────────────
# TX↔RX Length Protocol  (Bug 4 — single source of truth)
# ─────────────────────────────────────────────────────────────────────────────

def n_data_syms_for_payload(n_payload_bytes: int, modulation: str) -> int:
    """
    Number of DATA OFDM symbols needed to carry n_payload_bytes of payload.

    Derivation:
      wire_bits  = n_payload_bytes*8 + 32   # payload bits + 32-bit CRC
      coded_bits = wire_bits * 2            # rate-1/2 convolutional encoder
      N_sym      = ceil(coded_bits / (bps * NUM_DATA))

    This function is the SINGLE source of truth called by both:
      - waveform.assemble_packet()  (TX side: how much to pad)
      - demod.parse_signal_field()  (RX side: how many symbols to demodulate)

    Args:
        n_payload_bytes: Pre-CRC payload byte count (as encoded in SIGNAL field).
        modulation:      "BPSK" | "QPSK" | "16QAM".
    Returns:
        Integer number of full DATA OFDM symbols.
    """
    bps        = _BPS[modulation]
    wire_bits  = n_payload_bytes * 8 + 32   # payload + CRC-32
    coded_bits = wire_bits * 2              # rate-1/2
    return math.ceil(coded_bits / (bps * cfg.NUM_DATA))


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL Field Parser  (Bug 4)
# ─────────────────────────────────────────────────────────────────────────────

def parse_signal_field(Y_sig_fft: np.ndarray,
                       H_hat: np.ndarray) -> dict:
    """
    Decode the SIGNAL OFDM symbol and return payload length + modulation.

    TX packs into 50 info bits (see waveform.generate_signal_symbol):
        bits[0:4]  = rate indicator (4 bits, maps to _RATE_MAP)
        bits[4:16] = PSDU length in bytes, big-endian 12-bit unsigned
        bits[16:]  = zero (tail/reserved)
    Encoding: rate-1/2, NOT scrambled.

    Algorithm:
      1. ZF-equalize data subcarriers using H_hat.
      2. Hard-decision BPSK demap (SIGNAL is always BPSK).
      3. Viterbi decode 98 received coded bits (one per DATA SC) -> 50 info bits.
      4. Extract rate + length; look up modulation; compute n_data_syms.

    Args:
        Y_sig_fft: complex64 (FFT_SIZE,) — FFT of one SIGNAL OFDM symbol
                   after CFO correction.
        H_hat:     complex64 (NUM_ACTIVE,) — channel estimate from LTF.
    Returns:
        dict with keys:
            'n_payload_bytes' : int   — pre-CRC payload byte count
            'modulation'      : str   — "BPSK"/"QPSK"/"16QAM", or None if unknown
            'n_data_syms'     : int   — DATA OFDM symbols needed (0 if invalid)
            'rate_bits'       : tuple — raw 4-bit rate field (for diagnostics)
            'valid'           : bool  — False if rate code was not recognised
    """
    # Step 1 — ZF equalize data subcarriers
    H_data = H_hat[_DATA_MASK].astype(np.complex64)
    denom  = np.abs(H_data) ** 2 + 1e-10
    Y_data = np.array([Y_sig_fft[cfg.k_to_bin(k)] for k in cfg.DATA_INDICES],
                      dtype=np.complex64)
    equ    = (Y_data * np.conj(H_data) / denom)

    # Step 2 — Hard-decision BPSK (0→+1 mapped, 1→-1 mapped)
    rx_coded = (equ.real < 0).astype(np.uint8)   # shape (NUM_DATA=98,)

    # Step 3 — Viterbi decode
    # TX produced 100 coded bits from 50 info bits; we have 98 received coded
    # bits (one per DATA SC). Append two zeros to make 100 before Viterbi decode.
    padded = np.concatenate([rx_coded,
                             np.zeros(100 - len(rx_coded), dtype=np.uint8)])
    dec = scrambler.decode_half_rate(padded)    # → 50 info bits

    # Step 4 — Unpack rate + length
    rate_bits   = tuple(int(b) for b in dec[:4])
    length_bits = dec[4:16]          # 12 bits MSB-first → 0…4095 bytes

    # Re-pack 12 length bits into a 12-bit unsigned integer
    # np.packbits pads to byte boundary on the right, so prepend 4 zeros → 16 bits
    len_16 = np.packbits(
        np.concatenate([np.zeros(4, dtype=np.uint8), length_bits])
    ).view(np.uint16).byteswap()     # big-endian
    n_payload_bytes = int(len_16[0])

    modulation  = _RATE_MAP.get(rate_bits, None)
    n_data_syms = (n_data_syms_for_payload(n_payload_bytes, modulation)
                   if modulation else 0)

    return {
        "n_payload_bytes": n_payload_bytes,
        "modulation":      modulation,
        "n_data_syms":     n_data_syms,
        "rate_bits":       rate_bits,
        "valid":           modulation is not None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Equalizer — zero-forcing per subcarrier  (Section 7)
# ─────────────────────────────────────────────────────────────────────────────

def equalize(Y_data: np.ndarray, H_hat: np.ndarray) -> np.ndarray:
    """
    Zero-forcing equalization per data subcarrier (vectorized).

    Args:
        Y_data: complex64 (NUM_DATA,) — received data subcarrier values.
        H_hat:  complex64 (NUM_ACTIVE,) — channel estimate from LTF.
    Returns:
        Equalized complex symbols, shape (NUM_DATA,).
    """
    H_data = H_hat[_DATA_MASK].astype(np.complex64)
    denom  = np.abs(H_data) ** 2 + 1e-10
    return (Y_data * np.conj(H_data) / denom).astype(np.complex64)


# ─────────────────────────────────────────────────────────────────────────────
# Full RX demodulation chain
# ─────────────────────────────────────────────────────────────────────────────

def demodulate_packet(data_ffts: list,
                      H_hat: np.ndarray,
                      modulation: str = "BPSK",
                      n_payload_bytes: int = None) -> tuple:
    """
    Full demodulation: equalize → demap → Viterbi → descramble → CRC.

    Bug 4 fix: n_payload_bytes (from parse_signal_field) is used to trim the
    coded-bit stream to exactly wire_bits*2 bits before Viterbi decoding.
    Without this trim the decoder sees zero-padding as data and the CRC always
    fails.

    Args:
        data_ffts:       List of complex64 FFT arrays (one per DATA symbol).
        H_hat:           Channel estimate (NUM_ACTIVE,) from sync.py.
        modulation:      "BPSK" | "QPSK" | "16QAM".
        n_payload_bytes: Pre-CRC payload byte count from parse_signal_field().
                         Pass None to skip trimming (legacy / test mode).
    Returns:
        (payload_bits, crc_ok): recovered payload bits (CRC stripped) and bool.
    """
    # Equalize and collect data subcarrier values across all DATA symbols
    all_equ = []
    for Y in data_ffts:
        Y_data = np.array([Y[cfg.k_to_bin(k)] for k in cfg.DATA_INDICES],
                          dtype=np.complex64)
        all_equ.append(equalize(Y_data, H_hat))

    equ_syms  = np.array(all_equ, dtype=np.complex64)   # (Nsym, NUM_DATA)
    soft_bits = scrambler.demap_symbols_to_bits(equ_syms, modulation)

    # Trim to exact coded-bit count before Viterbi  (Bug 4 fix)
    if n_payload_bytes is not None:
        wire_bits   = n_payload_bytes * 8 + 32   # payload + CRC-32
        coded_count = wire_bits * 2              # rate-1/2
        soft_bits   = soft_bits[:coded_count]

    dec_bits = scrambler.decode_half_rate(soft_bits)
    rx_bits  = scrambler.descramble(dec_bits)

    payload_bits, crc_ok = scrambler.crc32_check(rx_bits)
    return payload_bits, crc_ok
