"""
scrambler.py — Scrambler/descrambler, convolutional encoder/decoder,
               bit interleaver, CRC-32, and QAM mapper for the custom
               128-pt OFDM PHY.

Generator polynomial: S(x) = x^7 + x^4 + 1  (same as 802.11 convention)
Convolutional code:   K=7, rate 1/2, generators [0o133, 0o171]
                      Puncturable to 3/4 (Section 6).

Fixes applied (v2):
  - interleave/deinterleave now invertible for all lengths (was [:len-pad] bug)
  - crc32_append / crc32_check added as canonical API
  - 16-QAM remapped to standard Gray-coded square constellation

Reference: phase2 (2).md — Section 6
"""

import binascii
import numpy as np
from typing import Optional
import config as cfg

# ─────────────────────────────────────────────────────────────────────────────
# 1. SCRAMBLER / DESCRAMBLER  (x^7 + x^4 + 1, LFSR length 127)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_SEED = 0b1011101   # 7-bit non-zero seed


# ─────────────────────────────────────────────────────────────────────────────
# 2. CRC-32  (IEEE 802.3 polynomial — canonical API for TX and RX)
# ─────────────────────────────────────────────────────────────────────────────

def crc32_append(bits: np.ndarray) -> np.ndarray:
    """
    Compute IEEE 802.3 CRC-32 over *bits* and append as 32 bits (MSB-first).
    Bit array is zero-padded to whole bytes before CRC; the same padding is
    applied consistently by crc32_check so the round-trip is exact.
    """
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    N    = len(bits)
    pad  = (-N) % 8
    data = np.packbits(np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])).tobytes()
    crc_val  = binascii.crc32(data) & 0xFFFF_FFFF
    # Use size-1 array to allow .view(np.uint8) (0-d arrays do not support view change)
    crc_bytes = np.array([crc_val], dtype=np.uint32).byteswap().view(np.uint8)
    crc_bits  = np.unpackbits(crc_bytes)
    return np.concatenate([bits, crc_bits]).astype(np.uint8)


def crc32_check(bits: np.ndarray) -> tuple:
    """
    Verify and strip 32-bit CRC appended by crc32_append().
    Returns: (data_bits, ok) — payload without CRC, and True if CRC matched.
    """
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    if len(bits) < 32:
        return bits, False
    data_bits = bits[:-32]
    expected  = crc32_append(data_bits)
    ok        = bool(np.all(bits[-32:] == expected[-32:]))
    return data_bits, ok


def _lfsr_seq(n_bits: int, seed: int = DEFAULT_SEED) -> np.ndarray:
    """Generate n_bits of the 127-length LFSR pseudo-random sequence."""
    state = seed & 0x7F
    seq   = np.empty(n_bits, dtype=np.uint8)
    for i in range(n_bits):
        bit     = ((state >> 6) ^ (state >> 3)) & 1   # x^7 XOR x^4
        seq[i]  = bit
        state   = ((state << 1) | bit) & 0x7F
    return seq


def scramble(bits: np.ndarray, seed: int = DEFAULT_SEED) -> np.ndarray:
    """XOR input bits with LFSR sequence."""
    pn = _lfsr_seq(len(bits), seed)
    return (bits ^ pn).astype(np.uint8)


def descramble(bits: np.ndarray, seed: int = DEFAULT_SEED) -> np.ndarray:
    """Descramble (identical operation to scramble for XOR scramblers)."""
    return scramble(bits, seed)


# ─────────────────────────────────────────────────────────────────────────────
# 2. CONVOLUTIONAL ENCODER (K=7, rate 1/2)
#    Generators: g0=0o133 (91 decimal), g1=0o171 (121 decimal)
# ─────────────────────────────────────────────────────────────────────────────

G0 = 0o133   # 1011011
G1 = 0o171   # 1111001


def _popcount(x: int) -> int:
    return bin(x).count('1')


def encode_half_rate(bits: np.ndarray) -> np.ndarray:
    """
    Rate-1/2 convolutional encoder, K=7.
    Input: binary array of length N.
    Output: binary array of length 2N (interleaved systematic + parity).
    """
    bits   = np.asarray(bits, dtype=np.uint8).ravel()
    n      = len(bits)
    out    = np.empty(2 * n, dtype=np.uint8)
    state  = 0
    K      = cfg.CONV_K          # 7
    mask   = (1 << K) - 1
    for i, b in enumerate(bits):
        state = ((state << 1) | int(b)) & mask
        out[2 * i]     = _popcount(state & G0) % 2
        out[2 * i + 1] = _popcount(state & G1) % 2
    return out


def puncture_to_3_4(coded: np.ndarray) -> np.ndarray:
    """
    Puncture a rate-1/2 stream to rate-3/4.
    Puncturing pattern per 802.11 convention:
      keep  bits: 1 1 1 0 0 1  (period 6, keep 4 → rate 3/4)
    Input length must be divisible by 6.
    """
    pattern = np.array([1, 1, 1, 0, 0, 1], dtype=bool)
    tiles   = np.tile(pattern, len(coded) // 6 + 1)[:len(coded)]
    return coded[tiles]


# ─────────────────────────────────────────────────────────────────────────────
# 3. VITERBI DECODER (K=7, rate 1/2)
# ─────────────────────────────────────────────────────────────────────────────

def _branch_metrics(state: int, bit: int) -> tuple:
    """Return (out0, out1) for given state and input bit (hard metric)."""
    s    = ((state << 1) | bit) & ((1 << cfg.CONV_K) - 1)
    out0 = _popcount(s & G0) % 2
    out1 = _popcount(s & G1) % 2
    return out0, out1


def decode_half_rate(coded_bits: np.ndarray) -> np.ndarray:
    """
    Hard-decision Viterbi decoder for rate-1/2 K=7 code.
    Input:  binary array of length 2N.
    Output: binary array of length N (decoded bits).
    """
    coded  = np.asarray(coded_bits, dtype=np.uint8).ravel()
    n      = len(coded) // 2
    n_st   = 1 << (cfg.CONV_K - 1)    # 64 states
    INF    = 9999

    # path_metric[state] = accumulated Hamming distance
    pm      = np.full(n_st, INF, dtype=int)
    pm[0]   = 0
    # history stores (predecessor_state, input_bit) for each (time, state)
    pred    = np.zeros((n, n_st), dtype=np.int32)   # predecessor state
    inbit   = np.zeros((n, n_st), dtype=np.uint8)   # input bit that caused transition

    for i in range(n):
        r0, r1  = int(coded[2 * i]), int(coded[2 * i + 1])
        new_pm  = np.full(n_st, INF, dtype=int)
        for s in range(n_st):
            if pm[s] == INF:
                continue
            for b in (0, 1):
                ns         = ((s << 1) | b) & (n_st - 1)
                out0, out1 = _branch_metrics(s, b)
                metric     = pm[s] + (out0 ^ r0) + (out1 ^ r1)
                if metric < new_pm[ns]:
                    new_pm[ns]  = metric
                    pred[i, ns] = s
                    inbit[i, ns] = b
        pm = new_pm

    # Traceback — recover decoded bits from input-bit history
    decoded = np.empty(n, dtype=np.uint8)
    state   = int(np.argmin(pm))
    for i in range(n - 1, -1, -1):
        decoded[i] = inbit[i, state]
        state      = pred[i, state]
    return decoded


# ─────────────────────────────────────────────────────────────────────────────
# 5. BIT INTERLEAVER / DE-INTERLEAVER
#    Block interleaver: write row-wise (n_rows × n_cols), read column-wise.
#
#    FIX: old deinterleave sliced [:len(bits)-pad], losing bits when pad>0.
#         Correct inverse is always [:N].  Verified: deinterleave(interleave(x))==x
#         for all lengths including non-multiples of n_cols.
# ─────────────────────────────────────────────────────────────────────────────

def _interleave_perm(N: int, n_cols: int) -> np.ndarray:
    """
    Return the permutation index array perm such that
    interleaved[i] = original[perm[i]].

    Algorithm: read a (n_rows x n_cols) matrix column-by-column,
    skipping padding positions (index >= N).
    """
    n_rows = (N + n_cols - 1) // n_cols
    perm   = []
    for col in range(n_cols):
        for row in range(n_rows):
            src = row * n_cols + col
            if src < N:
                perm.append(src)
    return np.array(perm, dtype=np.intp)


def interleave(bits: np.ndarray, n_cols: int = cfg.NUM_DATA) -> np.ndarray:
    """
    Block interleave bits: write (n_rows x n_cols) row-wise, read column-wise,
    skipping padding cells.

    Invariant: deinterleave(interleave(x)) == x  for all N >= 1.
    """
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    N    = len(bits)
    perm = _interleave_perm(N, n_cols)
    return bits[perm]


def deinterleave(bits: np.ndarray, n_cols: int = cfg.NUM_DATA) -> np.ndarray:
    """
    Block de-interleave bits (exact inverse of interleave).

    Invariant: deinterleave(interleave(x)) == x  for all N >= 1.
    """
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    N    = len(bits)
    perm  = _interleave_perm(N, n_cols)
    out   = np.empty(N, dtype=np.uint8)
    out[perm] = bits          # inverse permutation: out[src] = interleaved[i]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 6. QAM MAPPER / DEMAPPER
#
#    FIX: 16-QAM now uses standard Gray-coded square constellation.
#         b3 b2 → I-axis:  00→-3  01→-1  11→+1  10→+3
#         b1 b0 → Q-axis:  same mapping
#         Index = b3 b2 b1 b0 (MSB first, 0..15)
# ─────────────────────────────────────────────────────────────────────────────

_BPSK_MAP  = np.array([1. + 0j, -1. + 0j], dtype=np.complex64)
_QPSK_MAP  = np.array([1. + 1j, -1. + 1j, 1. - 1j, -1. - 1j],
                       dtype=np.complex64) / np.sqrt(2)

_GRAY_AXIS = {0b00: -3, 0b01: -1, 0b11: +1, 0b10: +3}   # 2-bit Gray → amplitude

def _build_qam16_gray() -> np.ndarray:
    pts = np.empty(16, dtype=np.complex64)
    for idx in range(16):
        I = _GRAY_AXIS[(idx >> 2) & 0x3]   # b3 b2
        Q = _GRAY_AXIS[ idx       & 0x3]   # b1 b0
        pts[idx] = complex(I, Q) / np.sqrt(10)
    return pts

_QAM16_MAP = _build_qam16_gray()


def map_bits_to_symbols(coded_bits: np.ndarray, modulation: str = "BPSK",
                         n_data: int = cfg.NUM_DATA) -> np.ndarray:
    """
    Map interleaved coded bits to QAM symbols.
    Returns complex64 array of shape (Nsym, n_data).
    """
    bps_map = {"BPSK": (1, _BPSK_MAP),
               "QPSK": (2, _QPSK_MAP),
               "16QAM": (4, _QAM16_MAP)}
    bps, cmap = bps_map[modulation]
    bits   = interleave(coded_bits)
    n_sym  = int(np.ceil(len(bits) / (bps * n_data)))
    bits   = np.concatenate([bits, np.zeros(n_sym * bps * n_data - len(bits),
                                             dtype=np.uint8)])
    syms   = np.empty((n_sym, n_data), dtype=np.complex64)
    for i in range(n_sym):
        blk = bits[i * bps * n_data:(i + 1) * bps * n_data].reshape(n_data, bps)
        for j in range(n_data):
            idx = int(''.join(str(b) for b in blk[j]), 2)
            syms[i, j] = cmap[idx]
    return syms


def demap_symbols_to_bits(syms: np.ndarray, modulation: str = "BPSK") -> np.ndarray:
    """
    Hard-decision QAM demapper.
    Input:  complex64 array of shape (Nsym, n_data).
    Output: binary array.
    """
    bps_map = {"BPSK": (1, _BPSK_MAP),
               "QPSK": (2, _QPSK_MAP),
               "16QAM": (4, _QAM16_MAP)}
    bps, cmap = bps_map[modulation]
    flat      = syms.ravel()
    bits_out  = []
    for sym in flat:
        idx = int(np.argmin(np.abs(sym - cmap)))
        bits_out.extend([int(b) for b in format(idx, f'0{bps}b')])
    return deinterleave(np.array(bits_out, dtype=np.uint8))
