"""
config.py — Single source of truth for all custom 128-point OFDM PHY numerology.
Custom Wi-Fi-like waveform (NOT IEEE 802.11a/g).

Reference: phase2 (2).md — Sections 3, 4, 10
"""

import numpy as np

# ─────────────────────────────────────────────
# OFDM Numerology (Section 3)
# ─────────────────────────────────────────────
FFT_SIZE        = 128           # N: 128-point FFT
CP_LEN          = 32            # Cyclic prefix length (1/4 ratio)
SYMBOL_LEN      = FFT_SIZE + CP_LEN  # 160 samples per OFDM symbol
FS_HW           = 25e6          # Hardware sample rate (USRP B210)
FS_FFT          = 20e6          # FFT-rate sample rate (post-resampling)
RESAMP_INTERP   = 4             # Rational resampler: interpolate by 4
RESAMP_DECIM    = 5             # Rational resampler: decimate by 5
SUBCARRIER_SPACING = FS_FFT / FFT_SIZE  # 156.25 kHz

# ─────────────────────────────────────────────
# Subcarrier Allocation (Section 4)
# Centered notation: k in {-64, ..., 63}
# ─────────────────────────────────────────────
# Guard bands: lower k in [-64, -54] (10 SCs) and upper k in [54, 63] (10 SCs)
LOWER_GUARD = list(range(-64, -53))   # 11 subcarriers (k=-64..-54 inclusive)
UPPER_GUARD = list(range(54, 64))     # 10 subcarriers (k=54..63)
DC_NULL     = [0]                      # DC subcarrier (nulled, k=0)

# Active region: k in {-53..+53} \ {0} — symmetric, 106 active subcarriers
# FIX (Bug 1): previous range(-54, 0) was asymmetric (54 negative, 53 positive).
# Correct allocation per original design doc: k in {-53..-1} u {1..53} = 106 total.
ACTIVE_SUBCARRIERS = list(range(-53, 0)) + list(range(1, 54))  # 106 total
NUM_ACTIVE = len(ACTIVE_SUBCARRIERS)  # 106

# Pilot subcarrier indices (Section 4)
# FIX (Bug 2): previous pilots [-52,-38,-24,-10,6,20,34,48] were asymmetric and
# conflicted with the original design doc spec of {±7, ±21, ±35, ±49}.
PILOT_INDICES = [-49, -35, -21, -7, 7, 21, 35, 49]   # symmetric ±7/±21/±35/±49
NUM_PILOTS    = len(PILOT_INDICES)    # 8

# Known BPSK polarity for pilots (fixed sequence, same pattern as before)
PILOT_POLARITY = np.array([1, 1, 1, 1, -1, 1, -1, 1], dtype=np.float32)

# Data subcarrier indices (active - pilots)
# 106 active − 8 pilots = 98 data subcarriers
DATA_INDICES = [k for k in ACTIVE_SUBCARRIERS if k not in PILOT_INDICES]
NUM_DATA     = len(DATA_INDICES)   # 98

# ─────────────────────────────────────────────
# Frame Structure (Section 5)
# ─────────────────────────────────────────────
STF_LEN  = 128   # 8 repetitions x 16-sample sequence
LTF_LEN  = 320   # 64-sample CP + 2 x 128-sample LTF
SIG_LEN  = 160   # 1 OFDM symbol (SIGNAL field, BPSK rate-1/2)
# DATA field: 160 x Nsym samples
# Hardware packet-size contract
# The detector supports at most 34 DATA OFDM symbols.
#
# With 98 data subcarriers/symbol and rate-1/2 coding:
#
#   coded_bits = 2 * (8 * payload_bytes + 32)
#
# Maximum payload depends on modulation because each data
# subcarrier carries a different number of coded bits.
#
# BPSK : 1 bit/subcarrier -> 204 bytes maximum
# QPSK : 2 bits/subcarrier -> 412 bytes maximum
# 16QAM: 4 bits/subcarrier -> 829 bytes maximum
MAX_PAYLOAD_BYTES = 829
MAX_DATA_SYMS      = 34

MAX_PAYLOAD_BYTES_BY_MODULATION = {
    "BPSK": 204,
    "QPSK": 412,
    "16QAM": 829,
}
# Packet detector persistence
# Require this many consecutive correlation-metric samples
# above DETECT_THRESH before declaring a packet.
#
# 8 samples at FS_FFT = 20 MHz corresponds to:
# 8 / 20e6 = 0.4 us.
#
# 3F-1C characterization:
#   Worst observed noise run   = 6 samples
#   Worst observed packet run  = 102 samples
# Therefore 8 samples provides measured separation in the
# tested AWGN detector-window characterization.
# Require this many consecutive correlation-metric samples
# above threshold before declaring an STF detection.
#
# At FS_FFT = 20 MHz:
#   8 samples  = 0.40 us
#   16 samples = 0.80 us
#
# 3F-1C detector-window characterization:
#   packet threshold-run minimum = 102 samples
#   noise threshold-run maximum   = 6 samples
#
# Exact 15 dB production-detector failed trial:
#   persistence 8  -> false detection at n=578
#   persistence 16 -> valid packet detection at n=998
DETECT_PERSISTENCE = 16
# ─────────────────────────────────────────────
# ZMQ Addresses (Section 8)
# ─────────────────────────────────────────────
ZMQ_TX_ADDR      = "tcp://127.0.0.1:5555"   # TX PUSH
ZMQ_RX_ADDR      = "tcp://127.0.0.1:5556"   # Channel bridge PUSH -> RX PULL
ZMQ_ANT1_ADDR    = "tcp://127.0.0.1:5557"   # Bridge -> RX: clean TX for ant-1 parallel channel
ZMQ_CHUNK_SIZE   = 4096                       # IQ samples per ZMQ message

# ─────────────────────────────────────────────
# Hardware (Section 9)  — 1-TX / 1-RX B210 OTA Deployment
# ─────────────────────────────────────────────
USRP_CENTER_FREQ = 2.412e9   # Channel 1 ISM band (2.412 GHz)
USRP_LO_OFFSET   = 8.75e6    # 8.75 MHz CORDIC digital off-tuning → k=-56 (inside lower guard k=-64..-54)
                              # CRITICAL: 5.0 MHz would map to k=-32 which IS an active data SC — DO NOT use 5 MHz.
USRP_TX_GAIN     = 30.0      # dB (calibrated for 0.5-1.0m LoS OTA deployment)
USRP_RX_GAIN     = 25.0      # dB (prevents AD9361 ADC clipping while preserving micro-reflection dynamic range)
USRP_SUBDEV_SPEC = "A:A"     # Single RX channel on Subdev A
NUM_RX_CHANNELS  = 1

# ─────────────────────────────────────────────
# Convolutional Coder (Section 6)
# ─────────────────────────────────────────────
CONV_K        = 7        # Constraint length
CONV_RATE_INV = 2        # Mother rate 1/2 (rate = 1/CONV_RATE_INV)
CONV_GEN      = [0o133, 0o171]  # Generator polynomials (octal), matching 802.11

# ─────────────────────────────────────────────
# HDF5 Storage (Section 14)
# ─────────────────────────────────────────────
HDF5_FLUSH_INTERVAL = 50   # Flush every N packets
HDF5_FILE_PATH      = "csi_data.h5"

# ─────────────────────────────────────────────
# Deployment mode flag
# ─────────────────────────────────────────────
MODE = "hardware"   # "simulation" | "hardware"

# ─────────────────────────────────────────────
# Guard Interval (Section 9 / B210 deployment)
# ─────────────────────────────────────────────
# Number of zero samples prepended to every TX packet burst at 25 MS/s.
# 1024 samples = 40.96 µs — ensures the RX Schmidl-Cox correlator has a
# clean blank reference window before the STF, eliminating the false-trigger-
# on-noise failure mode observed at < 128-sample lead-ins.
# Set to 0 only for back-to-back simulation benchmarks.
GUARD_SAMPLES    = 1024   # at FS_HW (25 MS/s)
GUARD_SAMPLES_BB = 820    # at FS_FFT (20 MS/s): ceil(GUARD_SAMPLES * 4/5) = ceil(819.2) = 820
# TX packet timing contract
# Packet starts are scheduled at a fixed 5 ms interval.
# This gives a nominal CSI sampling rate of 200 Hz.
TX_PACKET_PERIOD_S = 0.005
TX_PACKET_RATE_HZ  = 1.0 / TX_PACKET_PERIOD_S

# ─────────────────────────────────────────────
# Helper: centered-index -> FFT bin mapping
# k in {-64...63} -> bin in {0...127} via (k % FFT_SIZE)
# ─────────────────────────────────────────────
def k_to_bin(k: int) -> int:
    """Map centered subcarrier index k to NumPy FFT bin index."""
    return int(k % FFT_SIZE)

ACTIVE_BINS = np.array([k_to_bin(k) for k in ACTIVE_SUBCARRIERS])
PILOT_BINS  = np.array([k_to_bin(k) for k in PILOT_INDICES])
DATA_BINS   = np.array([k_to_bin(k) for k in DATA_INDICES])
