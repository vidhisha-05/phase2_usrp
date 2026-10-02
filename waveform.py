"""
waveform.py — STF/LTF/SIGNAL/DATA frame assembly for the custom 128-pt OFDM PHY.
Custom Wi-Fi-like waveform (NOT IEEE 802.11a/g).

Reference: phase2 (2).md — Sections 3, 4, 5
"""

import numpy as np
from scipy.signal import firwin, resample_poly
import config as cfg

# ─────────────────────────────────────────────────────────────────────────────
# 1. RATIONAL RESAMPLER (25 MS/s <-> 20 MS/s)
# ─────────────────────────────────────────────────────────────────────────────


def _build_resamp_filter(interp: int, decim: int, num_taps: int = 64) -> np.ndarray:
    """Design the common anti-aliasing FIR used by both rate converters."""
    cutoff = 1.0 / max(interp, decim)
    h = firwin(num_taps, cutoff, window=('kaiser', 8.0), pass_zero=True)
    return h.astype(np.float64)


RESAMP_FILTER = _build_resamp_filter(cfg.RESAMP_INTERP, cfg.RESAMP_DECIM)

# IMPORTANT:
# The 64-tap FIR has a 31.5-sample group delay at the *upsampled* FIR rate.
# It is NOT correct to subtract 31 samples (or 24 output samples) from every
# independent UHD block.  scipy.signal.resample_poly already centers the FIR
# and aligns the first output with the first input sample.
#
# Keep this name only for compatibility with older diagnostics/imports.
RESAMP_GROUP_DELAY_SAMPLES = (len(RESAMP_FILTER) - 1) // 2


def _resample_poly_one_shot(x: np.ndarray, up: int, down: int) -> np.ndarray:
    """Reference one-shot converter; intentionally no manual delay stripping."""
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("resampler expects a 1-D complex sample vector")
    y = resample_poly(x, up, down, window=RESAMP_FILTER)
    return y.astype(np.complex64, copy=False)


class StreamingRationalResampler:
    """Chunk-invariant streaming implementation of scipy's rational resampler.

    ``process()`` uses a short overlap history plus SciPy's compiled ``upfirdn``
    kernel.  The history is deliberately aligned to the decimation factor, so
    output phase is independent of where UHD happens to split its blocks.
    ``flush()`` releases the finite-stream tail for tests/offline use.
    """

    HISTORY_INPUT_SAMPLES = 40

    def __init__(self, up: int, down: int, taps: np.ndarray = RESAMP_FILTER):
        if up < 1 or down < 1:
            raise ValueError("up and down must be >= 1")
        self.up = int(up)
        self.down = int(down)
        self.taps = np.asarray(taps, dtype=np.float64)
        if self.taps.ndim != 1 or len(self.taps) == 0:
            raise ValueError("taps must be a non-empty 1-D array")

        # Mirror scipy.signal.resample_poly's FIR centering exactly.
        half_len = (len(self.taps) - 1) // 2
        self.pre_pad = self.down - (half_len % self.down)
        self.pre_remove = (half_len + self.pre_pad) // self.down
        self.h_padded = np.concatenate([
            np.zeros(self.pre_pad, dtype=np.float64),
            self.taps * self.up,
        ])

        self._history = np.empty(0, dtype=np.complex64)
        self._input_count = 0
        self._output_count = 0
        self._dtype = np.complex64
        self._closed = False
        self.flush_input_samples = int(np.ceil((len(self.h_padded) - 1) / self.up))

    @property
    def latency_samples(self) -> int:
        """Steady-state output alignment latency in output samples."""
        return self.pre_remove

    @property
    def input_count(self) -> int:
        return self._input_count

    @property
    def output_count(self) -> int:
        return self._output_count

    def reset(self) -> None:
        self._history = np.empty(0, dtype=np.complex64)
        self._input_count = 0
        self._output_count = 0
        self._dtype = np.complex64
        self._closed = False

    def process(self, x: np.ndarray) -> np.ndarray:
        """Consume one input chunk and return all currently stable output samples."""
        if self._closed:
            raise RuntimeError("resampler is closed; call reset() before reuse")
        x = np.asarray(x)
        if x.ndim != 1:
            raise ValueError("resampler expects a 1-D sample vector")
        if len(x) == 0:
            return np.empty(0, dtype=np.complex64)

        self._dtype = np.result_type(self._dtype, x.dtype, np.complex64)
        x = x.astype(self._dtype, copy=False)
        local = np.concatenate([self._history, x]) if len(self._history) else x

        # Choose the retained-history boundary on a decimation-factor multiple.
        # Then global output m maps to local output index
        # m - H*(up/down), with no fractional phase ambiguity.
        H = self._input_count - len(self._history)
        if H < 0 or H % self.down != 0:
            raise RuntimeError("internal resampler history alignment error")

        from scipy.signal import upfirdn
        raw = upfirdn(self.h_padded, local, self.up, self.down)
        offset = (H * self.up) // self.down

        new_input_count = self._input_count + len(x)
        stable_count = max(
            0,
            (new_input_count * self.up + self.down - 1) // self.down
            - self.pre_remove,
        )
        need = stable_count - self._output_count

        if need <= 0:
            out = np.empty(0, dtype=np.complex64)
        else:
            local_out_start = self._output_count - offset
            raw_start = self.pre_remove + local_out_start
            raw_stop = raw_start + need
            vals = raw[raw_start:raw_stop]
            if len(vals) != need:
                raise RuntimeError(
                    f"streaming resampler produced {len(vals)} samples, expected {need}")
            out = vals.astype(np.complex64, copy=False)
            self._output_count = stable_count

        self._input_count = new_input_count

        # Keep enough history for the FIR support, plus N mod down so that the
        # next local window starts on a decimation-phase boundary.
        keep = self.HISTORY_INPUT_SAMPLES + (self._input_count % self.down)
        self._history = local[-keep:].copy() if len(local) > keep else local.copy()
        return out

    def flush(self) -> np.ndarray:
        """Release the finite-stream tail; intended for offline/test streams."""
        if self._closed:
            return np.empty(0, dtype=np.complex64)
        target = (self._input_count * self.up + self.down - 1) // self.down
        zeros = np.zeros(self.flush_input_samples, dtype=self._dtype)
        out = self.process(zeros) if len(zeros) else np.empty(0, dtype=np.complex64)
        needed = max(0, target - (self._output_count - len(out)))
        out = out[:needed]
        self._output_count = target
        self._closed = True
        return out

def resample_25to20(x: np.ndarray) -> np.ndarray:
    """One-shot 25 MS/s -> 20 MS/s reference conversion (no delay stripping)."""
    return _resample_poly_one_shot(x, cfg.RESAMP_INTERP, cfg.RESAMP_DECIM)


# Backward-compatible alias.  The old name remains valid, but now has the
# correct zero-phase one-shot semantics.
upsample_25to20 = resample_25to20


def resample_20to25(x: np.ndarray) -> np.ndarray:
    """One-shot 20 MS/s -> 25 MS/s reference conversion (no delay stripping)."""
    return _resample_poly_one_shot(x, cfg.RESAMP_DECIM, cfg.RESAMP_INTERP)


# Backward-compatible alias.
downsample_20to25 = resample_20to25


def make_streaming_25to20() -> StreamingRationalResampler:
    """Create a persistent RX 25 MS/s -> 20 MS/s converter."""
    return StreamingRationalResampler(cfg.RESAMP_INTERP, cfg.RESAMP_DECIM)


def make_streaming_20to25() -> StreamingRationalResampler:
    """Create a persistent TX 20 MS/s -> 25 MS/s converter."""
    return StreamingRationalResampler(cfg.RESAMP_DECIM, cfg.RESAMP_INTERP)


# ─────────────────────────────────────────────────────────────────────────────
# 2. STF — Short Training Field (Section 5)
#    8 repetitions of a 16-sample Zadoff-Chu-like BPSK sequence.
#    128 samples total.
# ─────────────────────────────────────────────────────────────────────────────

def _build_stf_sequence() -> np.ndarray:
    """
    Build the 16-sample base sequence for the STF.
    Uses a Zadoff-Chu root-1 sequence of length 16, normalized to unit power.
    """
    N = 16
    u = 1  # Zadoff-Chu root
    n = np.arange(N)
    zc = np.exp(-1j * np.pi * u * n * (n + 1) / N).astype(np.complex64)
    zc /= np.sqrt(np.mean(np.abs(zc) ** 2))
    return zc


_STF_BASE = _build_stf_sequence()


def generate_stf() -> np.ndarray:
    """
    Generate the 128-sample STF: 8 identical repetitions of the 16-sample base.
    Returns complex64 array of shape (128,).
    """
    stf = np.tile(_STF_BASE, 8)
    assert len(stf) == cfg.STF_LEN
    return stf.astype(np.complex64)


# ─────────────────────────────────────────────────────────────────────────────
# 3. LTF — Long Training Field (Section 5)
#    64-sample CP + 2 x 128-sample LTF symbol = 320 samples.
#    LTF subcarrier values: +1/-1 BPSK on all active subcarriers.
# ─────────────────────────────────────────────────────────────────────────────

def _build_ltf_freq_domain() -> np.ndarray:
    """
    Build the 128-point LTF frequency-domain template.
    Active subcarriers carry BPSK {+1, -1}; guard bands and DC are 0.
    """
    X = np.zeros(cfg.FFT_SIZE, dtype=np.complex64)
    # BPSK assignment: alternating +1/-1 across active subcarriers
    for i, k in enumerate(cfg.ACTIVE_SUBCARRIERS):
        X[cfg.k_to_bin(k)] = 1.0 if (i % 2 == 0) else -1.0
    return X


LTF_FREQ = _build_ltf_freq_domain()


def generate_ltf() -> np.ndarray:
    """
    Generate the 320-sample LTF: 64-sample CP + 2 x 128-sample OFDM symbols.
    Returns complex64 array of shape (320,).
    """
    ltf_time = np.fft.ifft(LTF_FREQ, n=cfg.FFT_SIZE).astype(np.complex64)
    cp        = ltf_time[-64:]           # Double-length CP (64 samples)
    ltf_field = np.concatenate([cp, ltf_time, ltf_time])
    assert len(ltf_field) == cfg.LTF_LEN
    return ltf_field.astype(np.complex64)


# ─────────────────────────────────────────────────────────────────────────────
# 4. SIGNAL Field  (Section 5)
#    1 OFDM symbol, BPSK, rate-1/2 coded.  Carries rate/length metadata.
# ─────────────────────────────────────────────────────────────────────────────

def generate_signal_symbol(rate_bits: np.ndarray, length_bits: np.ndarray,
                           scrambler_mod, encoder_mod) -> np.ndarray:
    """
    Build the 160-sample SIGNAL OFDM symbol (CP=32 + FFT=128).

    SIGNAL bit-field protocol (Issue 9/10 — explicit contract):

        Info bits fed to the rate-1/2 encoder (50 bits total):
          bits[ 0: 4] — rate indicator (4 bits, maps to modulation)
          bits[ 4:16] — PSDU length in bytes, MSB-first unsigned 12-bit (0…4095)
          bit [16]    — tail/service bit, always 0 at TX
          bits[17:50] — zero padding (tail flush for Viterbi shift register)

        Encoder output: 100 coded bits.

        Transmitted: coded[0:98] on the 98 data subcarriers.
        Dropped:     coded[98]  — this bit is always 0 (tail), so dropping it
                     is lossless.  RX reconstructs 98 coded bits by appending
                     two zeros to reach 100 before Viterbi.  This IS an explicit contract.

        Pilots: same 8 fixed-polarity BPSK pilots as DATA symbols.
        Not scrambled (SIGNAL is always raw encoded, not scrambler-processed).

    Args:
        rate_bits:     np.uint8 array of shape (4,)  — rate indicator.
        length_bits:   np.uint8 array of shape (12,) — PSDU byte count, MSB-first.
        scrambler_mod: module exposing encode_half_rate() (for symmetry; SIGNAL
                       is NOT scrambled, only encoded).
        encoder_mod:   same reference (encode_half_rate() is called here).
    Returns:
        complex64 array of shape (160,) — CP + OFDM symbol.
    """
    # 4 rate + 12 length + 1 tail + 33 zero-padding = 50 bits → 100 coded bits
    raw_bits = np.zeros(50, dtype=np.uint8)
    raw_bits[:4]   = rate_bits
    raw_bits[4:16] = length_bits
    # raw_bits[16] = 0  (tail bit, already zero)
    # raw_bits[17:50] = 0  (Viterbi tail flush)
    coded = encoder_mod.encode_half_rate(raw_bits)  # → 100 coded bits
    # Transmit first 98 on the 98 data SCs (cfg.NUM_DATA=98); bit [98] is 0 (tail), dropped by contract.
    bpsk = 1.0 - 2.0 * coded[:cfg.NUM_DATA].astype(np.float32)  # 0→+1, 1→-1
    return _modulate_one_symbol(bpsk)


def _modulate_one_symbol(data_syms: np.ndarray) -> np.ndarray:
    """
    IFFT + CP for a single OFDM symbol (SIGNAL or DATA).

    Both SIGNAL and DATA symbols use identical pilot insertion: the 8 fixed
    BPSK polarity values from cfg.PILOT_POLARITY at cfg.PILOT_INDICES.
    Issue 11 fix: removed the dead `is_signal` parameter — it had no effect
    on the output and only created a misleading distinction.

    Args:
        data_syms: complex/float array of length NUM_DATA (98 values).
    Returns:
        complex64 of length SYMBOL_LEN (160).
    """
    X = np.zeros(cfg.FFT_SIZE, dtype=np.complex64)
    for i, k in enumerate(cfg.DATA_INDICES):
        X[cfg.k_to_bin(k)] = data_syms[i]
    for i, k in enumerate(cfg.PILOT_INDICES):
        X[cfg.k_to_bin(k)] = cfg.PILOT_POLARITY[i]
    x_time = np.fft.ifft(X, n=cfg.FFT_SIZE).astype(np.complex64)
    cp     = x_time[-cfg.CP_LEN:]
    symbol = np.concatenate([cp, x_time])
    assert len(symbol) == cfg.SYMBOL_LEN
    return symbol.astype(np.complex64)


# ─────────────────────────────────────────────────────────────────────────────
# 5. DATA Field — N_sym OFDM symbols
# ─────────────────────────────────────────────────────────────────────────────

def modulate_data_symbols(qam_symbols: np.ndarray) -> np.ndarray:
    """
    Modulate a block of QAM symbols into DATA OFDM symbols.

    Args:
        qam_symbols: complex array of shape (Nsym, NUM_DATA).
    Returns:
        complex64 array of shape (Nsym * SYMBOL_LEN,).
    """
    Nsym = qam_symbols.shape[0]
    out  = np.empty(Nsym * cfg.SYMBOL_LEN, dtype=np.complex64)
    for m in range(Nsym):
        out[m * cfg.SYMBOL_LEN:(m + 1) * cfg.SYMBOL_LEN] = \
            _modulate_one_symbol(qam_symbols[m])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 6. Full Packet Assembly
# ─────────────────────────────────────────────────────────────────────────────


def assemble_packet(payload_bits,
                    modulation='BPSK',
                    scrambler_mod=None, encoder_mod=None,
                    mapper_fn=None,
                    idle_samples=0):
    """
    Assemble a complete baseband packet:
      [idle zeros] + STF (128) + LTF (320) + SIGNAL (160) + DATA (160*Nsym)

    idle_samples: number of zero samples prepended at 20 MS/s before the STF.
      Set to cfg.GUARD_SAMPLES_BB for B210 TX so the RX Schmidl-Cox correlator
      always has a clean blank reference before the STF.
      Default = 0 (backwards compatible; simulation tests set this explicitly).

    TX<->RX length protocol  (Bug 4 fix — single source of truth):
      1.  wire_bits  = payload_bits + CRC-32                (N+32 bits)
      2.  coded_bits = encode_half_rate(scramble(wire_bits)) (2*(N+32) bits)
      3.  N_sym      = demod.n_data_syms_for_payload(N//8, mod)
      4.  coded_bits is ZERO-PADDED to exactly N_sym * NUM_DATA * bps bits
          before QAM mapping, so the DATA field always contains exactly N_sym
          complete OFDM symbols.
      5.  SIGNAL carries N//8 (pre-CRC bytes); RX calls the same formula to
          determine how many DATA symbols to demodulate.

    Args:
        payload_bits: 1-D array of raw information bits.
        modulation:   "BPSK", "QPSK", or "16QAM".
        scrambler_mod, encoder_mod: scrambler.py module reference.
        mapper_fn:    callable(coded_bits, mod) -> complex QAM symbols (Nsym, NUM_DATA).
        idle_samples: zeros prepended before STF (20 MS/s). Default 0.
    Returns:
        complex64 baseband samples at 20 MS/s.
    """
    import scrambler as _scr   # avoid circular import at module level
    import demod as _demod

    _BPS = {"BPSK": 1, "QPSK": 2, "16QAM": 4}

    stf = generate_stf()
    ltf = generate_ltf()

    if scrambler_mod and encoder_mod and mapper_fn:
        n_payload_bytes = len(payload_bits) // 8   # pre-CRC byte count

        # Step 1: append CRC
        payload_with_crc = _scr.crc32_append(payload_bits)   # N+32 bits

        # Step 2: scramble + encode
        scrambled   = scrambler_mod.scramble(payload_with_crc)
        coded       = encoder_mod.encode_half_rate(scrambled) # 2*(N+32) bits

        # Step 3: compute N_sym using shared formula (Bug 4)
        bps    = _BPS[modulation]
        n_sym  = _demod.n_data_syms_for_payload(n_payload_bytes, modulation)
        target = n_sym * cfg.NUM_DATA * bps   # total coded bits needed

        # Step 4: zero-pad coded bits to exact target length
        if len(coded) < target:
            coded = np.concatenate([coded,
                                    np.zeros(target - len(coded), dtype=np.uint8)])
        else:
            coded = coded[:target]

        # Step 5: QAM map and OFDM modulate
        qam_symbols = mapper_fn(coded, modulation)   # (n_sym, NUM_DATA)
        data_field  = modulate_data_symbols(qam_symbols)

        # SIGNAL field: encode pre-CRC byte count (RX uses same formula)
        rate_map = {"BPSK":  np.array([1, 0, 1, 1], dtype=np.uint8),
                    "QPSK":  np.array([0, 1, 0, 1], dtype=np.uint8),
                    "16QAM": np.array([1, 1, 0, 1], dtype=np.uint8)}
        len_bits = np.unpackbits(
            np.array([n_payload_bytes], dtype=np.uint16).byteswap().view(np.uint8))[-12:]
        sig = generate_signal_symbol(rate_map[modulation], len_bits,
                                     scrambler_mod, encoder_mod)
    else:
        # Minimal packet for loopback testing: no data field
        data_field = np.array([], dtype=np.complex64)
        sig        = np.zeros(cfg.SIG_LEN, dtype=np.complex64)

    guard = np.zeros(int(idle_samples), dtype=np.complex64)
    return np.concatenate([guard, stf, ltf, sig, data_field]).astype(np.complex64)

