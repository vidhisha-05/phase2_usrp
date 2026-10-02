"""
detector.py — STF-based sliding-window autocorrelation packet detector
              and coarse CFO estimation.

Bug fix: Advance past detected packet now uses a conservative maximum
(MAX_DATA_SYMS = 34 symbols = 200-byte BPSK payload) instead of the
previous hardcoded value of 17 (valid only for 100B BPSK). This ensures
the detector correctly skips the full packet regardless of payload size.

Reference: phase2 (2).md — Sections 5, 7.1, 12
"""

import numpy as np
import config as cfg

# Conservative maximum DATA symbols to advance past: 200B @ BPSK.
# n_data_syms_for_payload(200, 'BPSK') = ceil(2*(200*8+32)*2 / (98*1)) = 34.
# Using a constant avoids importing demod here (circular-import risk).
# Any real packet with <=200B payload will be fully skipped by this advance.
_MAX_DATA_SYMS = 34   # = n_data_syms_for_payload(200, 'BPSK')

# ─────────────────────────────────────────────────────────────────────────────
# Constants derived from STF structure
# ─────────────────────────────────────────────────────────────────────────────
STF_PERIOD    = 16      # 16-sample repeating unit in the STF
STF_REPS      = 8       # 8 repetitions
CORR_WINDOW   = STF_PERIOD   # autocorrelation window length
DETECT_THRESH = 0.65    # Normalized autocorrelation magnitude threshold


# ─────────────────────────────────────────────────────────────────────────────
# Sliding-window autocorrelation packet detector (Section 7.1 coarse CFO)
# ─────────────────────────────────────────────────────────────────────────────

class PacketDetector:
    """
    Streaming packet detector using STF sliding-window autocorrelation.

    Usage:
        det = PacketDetector()
        for chunk in sample_stream:
            pkts = det.process(chunk)   # list of (start_sample, coarse_cfo_hz) tuples
    """

    def __init__(self, threshold: float = DETECT_THRESH, fs: float = cfg.FS_FFT):
        self.threshold   = threshold
        self.fs          = fs
        self._buf        = np.array([], dtype=np.complex64)
        self._sample_idx = 0   # absolute sample index of buf[0]
        self._suppress_until = 0

    def process(self, samples: np.ndarray) -> list:
        """
        Append samples to internal buffer and scan for packet starts.

        Args:
            samples: complex64 array of new IQ samples at FS_FFT rate.
        Returns:
            List of (abs_sample_start, coarse_cfo_hz) tuples for detected packets.
        """
        self._buf = np.concatenate([self._buf, samples.astype(np.complex64)])
        detections = []

        # Need at least STF_LEN + look-ahead for full packet detection
        min_len = cfg.STF_LEN + cfg.LTF_LEN
        while len(self._buf) >= min_len:
            P, R = self._sliding_corr(self._buf[:min_len])
            metric = np.abs(P).astype(np.float64) ** 2 / (R.astype(np.float64) ** 2 + 1e-12)

            over_thresh = np.where(metric >= self.threshold)[0]
            if len(over_thresh) > 0:
                peak_idx   = int(over_thresh[0])
                abs_start  = self._sample_idx + peak_idx
                coarse_cfo = self._estimate_coarse_cfo(
                    self._buf[peak_idx:peak_idx + cfg.STF_LEN])
                detections.append((abs_start, coarse_cfo))
                # Advance past full preamble + max possible DATA payload.
                # _MAX_DATA_SYMS=34 covers the largest supported payload (200B BPSK).
                # Previous code had 17 hardcoded (100B BPSK only) — bug fixed.
                advance = min(
                    peak_idx
                    + cfg.STF_LEN
                    + cfg.LTF_LEN
                    + cfg.SIG_LEN
                    + _MAX_DATA_SYMS * cfg.SYMBOL_LEN,
                    len(self._buf)
                )
            else:
                advance = max(1, len(metric))

            self._buf        = self._buf[advance:]
            self._sample_idx += advance

        return detections

    # ── Internal helpers ──────────────────────────────────────────────────

    def _sliding_corr(self, buf: np.ndarray):
        """
        Compute Schmidl & Cox sliding autocorrelation over buf.
        Returns arrays P (cross-correlation) and R (energy).

        FIX: fully vectorized with NumPy stride tricks — no Python loop per sample.
        """
        L  = CORR_WINDOW
        N  = len(buf) - L   # correct: len-L valid windows of length L each
        if N <= 0:
            return np.zeros(1, dtype=np.complex64), np.zeros(1, dtype=np.float32)

        # Build (N, L) view without copying using stride tricks
        s  = buf.strides[0]
        A  = np.lib.stride_tricks.as_strided(buf,       shape=(N, L), strides=(s, s))
        B  = np.lib.stride_tricks.as_strided(buf[L:],   shape=(N, L), strides=(s, s))

        P = (A * np.conj(B)).sum(axis=1).astype(np.complex64)
        R = (np.abs(B).astype(np.float64) ** 2).sum(axis=1)   # float64 throughout: prevents overflow in R and R**2
        return P, R

    def _estimate_coarse_cfo(self, stf: np.ndarray) -> float:
        """
        Coarse CFO estimate from phase rotation between adjacent 16-sample
        STF repetitions (Section 7.1).

        Uses 5 repetitions (80 samples) to remain safely within the STF
        even when peak_idx jitters across the Schmidl-Cox plateau.
        """
        L   = STF_PERIOD
        reps = 5
        seg = stf[:L * reps]
        phi = np.angle(np.sum(seg[L:] * np.conj(seg[:-L])))
        cfo = phi / (2 * np.pi * L / self.fs)
        return float(cfo)
