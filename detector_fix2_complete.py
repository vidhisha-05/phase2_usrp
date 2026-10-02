"""
detector.py — STF-based sliding-window autocorrelation packet detector
              and coarse CFO estimation.

Fix 2A: absolute-index advance is bounded by the currently buffered samples.
Fix 2B: after a detection, suppress all candidate detections until the end
         of the maximum supported packet in absolute sample coordinates.
"""

import numpy as np
import config as cfg

_MAX_DATA_SYMS = 34   # n_data_syms_for_payload(200, 'BPSK')

STF_PERIOD    = 16
STF_REPS      = 8
CORR_WINDOW   = STF_PERIOD
DETECT_THRESH = 0.65


class PacketDetector:
    """Streaming STF detector with persistent absolute sample indexing."""

    def __init__(self, threshold: float = DETECT_THRESH, fs: float = cfg.FS_FFT):
        self.threshold = threshold
        self.fs = fs
        self._buf = np.array([], dtype=np.complex64)
        self._sample_idx = 0   # absolute sample index of buf[0]
        self._suppress_until = 0   # absolute coordinate: do not detect before this

    def process(self, samples: np.ndarray) -> list:
        """Append IQ samples and return (absolute_start, coarse_cfo_hz)."""
        self._buf = np.concatenate([self._buf, samples.astype(np.complex64)])
        detections = []

        min_len = cfg.STF_LEN + cfg.LTF_LEN
        while len(self._buf) >= min_len:
            # FIX 2B: if the current buffer begins inside a packet that was
            # already detected, consume samples up to the absolute end of the
            # maximum supported packet before attempting another correlation.
            if self._sample_idx < self._suppress_until:
                skip = min(
                    len(self._buf),
                    self._suppress_until - self._sample_idx,
                )
                self._buf = self._buf[skip:]
                self._sample_idx += skip
                continue

            P, R = self._sliding_corr(self._buf[:min_len])
            metric = np.abs(P).astype(np.float64) ** 2 / (R.astype(np.float64) ** 2 + 1e-12)

            over_thresh = np.where(metric >= self.threshold)[0]
            if len(over_thresh) > 0:
                peak_idx = int(over_thresh[0])
                abs_start = self._sample_idx + peak_idx
                coarse_cfo = self._estimate_coarse_cfo(
                    self._buf[peak_idx:peak_idx + cfg.STF_LEN]
                )
                detections.append((abs_start, coarse_cfo))

                # Maximum packet extent in absolute detector coordinates:
                # STF + LTF + SIGNAL + 34 DATA symbols =
                # 128 + 320 + 160 + 34*160 = 6048 samples.
                packet_len_max = (
                    cfg.STF_LEN
                    + cfg.LTF_LEN
                    + cfg.SIG_LEN
                    + _MAX_DATA_SYMS * cfg.SYMBOL_LEN
                )
                self._suppress_until = abs_start + packet_len_max

                # FIX 2A: consume only what is currently buffered.  If the
                # packet extends beyond this call's buffer, the next process()
                # call resumes at the same absolute coordinate and FIX 2B
                # continues suppression there.
                advance = min(
                    peak_idx + packet_len_max,
                    len(self._buf),
                )
            else:
                advance = max(1, len(metric))

            self._buf = self._buf[advance:]
            self._sample_idx += advance

        return detections

    def _sliding_corr(self, buf: np.ndarray):
        """Compute Schmidl & Cox sliding autocorrelation over buf."""
        L = CORR_WINDOW
        N = len(buf) - L
        if N <= 0:
            return np.zeros(1, dtype=np.complex64), np.zeros(1, dtype=np.float32)

        s = buf.strides[0]
        A = np.lib.stride_tricks.as_strided(
            buf, shape=(N, L), strides=(s, s)
        )
        B = np.lib.stride_tricks.as_strided(
            buf[L:], shape=(N, L), strides=(s, s)
        )

        P = (A * np.conj(B)).sum(axis=1).astype(np.complex64)
        R = (np.abs(B).astype(np.float64) ** 2).sum(axis=1)
        return P, R

    def _estimate_coarse_cfo(self, stf: np.ndarray) -> float:
        """Estimate CFO from phase rotation between 16-sample STF repetitions."""
        L = STF_PERIOD
        reps = 5
        seg = stf[:L * reps]
        phi = np.angle(np.sum(seg[L:] * np.conj(seg[:-L])))
        cfo = phi / (2 * np.pi * L / self.fs)
        return float(cfo)
