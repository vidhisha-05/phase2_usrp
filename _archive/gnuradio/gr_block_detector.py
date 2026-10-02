"""
gr_block_detector.py  --  GNU Radio Python Block: STF Packet Detector
======================================================================
Implements Schmidl & Cox sliding autocorrelation on a continuous IQ
stream. Outputs a tagged stream: passes all samples through, and
attaches a stream tag at the detected packet start with:
  - tag key: "pkt_start"
  - tag value: coarse CFO estimate (Hz, float)

In GNU Radio Companion:
  Type       : Python Block (Embedded / OOT)
  In ports   : 1 x complex64  (IQ stream at 20 MS/s baseband)
  Out ports  : 1 x complex64  (same stream, tagged at packet starts)
  Parameters : detect_threshold, corr_window
"""

import numpy as np
import sys, os
import pmt

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import config as cfg

try:
    import gnuradio.gr as gr
    _GR_AVAILABLE = True
except ImportError:
    _GR_AVAILABLE = False


# ---------------------------------------------------------------------------
# Core detector (adapted from detector.py — no ring buffer, stateful)
# ---------------------------------------------------------------------------

CORR_WINDOW  = 16      # STF period (16 samples)
DETECT_THRESH = 0.65   # Schmidl & Cox M^2/R^2 threshold


class StreamDetector:
    """
    Sliding-window STF autocorrelation.
    Call .push_samples(buf) -> list of (sample_index, coarse_cfo_hz).
    sample_index is relative to the beginning of the entire run.
    """
    def __init__(self, threshold: float = DETECT_THRESH):
        self.threshold   = threshold
        self._overlap    = np.zeros(2 * CORR_WINDOW - 1, dtype=np.complex64)
        self._total_idx  = 0   # running global sample counter
        self._last_det   = -cfg.STF_LEN   # prevent double detection

    def push(self, buf: np.ndarray):
        """
        Process one block of samples.
        Returns list of (global_sample_index, coarse_cfo_hz).
        """
        # Prepend overlap from previous call
        work = np.concatenate([self._overlap, buf.astype(np.complex64)])
        L    = CORR_WINDOW
        N    = len(work) - 2 * L
        if N <= 0:
            self._overlap = work[-(2 * L - 1):]
            self._total_idx += len(buf)
            return []

        detections = []
        i = 0
        while i < N:
            P = np.sum(work[i:i + L] * np.conj(work[i + L:i + 2 * L]))
            R = np.sum(np.abs(work[i + L:i + 2 * L]) ** 2) + 1e-12
            M2_R2 = (abs(P) ** 2) / (R ** 2)
            if M2_R2 > self.threshold:
                # Refine: find true peak in a ±8-sample window
                i0 = max(0, i - 8)
                i1 = min(N, i + 8)
                best_m, best_i = 0.0, i
                for j in range(i0, i1):
                    Pj = np.sum(work[j:j + L] * np.conj(work[j + L:j + 2 * L]))
                    Rj = np.sum(np.abs(work[j + L:j + 2 * L]) ** 2) + 1e-12
                    score = (abs(Pj) ** 2) / (Rj ** 2)
                    if score > best_m:
                        best_m, best_i = score, j
                i = best_i

                # Global sample index = total received - overlap prepend + i
                global_idx = self._total_idx - len(self._overlap) + i
                # Avoid double detection within one STF length
                if global_idx - self._last_det > cfg.STF_LEN:
                    # Coarse CFO
                    cfo = self._estimate_cfo(work[i:i + cfg.STF_LEN])
                    detections.append((global_idx, cfo))
                    self._last_det = global_idx
                i += cfg.STF_LEN   # skip to after STF
            else:
                i += 1

        self._overlap   = work[-(2 * L - 1):]
        self._total_idx += len(buf)
        return detections

    @staticmethod
    def _estimate_cfo(stf_seg: np.ndarray, fs: float = cfg.FS_FFT) -> float:
        L   = CORR_WINDOW
        seg = stf_seg[:min(len(stf_seg), cfg.STF_LEN)].astype(np.complex64)
        if len(seg) < 2 * L:
            return 0.0
        P   = np.sum(seg[:L] * np.conj(seg[L:2 * L]))
        phi = float(np.angle(P))
        return phi * fs / (2 * np.pi * L)


# ---------------------------------------------------------------------------
# GNU Radio block
# ---------------------------------------------------------------------------

if _GR_AVAILABLE:
    class blk(gr.sync_block):
        """
        GNU Radio Tagged-Stream Block: STF Packet Detector
        ---------------------------------------------------
        Passes samples through; attaches stream tags at detected starts.

        Tag key  : "pkt_start"
        Tag value: coarse CFO estimate (Hz) as float PMT

        Ports:
          Input  0: complex64 IQ stream
          Output 0: complex64 IQ stream (same samples, tagged)
        """

        def __init__(self,
                     detect_threshold: float = DETECT_THRESH,
                     corr_window:      int   = CORR_WINDOW):
            gr.sync_block.__init__(
                self,
                name   = "STF Packet Detector",
                in_sig = [np.complex64],
                out_sig= [np.complex64],
            )
            self._det    = StreamDetector(threshold=detect_threshold)
            self._n_dets = 0

        def work(self, input_items, output_items):
            inp = input_items[0]
            output_items[0][:] = inp   # pass-through

            dets = self._det.push(inp)
            for global_idx, cfo in dets:
                # Offset within THIS work() call
                local_off = global_idx - (self._det._total_idx - len(inp))
                local_off = max(0, min(local_off, len(inp) - 1))
                tag_off   = self.nitems_written(0) + local_off
                self.add_item_tag(
                    0,
                    tag_off,
                    pmt.intern("pkt_start"),
                    pmt.from_float(float(cfo)),
                )
                self._n_dets += 1

            return len(inp)
