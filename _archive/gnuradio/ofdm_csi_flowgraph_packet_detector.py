
# Custom 128-pt OFDM PHY constants (from config.py)
_PROJ       = "d:/phase2"
FFT_SIZE    = 128
CP_LEN      = 32
SYMBOL_LEN  = 160
FS_FFT      = 20e6
FS_HW       = 25e6
STF_LEN     = 128
LTF_LEN     = 320
SIG_LEN     = 160
NUM_ACTIVE  = 107
NUM_DATA    = 99
NUM_PILOTS  = 8

import sys as _sys
if _PROJ not in _sys.path:
    _sys.path.insert(0, _PROJ)

from gnuradio import gr
import pmt

import numpy as np

class blk(gr.sync_block):
    """Schmidl-Cox STF detector -- adds pkt_start stream tags"""

    def __init__(self, detect_threshold=0.65, corr_window=16):
        gr.sync_block.__init__(
            self,
            name='STF Packet Detector',
            in_sig=[np.complex64],
            out_sig=[np.complex64])
        self._thresh   = float(detect_threshold)
        self._L        = int(corr_window)
        self._overlap  = np.zeros(2 * int(corr_window) - 1, dtype=np.complex64)
        self._global   = 0
        self._last_det = -STF_LEN

    def work(self, input_items, output_items):
        inp = np.array(input_items[0], dtype=np.complex64)
        output_items[0][:] = inp
        L    = self._L
        work = np.concatenate([self._overlap, inp])
        N    = len(work) - 2 * L
        i    = 0
        while i < N:
            sa = work[i:i + L]
            sb = work[i + L:i + 2 * L]
            P  = np.sum(sa * np.conj(sb))
            R  = np.sum(np.abs(sb) ** 2) + 1e-12
            if (abs(P) ** 2) / (R ** 2) > self._thresh:
                g_idx = self._global - len(self._overlap) + i
                if g_idx - self._last_det > STF_LEN:
                    cfo      = float(np.angle(P)) * FS_FFT / (2.0 * 3.14159265 * L)
                    local_off = max(0, min(g_idx - (self._global - len(inp)),
                                          len(inp) - 1))
                    self.add_item_tag(
                        0,
                        self.nitems_written(0) + local_off,
                        pmt.intern('pkt_start'),
                        pmt.from_float(float(cfo)))
                    self._last_det = g_idx
                i += STF_LEN
            else:
                i += 1
        self._overlap  = work[-(2 * L - 1):]
        self._global  += len(inp)
        return len(inp)
