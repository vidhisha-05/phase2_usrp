
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
    """Phase-continuous AWGN + CFO + SCO + multipath impairment"""

    def __init__(self,
                 noise_voltage=0.005,
                 cfo_hz=0.0,
                 sco_ppm=0.0,
                 tap_string='1+0j'):
        gr.sync_block.__init__(
            self,
            name='Channel Impairment',
            in_sig=[np.complex64],
            out_sig=[np.complex64])
        self._noise = float(noise_voltage)
        self._cfo   = float(cfo_hz)
        self._sco   = float(sco_ppm)
        self._taps  = np.array(
            [complex(t.strip()) for t in str(tap_string).split(',') if t.strip()],
            dtype=np.complex64)
        self._idx   = 0

    def work(self, input_items, output_items):
        x   = input_items[0].astype(np.complex64).copy()
        N   = len(x)
        if len(self._taps) > 1:
            x = np.convolve(x, self._taps)[:N]
        n   = np.arange(self._idx, self._idx + N, dtype=np.float64)
        x   = x * np.exp(1j * 2.0 * np.pi * self._cfo / FS_FFT * n
                         ).astype(np.complex64)
        if self._sco != 0.0:
            x = x * np.exp(1j * 2.0 * np.pi * self._sco * 1e-6 * n / FS_FFT
                           ).astype(np.complex64)
        s   = self._noise / np.sqrt(2.0)
        noi = (s * np.random.randn(N).astype(np.float32) +
               1j * s * np.random.randn(N).astype(np.float32))
        output_items[0][:] = (x + noi.astype(np.complex64)).astype(np.complex64)
        self._idx += N
        return N
