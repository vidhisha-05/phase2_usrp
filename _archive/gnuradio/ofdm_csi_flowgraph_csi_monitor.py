
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
import time

_BAR = ' .:-=+#@$'

class blk(gr.basic_block):
    """Prints live ASCII |H_hat| amplitude bar per packet."""

    def __init__(self, update_every_n=5):
        gr.basic_block.__init__(
            self,
            name='CSI Monitor (Live)',
            in_sig=[],
            out_sig=[])
        self.message_port_register_in(pmt.intern('csi_in'))
        self.set_msg_handler(pmt.intern('csi_in'), self._on)
        self._n  = int(update_every_n)
        self._c  = 0
        self._t0 = time.monotonic()

    def _on(self, msg):
        if not pmt.is_dict(msg):
            return
        ok = pmt.to_bool(pmt.dict_ref(msg, pmt.intern('H_hat_valid'),
                                       pmt.from_bool(False)))
        if not ok:
            return
        self._c += 1
        if self._c % self._n != 0:
            return
        hv = pmt.dict_ref(msg, pmt.intern('H_hat'), pmt.PMT_NIL)
        if not pmt.is_c32vector(hv):
            return
        H   = np.abs(np.array(pmt.c32vector_elements(hv), dtype=np.complex64))
        cfo = pmt.to_double(pmt.dict_ref(msg, pmt.intern('cfo_hz'),
                                          pmt.from_double(0.0)))
        W   = 54
        bar = ''.join(
            _BAR[min(8, int(np.mean(H[int(i*len(H)/W):int((i+1)*len(H)/W)]) * 4))]
            for i in range(W))
        rate = self._c / max(time.monotonic() - self._t0, 0.001)
        print(f'\r  pkt#{self._c:>5}  cfo={cfo:>8.1f}Hz  [{bar}]  '
              f'mean={np.mean(H):.3f}  {rate:.1f}/s', end='', flush=True)
