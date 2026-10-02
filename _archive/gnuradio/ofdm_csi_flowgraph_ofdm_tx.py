
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

class blk(gr.sync_block):
    """Custom 128-pt OFDM TX -- generates STF+LTF+SIGNAL packets"""

    def __init__(self,
                 mod_scheme='BPSK',
                 payload_bytes=60,
                 packet_interval_s=0.01,
                 n_packets=0):
        gr.sync_block.__init__(
            self,
            name='Custom OFDM TX (128-pt)',
            in_sig=[],
            out_sig=[np.complex64])
        self._payload_bytes   = int(payload_bytes)
        self._interval_s      = float(packet_interval_s)
        self._n_packets       = int(n_packets)
        self._pkt_count       = 0
        self._buf             = np.array([], dtype=np.complex64)
        self._silence_pad_len = int(float(packet_interval_s) * FS_FFT)
        self._last_tx         = time.monotonic()
        self.set_output_multiple(SYMBOL_LEN)
        self._waveform = None

    def _get_waveform(self):
        if self._waveform is None:
            import waveform as _wv
            self._waveform = _wv
        return self._waveform

    def work(self, input_items, output_items):
        out = output_items[0]
        n   = len(out)
        ptr = 0
        wv  = self._get_waveform()
        while ptr < n:
            if len(self._buf) == 0:
                elapsed = time.monotonic() - self._last_tx
                if elapsed < self._interval_s:
                    zeros = min(n - ptr,
                                int((self._interval_s - elapsed) * FS_FFT))
                    out[ptr:ptr + zeros] = 0
                    ptr += zeros
                    continue
                if self._n_packets > 0 and self._pkt_count >= self._n_packets:
                    out[ptr:] = 0
                    return len(out)
                bits = np.random.randint(0, 2, self._payload_bytes * 8,
                                         dtype=np.uint8)
                pkt  = wv.assemble_packet(bits).astype(np.complex64)
                pad  = np.zeros(self._silence_pad_len, dtype=np.complex64)
                self._buf        = np.concatenate([pkt, pad])
                self._pkt_count += 1
                self._last_tx    = time.monotonic()
            chunk = min(n - ptr, len(self._buf))
            out[ptr:ptr + chunk] = self._buf[:chunk]
            self._buf = self._buf[chunk:]
            ptr += chunk
        return n
