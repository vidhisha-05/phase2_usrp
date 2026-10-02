"""
gr_block_tx.py  --  GNU Radio Python Block: Custom OFDM TX
=============================================================
Generates 128-pt OFDM packets (STF+LTF+SIGNAL+DATA) and outputs
them as a continuous stream of complex64 samples.

In GNU Radio Companion:
  Type       : Python Block (Embedded / OOT)
  Out ports  : 1 x complex64   (IQ sample stream)
  Parameters : mod_scheme, payload_bytes, packet_interval_s
"""

import numpy as np
import time
import sys
import os

# Ensure project root is importable inside GRC
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import config as cfg
import waveform
import scrambler

try:
    import gnuradio.gr as gr
    _GR_AVAILABLE = True
except ImportError:
    _GR_AVAILABLE = False


# ---------------------------------------------------------------------------
# Standalone callable (used both by GRC block and direct Python tests)
# ---------------------------------------------------------------------------

class OFDMPacketBuilder:
    """
    Stateless packet factory.
    Call .build(payload_bytes) -> complex64 ndarray.
    """
    MOD_BITS = {"BPSK": 1, "QPSK": 2, "QAM16": 4}

    def __init__(self, mod_scheme: str = "BPSK", payload_bytes: int = 60):
        self.mod    = mod_scheme
        self.n_bits = payload_bytes * 8

    def build(self) -> np.ndarray:
        """Return one IQ packet as complex64 (20 MS/s baseband)."""
        bits = np.random.randint(0, 2, self.n_bits, dtype=np.uint8)
        return waveform.assemble_packet(bits)


# ---------------------------------------------------------------------------
# GNU Radio sync_block wrapper
# ---------------------------------------------------------------------------

if _GR_AVAILABLE:
    class blk(gr.sync_block):
        """
        GNU Radio Source Block: 128-pt Custom OFDM TX
        ----------------------------------------------
        Outputs IQ samples for each packet back-to-back with
        configurable inter-packet silence padding.

        Ports:
          Output 0: complex64 stream
        """

        def __init__(self,
                     mod_scheme:         str   = "BPSK",
                     payload_bytes:      int   = 60,
                     packet_interval_s:  float = 0.01,
                     n_packets:          int   = 0):        # 0 = infinite
            gr.sync_block.__init__(
                self,
                name   = "Custom OFDM TX (128-pt)",
                in_sig = [],
                out_sig= [np.complex64],
            )
            self._builder      = OFDMPacketBuilder(mod_scheme, payload_bytes)
            self._interval_s   = packet_interval_s
            self._n_packets    = n_packets          # 0 = run forever
            self._pkt_count    = 0
            self._buf          = np.array([], dtype=np.complex64)
            self._silence_pad  = int(packet_interval_s * cfg.FS_FFT)
            self._last_tx      = time.monotonic()
            self.set_output_multiple(cfg.SYMBOL_LEN)

        def work(self, input_items, output_items):
            out  = output_items[0]
            n    = len(out)
            ptr  = 0

            while ptr < n:
                # Refill buffer if empty
                if len(self._buf) == 0:
                    # Inter-packet pacing
                    elapsed = time.monotonic() - self._last_tx
                    if elapsed < self._interval_s:
                        # Output zeros (silence) for remaining time
                        zeros = min(n - ptr,
                                    int((self._interval_s - elapsed) * cfg.FS_FFT))
                        out[ptr:ptr + zeros] = 0
                        ptr += zeros
                        continue

                    # Check packet limit
                    if self._n_packets > 0 and self._pkt_count >= self._n_packets:
                        out[ptr:] = 0
                        return len(out)

                    pkt = self._builder.build()
                    pad = np.zeros(self._silence_pad, dtype=np.complex64)
                    self._buf      = np.concatenate([pkt.astype(np.complex64), pad])
                    self._pkt_count += 1
                    self._last_tx   = time.monotonic()

                # Copy chunk
                chunk = min(n - ptr, len(self._buf))
                out[ptr:ptr + chunk] = self._buf[:chunk]
                self._buf = self._buf[chunk:]
                ptr += chunk

            return n
