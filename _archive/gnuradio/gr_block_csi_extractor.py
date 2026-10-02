"""
gr_block_csi_extractor.py  --  GNU Radio Python Block: CSI Extractor
=====================================================================
Reads stream tags ("pkt_start") from the detector block, extracts
a packet buffer, performs:
  1. Coarse CFO correction (from tag value)
  2. Fine CFO via LTF repetitions
  3. LTF-based CSI estimation (H_hat, 107 complex values)
  4. SCO pilot tracking

Outputs CSI records to a Python message port (pmt dict) for the
logger block downstream.

In GNU Radio Companion:
  Type       : Python Block (Embedded / OOT)
  In ports   : 1 x complex64  (tagged IQ stream)
  Out ports  : 0 sample ports  (uses message passing for CSI output)
  Msg out    : "csi_out"  (pmt dict per packet)
  Parameters : n_data_symbols
"""

import numpy as np
import sys, os
import pmt

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import config as cfg
from sync import (sync_packet, extract_csi, estimate_fine_cfo,
                  apply_cfo_correction)

try:
    import gnuradio.gr as gr
    _GR_AVAILABLE = True
except ImportError:
    _GR_AVAILABLE = False


# ---------------------------------------------------------------------------
# Packet buffer manager
# ---------------------------------------------------------------------------

class PacketBuffer:
    """
    Accumulates samples until a full packet is available.
    When pkt_start tag is seen, stores (global_offset, coarse_cfo) and
    starts collecting until buf_len samples are gathered.
    """
    def __init__(self, buf_len: int):
        self.buf_len    = buf_len
        self._pending   = []   # list of (coarse_cfo, ndarray_buffer)
        self._ready     = []   # completed (coarse_cfo, ndarray) items

    def start_packet(self, coarse_cfo: float):
        self._pending.append((coarse_cfo, np.array([], dtype=np.complex64)))

    def push(self, samples: np.ndarray):
        """Feed new samples; complete packets move to _ready."""
        for i, (cfo, buf) in enumerate(self._pending):
            need    = self.buf_len - len(buf)
            chunk   = samples[:need]
            buf     = np.concatenate([buf, chunk.astype(np.complex64)])
            self._pending[i] = (cfo, buf)
            if len(buf) >= self.buf_len:
                self._ready.append((cfo, buf[:self.buf_len]))

        # Prune completed
        self._pending = [(c, b) for c, b in self._pending
                         if len(b) < self.buf_len]

    def get_ready(self):
        """Return list of completed (coarse_cfo, buffer) and clear."""
        out          = list(self._ready)
        self._ready  = []
        return out


# ---------------------------------------------------------------------------
# GNU Radio block
# ---------------------------------------------------------------------------

if _GR_AVAILABLE:
    class blk(gr.sync_block):
        """
        GNU Radio Sink/Message Block: CSI Extractor
        ---------------------------------------------
        Reads tagged IQ stream, buffers packets, runs LTF sync + CFO
        correction + CSI extraction, and emits pmt dicts on "csi_out".

        Message output dict keys:
          "seq"        : int   packet sequence number
          "timestamp"  : float host time
          "cfo_hz"     : float total CFO (coarse + fine) in Hz
          "H_hat"      : c32vector  107 complex float32 values
          "H_hat_valid": bool

        Ports:
          Input  0: complex64 tagged stream
          Msg out: "csi_out"
        """

        def __init__(self, n_data_symbols: int = 0):
            gr.sync_block.__init__(
                self,
                name   = "CSI Extractor (128-pt OFDM)",
                in_sig = [np.complex64],
                out_sig= [],
            )
            self.message_port_register_out(pmt.intern("csi_out"))

            # Buffer enough samples: STF + LTF + SIGNAL + n_data * SYMBOL_LEN
            self._buf_len = (cfg.STF_LEN + cfg.LTF_LEN + cfg.SIG_LEN +
                             n_data_symbols * cfg.SYMBOL_LEN + 64)
            self._pbuf    = PacketBuffer(self._buf_len)
            self._seq     = 0

            # Tag tracking
            self._global_offset = 0   # nitems_read(0) at start of each work()

        def work(self, input_items, output_items):
            import time as _t
            inp = np.array(input_items[0], dtype=np.complex64)

            # -- Scan for pkt_start tags in this block -------------------
            base = self.nitems_read(0)
            tags = self.get_tags_in_window(0, 0, len(inp),
                                           pmt.intern("pkt_start"))
            for tag in tags:
                offset     = tag.offset - base            # local offset
                coarse_cfo = pmt.to_float(tag.value)
                # Deliver samples up to this tag to in-progress buffers
                self._pbuf.push(inp[:offset])
                self._pbuf.start_packet(coarse_cfo)
                inp = inp[offset:]   # advance past tag position

            # Feed remaining samples
            self._pbuf.push(inp)

            # -- Process completed packets --------------------------------
            for coarse_cfo, buf in self._pbuf.get_ready():
                try:
                    r = sync_packet(buf, coarse_cfo,
                                    n_data_symbols=0)
                    H      = r['H_hat']
                    valid  = (not np.any(np.isnan(H)) and
                              np.mean(np.abs(H)) > 0.05)

                    # Build PMT dict
                    d = pmt.make_dict()
                    d = pmt.dict_add(d, pmt.intern("seq"),
                                     pmt.from_long(self._seq))
                    d = pmt.dict_add(d, pmt.intern("timestamp"),
                                     pmt.from_double(_t.time()))
                    d = pmt.dict_add(d, pmt.intern("cfo_hz"),
                                     pmt.from_double(float(r['total_cfo'])))
                    d = pmt.dict_add(d, pmt.intern("H_hat_valid"),
                                     pmt.from_bool(bool(valid)))
                    if valid:
                        h_pmt = pmt.init_c32vector(
                            len(H),
                            [complex(v) for v in H.tolist()])
                        d = pmt.dict_add(d, pmt.intern("H_hat"), h_pmt)

                    self.message_port_pub(pmt.intern("csi_out"), d)
                    self._seq += 1
                except Exception as ex:
                    print(f"[CSI Extractor] Error: {ex}")

            return len(input_items[0])
