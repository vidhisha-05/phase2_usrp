"""
gr_block_csi_monitor.py  --  GNU Radio Python Block: Real-Time CSI Monitor
===========================================================================
Receives CSI pmt dicts and displays a live ASCII amplitude spectrum
on the terminal — one bar per active subcarrier.
Connect in parallel with the logger (same "csi_out" source).

In GNU Radio Companion:
  Type       : Python Block (Embedded / OOT)
  Msg in     : "csi_in"  (pmt dict)
  Parameters : update_every_n (print every N packets)
"""

import numpy as np
import sys, os, time
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
# ASCII bar-chart renderer
# ---------------------------------------------------------------------------

def render_csi_bar(H_abs: np.ndarray, width: int = 60) -> str:
    """
    Render |H[k]| for 107 subcarriers as a compressed ASCII bar.
    Groups subcarriers into 'width' buckets.
    """
    n   = len(H_abs)
    out = []
    for i in range(width):
        lo = int(i * n / width)
        hi = int((i + 1) * n / width)
        v  = np.mean(H_abs[lo:hi]) if hi > lo else H_abs[lo]
        # Scale 0-2.0 to 0-8 ASCII blocks
        level = min(8, int(v * 4))
        BLOCKS = " .:-=+#@$"
        out.append(BLOCKS[level])
    return "".join(out)


# ---------------------------------------------------------------------------
# GNU Radio block
# ---------------------------------------------------------------------------

if _GR_AVAILABLE:
    class blk(gr.basic_block):
        """
        GNU Radio Message Block: Real-Time CSI Monitor
        ------------------------------------------------
        Prints live |H_hat| amplitude spectrum to terminal.

        Ports:
          Msg in: "csi_in"  (pmt dict from CSI Extractor)
        """

        def __init__(self, update_every_n: int = 5):
            gr.basic_block.__init__(
                self,
                name   = "CSI Monitor (Live)",
                in_sig = [],
                out_sig= [],
            )
            self.message_port_register_in(pmt.intern("csi_in"))
            self.set_msg_handler(pmt.intern("csi_in"), self._handle)
            self._update_n  = update_every_n
            self._count     = 0
            self._t_start   = time.monotonic()

        def _handle(self, msg):
            if not pmt.is_dict(msg):
                return

            valid = pmt.dict_ref(msg, pmt.intern("H_hat_valid"),
                                 pmt.from_bool(False))
            if not pmt.to_bool(valid):
                return

            self._count += 1
            if self._count % self._update_n != 0:
                return

            h_pmt = pmt.dict_ref(msg, pmt.intern("H_hat"), pmt.PMT_NIL)
            if not pmt.is_c32vector(h_pmt):
                return
            H = np.abs(np.array(pmt.c32vector_elements(h_pmt),
                                dtype=np.complex64))
            cfo_pmt = pmt.dict_ref(msg, pmt.intern("cfo_hz"),
                                   pmt.from_double(0.0))
            elapsed = time.monotonic() - self._t_start
            rate    = self._count / max(elapsed, 0.001)
            bar     = render_csi_bar(H, width=54)

            print(f"\r  pkt#{self._count:>5}  "
                  f"cfo={pmt.to_double(cfo_pmt):>8.1f}Hz  "
                  f"|H|=[{bar}]  "
                  f"mean={np.mean(H):.3f}  "
                  f"rate={rate:.1f}/s",
                  end="", flush=True)
