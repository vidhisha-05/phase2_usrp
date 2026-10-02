
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
    """LTF sync + 2-stage CFO + H_hat[107]. Emits PMT dicts on csi_out."""

    def __init__(self, n_data_symbols=0):
        gr.sync_block.__init__(
            self,
            name='CSI Extractor (128-pt)',
            in_sig=[np.complex64],
            out_sig=[])
        self.message_port_register_out(pmt.intern('csi_out'))
        self._buf_len = (STF_LEN + LTF_LEN + SIG_LEN +
                         int(n_data_symbols) * SYMBOL_LEN + 64)
        self._pending = []
        self._seq     = 0
        self._sync    = None

    def _get_sync(self):
        if self._sync is None:
            import sync as _s
            self._sync = _s
        return self._sync

    def work(self, input_items, output_items):
        inp  = np.array(input_items[0], dtype=np.complex64)
        base = self.nitems_read(0)
        tags = self.get_tags_in_window(0, 0, len(inp), pmt.intern('pkt_start'))
        ptr  = 0
        for tag in sorted(tags, key=lambda t: t.offset):
            lo = int(tag.offset - base)
            self._feed(inp[ptr:lo])
            ptr = lo
            self._pending.append([pmt.to_float(tag.value),
                                   np.array([], dtype=np.complex64)])
        self._feed(inp[ptr:])
        self._flush()
        return len(input_items[0])

    def _feed(self, chunk):
        for e in self._pending:
            need = self._buf_len - len(e[1])
            if need > 0:
                e[1] = np.concatenate([e[1], chunk[:need].astype(np.complex64)])

    def _flush(self):
        done          = [(c, b) for c, b in self._pending if len(b) >= self._buf_len]
        self._pending = [(c, b) for c, b in self._pending if len(b) <  self._buf_len]
        s = self._get_sync()
        for cfo, buf in done:
            try:
                pc   = s.apply_cfo_correction(buf, cfo)
                s1   = STF_LEN + 64
                s2   = s1 + FFT_SIZE
                fine = s.estimate_fine_cfo(pc[s1:s1 + FFT_SIZE],
                                           pc[s2:s2 + FFT_SIZE])
                H    = s.extract_csi(pc, ltf_start=STF_LEN,
                                     total_cfo_hz=cfo + fine)
                valid = (not np.any(np.isnan(H)) and np.mean(np.abs(H)) > 0.05)
                d = pmt.make_dict()
                d = pmt.dict_add(d, pmt.intern('seq'),         pmt.from_long(self._seq))
                d = pmt.dict_add(d, pmt.intern('timestamp'),   pmt.from_double(time.time()))
                d = pmt.dict_add(d, pmt.intern('cfo_hz'),      pmt.from_double(float(cfo + fine)))
                d = pmt.dict_add(d, pmt.intern('H_hat_valid'), pmt.from_bool(bool(valid)))
                if valid:
                    h_v = pmt.init_c32vector(len(H), [complex(v) for v in H.tolist()])
                    d   = pmt.dict_add(d, pmt.intern('H_hat'), h_v)
                self.message_port_pub(pmt.intern('csi_out'), d)
                self._seq += 1
            except Exception:
                pass
