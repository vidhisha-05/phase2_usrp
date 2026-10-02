"""
gr_block_csi_logger.py  --  GNU Radio Python Block: HDF5 CSI Logger
====================================================================
Receives CSI pmt dicts on message port "csi_in" (from CSI Extractor)
and writes to HDF5 using the project's logger.py.
Also prints a live console status line.

In GNU Radio Companion:
  Type       : Python Block (Embedded / OOT)
  In ports   : 0 sample ports
  Msg in     : "csi_in"   (pmt dict from CSI Extractor)
  Parameters : hdf5_path, session_id, n_rx_channels, flush_interval
"""

import numpy as np
import sys, os, time, threading, queue

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import config as cfg
import pmt

try:
    import gnuradio.gr as gr
    _GR_AVAILABLE = True
except ImportError:
    _GR_AVAILABLE = False


# ---------------------------------------------------------------------------
# Minimal HDF5 writer (mirrors logger.py API but usable in-block)
# ---------------------------------------------------------------------------

class InlineHDF5Writer:
    """
    Thread-safe HDF5 CSI writer that runs in a background daemon thread.
    Receives dicts via a queue:
      {'H_hat': ndarray(107,), 'timestamp': float, 'seq': int, 'dropped': int}
    """
    def __init__(self, hdf5_path: str, session_id: str,
                 n_rx_channels: int = 1, flush_interval: int = 50):
        self._q           = queue.Queue(maxsize=2000)
        self._path        = hdf5_path
        self._session     = session_id
        self._n_ch        = n_rx_channels
        self._flush_n     = flush_interval
        self._running     = False
        self._thread      = None
        self._total       = 0

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def put(self, record: dict):
        """Non-blocking enqueue; drop if full (acquisition must not block)."""
        try:
            self._q.put_nowait(record)
        except queue.Full:
            pass

    def stop(self):
        self._running = False
        self._q.put(None)   # sentinel
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self):
        import h5py
        buf = []
        with h5py.File(self._path, 'a') as f:
            grp = f.require_group(f"sessions/{self._session}/csi")
            ds  = {}
            for ch in range(self._n_ch):
                key = f"antenna{ch}"
                if key not in grp:
                    ds[key] = grp.create_dataset(
                        key, shape=(0, cfg.NUM_ACTIVE),
                        maxshape=(None, cfg.NUM_ACTIVE),
                        dtype=np.complex64, chunks=(100, cfg.NUM_ACTIVE))
                else:
                    ds[key] = grp[key]

            print(f"[HDF5 Logger] Open: {self._path}  session={self._session}")

            while self._running or not self._q.empty():
                try:
                    rec = self._q.get(timeout=0.5)
                except queue.Empty:
                    continue
                if rec is None:
                    break
                buf.append(rec)
                if len(buf) >= self._flush_n:
                    self._flush(buf, ds)
                    buf.clear()

            if buf:
                self._flush(buf, ds)
            print(f"[HDF5 Logger] Closed. Total packets: {self._total}")

    def _flush(self, records, ds):
        import h5py
        for ch_idx, key in enumerate(ds):
            rows = []
            for r in records:
                h = r.get('H_hat') if ch_idx == 0 else r.get('H_hat_ant1')
                if h is not None and len(h) == cfg.NUM_ACTIVE:
                    rows.append(h)
            if rows:
                arr = np.array(rows, dtype=np.complex64)
                cur = ds[key].shape[0]
                ds[key].resize(cur + len(arr), axis=0)
                ds[key][cur:cur + len(arr)] = arr
        ds[list(ds.keys())[0]].file.flush()
        self._total += len(records)
        print(f"[HDF5 Logger] Flushed {len(records)} pkts  total={self._total}")


# ---------------------------------------------------------------------------
# GNU Radio block
# ---------------------------------------------------------------------------

if _GR_AVAILABLE:
    class blk(gr.basic_block):
        """
        GNU Radio Message Block: HDF5 CSI Logger
        ------------------------------------------
        Receives "csi_out" PMT dicts from CSI Extractor.
        Writes H_hat to HDF5 via background thread.

        Ports:
          Msg in: "csi_in"  (pmt dict)
        """

        def __init__(self,
                     hdf5_path:      str = "gr_csi_output.h5",
                     session_id:     str = "gr_session_001",
                     n_rx_channels:  int = 1,
                     flush_interval: int = 50):
            gr.basic_block.__init__(
                self,
                name   = "HDF5 CSI Logger",
                in_sig = [],
                out_sig= [],
            )
            self.message_port_register_in(pmt.intern("csi_in"))
            self.set_msg_handler(pmt.intern("csi_in"), self._handle_csi)

            self._writer = InlineHDF5Writer(
                hdf5_path, session_id, n_rx_channels, flush_interval)
            self._writer.start()
            self._pkt_count = 0

        def _handle_csi(self, msg):
            if not pmt.is_dict(msg):
                return

            valid = pmt.dict_ref(msg, pmt.intern("H_hat_valid"),
                                 pmt.from_bool(False))
            if not pmt.to_bool(valid):
                return

            h_pmt = pmt.dict_ref(msg, pmt.intern("H_hat"), pmt.PMT_NIL)
            if pmt.is_c32vector(h_pmt):
                H = np.array(pmt.c32vector_elements(h_pmt), dtype=np.complex64)
            else:
                return

            seq_pmt = pmt.dict_ref(msg, pmt.intern("seq"),
                                   pmt.from_long(0))
            ts_pmt  = pmt.dict_ref(msg, pmt.intern("timestamp"),
                                   pmt.from_double(0.0))
            cfo_pmt = pmt.dict_ref(msg, pmt.intern("cfo_hz"),
                                   pmt.from_double(0.0))

            record = {
                'H_hat':     H,
                'timestamp': pmt.to_double(ts_pmt),
                'seq':       pmt.to_long(seq_pmt),
                'dropped':   0,
                'cfo_hz':    pmt.to_double(cfo_pmt),
            }
            self._writer.put(record)
            self._pkt_count += 1
            if self._pkt_count % 10 == 0:
                print(f"[CSI Logger] Received pkt #{self._pkt_count}"
                      f"  cfo={pmt.to_double(cfo_pmt):.1f} Hz")

        def stop(self):
            self._writer.stop()
            return True
