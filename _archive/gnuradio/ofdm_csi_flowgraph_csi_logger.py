
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
import threading
import queue

class blk(gr.basic_block):
    """Receives csi_in PMT dicts and writes H_hat to HDF5."""

    def __init__(self,
                 hdf5_path='gr_csi_output.h5',
                 session_id='gr_session_001',
                 n_rx_channels=1,
                 flush_interval=50):
        gr.basic_block.__init__(
            self,
            name='HDF5 CSI Logger',
            in_sig=[],
            out_sig=[])
        self.message_port_register_in(pmt.intern('csi_in'))
        self.set_msg_handler(pmt.intern('csi_in'), self._on_csi)
        self._path    = str(hdf5_path)
        self._session = str(session_id)
        self._flush_n = int(flush_interval)
        self._q       = queue.Queue(maxsize=2000)
        self._running = True
        self._count   = 0
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _on_csi(self, msg):
        if not pmt.is_dict(msg):
            return
        ok = pmt.to_bool(pmt.dict_ref(msg, pmt.intern('H_hat_valid'),
                                      pmt.from_bool(False)))
        if not ok:
            return
        hv = pmt.dict_ref(msg, pmt.intern('H_hat'), pmt.PMT_NIL)
        if not pmt.is_c32vector(hv):
            return
        H  = np.array(pmt.c32vector_elements(hv), dtype=np.complex64)
        ts = pmt.to_double(pmt.dict_ref(msg, pmt.intern('timestamp'),
                                         pmt.from_double(0.0)))
        sq = pmt.to_long(pmt.dict_ref(msg, pmt.intern('seq'),
                                       pmt.from_long(0)))
        try:
            self._q.put_nowait({'H_hat': H, 'ts': ts, 'seq': sq})
        except queue.Full:
            pass
        self._count += 1
        if self._count % 10 == 0:
            print(f'[Logger] {self._count} pkts', end='\r', flush=True)

    def _run(self):
        import h5py
        buf = []
        with h5py.File(self._path, 'a') as hf:
            grp = hf.require_group(f'sessions/{self._session}/csi')
            if 'antenna0' not in grp:
                ds = grp.create_dataset(
                    'antenna0', shape=(0, NUM_ACTIVE),
                    maxshape=(None, NUM_ACTIVE),
                    dtype=np.complex64, chunks=(100, NUM_ACTIVE))
            else:
                ds = grp['antenna0']
            print(f'[HDF5] {self._path}  session={self._session}')
            while self._running or not self._q.empty():
                try:
                    buf.append(self._q.get(timeout=0.5))
                except Exception:
                    continue
                if len(buf) >= self._flush_n:
                    rows = np.array([r['H_hat'] for r in buf], dtype=np.complex64)
                    n0   = ds.shape[0]
                    ds.resize(n0 + len(rows), axis=0)
                    ds[n0:n0 + len(rows)] = rows
                    hf.flush()
                    print(f'[HDF5] Flushed {len(buf)} total={n0+len(rows)}')
                    buf.clear()
            if buf:
                rows = np.array([r['H_hat'] for r in buf], dtype=np.complex64)
                n0   = ds.shape[0]
                ds.resize(n0 + len(rows), axis=0)
                ds[n0:n0 + len(rows)] = rows
                hf.flush()
            print(f'[HDF5] Closed total={ds.shape[0]}')

    def stop(self):
        self._running = False
        self._thread.join(timeout=5)
        return True
