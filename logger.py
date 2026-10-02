"""
logger.py — Buffered CSI / metadata writer to HDF5.

Runs in a dedicated thread; consumes CSI records from a queue and flushes
to disk every HDF5_FLUSH_INTERVAL packets to keep disk I/O off the
acquisition/processing path.

Fully supports single-channel (1-TX/1-RX) and dual-channel logging, as well as
sanitized CSI phase dataset persistence for Human Activity Recognition (HAR).

Reference: SYSTEM_SPEC.md — Section 14, 15
"""

import queue
import threading
import time
import numpy as np
import h5py
import config as cfg

class CSILogger:
    """
    Thread-safe HDF5 CSI logger supporting single and dual antenna records,
    as well as single-antenna phase-sanitized CSI persistence.
    """

    def __init__(self,
                 csi_queue:      queue.Queue,
                 hdf5_path:      str  = cfg.HDF5_FILE_PATH,
                 session_id:     str  = "session_001",
                 n_rx_channels:  int  = 1,
                 flush_interval: int  = cfg.HDF5_FLUSH_INTERVAL,
                 metadata:       dict = None):
        self._q              = csi_queue
        self._path           = hdf5_path
        self._sid            = session_id
        self._n_ch           = n_rx_channels
        self._flush_interval = flush_interval
        self._metadata       = metadata or {}
        self._stop           = threading.Event()

        self._buf_ant0:          list = []
        self._buf_sanitized:     list = []
        self._buf_phase_san:     list = []
        self._buf_ant1:          list = []
        self._buf_ts:            list = []
        self._buf_abs_s:         list = []
        self._buf_seq:           list = []
        self._buf_cfo:           list = []

        # Open HDF5 file and create groups
        self._file = h5py.File(hdf5_path, 'a')
        base       = f"/sessions/{session_id}"
        if base not in self._file:
            grp = self._file.create_group(base)
            grp.attrs.update(self._metadata)

        csi_grp = f"{base}/csi"
        if csi_grp not in self._file:
            self._file.create_group(csi_grp)

        # Raw Antenna 0 CSI dataset
        self._ds_ant0 = self._file.require_dataset(
            f"{csi_grp}/antenna0",
            shape=(0, cfg.NUM_ACTIVE),
            maxshape=(None, cfg.NUM_ACTIVE),
            dtype=np.complex64,
            chunks=(cfg.HDF5_FLUSH_INTERVAL, cfg.NUM_ACTIVE))

        # Phase-Sanitized CSI dataset (HAR micro-activity)
        self._ds_sanitized = self._file.require_dataset(
            f"{csi_grp}/sanitized",
            shape=(0, cfg.NUM_ACTIVE),
            maxshape=(None, cfg.NUM_ACTIVE),
            dtype=np.complex64,
            chunks=(cfg.HDF5_FLUSH_INTERVAL, cfg.NUM_ACTIVE))

        # Detrended Phase dataset (radians)
        self._ds_phase_san = self._file.require_dataset(
            f"{csi_grp}/phase_sanitized",
            shape=(0, cfg.NUM_ACTIVE),
            maxshape=(None, cfg.NUM_ACTIVE),
            dtype=np.float64,
            chunks=(cfg.HDF5_FLUSH_INTERVAL, cfg.NUM_ACTIVE))

        # Antenna 1 CSI dataset (optional for multi-channel mode)
        if n_rx_channels >= 2:
            self._ds_ant1 = self._file.require_dataset(
                f"{csi_grp}/antenna1",
                shape=(0, cfg.NUM_ACTIVE),
                maxshape=(None, cfg.NUM_ACTIVE),
                dtype=np.complex64,
                chunks=(cfg.HDF5_FLUSH_INTERVAL, cfg.NUM_ACTIVE))
        else:
            self._ds_ant1 = None

        # Timestamp dataset
        self._ds_ts = self._file.require_dataset(
            f"{base}/timestamps",
            shape=(0,), maxshape=(None,), dtype=np.float64,
            chunks=(cfg.HDF5_FLUSH_INTERVAL,))

        # Absolute BB sample-coordinate dataset
        self._ds_abs_s = self._file.require_dataset(
            f"{base}/abs_s",
            shape=(0,), maxshape=(None,), dtype=np.int64,
            chunks=(cfg.HDF5_FLUSH_INTERVAL,))

        # Sequence dataset
        self._ds_seq = self._file.require_dataset(
            f"{base}/seq",
            shape=(0,), maxshape=(None,), dtype=np.int64,
            chunks=(cfg.HDF5_FLUSH_INTERVAL,))

        # Per-packet CFO estimate (Hz)
        self._ds_cfo = self._file.require_dataset(
            f"{base}/cfo_hz",
            shape=(0,), maxshape=(None,), dtype=np.float64,
            chunks=(cfg.HDF5_FLUSH_INTERVAL,))

        print(f"[logger] HDF5 opened: {hdf5_path} | session={session_id} | channels={n_rx_channels}")

    def run(self):
        """Main logger loop — consumes queue records until stop event."""
        while not self._stop.is_set() or not self._q.empty():
            try:
                record = self._q.get(timeout=0.1)
            except queue.Empty:
                continue

            # F-04 FIX: Only log CSI for CRC-passing packets (matches design contract).
            # CRC failures indicate corrupted timing/sync — CSI from those frames is
            # unreliable and must NOT be written to the HAR training dataset.
            if not record.get('crc_ok', True):
                continue

            self._buf_ant0.append(record['H_hat'])
            self._buf_sanitized.append(record.get('H_sanitized', record['H_hat']))
            self._buf_phase_san.append(record.get('phase_sanitized', np.angle(record['H_hat'])))
            self._buf_ts.append(record['timestamp'])
            self._buf_abs_s.append(int(record['abs_s']))
            self._buf_seq.append(record.get('seq', 0))
            self._buf_cfo.append(float(record.get('cfo_hz', 0.0)))

            if self._n_ch >= 2 and 'H_hat_ant1' in record:
                self._buf_ant1.append(record['H_hat_ant1'])

            if record.get('dropped', 0) > 0:
                self._file[f"/sessions/{self._sid}"].attrs['ring_drops'] = record['dropped']

            if len(self._buf_ant0) >= self._flush_interval:
                self._flush()

        self._flush()
        self._file.close()
        print("[logger] HDF5 dataset successfully closed.")

    def stop(self):
        self._stop.set()

    def _flush(self):
        if not self._buf_ant0:
            return

        n    = len(self._buf_ant0)
        data = np.array(self._buf_ant0, dtype=np.complex64)
        san  = np.array(self._buf_sanitized, dtype=np.complex64)
        psan = np.array(self._buf_phase_san, dtype=np.float64)
        ts   = np.array(self._buf_ts, dtype=np.float64)
        abs_s = np.array(self._buf_abs_s, dtype=np.int64)

        old = self._ds_ant0.shape[0]

        self._ds_ant0.resize(old + n, axis=0)
        self._ds_ant0[old:old + n] = data

        self._ds_sanitized.resize(old + n, axis=0)
        self._ds_sanitized[old:old + n] = san

        self._ds_phase_san.resize(old + n, axis=0)
        self._ds_phase_san[old:old + n] = psan

        self._ds_ts.resize(old + n, axis=0)
        self._ds_ts[old:old + n] = ts

        self._ds_abs_s.resize(old + n, axis=0)
        self._ds_abs_s[old:old + n] = abs_s

        seq_arr = np.array(self._buf_seq, dtype=np.int64)
        self._ds_seq.resize(old + n, axis=0)
        self._ds_seq[old:old + n] = seq_arr

        cfo_arr = np.array(self._buf_cfo, dtype=np.float64)
        self._ds_cfo.resize(old + n, axis=0)
        self._ds_cfo[old:old + n] = cfo_arr

        if self._ds_ant1 is not None and self._buf_ant1:
            data1 = np.array(self._buf_ant1, dtype=np.complex64)
            n1    = len(data1)           # may differ from n if some pkts had no ant-1 CSI
            old1  = self._ds_ant1.shape[0]
            self._ds_ant1.resize(old1 + n1, axis=0)
            self._ds_ant1[old1:old1 + n1] = data1

        self._file.flush()
        print(f"[logger] Flushed {n} records to HDF5 | Total records: {old + n}")

        self._buf_ant0.clear()
        self._buf_sanitized.clear()
        self._buf_phase_san.clear()
        self._buf_ant1.clear()
        self._buf_ts.clear()
        self._buf_abs_s.clear()
        self._buf_seq.clear()
        self._buf_cfo.clear()

    def log_trial(self, trial_id: str, activity: str,
                  start_ts: float, end_ts: float,
                  uhd_start_ts: float = None, uhd_end_ts: float = None):
        grp = self._file.require_group(f"/sessions/{self._sid}/trials/{trial_id}")
        grp.attrs['activity']   = activity
        grp.attrs['start_host'] = start_ts
        grp.attrs['end_host']   = end_ts
        if uhd_start_ts is not None:
            grp.attrs['start_uhd'] = uhd_start_ts
        if uhd_end_ts is not None:
            grp.attrs['end_uhd']   = uhd_end_ts
        self._file.flush()
