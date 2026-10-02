"""
rx_sim.py — RX process: simulation mode (ZMQ PULL).

Pulls IQ samples from channel_bridge, runs packet detection, synchronization,
and CSI extraction.  Passes CSI records to logger.py via a queue.

Reference: phase2 (2).md — Sections 8, 12, 15, 17
"""

import time
import struct
import queue
import threading
import numpy as np
import zmq
import config as cfg
from detector import PacketDetector
from sync import sync_packet

_HEADER_FMT  = "<Qd"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)


def _unpack(msg: bytes):
    seq, ts = struct.unpack(_HEADER_FMT, msg[:_HEADER_SIZE])
    samples = np.frombuffer(msg[_HEADER_SIZE:], dtype=np.complex64)
    return seq, ts, samples


# ─────────────────────────────────────────────────────────────────────────────
# Ring Buffer  (single-producer / single-consumer, Section 15)
# ─────────────────────────────────────────────────────────────────────────────

class RingBuffer:
    """Array-backed ring buffer for complex64 samples.

    FIX: write() and read() are fully vectorized with NumPy slice operations,
         replacing the previous per-sample Python loops (O(n) overhead per sample).
    """

    def __init__(self, capacity: int = 1 << 20):   # 1 M samples default
        self._buf     = np.zeros(capacity, dtype=np.complex64)
        self._cap     = capacity
        self._write   = 0
        self._read    = 0
        self._count   = 0
        self._dropped = 0
        self._lock    = threading.Lock()

    def write(self, samples: np.ndarray):
        with self._lock:
            n = len(samples)
            if n > self._cap - self._count:
                overflow = n - (self._cap - self._count)
                self._read   = (self._read + overflow) % self._cap
                self._count -= overflow
                self._dropped += overflow
            # Vectorized wrap-around write
            first = min(n, self._cap - self._write)
            self._buf[self._write:self._write + first] = samples[:first]
            if first < n:
                self._buf[:n - first] = samples[first:]
            self._write = (self._write + n) % self._cap
            self._count += n

    def read(self, n: int) -> np.ndarray:
        with self._lock:
            available = min(n, self._count)
            if available == 0:
                return np.array([], dtype=np.complex64)
            out = self._linear_read(available)
            self._read  = (self._read + available) % self._cap
            self._count -= available
            return out

    def peek(self, n: int) -> np.ndarray:
        """Read without consuming."""
        with self._lock:
            return self._linear_read(min(n, self._count))

    def _linear_read(self, n: int) -> np.ndarray:
        """Non-locking vectorized read of n samples (caller holds lock)."""
        first = min(n, self._cap - self._read)
        if first == n:
            return self._buf[self._read:self._read + n].copy()
        return np.concatenate([
            self._buf[self._read:self._read + first],
            self._buf[:n - first]
        ])

    @property
    def occupancy(self) -> float:
        """Return buffer fill as a ratio in [0.0, 1.0].
        FIX (Bug 3): previously returned raw integer self._count;
        callers need a normalised ratio for overflow monitoring."""
        with self._lock:
            return self._count / self._cap

    @property
    def dropped(self) -> int:
        with self._lock:
            return self._dropped



# ─────────────────────────────────────────────────────────────────────────────
# RX Simulation Process
# ─────────────────────────────────────────────────────────────────────────────

# Packet window: STF + LTF + SIGNAL + some DATA symbols (look-ahead budget)
_PKT_WINDOW = cfg.STF_LEN + cfg.LTF_LEN + cfg.SIG_LEN + 4 * cfg.SYMBOL_LEN

# Default ant-1 channel parameters (Issue 12: configurable via kwarg)
_DEFAULT_ANT1_CH = dict(noise_voltage=0.008, cfo_hz=120.0, sco_ppm=0.5, seed=137)


def run_rx_sim(csi_queue: queue.Queue,
               pull_addr:       str   = cfg.ZMQ_RX_ADDR,
               n_packets:       int   = 0,
               n_rx_channels:   int   = 1,
               ant1_pull_addr:  str   = cfg.ZMQ_ANT1_ADDR,
               ant1_channel_params: dict = None):
    """
    Pull IQ samples from channel_bridge; detect packets; extract CSI.

    Issue 1 fix — unified stream index:
      buf_abs_start is derived from det._sample_idx after each process() call,
      not maintained as an independent counter.  This eliminates drift.

    Issue 2 fix — accurate metadata timestamp:
      A deque of (chunk_end_abs, seq, ts) records is appended whenever a chunk
      is written to the ring.  At detection time, we look up the earliest entry
      whose chunk_end_abs >= abs_start to get the correct (seq, ts).

    Issue 3 fix — sample-aligned antenna streams:
      Ant-0 and ant-1 ZMQ messages are paired in a single list of
      (samples_a0, samples_a1) tuples collected in the same poll cycle.
      Both ring buffers are written atomically from the same pair.

    Issue 12 fix — configurable ant-1 channel:
      ant1_channel_params dict overrides _DEFAULT_ANT1_CH.

    Args:
        csi_queue:           Queue consumed by logger.py.
        pull_addr:           ZMQ address of ant-0 impaired stream.
        n_packets:           Stop after N CSI records (0 = infinite).
        n_rx_channels:       1 or 2.
        ant1_pull_addr:      ZMQ address for clean TX stream (ant-1 parallel).
        ant1_channel_params: dict of ChannelModel kwargs for ant-1.
    """
    import collections
    from channel_bridge import ChannelModel

    # ── ZMQ sockets ──────────────────────────────────────────────────────────
    ctx       = zmq.Context()
    sock_ant0 = ctx.socket(zmq.PULL)
    sock_ant0.connect(pull_addr)

    sock_ant1 = None
    ring_ant1 = None
    _ch_ant1  = None
    if n_rx_channels >= 2:
        sock_ant1 = ctx.socket(zmq.PULL)
        sock_ant1.connect(ant1_pull_addr)
        ring_ant1 = RingBuffer()
        params    = ant1_channel_params or _DEFAULT_ANT1_CH
        _ch_ant1  = ChannelModel(**params)

    # ── Ring buffers and detector ─────────────────────────────────────────────
    ring    = RingBuffer()       # ant-0 impaired samples
    det     = PacketDetector()   # tracks det._sample_idx internally
    pkt_cnt = 0

    # look-ahead: last _PKT_WINDOW samples preserved across chunks
    look_ahead    = np.array([], dtype=np.complex64)
    look_ahead_a1 = np.array([], dtype=np.complex64)

    # Issue 2: metadata deque — records (chunk_end_abs_idx, seq, ts)
    # chunk_end_abs_idx = stream-absolute index of the last sample in that chunk
    _meta_deque: collections.deque = collections.deque(maxlen=256)
    _meta_lock  = threading.Lock()
    _stream_abs = [0]   # absolute sample counter for ring writes

    # Issue 3: paired-chunk queue for aligned ant-0 / ant-1 samples
    _paired: list = []          # list of (samples_a0, samples_a1 | None)
    _paired_lock = threading.Lock()

    print(f"[rx_sim] ant0={pull_addr}  n_rx_channels={n_rx_channels}")
    if sock_ant1:
        print(f"[rx_sim] ant1 clean stream={ant1_pull_addr}  "
              f"ch_params={ant1_channel_params or _DEFAULT_ANT1_CH}")

    # ── Acquisition thread ────────────────────────────────────────────────────
    _stop_event = threading.Event()

    def _acquire():
        """
        Issue 3: pair ant-0 and ant-1 ZMQ messages in the same poll cycle.
        Both messages (if available) are appended to _paired as a tuple so the
        processing loop can read them together, guaranteeing alignment.
        """
        while not _stop_event.is_set():
            a0_msg = None
            a1_msg = None
            try:
                a0_msg = sock_ant0.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                pass
            if sock_ant1 is not None:
                try:
                    a1_msg = sock_ant1.recv(flags=zmq.NOBLOCK)
                except zmq.Again:
                    pass
            if a0_msg is not None:
                seq0, ts0, s0 = _unpack(a0_msg)
                s1 = None
                if a1_msg is not None:
                    _, _, s1 = _unpack(a1_msg)
                with _paired_lock:
                    _paired.append((s0, s1, int(seq0), float(ts0)))
            elif a1_msg is not None:
                # Ant-1 arrived without ant-0 (rare); buffer it separately
                _, _, s1 = _unpack(a1_msg)
                with _paired_lock:
                    _paired.append((None, s1, 0, 0.0))
            time.sleep(0.0001)

    acq_thread = threading.Thread(target=_acquire, daemon=True)
    acq_thread.start()

    # ── Processing loop ───────────────────────────────────────────────────────
    try:
        while True:
            # Drain the paired-chunk list into ring buffers
            with _paired_lock:
                batch = list(_paired)
                _paired.clear()

            if not batch:
                time.sleep(0.0005)
                continue

            for (s0, s1, seq, ts) in batch:
                if s0 is not None:
                    ring.write(s0)
                    with _meta_lock:
                        _stream_abs[0] += len(s0)
                        _meta_deque.append((_stream_abs[0] - 1, seq, ts))
                if s1 is not None and ring_ant1 is not None:
                    ring_ant1.write(s1)

            # Read ant-0 chunk
            chunk = ring.read(cfg.ZMQ_CHUNK_SIZE)
            if len(chunk) < cfg.STF_LEN + cfg.LTF_LEN:
                continue

            # Combine look-ahead with new chunk
            buf = np.concatenate([look_ahead, chunk]) if len(look_ahead) else chunk

            # Read matching ant-1 chunk (same size as ant-0 chunk)
            buf_a1 = None
            if ring_ant1 is not None:
                chunk_a1 = ring_ant1.read(cfg.ZMQ_CHUNK_SIZE)
                if len(chunk_a1) > 0:
                    buf_a1 = (np.concatenate([look_ahead_a1, chunk_a1])
                              if len(look_ahead_a1) else chunk_a1)
                    look_ahead_a1 = (buf_a1[-_PKT_WINDOW:]
                                     if len(buf_a1) >= _PKT_WINDOW else buf_a1)

            # ── Packet detection ──────────────────────────────────────────────
            # Issue 1 fix (CORRECTED): feed ONLY the new chunk to the detector.
            # look_ahead was already processed in the previous iteration;
            # re-feeding it would double-count and misplace abs_start indices.
            #
            # chunk_abs_start = absolute stream index of chunk[0].
            # Formula: det._sample_idx = abs idx of det._buf[0] (remaining unprocessed);
            #          det._buf has len(det._buf) samples left from previous call;
            #          therefore chunk[0] is at abs idx = det._sample_idx + len(det._buf).
            chunk_abs_start = det._sample_idx + len(det._buf)  # capture BEFORE process
            detections = det.process(chunk)                      # only NEW samples

            # buf = look_ahead + chunk, used for packet extraction only
            buf = np.concatenate([look_ahead, chunk]) if len(look_ahead) else chunk
            # buf[0] is look_ahead[0] which is at abs idx chunk_abs_start - len(look_ahead)
            buf_abs_start = chunk_abs_start - len(look_ahead)

            for (abs_start, coarse_cfo) in detections:
                # Convert stream-absolute → buf-relative
                start_rel = int(abs_start) - int(buf_abs_start)
                if start_rel < 0 or start_rel + _PKT_WINDOW > len(buf):
                    continue   # packet not fully in this buf; look-ahead will catch it

                pkt_buf = buf[start_rel:start_rel + _PKT_WINDOW]
                result  = sync_packet(pkt_buf, coarse_cfo, n_data_symbols=0)
                H_hat   = result['H_hat']

                if np.any(np.isnan(H_hat)):
                    continue

                # Issue 2 fix: look up (seq, ts) for this abs_start from the deque
                seq_for_pkt, ts_for_pkt = 0, 0.0
                with _meta_lock:
                    for (end_abs, sq, t) in _meta_deque:
                        if end_abs >= abs_start:
                            seq_for_pkt, ts_for_pkt = sq, t
                            break

                csi_record = {
                    'H_hat':     H_hat,
                    'timestamp': ts_for_pkt,
                    'seq':       seq_for_pkt,
                    'pkt_idx':   pkt_cnt,
                    'cfo_hz':    result.get('total_cfo', 0.0),
                    'ring_occ':  ring.occupancy,  # F-05 FIX: occupancy already returns fraction 0.0-1.0
                    'dropped':   ring.dropped,
                }

                # Issue 3 / Bug 3: ant-1 uses CLEAN TX samples → independent CH
                if _ch_ant1 is not None and buf_a1 is not None:
                    if start_rel + _PKT_WINDOW <= len(buf_a1):
                        clean_pkt   = buf_a1[start_rel:start_rel + _PKT_WINDOW]
                        impaired_a1 = _ch_ant1.apply(clean_pkt)
                        r1 = sync_packet(impaired_a1, coarse_cfo, n_data_symbols=0)
                        if not np.any(np.isnan(r1['H_hat'])):
                            csi_record['H_hat_ant1'] = r1['H_hat']

                csi_queue.put(csi_record)
                pkt_cnt += 1
                print(f"[rx_sim] CSI #{pkt_cnt}  "
                      f"cfo={result.get('total_cfo',0):.1f}Hz  "
                      f"dropped={ring.dropped}")

                if n_packets and pkt_cnt >= n_packets:
                    return

            # Update look-ahead — preserve last _PKT_WINDOW samples
            look_ahead = buf[-_PKT_WINDOW:] if len(buf) >= _PKT_WINDOW else buf

    except KeyboardInterrupt:
        print("[rx_sim] Shutting down.")
    finally:
        _stop_event.set()
        sock_ant0.close()
        if sock_ant1:
            sock_ant1.close()
        ctx.term()


if __name__ == "__main__":
    import argparse
    from logger import CSILogger
    p = argparse.ArgumentParser()
    p.add_argument("--n_packets",  type=int, default=0)
    p.add_argument("--hdf5",       type=str, default=cfg.HDF5_FILE_PATH)
    p.add_argument("--n_channels", type=int, default=1,
                   help="1 or 2 RX channels")
    a = p.parse_args()

    q      = queue.Queue(maxsize=1000)
    logger = CSILogger(q, a.hdf5, n_rx_channels=a.n_channels)
    t      = threading.Thread(target=logger.run, daemon=True)
    t.start()
    run_rx_sim(q, n_packets=a.n_packets, n_rx_channels=a.n_channels)

