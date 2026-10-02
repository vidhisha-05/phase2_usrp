#!/usr/bin/env python3
"""
generate_grc.py  --  Generates a valid GRC 3.10 flowgraph file
================================================================
Uses json.dumps() to properly escape Python source code into
YAML-safe double-quoted strings -- the exact format GRC requires.

Run once:
    cd d:\phase2\gnuradio
    python generate_grc.py
    -> ofdm_csi_flowgraph.grc  (open in gnuradio-companion)
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Embedded block source code (each class blk(gr.xxx) for GRC's epy_block)
# GRC injects:  from gnuradio import gr, pmt  BEFORE running _source_code.
# ---------------------------------------------------------------------------

SRC = {}

SRC['ofdm_tx'] = r"""
import numpy as np
import time
import sys
import os

_PROJ = r"d:\phase2"
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

import config as cfg
import waveform

class blk(gr.sync_block):
    """Custom OFDM TX -- generates 128-pt STF+LTF+SIGNAL packets."""

    def __init__(self,
                 mod_scheme='BPSK',
                 payload_bytes=60,
                 packet_interval_s=0.01,
                 n_packets=0):
        gr.sync_block.__init__(
            self,
            name='Custom OFDM TX (128-pt)',
            in_sig=[],
            out_sig=[np.complex64],
        )
        self._payload_bits = payload_bytes * 8
        self._interval_s   = packet_interval_s
        self._n_packets    = n_packets
        self._pkt_count    = 0
        self._buf          = np.array([], dtype=np.complex64)
        self._silence_pad  = int(packet_interval_s * cfg.FS_FFT)
        self._last_tx      = time.monotonic()
        self.set_output_multiple(cfg.SYMBOL_LEN)

    def work(self, input_items, output_items):
        out = output_items[0]
        n   = len(out)
        ptr = 0
        while ptr < n:
            if len(self._buf) == 0:
                elapsed = time.monotonic() - self._last_tx
                if elapsed < self._interval_s:
                    zeros = min(n - ptr,
                                int((self._interval_s - elapsed) * cfg.FS_FFT))
                    out[ptr:ptr + zeros] = 0
                    ptr += zeros
                    continue
                if self._n_packets > 0 and self._pkt_count >= self._n_packets:
                    out[ptr:] = 0
                    return len(out)
                bits = np.random.randint(0, 2, self._payload_bits, dtype=np.uint8)
                pkt  = waveform.assemble_packet(bits)
                pad  = np.zeros(self._silence_pad, dtype=np.complex64)
                self._buf       = np.concatenate([pkt.astype(np.complex64), pad])
                self._pkt_count += 1
                self._last_tx   = time.monotonic()
            chunk = min(n - ptr, len(self._buf))
            out[ptr:ptr + chunk] = self._buf[:chunk]
            self._buf = self._buf[chunk:]
            ptr += chunk
        return n
"""

SRC['channel_impairment'] = r"""
import numpy as np
import sys

_PROJ = r"d:\phase2"
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

import config as cfg

class blk(gr.sync_block):
    """Channel: AWGN + phase-continuous CFO + SCO + multipath."""

    def __init__(self,
                 noise_voltage=0.005,
                 cfo_hz=0.0,
                 sco_ppm=0.0,
                 tap_string='1+0j'):
        gr.sync_block.__init__(
            self,
            name='Channel Impairment',
            in_sig=[np.complex64],
            out_sig=[np.complex64],
        )
        taps = [complex(t.strip()) for t in tap_string.split(',') if t.strip()]
        self._taps  = np.array(taps or [1.0+0j], dtype=np.complex64)
        self._noise = noise_voltage
        self._cfo   = cfo_hz
        self._sco   = sco_ppm
        self._fs    = cfg.FS_FFT
        self._idx   = 0

    def work(self, input_items, output_items):
        x = input_items[0].astype(np.complex64).copy()
        if len(self._taps) > 1:
            x = np.convolve(x, self._taps)[:len(input_items[0])]
        n   = np.arange(self._idx, self._idx + len(x), dtype=np.float64)
        x   = x * np.exp(1j * 2 * np.pi * self._cfo / self._fs * n).astype(np.complex64)
        if self._sco != 0.0:
            x = x * np.exp(1j * 2 * np.pi * self._sco * 1e-6 * n / self._fs).astype(np.complex64)
        noise = (self._noise / np.sqrt(2)) * (
            np.random.randn(len(x)).astype(np.float32) +
            1j * np.random.randn(len(x)).astype(np.float32))
        output_items[0][:] = (x + noise.astype(np.complex64)).astype(np.complex64)
        self._idx += len(x)
        return len(input_items[0])
"""

SRC['packet_detector'] = r"""
import numpy as np
import sys
import pmt

_PROJ = r"d:\phase2"
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

import config as cfg

CORR_WINDOW = 16

class blk(gr.sync_block):
    """STF Detector: Schmidl & Cox. Attaches pkt_start stream tags."""

    def __init__(self, detect_threshold=0.65, corr_window=16):
        gr.sync_block.__init__(
            self,
            name='STF Packet Detector',
            in_sig=[np.complex64],
            out_sig=[np.complex64],
        )
        self._thresh   = detect_threshold
        self._L        = corr_window
        self._overlap  = np.zeros(2 * corr_window - 1, dtype=np.complex64)
        self._global   = 0
        self._last_det = -cfg.STF_LEN

    def work(self, input_items, output_items):
        inp = input_items[0]
        output_items[0][:] = inp
        L    = self._L
        work = np.concatenate([self._overlap, inp.astype(np.complex64)])
        N    = len(work) - 2 * L
        i    = 0
        while i < N:
            seg_a = work[i:i + L]
            seg_b = work[i + L:i + 2 * L]
            P  = np.sum(seg_a * np.conj(seg_b))
            R  = np.sum(np.abs(seg_b) ** 2) + 1e-12
            M2 = (abs(P) ** 2) / (R ** 2)
            if M2 > self._thresh:
                g_idx = self._global - len(self._overlap) + i
                if g_idx - self._last_det > cfg.STF_LEN:
                    phi = float(np.angle(P))
                    cfo = phi * cfg.FS_FFT / (2 * np.pi * L)
                    local_off = max(0, min(
                        g_idx - (self._global - len(inp)), len(inp) - 1))
                    self.add_item_tag(
                        0,
                        self.nitems_written(0) + local_off,
                        pmt.intern('pkt_start'),
                        pmt.from_float(float(cfo)),
                    )
                    self._last_det = g_idx
                i += cfg.STF_LEN
            else:
                i += 1
        self._overlap  = work[-(2 * L - 1):]
        self._global  += len(inp)
        return len(inp)
"""

SRC['csi_extractor'] = r"""
import numpy as np
import time
import sys
import pmt

_PROJ = r"d:\phase2"
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

import config as cfg
from sync import sync_packet, estimate_fine_cfo, apply_cfo_correction, extract_csi

class blk(gr.sync_block):
    """CSI Extractor: reads pkt_start tags -> LTF sync -> H_hat[107]."""

    def __init__(self, n_data_symbols=0):
        gr.sync_block.__init__(
            self,
            name='CSI Extractor (128-pt)',
            in_sig=[np.complex64],
            out_sig=[],
        )
        self.message_port_register_out(pmt.intern('csi_out'))
        self._buf_len = (cfg.STF_LEN + cfg.LTF_LEN + cfg.SIG_LEN +
                         n_data_symbols * cfg.SYMBOL_LEN + 64)
        self._pending = []   # list of [coarse_cfo, buffer_ndarray]
        self._seq     = 0

    def work(self, input_items, output_items):
        inp  = np.array(input_items[0], dtype=np.complex64)
        base = self.nitems_read(0)
        tags = self.get_tags_in_window(0, 0, len(inp), pmt.intern('pkt_start'))
        ptr  = 0
        for tag in sorted(tags, key=lambda t: t.offset):
            local = int(tag.offset - base)
            self._feed_pending(inp[ptr:local])
            ptr = local
            cfo = pmt.to_float(tag.value)
            self._pending.append([cfo, np.array([], dtype=np.complex64)])
        self._feed_pending(inp[ptr:])
        self._process_ready()
        return len(input_items[0])

    def _feed_pending(self, chunk):
        if len(chunk) == 0:
            return
        for entry in self._pending:
            need = self._buf_len - len(entry[1])
            if need > 0:
                entry[1] = np.concatenate(
                    [entry[1], chunk[:need].astype(np.complex64)])

    def _process_ready(self):
        ready         = [(c, b) for c, b in self._pending if len(b) >= self._buf_len]
        self._pending = [(c, b) for c, b in self._pending if len(b) < self._buf_len]
        for coarse_cfo, buf in ready:
            try:
                p_cc  = apply_cfo_correction(buf, coarse_cfo)
                s1    = cfg.STF_LEN + 64
                s2    = s1 + cfg.FFT_SIZE
                fine  = estimate_fine_cfo(p_cc[s1:s1 + cfg.FFT_SIZE],
                                          p_cc[s2:s2 + cfg.FFT_SIZE])
                total = coarse_cfo + fine
                H     = extract_csi(p_cc, ltf_start=cfg.STF_LEN,
                                    total_cfo_hz=total)
                valid = (not np.any(np.isnan(H)) and np.mean(np.abs(H)) > 0.05)
                d = pmt.make_dict()
                d = pmt.dict_add(d, pmt.intern('seq'),
                                 pmt.from_long(self._seq))
                d = pmt.dict_add(d, pmt.intern('timestamp'),
                                 pmt.from_double(time.time()))
                d = pmt.dict_add(d, pmt.intern('cfo_hz'),
                                 pmt.from_double(float(total)))
                d = pmt.dict_add(d, pmt.intern('H_hat_valid'),
                                 pmt.from_bool(bool(valid)))
                if valid:
                    h_v = pmt.init_c32vector(
                        len(H), [complex(v) for v in H.tolist()])
                    d = pmt.dict_add(d, pmt.intern('H_hat'), h_v)
                self.message_port_pub(pmt.intern('csi_out'), d)
                self._seq += 1
            except Exception:
                pass
"""

SRC['csi_logger'] = r"""
import numpy as np
import sys
import threading
import queue
import pmt

_PROJ = r"d:\phase2"
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

import config as cfg

class blk(gr.basic_block):
    """HDF5 CSI Logger: writes H_hat from csi_in message port."""

    def __init__(self,
                 hdf5_path='gr_csi_output.h5',
                 session_id='gr_session_001',
                 n_rx_channels=1,
                 flush_interval=50):
        gr.basic_block.__init__(
            self, name='HDF5 CSI Logger', in_sig=[], out_sig=[])
        self.message_port_register_in(pmt.intern('csi_in'))
        self.set_msg_handler(pmt.intern('csi_in'), self._on_csi)
        self._q       = queue.Queue(maxsize=2000)
        self._path    = hdf5_path
        self._session = session_id
        self._n_ch    = n_rx_channels
        self._flush_n = flush_interval
        self._running = True
        self._count   = 0
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _on_csi(self, msg):
        if not pmt.is_dict(msg):
            return
        valid = pmt.to_bool(
            pmt.dict_ref(msg, pmt.intern('H_hat_valid'), pmt.from_bool(False)))
        if not valid:
            return
        h_v = pmt.dict_ref(msg, pmt.intern('H_hat'), pmt.PMT_NIL)
        if not pmt.is_c32vector(h_v):
            return
        H  = np.array(pmt.c32vector_elements(h_v), dtype=np.complex64)
        ts = pmt.to_double(
            pmt.dict_ref(msg, pmt.intern('timestamp'), pmt.from_double(0.0)))
        sq = pmt.to_long(
            pmt.dict_ref(msg, pmt.intern('seq'), pmt.from_long(0)))
        try:
            self._q.put_nowait({'H_hat': H, 'timestamp': ts, 'seq': sq})
        except queue.Full:
            pass
        self._count += 1
        if self._count % 10 == 0:
            print(f'[Logger] {self._count} packets received', end='\r', flush=True)

    def _run(self):
        import h5py
        buf = []
        with h5py.File(self._path, 'a') as hf:
            grp = hf.require_group(f'sessions/{self._session}/csi')
            if 'antenna0' not in grp:
                ds = grp.create_dataset(
                    'antenna0', shape=(0, cfg.NUM_ACTIVE),
                    maxshape=(None, cfg.NUM_ACTIVE),
                    dtype=np.complex64, chunks=(100, cfg.NUM_ACTIVE))
            else:
                ds = grp['antenna0']
            print(f'[HDF5] Open: {self._path}  session={self._session}')
            while self._running or not self._q.empty():
                try:
                    rec = self._q.get(timeout=0.5)
                except Exception:
                    continue
                buf.append(rec)
                if len(buf) >= self._flush_n:
                    rows = np.array([r['H_hat'] for r in buf], dtype=np.complex64)
                    n0   = ds.shape[0]
                    ds.resize(n0 + len(rows), axis=0)
                    ds[n0:n0 + len(rows)] = rows
                    hf.flush()
                    print(f'[HDF5] Flushed {len(buf)} pkts  total={n0 + len(rows)}')
                    buf.clear()
            if buf:
                rows = np.array([r['H_hat'] for r in buf], dtype=np.complex64)
                n0   = ds.shape[0]
                ds.resize(n0 + len(rows), axis=0)
                ds[n0:n0 + len(rows)] = rows
                hf.flush()
            print(f'[HDF5] Closed. Total: {ds.shape[0]}')

    def stop(self):
        self._running = False
        self._thread.join(timeout=5)
        return True
"""

SRC['csi_monitor'] = r"""
import numpy as np
import sys
import time
import pmt

_PROJ = r"d:\phase2"
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

import config as cfg

class blk(gr.basic_block):
    """Live terminal CSI display: ASCII amplitude bar per packet."""

    def __init__(self, update_every_n=5):
        gr.basic_block.__init__(
            self, name='CSI Monitor (Live)', in_sig=[], out_sig=[])
        self.message_port_register_in(pmt.intern('csi_in'))
        self.set_msg_handler(pmt.intern('csi_in'), self._on_csi)
        self._n     = update_every_n
        self._count = 0
        self._t0    = time.monotonic()

    def _on_csi(self, msg):
        if not pmt.is_dict(msg):
            return
        valid = pmt.to_bool(
            pmt.dict_ref(msg, pmt.intern('H_hat_valid'), pmt.from_bool(False)))
        if not valid:
            return
        self._count += 1
        if self._count % self._n != 0:
            return
        h_v = pmt.dict_ref(msg, pmt.intern('H_hat'), pmt.PMT_NIL)
        if not pmt.is_c32vector(h_v):
            return
        H    = np.abs(np.array(pmt.c32vector_elements(h_v), dtype=np.complex64))
        cfo  = pmt.to_double(
            pmt.dict_ref(msg, pmt.intern('cfo_hz'), pmt.from_double(0.0)))
        W    = 54
        BLKS = ' .:-=+#@$'
        bar  = ''.join(
            BLKS[min(8, int(np.mean(H[int(i*len(H)/W):int((i+1)*len(H)/W)])*4))]
            for i in range(W))
        rate = self._count / max(time.monotonic() - self._t0, 0.001)
        print(
            f'\r  pkt#{self._count:>5}  cfo={cfo:>8.1f}Hz'
            f'  [{bar}]  mean={np.mean(H):.3f}  {rate:.1f}/s',
            end='', flush=True)
"""

# ---------------------------------------------------------------------------
# GRC 3.10 block template
# _source_code MUST be a JSON-encoded string (double-quoted, \n-escaped)
# ---------------------------------------------------------------------------

def epy_block(name, x, y, extra_params, src_key):
    """Produce a YAML epy_block entry with properly encoded source code."""
    # json.dumps produces "...\n..." which is valid YAML double-quoted string
    src_encoded = json.dumps(SRC[src_key].strip() + '\n')
    extra = ''.join(f'    {k}: {v}\n' for k, v in extra_params.items())
    return f"""\
- name: {name}
  id: epy_block
  parameters:
    _source_code: {src_encoded}
    affinity: ''
    alias: {name}
    comment: ''
    maxoutbuf: '0'
    minoutbuf: '0'
{extra}\
  states:
    bus_sink: false
    bus_source: false
    bus_structure: null
    coordinate: [{x}, {y}]
    rotation: 0
    state: enabled

"""


# ---------------------------------------------------------------------------
# Assemble the flowgraph
# ---------------------------------------------------------------------------

grc = """\
options:
  parameters:
    author: Custom OFDM PHY
    catch_exceptions: 'True'
    category: '[GRC Hier Blocks]'
    cmake_opt: ''
    comment: ''
    copyright: ''
    description: Custom 128-pt OFDM PHY -- real-time CSI sensing
    gen_cmake: 'On'
    gen_linking: dynamic
    hier_block_src_path: '.:'
    id: ofdm_csi_flowgraph
    max_nouts: '0'
    output_language: python
    placement: (0, 0)
    qt_qss_theme: ''
    realtime_scheduling: ''
    run: run_now
    run_command: '{python} -u {filename}'
    run_options: prompt
    sizing_mode: fixed
    thread_safe_setters: ''
    title: 128-pt Custom OFDM PHY -- CSI Simulation
    window_size: (1400, 1000)
  states:
    bus_sink: false
    bus_source: false
    bus_structure: null
    coordinate: [8, 8]
    rotation: 0
    state: enabled

blocks:
"""

# Variables
VARS = [
    ('noise_voltage', '0.005',              200,  'AWGN noise voltage'),
    ('cfo_hz',        '500.0',              360,  'Injected CFO Hz'),
    ('sco_ppm',       '1.0',               520,  'Injected SCO ppm'),
    ('tap_string',    '"1+0j"',            680,  'Multipath taps'),
    ('hdf5_path',     '"gr_csi_output.h5"', 880,  'Output HDF5 path'),
    ('session_id',    '"gr_session_001"',  1080,  'HDF5 session ID'),
]
for vname, vval, vx, vcmt in VARS:
    grc += f"""\
- name: {vname}
  id: variable
  parameters:
    comment: '{vcmt}'
    value: '{vval}'
  states:
    bus_sink: false
    bus_source: false
    bus_structure: null
    coordinate: [{vx}, 8]
    rotation: 0
    state: enabled

"""

# Blocks
grc += epy_block('ofdm_tx', 40, 200,
    {'mod_scheme': "'BPSK'",
     'payload_bytes': "'60'",
     'packet_interval_s': "'0.01'",
     'n_packets': "'0'"},
    'ofdm_tx')

grc += epy_block('channel_impairment', 280, 200,
    {'noise_voltage': 'noise_voltage',
     'cfo_hz':        'cfo_hz',
     'sco_ppm':       'sco_ppm',
     'tap_string':    'tap_string'},
    'channel_impairment')

grc += epy_block('packet_detector', 520, 200,
    {'detect_threshold': "'0.65'",
     'corr_window':      "'16'"},
    'packet_detector')

grc += epy_block('csi_extractor', 760, 200,
    {'n_data_symbols': "'0'"},
    'csi_extractor')

grc += epy_block('csi_logger', 1000, 320,
    {'hdf5_path':      'hdf5_path',
     'session_id':     'session_id',
     'n_rx_channels':  "'1'",
     'flush_interval': "'50'"},
    'csi_logger')

grc += epy_block('csi_monitor', 1000, 80,
    {'update_every_n': "'5'"},
    'csi_monitor')

# Connections
grc += """\
connections:
- [ofdm_tx, '0', channel_impairment, '0']
- [channel_impairment, '0', packet_detector, '0']
- [packet_detector, '0', csi_extractor, '0']
- [csi_extractor, csi_out, csi_logger, csi_in]
- [csi_extractor, csi_out, csi_monitor, csi_in]

metadata:
  file_format: 1
  grc_version: 3.10.0.0
"""

# Write file
out = os.path.join(HERE, 'ofdm_csi_flowgraph.grc')
with open(out, 'w', encoding='utf-8') as f:
    f.write(grc)

# Verify it parses as valid YAML
import yaml
with open(out, encoding='utf-8') as f:
    doc = yaml.safe_load(f)
print(f"[OK] Written: {out}")
print(f"     Blocks:  {len(doc['blocks'])}")
print(f"     Lines:   {grc.count(chr(10))}")
print()
print("Open in GNU Radio Companion:")
print("  gnuradio-companion ofdm_csi_flowgraph.grc")
print()
print("Or run directly (no GRC needed):")
print("  cd d:\\phase2\\gnuradio && python gr_flowgraph.py")
