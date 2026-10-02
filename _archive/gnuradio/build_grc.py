"""
build_grc.py  --  Generates ofdm_csi_flowgraph.grc (GRC 3.10 format)
======================================================================
Run:  C:/Users/Vidhisha/radioconda/python.exe build_grc.py
Output: ofdm_csi_flowgraph.grc  (open with gnuradio-companion)

What you will see in simulation:
  - QT GUI Time Sink  : OFDM waveform (I/Q)
  - QT GUI Freq Sink  : 128-pt OFDM spectrum post-channel
  - Terminal console  : live CSI stats from csi_monitor block
  - HDF5 file         : gr_csi_output.h5 (CSI data logged)
"""

import yaml
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "ofdm_csi_flowgraph.grc")
PROJ_PATH = "d:/phase2"

# =============================================================================
# PHY CONSTANTS HEADER  (prepended to every epy_block source)
# No top-level project imports here -- GRC execs this during port discovery
# =============================================================================
PHY_CONSTS = f"""
# Custom 128-pt OFDM PHY constants (from config.py)
_PROJ       = "{PROJ_PATH}"
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
"""

# =============================================================================
# BLOCK SOURCE CODE
# =============================================================================

SRC_TX = PHY_CONSTS + """
import numpy as np
import time

class blk(gr.sync_block):
    \"\"\"Custom 128-pt OFDM TX -- generates STF+LTF+SIGNAL packets\"\"\"

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
"""

SRC_CHANNEL = PHY_CONSTS + """
import numpy as np

class blk(gr.sync_block):
    \"\"\"Phase-continuous AWGN + CFO + SCO + multipath impairment\"\"\"

    def __init__(self,
                 noise_voltage=0.005,
                 cfo_hz=0.0,
                 sco_ppm=0.0,
                 tap_string='1+0j'):
        gr.sync_block.__init__(
            self,
            name='Channel Impairment',
            in_sig=[np.complex64],
            out_sig=[np.complex64])
        self._noise = float(noise_voltage)
        self._cfo   = float(cfo_hz)
        self._sco   = float(sco_ppm)
        self._taps  = np.array(
            [complex(t.strip()) for t in str(tap_string).split(',') if t.strip()],
            dtype=np.complex64)
        self._idx   = 0

    def work(self, input_items, output_items):
        x   = input_items[0].astype(np.complex64).copy()
        N   = len(x)
        if len(self._taps) > 1:
            x = np.convolve(x, self._taps)[:N]
        n   = np.arange(self._idx, self._idx + N, dtype=np.float64)
        x   = x * np.exp(1j * 2.0 * np.pi * self._cfo / FS_FFT * n
                         ).astype(np.complex64)
        if self._sco != 0.0:
            x = x * np.exp(1j * 2.0 * np.pi * self._sco * 1e-6 * n / FS_FFT
                           ).astype(np.complex64)
        s   = self._noise / np.sqrt(2.0)
        noi = (s * np.random.randn(N).astype(np.float32) +
               1j * s * np.random.randn(N).astype(np.float32))
        output_items[0][:] = (x + noi.astype(np.complex64)).astype(np.complex64)
        self._idx += N
        return N
"""

SRC_DETECTOR = PHY_CONSTS + """
import numpy as np

class blk(gr.sync_block):
    \"\"\"Schmidl-Cox STF detector -- adds pkt_start stream tags\"\"\"

    def __init__(self, detect_threshold=0.65, corr_window=16):
        gr.sync_block.__init__(
            self,
            name='STF Packet Detector',
            in_sig=[np.complex64],
            out_sig=[np.complex64])
        self._thresh   = float(detect_threshold)
        self._L        = int(corr_window)
        self._overlap  = np.zeros(2 * int(corr_window) - 1, dtype=np.complex64)
        self._global   = 0
        self._last_det = -STF_LEN

    def work(self, input_items, output_items):
        inp = np.array(input_items[0], dtype=np.complex64)
        output_items[0][:] = inp
        L    = self._L
        work = np.concatenate([self._overlap, inp])
        N    = len(work) - 2 * L
        i    = 0
        while i < N:
            sa = work[i:i + L]
            sb = work[i + L:i + 2 * L]
            P  = np.sum(sa * np.conj(sb))
            R  = np.sum(np.abs(sb) ** 2) + 1e-12
            if (abs(P) ** 2) / (R ** 2) > self._thresh:
                g_idx = self._global - len(self._overlap) + i
                if g_idx - self._last_det > STF_LEN:
                    cfo      = float(np.angle(P)) * FS_FFT / (2.0 * 3.14159265 * L)
                    local_off = max(0, min(g_idx - (self._global - len(inp)),
                                          len(inp) - 1))
                    self.add_item_tag(
                        0,
                        self.nitems_written(0) + local_off,
                        pmt.intern('pkt_start'),
                        pmt.from_float(float(cfo)))
                    self._last_det = g_idx
                i += STF_LEN
            else:
                i += 1
        self._overlap  = work[-(2 * L - 1):]
        self._global  += len(inp)
        return len(inp)
"""

SRC_EXTRACTOR = PHY_CONSTS + """
import numpy as np
import time

class blk(gr.sync_block):
    \"\"\"LTF sync + 2-stage CFO + H_hat[107]. Emits PMT dicts on csi_out.\"\"\"

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
"""

SRC_LOGGER = PHY_CONSTS + """
import numpy as np
import threading
import queue

class blk(gr.basic_block):
    \"\"\"Receives csi_in PMT dicts and writes H_hat to HDF5.\"\"\"

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
            print(f'[Logger] {self._count} pkts', end='\\r', flush=True)

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
"""

SRC_MONITOR = PHY_CONSTS + """
import numpy as np
import time

_BAR = ' .:-=+#@$'

class blk(gr.basic_block):
    \"\"\"Prints live ASCII |H_hat| amplitude bar per packet.\"\"\"

    def __init__(self, update_every_n=5):
        gr.basic_block.__init__(
            self,
            name='CSI Monitor (Live)',
            in_sig=[],
            out_sig=[])
        self.message_port_register_in(pmt.intern('csi_in'))
        self.set_msg_handler(pmt.intern('csi_in'), self._on)
        self._n  = int(update_every_n)
        self._c  = 0
        self._t0 = time.monotonic()

    def _on(self, msg):
        if not pmt.is_dict(msg):
            return
        ok = pmt.to_bool(pmt.dict_ref(msg, pmt.intern('H_hat_valid'),
                                       pmt.from_bool(False)))
        if not ok:
            return
        self._c += 1
        if self._c % self._n != 0:
            return
        hv = pmt.dict_ref(msg, pmt.intern('H_hat'), pmt.PMT_NIL)
        if not pmt.is_c32vector(hv):
            return
        H   = np.abs(np.array(pmt.c32vector_elements(hv), dtype=np.complex64))
        cfo = pmt.to_double(pmt.dict_ref(msg, pmt.intern('cfo_hz'),
                                          pmt.from_double(0.0)))
        W   = 54
        bar = ''.join(
            _BAR[min(8, int(np.mean(H[int(i*len(H)/W):int((i+1)*len(H)/W)]) * 4))]
            for i in range(W))
        rate = self._c / max(time.monotonic() - self._t0, 0.001)
        print(f'\\r  pkt#{self._c:>5}  cfo={cfo:>8.1f}Hz  [{bar}]  '
              f'mean={np.mean(H):.3f}  {rate:.1f}/s', end='', flush=True)
"""

# =============================================================================
# GRC DOCUMENT ASSEMBLY
# =============================================================================

def var_blk(name, value, x, comment=""):
    return {
        "name": name, "id": "variable",
        "parameters": {"comment": comment, "value": value},
        "states": {"bus_sink": False, "bus_source": False, "bus_structure": None,
                   "coordinate": [x, 8], "rotation": 0, "state": "enabled"},
    }

def epy_blk(name, x, y, params, src):
    p = {"_source_code": src, "affinity": "", "alias": name,
         "comment": "", "maxoutbuf": "0", "minoutbuf": "0"}
    p.update(params)
    return {
        "name": name, "id": "epy_block", "parameters": p,
        "states": {"bus_sink": False, "bus_source": False, "bus_structure": None,
                   "coordinate": [x, y], "rotation": 0, "state": "enabled"},
    }

doc = {
    "options": {
        "parameters": {
            "author":             "Custom OFDM PHY",
            "catch_exceptions":   "True",
            "category":           "[GRC Hier Blocks]",
            "cmake_opt":          "",
            "comment":            "",
            "copyright":          "",
            "description":        "Custom 128-pt OFDM PHY real-time CSI sensing",
            "gen_cmake":          "On",
            "gen_linking":        "dynamic",
            "generate_options":   "qt_gui",
            "hier_block_src_path":".:",
            "id":                 "ofdm_csi_flowgraph",
            "max_nouts":          "0",
            "output_language":    "python",
            "placement":          "(0, 0)",
            "qt_qss_theme":       "",
            "realtime_scheduling":"",
            "run":                "True",
            "run_command":        "{python} -u {filename}",
            "run_options":        "prompt",
            "sizing_mode":        "fixed",
            "thread_safe_setters":"",
            "title":              "128-pt Custom OFDM PHY -- CSI Simulation",
            "window_size":        "(1400, 1000)",
        },
        "states": {
            "bus_sink": False, "bus_source": False, "bus_structure": None,
            "coordinate": [8, 8], "rotation": 0, "state": "enabled",
        },
    },

    "blocks": [
        # ── Variables ─────────────────────────────────────────────────────
        var_blk("samp_rate",     "20e6",               40,  "Sample rate (20 MHz = FS_FFT)"),
        var_blk("noise_voltage", "0.005",             200,  "AWGN noise voltage (0.001-0.1)"),
        var_blk("cfo_hz",        "500.0",             380,  "Injected CFO in Hz"),
        var_blk("sco_ppm",       "1.0",               560,  "Injected SCO in ppm"),
        var_blk("tap_string",    '"1+0j"',             740,  "Multipath taps (comma-sep complex)"),
        var_blk("hdf5_path",     '"gr_csi_output.h5"', 920, "Output HDF5 file path"),
        var_blk("session_id",    '"gr_session_001"',  1100, "HDF5 session group name"),

        # ── Throttle: rate-limits to real-time 20 MHz ─────────────────────
        {
            "name": "throttle",
            "id": "blocks_throttle",
            "parameters": {
                "affinity": "", "alias": "throttle", "comment": "",
                "ignoretag": "True", "maxoutbuf": "0", "minoutbuf": "0",
                "samples_per_second": "samp_rate",
                "type": "complex", "vlen": "1",
            },
            "states": {
                "bus_sink": False, "bus_source": False, "bus_structure": None,
                "coordinate": [220, 280], "rotation": 0, "state": "enabled",
            },
        },

        # ── QT Time Sink: OFDM waveform (I+jQ) ─────────────────────────
        {
            "name": "qtgui_time_sink",
            "id": "qtgui_time_sink_x",
            "parameters": {
                "affinity": "", "alias": "", "comment": "",
                "alpha1": "1.0", "alpha2": "1.0", "alpha3": "1.0",
                "alpha4": "1.0", "alpha5": "1.0",
                "autoscale": "True", "axislabels": "True",
                "color1": '"blue"', "color2": '"red"', "color3": '"green"',
                "color4": '"black"', "color5": '"cyan"',
                "ctrlpanel": "False", "entags": "True", "grid": "True",
                "gui_hint": "",
                "label1": '"I"', "label2": '"Q"',
                "label3": '""', "label4": '""', "label5": '""',
                "legend": "True",
                "marker1": "-1", "marker2": "-1", "marker3": "-1",
                "marker4": "-1", "marker5": "-1",
                "maxoutbuf": "0", "minoutbuf": "0",
                "name": '"OFDM TX Waveform"',
                "nconnections": "1", "nsamps": "1280",
                "showports": "False", "stem": "False",
                "style1": "1", "style2": "1", "style3": "1",
                "style4": "1", "style5": "1",
                "tr_chan": "0", "tr_delay": "0", "tr_level": "0.0",
                "tr_mode": "qtgui.TRIG_MODE_FREE",
                "tr_slope": "qtgui.TRIG_SLOPE_POS", "tr_tag": '""',
                "type": "complex", "update_time": "0.10",
                "width1": "1", "width2": "1", "width3": "1",
                "width4": "1", "width5": "1",
                "ylabel": "Amplitude", "ymax": "2", "ymin": "-2", "yunit": '""',
            },
            "states": {
                "bus_sink": False, "bus_source": False, "bus_structure": None,
                "coordinate": [220, 480], "rotation": 0, "state": "enabled",
            },
        },

        # ── QT Frequency Sink: 128-pt OFDM spectrum ───────────────────────
        {
            "name": "qtgui_freq_sink",
            "id": "qtgui_freq_sink_x",
            "parameters": {
                "affinity": "", "alias": "", "comment": "",
                "alpha1": "1.0", "alpha2": "1.0", "alpha3": "1.0",
                "alpha4": "1.0", "alpha5": "1.0",
                "autoscale": "False", "average": "0.2", "axislabels": "True",
                "bw": "samp_rate",
                "color1": '"blue"', "color2": '"red"', "color3": '"green"',
                "color4": '"black"', "color5": '"cyan"',
                "ctrlpanel": "False", "fc": "0", "fftsize": "128",
                "freqhalf": "True", "grid": "True", "gui_hint": "",
                "label": "Relative Gain", "label1": '"Post-Channel"',
                "legend": "True", "maxoutbuf": "0", "minoutbuf": "0",
                "name": '"OFDM Spectrum (128-pt)"',
                "nconnections": "1", "norm_window": "False",
                "showlog": "False", "showports": "False",
                "type": "complex", "update_time": "0.10",
                "win_size": "", "wintype": "window.WIN_BLACKMAN_hARRIS",
                "ymax": "10", "ymin": "-140",
            },
            "states": {
                "bus_sink": False, "bus_source": False, "bus_structure": None,
                "coordinate": [500, 480], "rotation": 0, "state": "enabled",
            },
        },

        # ── EPY processing blocks ─────────────────────────────────────────
        epy_blk("ofdm_tx", 40, 280,
                {"mod_scheme":        "'BPSK'",
                 "payload_bytes":     "60",
                 "packet_interval_s": "0.01",
                 "n_packets":         "0"},
                SRC_TX),

        epy_blk("channel_impairment", 440, 280,
                {"noise_voltage": "noise_voltage",
                 "cfo_hz":        "cfo_hz",
                 "sco_ppm":       "sco_ppm",
                 "tap_string":    "tap_string"},
                SRC_CHANNEL),

        epy_blk("packet_detector", 680, 280,
                {"detect_threshold": "0.65",
                 "corr_window":      "16"},
                SRC_DETECTOR),

        epy_blk("csi_extractor", 880, 280,
                {"n_data_symbols": "0"},
                SRC_EXTRACTOR),

        epy_blk("csi_logger", 1100, 440,
                {"hdf5_path":      "hdf5_path",
                 "session_id":     "session_id",
                 "n_rx_channels":  "1",
                 "flush_interval": "50"},
                SRC_LOGGER),

        epy_blk("csi_monitor", 1100, 120,
                {"update_every_n": "5"},
                SRC_MONITOR),
    ],

    "connections": [
        # Sample stream chain
        ["ofdm_tx",           "0",       "throttle",           "0"],
        ["throttle",          "0",       "channel_impairment", "0"],
        ["throttle",          "0",       "qtgui_time_sink",    "0"],  # waveform plot
        ["channel_impairment","0",       "packet_detector",    "0"],
        ["channel_impairment","0",       "qtgui_freq_sink",    "0"],  # spectrum plot
        ["packet_detector",   "0",       "csi_extractor",      "0"],
        # Message stream chain
        ["csi_extractor",     "csi_out", "csi_logger",         "csi_in"],
        ["csi_extractor",     "csi_out", "csi_monitor",        "csi_in"],
    ],

    "metadata": {
        "file_format":  1,
        "grc_version":  "3.10.0.0",
    },
}

# =============================================================================
# Write and validate
# =============================================================================
with open(OUT, "w", encoding="utf-8") as f:
    yaml.dump(doc, f, default_flow_style=False, allow_unicode=True,
              sort_keys=False, width=120)

with open(OUT, encoding="utf-8") as f:
    check = yaml.safe_load(f)

epy_blocks = [b for b in check["blocks"] if b["id"] == "epy_block"]
gui_blocks = [b for b in check["blocks"] if b["id"] in
              ("qtgui_time_sink_x", "qtgui_freq_sink_x", "blocks_throttle")]

print(f"[OK] Written : {OUT}")
print(f"     Blocks  : {len(check['blocks'])}  "
      f"(epy={len(epy_blocks)}, gui={len(gui_blocks)})")
print(f"     Connects: {len(check['connections'])}")
print()
print("  GUI display blocks:")
for b in gui_blocks:
    print(f"    {b['name']:25} id={b['id']}")
print()
print("  EPY block checks:")
all_ok = True
for b in epy_blocks:
    src    = b["parameters"]["_source_code"]
    ok_cls = "class blk" in src
    ok_gr  = "from gnuradio import gr" in src
    ok_str = isinstance(src, str)
    status = "[OK]" if (ok_cls and ok_gr and ok_str) else "[!!]"
    print(f"  {status}  {b['name']:30}  class={ok_cls}  gr={ok_gr}  chars={len(src)}")
    if not (ok_cls and ok_gr and ok_str):
        all_ok = False

print()
if all_ok:
    print("All blocks valid.")
    print()
    print("Open in GRC:")
    print(f"  C:/Users/Vidhisha/radioconda/Scripts/gnuradio-companion.exe  {OUT}")
    print()
    print("What you will see when you press Run (F6):")
    print("  [GUI window]  OFDM TX Waveform   -- time-domain I/Q plot")
    print("  [GUI window]  OFDM Spectrum       -- 128-pt FFT spectrum")
    print("  [GRC console] pkt# / cfo / |H|   -- per-packet CSI stats")
    print("  [HDF5 file]   gr_csi_output.h5   -- CSI data saved to disk")
else:
    print("[!!] Some blocks failed -- see above")
