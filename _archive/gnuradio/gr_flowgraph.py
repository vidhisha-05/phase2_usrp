#!/usr/bin/env python3
"""
gr_flowgraph.py  --  Complete GNU Radio Flowgraph
==================================================
Custom 128-pt OFDM PHY — Real-Time Simulation + CSI Extraction

This file is the SINGLE entry point to run the entire system in GNU Radio.
It wires every block together exactly as the GRC flowgraph diagram shows.

USAGE
-----
Simulation (no hardware):
    python gr_flowgraph.py

With impairments:
    python gr_flowgraph.py --noise 0.02 --cfo 1500 --sco 2.0

Change output file:
    python gr_flowgraph.py --hdf5 my_session.h5 --session_id run_001

Stop after N packets:
    python gr_flowgraph.py --n_packets 200

Hardware (USRP B210):
    python gr_flowgraph.py --mode hardware --freq 2.462e9

FLOWGRAPH DIAGRAM
-----------------

  [OFDM TX Block]
       |
       | complex64 IQ stream (20 MS/s)
       v
  [Channel Impairment Block]   <-- AWGN + CFO + SCO + Multipath
       |
       | complex64 IQ stream (impaired)
       v
  [STF Packet Detector Block]  <-- Schmidl & Cox; attaches stream tags
       |
       | complex64 tagged stream
       v
  [CSI Extractor Block]        <-- LTF sync + 2-stage CFO + H_hat[107]
       |
       | PMT message: {seq, timestamp, cfo_hz, H_hat[107], H_hat_valid}
       +-------+--------+
       |                |
       v                v
[HDF5 Logger]    [CSI Monitor]
  (file)           (terminal)

"""

import sys
import os
import argparse
import signal
import time

# -- Project path setup -------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)   # d:\phase2
for p in [_HERE, _PROJ]:
    if p not in sys.path:
        sys.path.insert(0, p)

import config as cfg

# -- Check GNU Radio availability ---------------------------------------------
try:
    import gnuradio.gr as gr
    from gnuradio import blocks
    import pmt
    _GR_OK = True
except ImportError:
    _GR_OK = False
    print("[ERROR] GNU Radio not found.")
    print("  Install: sudo apt install gnuradio  (Linux)")
    print("           or see https://github.com/ryanvolz/radioconda (Windows/Mac)")
    sys.exit(1)

# -- Import all custom blocks -------------------------------------------------
from gr_block_tx           import blk as TxBlock
from gr_block_channel      import blk as ChannelBlock
from gr_block_detector     import blk as DetectorBlock
from gr_block_csi_extractor import blk as CSIExtractorBlock
from gr_block_csi_logger   import blk as LoggerBlock
from gr_block_csi_monitor  import blk as MonitorBlock


# =============================================================================
# Flowgraph class
# =============================================================================

class OFDMSimFlowgraph(gr.top_block):
    """
    Complete GNU Radio top-level flowgraph:
      TX -> Channel -> Detector -> CSI Extractor -> Logger + Monitor
    """

    def __init__(self, args):
        gr.top_block.__init__(self, "128pt OFDM PHY Simulation")

        # ------------------------------------------------------------------
        # BLOCK 1: TX  (OFDM packet source)
        # ------------------------------------------------------------------
        self.tx = TxBlock(
            mod_scheme        = args.mod,
            payload_bytes     = args.payload_bytes,
            packet_interval_s = args.interval,
            n_packets         = args.n_packets,
        )

        # ------------------------------------------------------------------
        # BLOCK 2: Channel Impairment
        # ------------------------------------------------------------------
        self.channel = ChannelBlock(
            noise_voltage = args.noise,
            cfo_hz        = args.cfo,
            sco_ppm       = args.sco,
            tap_string    = args.taps,
        )

        # ------------------------------------------------------------------
        # BLOCK 3: STF Packet Detector (tagged stream)
        # ------------------------------------------------------------------
        self.detector = DetectorBlock(
            detect_threshold = args.det_thresh,
        )

        # ------------------------------------------------------------------
        # BLOCK 4: CSI Extractor (message output)
        # ------------------------------------------------------------------
        self.csi_ext = CSIExtractorBlock(
            n_data_symbols = 0,   # STF+LTF only (CSI sensing, no demod)
        )

        # ------------------------------------------------------------------
        # BLOCK 5: HDF5 Logger (message sink)
        # ------------------------------------------------------------------
        self.logger = LoggerBlock(
            hdf5_path      = args.hdf5,
            session_id     = args.session_id,
            n_rx_channels  = 1,
            flush_interval = 50,
        )

        # ------------------------------------------------------------------
        # BLOCK 6: CSI Monitor (message sink, terminal display)
        # ------------------------------------------------------------------
        self.monitor = MonitorBlock(
            update_every_n = 5,
        )

        # ------------------------------------------------------------------
        # WIRE: sample stream connections
        # TX -> Channel -> Detector -> CSI Extractor
        # ------------------------------------------------------------------
        self.connect(self.tx,       self.channel)
        self.connect(self.channel,  self.detector)
        self.connect(self.detector, self.csi_ext)

        # ------------------------------------------------------------------
        # WIRE: message connections
        # CSI Extractor -> Logger
        # CSI Extractor -> Monitor
        # ------------------------------------------------------------------
        self.msg_connect(self.csi_ext, pmt.intern("csi_out"),
                         self.logger,  pmt.intern("csi_in"))
        self.msg_connect(self.csi_ext, pmt.intern("csi_out"),
                         self.monitor, pmt.intern("csi_in"))

        print_config(args)


# =============================================================================
# Hardware flowgraph (extends sim with UHD source/sink)
# =============================================================================

class OFDMHardwareFlowgraph(gr.top_block):
    """
    Hardware flowgraph: replaces TX+Channel with USRP B210.
    TX  : USRP sink  (separate terminal — run main_tx.py --mode hardware)
    RX  : USRP source -> Detector -> CSI Extractor -> Logger + Monitor
    """

    def __init__(self, args):
        gr.top_block.__init__(self, "128pt OFDM PHY Hardware")

        try:
            from gnuradio import uhd
        except ImportError:
            print("[ERROR] gr-uhd not installed.")
            print("  Install: sudo apt install gnuradio gr-uhd")
            sys.exit(1)

        # ------------------------------------------------------------------
        # UHD Source (2 channels at 25 MS/s)
        # ------------------------------------------------------------------
        self.uhd_src = uhd.usrp_source(
            ",".join(("", "")),  # auto-find device
            uhd.stream_args(
                cpu_format  = "fc32",
                args        = "",
                channels    = list(range(cfg.NUM_RX_CHANNELS)),
            ),
        )
        self.uhd_src.set_samp_rate(cfg.FS_HW)
        self.uhd_src.set_center_freq(args.freq, 0)
        self.uhd_src.set_gain(args.rx_gain, 0)
        self.uhd_src.set_antenna("RX2", 0)
        if cfg.NUM_RX_CHANNELS > 1:
            self.uhd_src.set_center_freq(args.freq, 1)
            self.uhd_src.set_gain(args.rx_gain, 1)
            self.uhd_src.set_antenna("RX2", 1)

        # ------------------------------------------------------------------
        # Resampler: 25 MS/s -> 20 MS/s  (rational 4/5)
        # ------------------------------------------------------------------
        from gnuradio import filter as gr_filter
        from gnuradio.filter import firdes
        resamp_taps = firdes.low_pass(
            gain          = float(cfg.RESAMP_INTERP),
            sampling_freq = cfg.FS_HW * cfg.RESAMP_INTERP,
            cutoff_freq   = cfg.FS_FFT / 2.0,
            transition_bw = 1e6,
            window        = firdes.WIN_KAISER,
            beta          = 8.0,
        )
        self.resamp = gr_filter.rational_resampler_ccc(
            interpolation = cfg.RESAMP_INTERP,
            decimation    = cfg.RESAMP_DECIM,
            taps          = resamp_taps,
        )

        # ------------------------------------------------------------------
        # Detector, CSI Extractor, Logger, Monitor  (same as sim)
        # ------------------------------------------------------------------
        self.detector = DetectorBlock(detect_threshold=args.det_thresh)
        self.csi_ext  = CSIExtractorBlock(n_data_symbols=0)
        self.logger   = LoggerBlock(
            hdf5_path      = args.hdf5,
            session_id     = args.session_id,
            n_rx_channels  = cfg.NUM_RX_CHANNELS,
            flush_interval = 50,
        )
        self.monitor  = MonitorBlock(update_every_n=5)

        # ------------------------------------------------------------------
        # Wire: UHD ch0 -> resamp -> detector -> csi_ext
        # ------------------------------------------------------------------
        self.connect((self.uhd_src, 0), self.resamp)
        self.connect(self.resamp,       self.detector)
        self.connect(self.detector,     self.csi_ext)

        # ------------------------------------------------------------------
        # Message connections
        # ------------------------------------------------------------------
        self.msg_connect(self.csi_ext, pmt.intern("csi_out"),
                         self.logger,  pmt.intern("csi_in"))
        self.msg_connect(self.csi_ext, pmt.intern("csi_out"),
                         self.monitor, pmt.intern("csi_in"))

        print_config(args)


# =============================================================================
# Helpers
# =============================================================================

def print_config(args):
    print("\n" + "="*60)
    print("  Custom 128-pt OFDM PHY -- GNU Radio Flowgraph")
    print("="*60)
    print(f"  Mode         : {args.mode}")
    print(f"  Modulation   : {args.mod}")
    print(f"  Noise voltage: {args.noise}")
    print(f"  CFO inject   : {args.cfo} Hz")
    print(f"  SCO inject   : {args.sco} ppm")
    print(f"  Taps         : {args.taps}")
    print(f"  HDF5 output  : {args.hdf5}")
    print(f"  Session ID   : {args.session_id}")
    print(f"  n_packets    : {'unlimited' if args.n_packets == 0 else args.n_packets}")
    print("="*60 + "\n")
    print("  FLOWGRAPH:")
    print("  [TX] -> [Channel] -> [Detector] -> [CSI Extractor]")
    print("                                           |")
    print("                              +------------+------------+")
    print("                              |                         |")
    print("                        [HDF5 Logger]           [CSI Monitor]")
    print(f"                        {args.hdf5}          (terminal)\n")


def parse_args():
    ap = argparse.ArgumentParser(
        description="128-pt OFDM PHY GNU Radio Simulation/Hardware Flowgraph")
    ap.add_argument("--mode",          default="simulation",
                    choices=["simulation", "hardware"])
    ap.add_argument("--mod",           default="BPSK",
                    choices=["BPSK", "QPSK", "QAM16"])
    ap.add_argument("--payload_bytes", type=int,   default=60)
    ap.add_argument("--interval",      type=float, default=0.01,
                    help="Inter-packet interval (s)")
    ap.add_argument("--n_packets",     type=int,   default=0,
                    help="Total packets to send (0=unlimited)")
    ap.add_argument("--noise",         type=float, default=0.005,
                    help="AWGN noise voltage amplitude")
    ap.add_argument("--cfo",           type=float, default=0.0,
                    help="Injected CFO (Hz)")
    ap.add_argument("--sco",           type=float, default=0.0,
                    help="Injected SCO (ppm)")
    ap.add_argument("--taps",          default="1+0j",
                    help="Multipath taps, e.g. '1+0j,0.3-0.1j'")
    ap.add_argument("--det_thresh",    type=float, default=0.65,
                    help="Schmidl & Cox detection threshold (0-1)")
    ap.add_argument("--hdf5",          default="gr_csi_output.h5")
    ap.add_argument("--session_id",    default="gr_session_001")
    ap.add_argument("--freq",          type=float, default=cfg.USRP_CENTER_FREQ,
                    help="Hardware center frequency (Hz)")
    ap.add_argument("--rx_gain",       type=float, default=cfg.USRP_RX_GAIN,
                    help="Hardware RX gain (dB)")
    return ap.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    if args.mode == "simulation":
        fg = OFDMSimFlowgraph(args)
    else:
        fg = OFDMHardwareFlowgraph(args)

    # -- Graceful Ctrl+C shutdown -------------------------------------------
    def _sigint(sig, frame):
        print("\n\n  [STOP] Ctrl+C received — stopping flowgraph...")
        fg.stop()
        fg.wait()
        sys.exit(0)
    signal.signal(signal.SIGINT, _sigint)

    # -- Run -------------------------------------------------------------------
    print("  Starting flowgraph. Press Ctrl+C to stop.\n")
    fg.start()

    # If n_packets is set, monitor and stop when done
    if args.n_packets > 0:
        poll_interval = 0.5
        while True:
            time.sleep(poll_interval)
            # CSI extractor sequence number is our packet counter
            if hasattr(fg, 'csi_ext') and fg.csi_ext._seq >= args.n_packets:
                print(f"\n\n  [DONE] {args.n_packets} packets processed.")
                fg.stop()
                fg.wait()
                break
    else:
        fg.wait()

    print(f"\n  CSI saved to: {args.hdf5}\n")


if __name__ == "__main__":
    main()
