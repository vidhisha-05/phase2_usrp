#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: 128-pt Custom OFDM PHY -- CSI Simulation
# Author: Custom OFDM PHY
# Description: Custom 128-pt OFDM PHY real-time CSI sensing
# GNU Radio version: 3.10.12.0

from PyQt5 import Qt
from gnuradio import qtgui
from gnuradio import blocks
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from PyQt5 import Qt
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
import ofdm_csi_flowgraph_channel_impairment as channel_impairment  # embedded python block
import ofdm_csi_flowgraph_csi_extractor as csi_extractor  # embedded python block
import ofdm_csi_flowgraph_csi_logger as csi_logger  # embedded python block
import ofdm_csi_flowgraph_csi_monitor as csi_monitor  # embedded python block
import ofdm_csi_flowgraph_ofdm_tx as ofdm_tx  # embedded python block
import ofdm_csi_flowgraph_packet_detector as packet_detector  # embedded python block
import sip
import threading



class ofdm_csi_flowgraph(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "128-pt Custom OFDM PHY -- CSI Simulation", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("128-pt Custom OFDM PHY -- CSI Simulation")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except BaseException as exc:
            print(f"Qt GUI: Could not set Icon: {str(exc)}", file=sys.stderr)
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("gnuradio/flowgraphs", "ofdm_csi_flowgraph")

        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)
        self.flowgraph_started = threading.Event()

        ##################################################
        # Variables
        ##################################################
        self.tap_string = tap_string = "1+0j"
        self.session_id = session_id = "gr_session_001"
        self.sco_ppm = sco_ppm = 1.0
        self.samp_rate = samp_rate = 20e6
        self.noise_voltage = noise_voltage = 0.005
        self.hdf5_path = hdf5_path = "gr_csi_output.h5"
        self.cfo_hz = cfo_hz = 500.0

        ##################################################
        # Blocks
        ##################################################

        self.throttle = blocks.throttle(gr.sizeof_gr_complex*1, samp_rate,True)
        self.throttle.set_block_alias("throttle")
        self.qtgui_time_sink = qtgui.time_sink_c(
            1024, #size
            samp_rate, #samp_rate
            "OFDM TX Waveform", #name
            1, #number of inputs
            None # parent
        )
        self.qtgui_time_sink.set_update_time(0.10)
        self.qtgui_time_sink.set_y_axis(-2, 2)

        self.qtgui_time_sink.set_y_label('Amplitude', "")

        self.qtgui_time_sink.enable_tags(True)
        self.qtgui_time_sink.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, 0, "")
        self.qtgui_time_sink.enable_autoscale(True)
        self.qtgui_time_sink.enable_grid(True)
        self.qtgui_time_sink.enable_axis_labels(True)
        self.qtgui_time_sink.enable_control_panel(False)
        self.qtgui_time_sink.enable_stem_plot(False)


        labels = ["I", "Q", "", "", "",
            'Signal 6', 'Signal 7', 'Signal 8', 'Signal 9', 'Signal 10']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ['blue', 'red', 'green', 'black', 'cyan',
            'magenta', 'yellow', 'dark red', 'dark green', 'dark blue']
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]
        styles = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        markers = [-1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1]


        for i in range(2):
            if len(labels[i]) == 0:
                if (i % 2 == 0):
                    self.qtgui_time_sink.set_line_label(i, "Re{{Data {0}}}".format(i/2))
                else:
                    self.qtgui_time_sink.set_line_label(i, "Im{{Data {0}}}".format(i/2))
            else:
                self.qtgui_time_sink.set_line_label(i, labels[i])
            self.qtgui_time_sink.set_line_width(i, widths[i])
            self.qtgui_time_sink.set_line_color(i, colors[i])
            self.qtgui_time_sink.set_line_style(i, styles[i])
            self.qtgui_time_sink.set_line_marker(i, markers[i])
            self.qtgui_time_sink.set_line_alpha(i, alphas[i])

        self._qtgui_time_sink_win = sip.wrapinstance(self.qtgui_time_sink.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_time_sink_win)
        self.qtgui_freq_sink = qtgui.freq_sink_c(
            128, #size
            window.WIN_BLACKMAN_hARRIS, #wintype
            0, #fc
            samp_rate, #bw
            "OFDM Spectrum (128-pt)", #name
            1,
            None # parent
        )
        self.qtgui_freq_sink.set_update_time(0.10)
        self.qtgui_freq_sink.set_y_axis((-140), 10)
        self.qtgui_freq_sink.set_y_label('Relative Gain', 'dB')
        self.qtgui_freq_sink.set_trigger_mode(qtgui.TRIG_MODE_FREE, 0.0, 0, "")
        self.qtgui_freq_sink.enable_autoscale(False)
        self.qtgui_freq_sink.enable_grid(True)
        self.qtgui_freq_sink.set_fft_average(0.2)
        self.qtgui_freq_sink.enable_axis_labels(True)
        self.qtgui_freq_sink.enable_control_panel(False)
        self.qtgui_freq_sink.set_fft_window_normalized(False)



        labels = ["Post-Channel", '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "green", "black", "cyan",
            "magenta", "yellow", "dark red", "dark green", "dark blue"]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_freq_sink.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_freq_sink.set_line_label(i, labels[i])
            self.qtgui_freq_sink.set_line_width(i, widths[i])
            self.qtgui_freq_sink.set_line_color(i, colors[i])
            self.qtgui_freq_sink.set_line_alpha(i, alphas[i])

        self._qtgui_freq_sink_win = sip.wrapinstance(self.qtgui_freq_sink.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_freq_sink_win)
        self.packet_detector = packet_detector.blk(detect_threshold=0.65, corr_window=16)
        self.packet_detector.set_block_alias("packet_detector")
        self.ofdm_tx = ofdm_tx.blk(mod_scheme='BPSK', payload_bytes=60, packet_interval_s=0.01, n_packets=0)
        self.ofdm_tx.set_block_alias("ofdm_tx")
        self.csi_monitor = csi_monitor.blk(update_every_n=5)
        self.csi_monitor.set_block_alias("csi_monitor")
        self.csi_logger = csi_logger.blk(hdf5_path=hdf5_path, session_id=session_id, n_rx_channels=1, flush_interval=50)
        self.csi_logger.set_block_alias("csi_logger")
        self.csi_extractor = csi_extractor.blk(n_data_symbols=0)
        self.csi_extractor.set_block_alias("csi_extractor")
        self.channel_impairment = channel_impairment.blk(noise_voltage=noise_voltage, cfo_hz=cfo_hz, sco_ppm=sco_ppm, tap_string=tap_string)
        self.channel_impairment.set_block_alias("channel_impairment")


        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.csi_extractor, 'csi_out'), (self.csi_logger, 'csi_in'))
        self.msg_connect((self.csi_extractor, 'csi_out'), (self.csi_monitor, 'csi_in'))
        self.connect((self.channel_impairment, 0), (self.packet_detector, 0))
        self.connect((self.channel_impairment, 0), (self.qtgui_freq_sink, 0))
        self.connect((self.ofdm_tx, 0), (self.throttle, 0))
        self.connect((self.packet_detector, 0), (self.csi_extractor, 0))
        self.connect((self.throttle, 0), (self.channel_impairment, 0))
        self.connect((self.throttle, 0), (self.qtgui_time_sink, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("gnuradio/flowgraphs", "ofdm_csi_flowgraph")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_tap_string(self):
        return self.tap_string

    def set_tap_string(self, tap_string):
        self.tap_string = tap_string

    def get_session_id(self):
        return self.session_id

    def set_session_id(self, session_id):
        self.session_id = session_id

    def get_sco_ppm(self):
        return self.sco_ppm

    def set_sco_ppm(self, sco_ppm):
        self.sco_ppm = sco_ppm

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.qtgui_freq_sink.set_frequency_range(0, self.samp_rate)
        self.qtgui_time_sink.set_samp_rate(self.samp_rate)
        self.throttle.set_sample_rate(self.samp_rate)

    def get_noise_voltage(self):
        return self.noise_voltage

    def set_noise_voltage(self, noise_voltage):
        self.noise_voltage = noise_voltage

    def get_hdf5_path(self):
        return self.hdf5_path

    def set_hdf5_path(self, hdf5_path):
        self.hdf5_path = hdf5_path

    def get_cfo_hz(self):
        return self.cfo_hz

    def set_cfo_hz(self, cfo_hz):
        self.cfo_hz = cfo_hz




def main(top_block_cls=ofdm_csi_flowgraph, options=None):

    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()
    tb.flowgraph_started.set()

    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()

if __name__ == '__main__':
    main()
