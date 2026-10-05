import json
import queue

import numpy as np
import pmt
from gnuradio import gr


class csi_parser(gr.sync_block):
    """
    ZMQ JSON CSI message -> GNU Radio vector streams.

    Input message:
        {
            "timestamp": ...,
            "seq": ...,
            "csi_mag": [106 values],
            "csi_phase": [106 values],
            "crc_ok": true
        }

    Outputs:
        port 0: CSI magnitude, 106 floats
        port 1: CSI phase,     106 floats
    """

    def __init__(self, n_sc=106):
        self.n_sc = int(n_sc)

        gr.sync_block.__init__(
            self,
            name="CSI JSON Parser",
            in_sig=None,
            out_sig=[
                (np.float32, self.n_sc),
                (np.float32, self.n_sc),
            ],
        )

        self.message_port_register_in(
            pmt.intern("in")
        )

        self.set_msg_handler(
            pmt.intern("in"),
            self.handle_msg
        )

        self.mag = np.ones(
            self.n_sc,
            dtype=np.float32
        )

        self.phase = np.zeros(
            self.n_sc,
            dtype=np.float32
        )

        self.lock = __import__("threading").Lock()

    def handle_msg(self, msg):
        try:
            # ZMQ Message Source normally gives:
            # PMT pair(metadata, payload)
            if pmt.is_pair(msg):
                msg = pmt.cdr(msg)

            if pmt.is_u8vector(msg):
                raw = bytes(
                    pmt.u8vector_elements(msg)
                )

            elif pmt.is_blob(msg):
                raw = bytes(
                    pmt.blob_data(msg)
                )

            elif pmt.is_symbol(msg):
                raw = pmt.symbol_to_string(msg).encode(
                    "utf-8"
                )

            else:
                value = pmt.to_python(msg)

                if isinstance(value, bytes):
                    raw = value
                elif isinstance(value, str):
                    raw = value.encode("utf-8")
                else:
                    return

            data = json.loads(
                raw.decode("utf-8")
            )

            mag = np.asarray(
                data["csi_mag"],
                dtype=np.float32
            )

            phase = np.asarray(
                data["csi_phase"],
                dtype=np.float32
            )

            if mag.size != self.n_sc:
                return

            if phase.size != self.n_sc:
                return

            if not np.all(np.isfinite(mag)):
                return

            if not np.all(np.isfinite(phase)):
                return

            with self.lock:
                self.mag = mag.copy()
                self.phase = phase.copy()

        except Exception:
            # Ignore malformed messages without stopping
            # the GNU Radio flowgraph.
            return

    def work(self, input_items, output_items):
        with self.lock:
            mag = self.mag.copy()
            phase = self.phase.copy()

        # One vector is produced on every scheduler call.
        output_items[0][0, :] = mag
        output_items[1][0, :] = phase

        return 1