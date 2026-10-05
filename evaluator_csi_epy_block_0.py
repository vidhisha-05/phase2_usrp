import json
import threading

import numpy as np
import pmt
from gnuradio import gr


class blk(gr.sync_block):
    """
    ZMQ JSON CSI message parser.

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
        port 1: CSI phase, 106 floats
    """

    def __init__(self):
        self.n_sc = 106

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

        self.lock = threading.Lock()

    def handle_msg(self, msg):
        try:
            # ---------------------------------------------------------
            # Extract raw JSON bytes from the PMT message
            # ---------------------------------------------------------
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

            # ---------------------------------------------------------
            # Decode JSON
            # ---------------------------------------------------------
            data = json.loads(
                raw.decode("utf-8")
            )

            # ---------------------------------------------------------
            # Convert CSI arrays to NumPy
            # ---------------------------------------------------------
            mag = np.asarray(
                data["csi_mag"],
                dtype=np.float32
            )

            phase = np.asarray(
                data["csi_phase"],
                dtype=np.float32
            )

            # ---------------------------------------------------------
            # Validate vector lengths
            # ---------------------------------------------------------
            if mag.size != self.n_sc:
                return

            if phase.size != self.n_sc:
                return

            # ---------------------------------------------------------
            # Validate numerical values
            # ---------------------------------------------------------
            if not np.all(np.isfinite(mag)):
                return

            if not np.all(np.isfinite(phase)):
                return

            # ---------------------------------------------------------
            # DEBUG: verify that the parser receives the actual
            # subcarrier-dependent CSI values.
            # ---------------------------------------------------------
            print(
                "CSI DEBUG:",
                "mag=",
                mag[0],
                mag[53],
                mag[-1],
                "phase=",
                phase[0],
                phase[53],
                phase[-1],
                flush=True
            )

            # ---------------------------------------------------------
            # Update the latest CSI frame
            # ---------------------------------------------------------
            with self.lock:
                self.mag = mag.copy()
                self.phase = phase.copy()

        except Exception as exc:
            print(
                "CSI PARSER ERROR:",
                repr(exc),
                flush=True
            )
            return

    def work(self, input_items, output_items):
        # -------------------------------------------------------------
        # Copy the latest CSI frame safely
        # -------------------------------------------------------------
        with self.lock:
            mag = self.mag.copy()
            phase = self.phase.copy()

        # -------------------------------------------------------------
        # Output:
        #   port 0 -> magnitude vector
        #   port 1 -> phase vector
        # -------------------------------------------------------------
        output_items[0][0, :] = mag
        output_items[1][0, :] = phase

        return 1