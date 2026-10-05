import json
import queue
import threading
import time

import numpy as np
import zmq

import config as cfg
from tx_sim import run_tx_sim
from channel_bridge import ChannelModel, run_bridge
from rx_sim import run_rx_sim


# =============================================================================
# Evaluator simulation configuration
# =============================================================================

N_SC = len(cfg.ACTIVE_SUBCARRIERS)

PUB_ADDR = "tcp://127.0.0.1:5558"

PAYLOAD_BYTES = 100
MODULATION = "BPSK"

# 5 ms packet start-to-start period = 200 packets/s.
# This is already consistent with the validated simulation timing.
PACKET_INTERVAL_S = 0.005

N_PACKETS = 200

# Existing channel model:
# default noise_voltage = 0.01
# flat single-tap channel
# zero CFO
# zero SCO
CHANNEL_NOISE = 0.01
CHANNEL_TAPS = [1.0 + 0.0j, 0.5 + 0.2j]
CHANNEL_CFO_HZ = 0.0
CHANNEL_SCO_PPM = 0.0


# =============================================================================
# CSI publisher
# =============================================================================

class CSIPublisher:
    """
    Converts CSI records from rx_sim.py into PMT-serialized JSON messages
    consumed by the GNU Radio ZMQ SUB Message Source.

    Published fields:

        timestamp
        seq
        csi_mag
        csi_phase
        crc_ok

    Note:
        rx_sim currently performs CSI extraction but does not perform the
        payload CRC decode in this path. Therefore crc_ok is published as
        None rather than falsely claiming a CRC result.
    """

    def __init__(self, address=PUB_ADDR):
        self.address = address

        self.ctx = zmq.Context()

        self.sock = self.ctx.socket(
            zmq.PUB
        )

        self.sock.bind(
            self.address
        )

        self.count = 0

    def publish(self, record):

        H_hat = np.asarray(
            record["H_hat"],
            dtype=np.complex64
        )

        if H_hat.size != N_SC:
            raise RuntimeError(
                f"Unexpected CSI length: "
                f"{H_hat.size}, expected {N_SC}"
            )

        if not np.all(np.isfinite(H_hat)):
            return

        magnitude = np.abs(
            H_hat
        ).astype(
            np.float32
        )

        phase = np.angle(
            H_hat
        ).astype(
            np.float32
        )

        payload = {
            "timestamp": float(
                record.get(
                    "timestamp",
                    time.time()
                )
            ),

            "seq": int(
                record.get(
                    "seq",
                    self.count
                )
            ),

            "csi_mag": magnitude.tolist(),

            "csi_phase": phase.tolist(),

            # rx_sim does not perform the CRC decode in this CSI path.
            "crc_ok": None,
        }

        json_bytes = json.dumps(
            payload
        ).encode(
            "utf-8"
        )

        # -------------------------------------------------------------
        # GNU Radio ZMQ SUB Message Source expects a serialized PMT.
        # -------------------------------------------------------------
        import pmt

        pmt_msg = pmt.init_u8vector(
            len(json_bytes),
            list(json_bytes)
        )

        serialized = pmt.serialize_str(
            pmt_msg
        )

        self.sock.send(
            serialized
        )

        self.count += 1

        if self.count <= 5 or self.count % 20 == 0:

            print(
                f"[evaluator] CSI #{self.count:04d} | "
                f"seq={payload['seq']} | "
                f"|H| mean={np.mean(magnitude):.4f} | "
                f"|H| std={np.std(magnitude):.4f} | "
                f"phase range="
                f"[{np.min(phase):+.3f}, "
                f"{np.max(phase):+.3f}]"
            )

    def close(self):

        self.sock.close()
        self.ctx.term()


# =============================================================================
# RX worker
# =============================================================================

def rx_worker(
    csi_queue,
    done_event
):
    """
    Run the project's existing rx_sim.py CSI extraction path.
    """

    try:

        run_rx_sim(
            csi_queue,
            n_packets=N_PACKETS,
            n_rx_channels=1
        )

    finally:

        done_event.set()


# =============================================================================
# Main
# =============================================================================

def main():

    print("=" * 70)
    print("REAL PROJECT SIMULATION -> CSI EVALUATOR")
    print("=" * 70)

    print(
        f"Active subcarriers : {N_SC}"
    )

    print(
        f"Payload            : "
        f"{PAYLOAD_BYTES} bytes"
    )

    print(
        f"Modulation         : "
        f"{MODULATION}"
    )

    print(
        f"Packet interval    : "
        f"{PACKET_INTERVAL_S * 1000:.1f} ms"
    )

    print(
        f"Packet rate        : "
        f"{1.0 / PACKET_INTERVAL_S:.1f} Hz"
    )

    print(
        f"Number of packets  : "
        f"{N_PACKETS}"
    )

    print(
        f"Channel noise      : "
        f"{CHANNEL_NOISE}"
    )

    print(
        f"CFO                : "
        f"{CHANNEL_CFO_HZ:.1f} Hz"
    )

    print(
        f"SCO                : "
        f"{CHANNEL_SCO_PPM:.1f} ppm"
    )

    print(
        f"GUI PUB address    : "
        f"{PUB_ADDR}"
    )

    print("=" * 70)

    # -------------------------------------------------------------------------
    # Queue between rx_sim and this evaluator
    # -------------------------------------------------------------------------

    csi_queue = queue.Queue(
        maxsize=1000
    )

    # -------------------------------------------------------------------------
    # Start CSI publisher
    # -------------------------------------------------------------------------

    publisher = CSIPublisher(
        PUB_ADDR
    )

    # -------------------------------------------------------------------------
    # Start the existing channel bridge.
    #
    # TX -> channel bridge -> RX
    # -------------------------------------------------------------------------

    channel = ChannelModel(
        noise_voltage=CHANNEL_NOISE,
        taps=CHANNEL_TAPS,
        cfo_hz=CHANNEL_CFO_HZ,
        sco_ppm=CHANNEL_SCO_PPM
    )

    bridge_thread = threading.Thread(
        target=run_bridge,
        args=(channel,),
        daemon=True,
        name="channel_bridge"
    )

    bridge_thread.start()

    # -------------------------------------------------------------------------
    # Start the existing RX simulation.
    # -------------------------------------------------------------------------

    done_event = threading.Event()

    rx_thread = threading.Thread(
        target=rx_worker,
        args=(
            csi_queue,
            done_event
        ),
        daemon=True,
        name="rx_sim"
    )

    rx_thread.start()

    # -------------------------------------------------------------------------
    # Start the existing TX simulation.
    #
    # This function generates the actual OFDM packets using waveform.py.
    # -------------------------------------------------------------------------

    tx_thread = threading.Thread(
        target=run_tx_sim,
        args=(
            PAYLOAD_BYTES,
            MODULATION,
            PACKET_INTERVAL_S,
            N_PACKETS
        ),
        daemon=True,
        name="tx_sim"
    )

    tx_thread.start()

    # -------------------------------------------------------------------------
    # Consume CSI records and publish them to GNU Radio.
    # -------------------------------------------------------------------------

    published = 0

    try:

        while not done_event.is_set():

            try:

                record = csi_queue.get(
                    timeout=0.5
                )

            except queue.Empty:

                continue

            publisher.publish(
                record
            )

            published += 1

    except KeyboardInterrupt:

        print(
            "\n[evaluator] Keyboard interrupt."
        )

    finally:

        print()
        print("=" * 70)
        print(
            f"[evaluator] Published CSI records: "
            f"{published}"
        )
        print("=" * 70)

        publisher.close()


if __name__ == "__main__":

    main()