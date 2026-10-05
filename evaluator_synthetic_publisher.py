import json
import time

import numpy as np
import pmt
import zmq


PUB_ADDR = "tcp://127.0.0.1:5555"

N_SC = 106
FPS = 20.0


ctx = zmq.Context()

sock = ctx.socket(zmq.PUB)
sock.bind(PUB_ADDR)


print("Synthetic CSI publisher")
print(f"Address      : {PUB_ADDR}")
print(f"Subcarriers  : {N_SC}")
print(f"Rate         : {FPS:.1f} Hz")
print("Format       : PMT-serialized u8vector containing JSON")
print("Press Ctrl+C to stop.")


phase0 = np.linspace(
    -np.pi,
    np.pi,
    N_SC
)


try:

    while True:

        t = time.monotonic()

        # ---------------------------------------------------------
        # Synthetic CSI magnitude
        # ---------------------------------------------------------
        magnitude = (
            1.0
            + 0.08 * np.sin(
                2.0 * np.pi * 1.0 * t
            )
            + 0.03 * np.sin(
                2.0 * np.pi * 3.0 * t
                + phase0
            )
        )

        # ---------------------------------------------------------
        # Synthetic CSI phase
        # ---------------------------------------------------------
        phase = (
            phase0
            + 0.35 * np.sin(
                2.0 * np.pi * 1.5 * t
            )
            + 0.08 * np.sin(
                2.0 * np.pi * 4.0 * t
                + phase0
            )
        )

        # ---------------------------------------------------------
        # JSON payload
        # ---------------------------------------------------------
        payload = {
            "timestamp": time.time(),
            "seq": int(t * FPS),

            "csi_mag": magnitude.astype(
                np.float32
            ).tolist(),

            "csi_phase": phase.astype(
                np.float32
            ).tolist(),

            "crc_ok": True,
        }

        json_bytes = json.dumps(
            payload
        ).encode("utf-8")

        # ---------------------------------------------------------
        # Convert JSON bytes into a GNU Radio PMT u8vector
        # ---------------------------------------------------------
        pmt_msg = pmt.init_u8vector(
            len(json_bytes),
            list(json_bytes)
        )

        # ---------------------------------------------------------
        # Serialize PMT before sending through ZMQ.
        #
        # GNU Radio ZMQ SUB Message Source expects
        # serialized PMT messages.
        # ---------------------------------------------------------
        serialized = pmt.serialize_str(
            pmt_msg
        )

        sock.send(
            serialized
        )

        time.sleep(
            1.0 / FPS
        )


except KeyboardInterrupt:

    print("\nStopping publisher...")


finally:

    sock.close()
    ctx.term()