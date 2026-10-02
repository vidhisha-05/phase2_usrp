"""
tx_sim.py — TX process: simulation mode (ZMQ PUSH).

Builds baseband packets using waveform.py / scrambler.py and pushes them
via ZMQ PUSH socket to the channel_bridge.

Reference: phase2 (2).md — Sections 8, 12, 17
"""

import time
import struct
import numpy as np
import zmq
import config as cfg
import waveform
import scrambler

_HEADER_FMT  = "<Qd"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)


def _pack(seq: int, ts: float, samples: np.ndarray) -> bytes:
    return struct.pack(_HEADER_FMT, seq, ts) + samples.astype(np.complex64).tobytes()


def run_tx_sim(payload_bytes: int = 100,
               modulation: str = "BPSK",
               packet_interval_s: float = 0.005,
               n_packets: int = 0):
    """
    Continuously transmit packets via ZMQ PUSH.

    Args:
        payload_bytes:      Payload size per packet in bytes.
        modulation:         "BPSK", "QPSK", or "16QAM".
        packet_interval_s:  Inter-packet gap (seconds). 0 = as fast as possible.
        n_packets:          Stop after N packets. 0 = run forever.
    """
    ctx  = zmq.Context()
    sock = ctx.socket(zmq.PUSH)
    sock.bind(cfg.ZMQ_TX_ADDR)
    print(f"[tx_sim] Bound to {cfg.ZMQ_TX_ADDR}  mod={modulation}")

    seq = 0
    try:
        while True:
            # Random payload
            payload_bits = np.random.randint(0, 2,
                                              payload_bytes * 8,
                                              dtype=np.uint8)

            # Build packet (20 MS/s baseband samples)
            packet_20ms = waveform.assemble_packet(
                payload_bits,
                modulation=modulation,
                scrambler_mod=scrambler,
                encoder_mod=scrambler,
                mapper_fn=scrambler.map_bits_to_symbols)

            # Optionally resample to 25 MS/s for HW compatibility
            # packet_25ms = waveform.downsample_20to25(packet_20ms)  # HW path

            ts  = time.monotonic()
            msg = _pack(seq, ts, packet_20ms)
            sock.send(msg)
            seq += 1
            print(f"[tx_sim] TX pkt #{seq}  len={len(packet_20ms)} samples")

            if n_packets and seq >= n_packets:
                break
            if packet_interval_s > 0:
                time.sleep(packet_interval_s)
    except KeyboardInterrupt:
        print("[tx_sim] Shutting down.")
    finally:
        sock.close()
        ctx.term()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--payload",   type=int,   default=100)
    p.add_argument("--mod",       type=str,   default="BPSK")
    p.add_argument("--interval",  type=float, default=0.005)
    p.add_argument("--n_packets", type=int,   default=0)
    a = p.parse_args()
    run_tx_sim(a.payload, a.mod, a.interval, a.n_packets)
