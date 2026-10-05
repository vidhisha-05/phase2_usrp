"""
main_tx.py — Entry point for the TX process.

Config-driven switch between simulation (ZMQ) and hardware (UHD) mode.
Set MODE in config.py to "simulation" or "hardware".

Reference: phase2 (2).md — Sections 12, 17
"""

import argparse
import config as cfg

def main():
    p = argparse.ArgumentParser(description="Custom 128-pt OFDM PHY — TX")
    p.add_argument("--mode",       type=str,   default=cfg.MODE,
                   choices=["simulation", "hardware"],
                   help="Deployment mode")
    p.add_argument("--payload",    type=int,   default=100,
                   help="Payload size (bytes)")
    p.add_argument("--mod",        type=str,   default="BPSK",
                   choices=["BPSK", "QPSK", "16QAM"],
                   help="Modulation scheme")
    p.add_argument("--interval",   type=float, default=0.005,
                   help="Inter-packet interval (s). 0 = max rate")
    p.add_argument("--n_packets",  type=int,   default=0,
                   help="Number of packets to send. 0 = run forever")
    a = p.parse_args()

    if a.mode == "simulation":
        from tx_sim import run_tx_sim
        run_tx_sim(a.payload, a.mod, a.interval, a.n_packets)
    else:
        from tx_hardware import run_tx_hardware
        run_tx_hardware(a.payload, a.mod, a.interval, a.n_packets)


if __name__ == "__main__":
    main()
