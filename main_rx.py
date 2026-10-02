"""
main_rx.py — Entry point for the RX process.

Config-driven switch between simulation (ZMQ) and hardware (UHD) mode.
Set MODE in config.py to "simulation" or "hardware".

Reference: phase2 (2).md — Sections 12, 17
"""

import argparse
import queue
import threading
import config as cfg
from logger import CSILogger


def main():
    p = argparse.ArgumentParser(description="Custom 128-pt OFDM PHY — RX / CSI Logger")
    p.add_argument("--mode",       type=str,   default=cfg.MODE,
                   choices=["simulation", "hardware"],
                   help="Deployment mode")
    p.add_argument("--n_packets",  type=int,   default=0,
                   help="Stop after N CSI records. 0 = run forever")
    p.add_argument("--hdf5",       type=str,   default=cfg.HDF5_FILE_PATH,
                   help="Output HDF5 file path")
    p.add_argument("--session_id", type=str,   default="session_001")
    p.add_argument("--n_channels", type=int,   default=1,
                   help="Number of RX antenna channels (1 or 2)")
    a = p.parse_args()

    # F-26 FIX: was hardcoded n_ch=2 in hardware mode, but NUM_RX_CHANNELS=1
    # Using cfg.NUM_RX_CHANNELS ensures consistency with config and logger
    n_ch = a.n_channels if a.mode == "simulation" else cfg.NUM_RX_CHANNELS

    csi_q  = queue.Queue(maxsize=2000)
    logger = CSILogger(csi_q,
                       hdf5_path=a.hdf5,
                       session_id=a.session_id,
                       n_rx_channels=n_ch)
    log_t  = threading.Thread(target=logger.run, daemon=True, name="logger")
    log_t.start()

    try:
        if a.mode == "simulation":
            from rx_sim import run_rx_sim
            run_rx_sim(csi_q, n_packets=a.n_packets, n_rx_channels=n_ch)
        else:
            # F-27 FIX: validate_pipeline.py does not exist (missing module).
            # Replaced with subprocess call to validate_stage1.py.
            # NOTE: For true single-B210 full-duplex, use main_hardware.py instead.
            #       This file's hardware path still creates a separate MultiUSRP()
            #       which will fail if tx_hardware.py is also running on the same device.
            print("[main_rx] Running Stage 1 simulation gate before hardware start ...")
            import subprocess, sys as _sys
            _result = subprocess.run(
                [_sys.executable, 'validate_stage1.py'],
                capture_output=True, timeout=30, cwd='d:/phase2')
            if _result.returncode != 0:
                print("[main_rx] Stage 1 FAIL -- hardware mode aborted.")
                _out = _result.stdout.decode(errors='replace')
                print(_out[-2000:] if len(_out) > 2000 else _out)
                return
            print("[main_rx] Stage 1 PASS -- proceeding to hardware.")
            from rx_hardware import run_rx_hardware
            run_rx_hardware(csi_q, n_packets=a.n_packets)
    finally:
        logger.stop()
        log_t.join(timeout=10)
        print("[main_rx] Done.")



if __name__ == "__main__":
    main()
