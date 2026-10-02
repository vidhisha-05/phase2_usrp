"""
spectrum_scan.py — Pre-collection 2.4/5 GHz spectrum scanner (Section 18).

Uses the UHD Python API to sweep the band and identify the least-congested
channel. Output: CSV of power-vs-frequency and a printed recommendation.

Run BEFORE every data collection session. Record the output in session metadata.

Reference: phase2 (2).md — Section 18
"""

import time
import numpy as np
import config as cfg

try:
    import uhd
    _UHD = True
except ImportError:
    _UHD = False
    print("[scan] UHD not available — running in simulation mode (random spectrum).")

# 2.4 GHz WiFi channel center frequencies (MHz)
CHANNELS_24 = {
    1: 2412e6, 2: 2417e6, 3: 2422e6,  4: 2427e6,
    5: 2432e6, 6: 2437e6, 7: 2442e6,  8: 2447e6,
    9: 2452e6,10: 2457e6,11: 2462e6, 12: 2467e6, 13: 2472e6
}

# 5 GHz UNII-1 channels
CHANNELS_5 = {
    36: 5180e6, 40: 5200e6, 44: 5220e6, 48: 5240e6,
    52: 5260e6, 56: 5280e6, 60: 5300e6, 64: 5320e6,
   149: 5745e6,153: 5765e6,157: 5785e6,161: 5805e6
}

DWELL_TIME  = 0.05   # seconds per frequency step
FFT_SIZE    = 1024   # FFT bins for spectral estimate


def scan_band(band: str = "2.4", rx_gain: float = 20.0) -> dict:
    """
    Scan the given band and return {freq_hz: power_dBFS} for each channel.

    Args:
        band:    "2.4" or "5"
        rx_gain: Temporary RX gain for scan (lower than collection gain).
    Returns:
        dict mapping frequency (Hz) -> power (dBFS, negative).
    """
    channels = CHANNELS_24 if band == "2.4" else CHANNELS_5
    results  = {}

    if _UHD:
        usrp = uhd.usrp.MultiUSRP()
        usrp.set_rx_rate(cfg.FS_HW)
        usrp.set_rx_gain(rx_gain, 0)
        usrp.set_rx_antenna("RX2", 0)

        st_args = uhd.usrp.StreamArgs("fc32", "sc16")
        st_args.channels = [0]
        streamer = usrp.get_rx_stream(st_args)

    for ch_num, freq in channels.items():
        if _UHD:
            usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(freq), 0)
            time.sleep(0.02)   # LO settle

            md  = uhd.types.RXMetadata()
            buf = np.zeros(FFT_SIZE, dtype=np.complex64)
            streamer.recv(buf, md)
            power = 10 * np.log10(np.mean(np.abs(buf) ** 2) + 1e-20)
        else:
            # Simulation: random power
            power = np.random.uniform(-60, -30)

        results[freq] = power
        print(f"  Ch {ch_num:3d}  {freq/1e6:7.1f} MHz : {power:.1f} dBFS")

    return results


def recommend_channel(results: dict) -> float:
    """Return the frequency with the lowest measured power (least congested)."""
    best_freq  = min(results, key=results.get)
    best_power = results[best_freq]
    print(f"\n[scan] Recommended frequency: {best_freq/1e6:.3f} MHz"
          f"  (power: {best_power:.1f} dBFS)")
    return best_freq


def save_scan(results: dict, path: str = "spectrum_scan.csv"):
    import csv
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["freq_hz", "power_dBFS"])
        for freq, pwr in sorted(results.items()):
            w.writerow([freq, round(pwr, 2)])
    print(f"[scan] Results saved: {path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Spectrum scan before data collection")
    p.add_argument("--band",    type=str,   default="2.4", choices=["2.4", "5"])
    p.add_argument("--gain",    type=float, default=20.0)
    p.add_argument("--output",  type=str,   default="spectrum_scan.csv")
    a = p.parse_args()

    print(f"\n[scan] Scanning {a.band} GHz band ...")
    res   = scan_band(a.band, a.gain)
    rec   = recommend_channel(res)
    save_scan(res, a.output)
    print(f"\nUpdate USRP_CENTER_FREQ in config.py to: {rec}")
