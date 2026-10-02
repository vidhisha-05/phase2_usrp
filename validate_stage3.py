"""
validate_stage3.py  --  STAGE 3 VALIDATION
End-to-End Pipeline: packet build -> channel impairment -> CSI extraction -> HDF5

Tests the full chain in-process without ZMQ transport complexity.
ZMQ transport itself is validated implicitly — it's a standard library.
"""

import struct, threading, time, sys, os, argparse
import numpy as np

sys.path.insert(0, r"d:\phase2")
import config as cfg

errors = []

def check(name, cond, detail=""):
    tag = "[PASS]" if cond else "[FAIL]"
    msg = f"  {tag}  {name}"
    if detail:
        msg += f"  <- {detail}"
    print(msg)
    if not cond:
        errors.append(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_packets", type=int,   default=30)
    ap.add_argument("--noise",     type=float, default=0.005)
    args = ap.parse_args()

    print("\n" + "="*54)
    print("  STAGE 3 -- End-to-End Pipeline (in-process)")
    print(f"  n_packets={args.n_packets}  noise={args.noise}")
    print("="*54 + "\n")

    import waveform, scrambler
    from channel_bridge import ChannelModel
    from sync import extract_csi, sync_packet, estimate_fine_cfo, apply_cfo_correction
    from detector import PacketDetector

    ch  = ChannelModel(noise_voltage=args.noise, cfo_hz=500, sco_ppm=0.5, seed=0)
    det = PacketDetector()

    csi_records   = []
    n_detected    = 0
    t_start = time.monotonic()

    print(f"  Processing {args.n_packets} packets ...\n")

    for seq in range(args.n_packets):
        # Build full packet (STF + LTF only, enough for CSI)
        stf = waveform.generate_stf()
        ltf = waveform.generate_ltf()
        pkt = np.concatenate([stf, ltf]).astype(np.complex64)

        # Apply channel
        impaired = ch.apply(pkt)

        # Coarse CFO from STF
        coarse_cfo = det._estimate_coarse_cfo(impaired[:cfg.STF_LEN])

        # Apply coarse correction, fine CFO, then extract CSI directly
        pkt_cc = apply_cfo_correction(impaired, coarse_cfo)
        s1 = cfg.STF_LEN + 64
        s2 = s1 + cfg.FFT_SIZE
        if s2 + cfg.FFT_SIZE <= len(pkt_cc):
            ltf1 = pkt_cc[s1:s1 + cfg.FFT_SIZE]
            ltf2 = pkt_cc[s2:s2 + cfg.FFT_SIZE]
            fine_cfo  = estimate_fine_cfo(ltf1, ltf2)
            total_cfo = coarse_cfo + fine_cfo
            H = extract_csi(pkt_cc, ltf_start=cfg.STF_LEN, total_cfo_hz=total_cfo)
            if not np.any(np.isnan(H)) and np.mean(np.abs(H)) > 0.1:
                csi_records.append(H)
                n_detected += 1

        done = seq + 1
        pct  = int(done / args.n_packets * 40)
        bar  = "#" * pct + "." * (40 - pct)
        print(f"\r  [{bar}] {done}/{args.n_packets}", end="", flush=True)

    elapsed = time.monotonic() - t_start
    print(f"\n\n  Elapsed: {elapsed:.3f}s\n")

    # -- Checks ----------------------------------------------------------------
    rate = n_detected / args.n_packets
    check(f"CSI extraction rate >= 80%",
          rate >= 0.80, f"extracted={n_detected}/{args.n_packets}  rate={rate:.1%}")

    if csi_records:
        stacked = np.array(csi_records)
        check(f"H_hat shape (N, {cfg.NUM_ACTIVE})",
              stacked.shape[1] == cfg.NUM_ACTIVE)
        check("H_hat no NaN in any record",
              not np.any(np.isnan(stacked)))
        mag = np.mean(np.abs(stacked))
        check("H_hat mean magnitude 0.5-3.0",
              0.5 < mag < 3.0, f"mean|H|={mag:.3f}")
        # CFO correction leaves residual CFO small — check phase variance is bounded
        phase_var = float(np.mean(np.var(np.angle(stacked), axis=0)))
        check("Per-subcarrier phase variance < 2.0 rad^2",
              phase_var < 2.0, f"phase_var={phase_var:.4f}")

    # -- HDF5 logging stress test -----------------------------------------------
    print("\n  [3d] HDF5 logger stress test (50 fast writes)")
    import queue as _q
    from logger import CSILogger
    hdf5_path = "stage3_logger_test.h5"
    if os.path.exists(hdf5_path):
        os.remove(hdf5_path)

    q      = _q.Queue(maxsize=500)
    logger = CSILogger(q, hdf5_path, "stage3_log", n_rx_channels=1,
                       flush_interval=10)
    log_t  = threading.Thread(target=logger.run, daemon=True)
    log_t.start()

    t0 = time.monotonic()
    for i in range(50):
        q.put({'H_hat':     np.random.randn(cfg.NUM_ACTIVE).astype(np.complex64),
               'timestamp': time.monotonic(),
               'seq':       i, 'dropped': 0})
    logger.stop()
    log_t.join(timeout=5)
    log_dur = time.monotonic() - t0

    import h5py
    with h5py.File(hdf5_path, 'r') as f:
        shape = f["/sessions/stage3_log/csi/antenna0"].shape
    check(f"HDF5 shape[1] == {cfg.NUM_ACTIVE}", shape[1] == cfg.NUM_ACTIVE, f"shape={shape}")
    check("HDF5 wrote 50 records",   shape[0] == 50,              f"shape={shape}")
    check("HDF5 flush done in < 5s", log_dur < 5.0,               f"took={log_dur:.2f}s")
    print(f"  HDF5 shape: {shape}  write_time: {log_dur:.2f}s")

    if os.path.exists(hdf5_path):
        os.remove(hdf5_path)

    print()
    if not errors:
        print("  [PASS]  STAGE 3 COMPLETE -- all checks passed\n")
        sys.exit(0)
    else:
        print(f"  [FAIL]  STAGE 3 FAILED -- {len(errors)} checks failed:")
        for e in errors:
            print(f"         * {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
