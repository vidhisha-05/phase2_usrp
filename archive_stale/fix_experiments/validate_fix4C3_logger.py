"""
validate_fix4C3_logger.py

Fix 4C-3 validation:
Verify that CSILogger persists abs_s and keeps it row-aligned
with timestamps, seq, CSI, and CFO.
"""

import os
import tempfile
import queue
import numpy as np
import h5py

import config as cfg
from logger import CSILogger


def make_record(abs_s, seq, timestamp, cfo_hz, crc_ok=True):
    return {
        "H_hat": np.ones(cfg.NUM_ACTIVE, dtype=np.complex64) * (1.0 + 0.1j),
        "H_sanitized": np.ones(
            cfg.NUM_ACTIVE, dtype=np.complex64
        ) * (2.0 + 0.2j),
        "phase_sanitized": np.ones(
            cfg.NUM_ACTIVE, dtype=np.float64
        ) * 0.3,
        "timestamp": float(timestamp),
        "abs_s": int(abs_s),
        "seq": int(seq),
        "cfo_hz": float(cfo_hz),
        "crc_ok": bool(crc_ok),
        "dropped": 0,
    }


def main():
    fd, path = tempfile.mkstemp(
        prefix="fix4c3_logger_",
        suffix=".h5"
    )
    os.close(fd)

    try:
        q = queue.Queue()

        records = [
            make_record(997,     0, 100.000000000,  100.0),
            make_record(100997,  1, 100.005000000,  101.0),
            make_record(200997,  2, 100.010000000,  102.0),
            make_record(300997,  3, 100.015000000,  103.0),
        ]

        for r in records:
            q.put(r)

        logger = CSILogger(
            csi_queue=q,
            hdf5_path=path,
            session_id="session_001",
            n_rx_channels=1,
            flush_interval=2,
        )

        logger.stop()
        logger.run()

        with h5py.File(path, "r") as f:
            base = "/sessions/session_001"

            # ---------------------------------------------------------
            # 1. Dataset existence
            # ---------------------------------------------------------
            required = [
                f"{base}/csi/antenna0",
                f"{base}/csi/sanitized",
                f"{base}/csi/phase_sanitized",
                f"{base}/timestamps",
                f"{base}/abs_s",
                f"{base}/seq",
                f"{base}/cfo_hz",
            ]

            for dataset in required:
                assert dataset in f, f"Missing dataset: {dataset}"

            print("[PASS] All required datasets exist")

            # ---------------------------------------------------------
            # 2. Dataset lengths
            # ---------------------------------------------------------
            n = len(records)

            lengths = {
                "antenna0": f[f"{base}/csi/antenna0"].shape[0],
                "sanitized": f[f"{base}/csi/sanitized"].shape[0],
                "phase_sanitized": f[
                    f"{base}/csi/phase_sanitized"
                ].shape[0],
                "timestamps": f[f"{base}/timestamps"].shape[0],
                "abs_s": f[f"{base}/abs_s"].shape[0],
                "seq": f[f"{base}/seq"].shape[0],
                "cfo_hz": f[f"{base}/cfo_hz"].shape[0],
            }

            for name, length in lengths.items():
                assert length == n, (
                    f"{name} length={length}, expected {n}"
                )

            print("[PASS] All datasets have identical record counts")

            # ---------------------------------------------------------
            # 3. abs_s values
            # ---------------------------------------------------------
            abs_s = f[f"{base}/abs_s"][:]
            expected_abs_s = np.array(
                [997, 100997, 200997, 300997],
                dtype=np.int64,
            )

            assert abs_s.dtype == np.int64
            assert np.array_equal(abs_s, expected_abs_s)

            print("[PASS] abs_s values and dtype are correct")

            # ---------------------------------------------------------
            # 4. seq values
            # ---------------------------------------------------------
            seq = f[f"{base}/seq"][:]
            expected_seq = np.array(
                [0, 1, 2, 3],
                dtype=np.int64,
            )

            assert seq.dtype == np.int64
            assert np.array_equal(seq, expected_seq)

            print("[PASS] seq values remain correct")

            # ---------------------------------------------------------
            # 5. Timestamp values
            # ---------------------------------------------------------
            timestamps = f[f"{base}/timestamps"][:]
            expected_timestamps = np.array(
                [
                    100.000000000,
                    100.005000000,
                    100.010000000,
                    100.015000000,
                ],
                dtype=np.float64,
            )

            assert np.allclose(
                timestamps,
                expected_timestamps,
                rtol=0.0,
                atol=1e-12,
            )

            print("[PASS] timestamp values remain correct")

            # ---------------------------------------------------------
            # 6. CFO values
            # ---------------------------------------------------------
            cfo = f[f"{base}/cfo_hz"][:]
            expected_cfo = np.array(
                [100.0, 101.0, 102.0, 103.0],
                dtype=np.float64,
            )

            assert np.allclose(
                cfo,
                expected_cfo,
                rtol=0.0,
                atol=1e-12,
            )

            print("[PASS] CFO values remain correct")

            # ---------------------------------------------------------
            # 7. CSI row alignment
            # ---------------------------------------------------------
            H = f[f"{base}/csi/antenna0"][:]

            assert H.shape == (
                n,
                cfg.NUM_ACTIVE,
            )

            expected_value = 1.0 + 0.1j

            assert np.allclose(
                H,
                expected_value,
                rtol=0.0,
                atol=1e-7,
            )

            print("[PASS] CSI rows remain aligned")

            # ---------------------------------------------------------
            # 8. abs_s → timestamp relationship
            #
            # Here we deliberately use:
            #
            # timestamp = 100 + abs_s / 20e6
            #
            # relative to the first abs_s coordinate.
            # ---------------------------------------------------------
            fs = float(cfg.FS_FFT)

            reconstructed_dt = (
                abs_s.astype(np.float64)
                - float(abs_s[0])
            ) / fs

            actual_dt = timestamps - timestamps[0]

            assert np.allclose(
                reconstructed_dt,
                actual_dt,
                rtol=0.0,
                atol=1e-12,
            )

            print("[PASS] abs_s ↔ timestamp alignment is correct")

            # ---------------------------------------------------------
            # 9. Check monotonicity
            # ---------------------------------------------------------
            assert np.all(np.diff(abs_s) > 0)
            assert np.all(np.diff(timestamps) > 0)
            assert np.all(np.diff(seq) > 0)

            print("[PASS] abs_s / timestamp / seq are monotonic")

            # ---------------------------------------------------------
            # 10. Check expected 5 ms spacing
            # ---------------------------------------------------------
            abs_spacing = np.diff(abs_s)
            expected_spacing = int(round(
                cfg.FS_FFT * cfg.TX_PACKET_PERIOD_S
            ))

            assert np.all(abs_spacing == expected_spacing), (
                f"abs_s spacing={abs_spacing}, "
                f"expected {expected_spacing}"
            )

            print(
                "[PASS] abs_s packet spacing matches "
                f"{cfg.TX_PACKET_PERIOD_S * 1000:.3f} ms schedule"
            )

        print("\nFIX 4C-3 COMPLETE")

    finally:
        if os.path.exists(path):
            os.remove(path)


if __name__ == "__main__":
    main()