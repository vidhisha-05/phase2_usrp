import queue
import tempfile
from pathlib import Path

import h5py
import numpy as np

import config as cfg
from logger import CSILogger


NUM_RECORDS_VALID = 3
NUM_ACTIVE = cfg.NUM_ACTIVE


def make_record(crc_value, seq, abs_s):
    H = (
        np.arange(NUM_ACTIVE, dtype=np.float32)
        + 1j * np.arange(NUM_ACTIVE, dtype=np.float32)
    ).astype(np.complex64)

    return {
        "H_hat": H,
        "H_sanitized": H,
        "phase_sanitized": np.angle(H),
        "timestamp": float(abs_s) / cfg.FS_FFT,
        "abs_s": int(abs_s),
        "seq": int(seq),
        "cfo_hz": 10.0,
        "crc_ok": crc_value,
    }


def main():
    print("=" * 72)
    print("FIX 4C-5 — STRICT CRC GATE")
    print("=" * 72)

    q = queue.Queue()

    with tempfile.TemporaryDirectory(prefix="fix4c5_crc_") as td:
        h5_path = str(Path(td) / "crc_gate_test.h5")

        logger = CSILogger(
            csi_queue=q,
            hdf5_path=h5_path,
            session_id="session_001",
            n_rx_channels=1,
            flush_interval=10,
        )

        # Valid record: MUST be logged.
        q.put(make_record(True, 0, 1000))

        # Invalid record: MUST be discarded.
        q.put(make_record(False, 1, 2000))

        # Missing crc_ok: MUST be discarded.
        missing = make_record(True, 2, 3000)
        del missing["crc_ok"]
        q.put(missing)

        # None: MUST be discarded.
        q.put(make_record(None, 3, 4000))

        # Integer zero: MUST be discarded.
        q.put(make_record(0, 4, 5000))

        # String "True": MUST be discarded.
        q.put(make_record("True", 5, 6000))

        logger.stop()
        logger.run()

        with h5py.File(h5_path, "r") as f:
            base = "/sessions/session_001"

            n = f[f"{base}/timestamps"].shape[0]
            seq = f[f"{base}/seq"][:]
            abs_s = f[f"{base}/abs_s"][:]

            print(f"[INFO] Logged records: {n}")
            print(f"[INFO] Logged seq:      {seq}")
            print(f"[INFO] Logged abs_s:    {abs_s}")

            assert n == 1, (
                f"Expected exactly 1 CRC-valid record, got {n}"
            )

            assert np.array_equal(seq, np.array([0], dtype=np.int64))
            assert np.array_equal(abs_s, np.array([1000], dtype=np.int64))

            print("[PASS] crc_ok=True is logged.")
            print("[PASS] crc_ok=False is discarded.")
            print("[PASS] Missing crc_ok is discarded.")
            print("[PASS] crc_ok=None is discarded.")
            print("[PASS] crc_ok=0 is discarded.")
            print('[PASS] crc_ok="True" is discarded.')
            print("[PASS] Strict CRC gate is enforced.")
            print()
            print("=" * 72)
            print("FIX 4C-5 COMPLETE")
            print("=" * 72)


if __name__ == "__main__":
    main()