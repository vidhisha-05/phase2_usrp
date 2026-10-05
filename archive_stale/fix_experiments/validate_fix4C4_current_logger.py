"""
FIX 4C-4 Phase B-2
Fresh HDF5 artifact generated through the CURRENT CSILogger implementation.

This test:
    1. Uses the real current logger.py.
    2. Feeds records through queue.Queue -> CSILogger.run().
    3. Forces multiple flushes using flush_interval=3.
    4. Produces 7 accepted records.
    5. Reopens the resulting HDF5 file.
    6. Verifies schema, dimensions, dtypes, counts and alignment.

No production file is modified.
"""

from pathlib import Path
import queue
import threading
import tempfile

import h5py
import numpy as np

import config as cfg
from logger import CSILogger


# ---------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------

N_RECORDS = 7
FLUSH_INTERVAL = 3

T0 = 123456.0

FIRST_ABS_S = 997
PACKET_SPACING_BB = 100_000

EXPECTED_ABS_S = np.array(
    [
        FIRST_ABS_S + i * PACKET_SPACING_BB
        for i in range(N_RECORDS)
    ],
    dtype=np.int64,
)

EXPECTED_TIMESTAMPS = (
    T0 + EXPECTED_ABS_S.astype(np.float64) / float(cfg.FS_FFT)
)

EXPECTED_SEQ = np.arange(
    N_RECORDS,
    dtype=np.int64,
)

EXPECTED_CFO = np.array(
    [1000.0 + 10.0 * i for i in range(N_RECORDS)],
    dtype=np.float64,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def make_record(i: int):
    """
    Construct one record matching the current main_hardware.py
    logger contract.
    """

    # Deterministic complex CSI.
    #
    # Shape:
    #     (NUM_ACTIVE,)
    #
    # Current system:
    #     NUM_ACTIVE = 106
    #
    H_hat = np.full(
        (cfg.NUM_ACTIVE,),
        complex(i + 1.0, -(i + 0.5)),
        dtype=np.complex64,
    )

    H_sanitized = np.full(
        (cfg.NUM_ACTIVE,),
        complex(i + 2.0, -(i + 0.25)),
        dtype=np.complex64,
    )

    phase_sanitized = np.full(
        (cfg.NUM_ACTIVE,),
        0.01 * i,
        dtype=np.float64,
    )

    return {
        "H_hat": H_hat,
        "H_sanitized": H_sanitized,
        "phase_sanitized": phase_sanitized,

        "phase_slope": 0.001 * i,
        "phase_intercept": 0.002 * i,

        "rx_bits": np.array([0, 1, 0, 1], dtype=np.uint8),

        "crc_ok": True,

        "cfo_hz": EXPECTED_CFO[i],

        "timestamp": EXPECTED_TIMESTAMPS[i],

        "abs_s": int(EXPECTED_ABS_S[i]),

        "seq": int(EXPECTED_SEQ[i]),

        "dropped": 0,

        "ring_occ": i,
    }


# ---------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------

print("=" * 72)
print("FIX 4C-4 PHASE B-2 — CURRENT LOGGER → FRESH HDF5")
print("=" * 72)

print(f"[INFO] cfg.NUM_ACTIVE = {cfg.NUM_ACTIVE}")
print(f"[INFO] cfg.FS_FFT     = {cfg.FS_FFT}")
print(f"[INFO] N_RECORDS      = {N_RECORDS}")
print(f"[INFO] FLUSH_INTERVAL = {FLUSH_INTERVAL}")
print(f"[INFO] FIRST_ABS_S    = {FIRST_ABS_S}")
print(f"[INFO] BB spacing     = {PACKET_SPACING_BB}")


# ---------------------------------------------------------------------
# Basic configuration checks
# ---------------------------------------------------------------------

if cfg.NUM_ACTIVE != 106:
    raise AssertionError(
        f"Expected NUM_ACTIVE=106, got {cfg.NUM_ACTIVE}"
    )

if float(cfg.FS_FFT) != 20e6:
    raise AssertionError(
        f"Expected FS_FFT=20e6, got {cfg.FS_FFT}"
    )

print("[PASS] Current PHY configuration is NUM_ACTIVE=106, FS_FFT=20 MHz.")


# ---------------------------------------------------------------------
# Temporary HDF5 file
# ---------------------------------------------------------------------

with tempfile.TemporaryDirectory(
    prefix="fix4c4_current_logger_"
) as tmpdir:

    h5_path = Path(tmpdir) / "current_logger_test.h5"

    print(f"[INFO] Temporary HDF5: {h5_path}")

    q = queue.Queue()

    logger = CSILogger(
        csi_queue=q,
        hdf5_path=str(h5_path),
        session_id="session_001",
        n_rx_channels=1,
        flush_interval=FLUSH_INTERVAL,
        metadata={
            "test_name": "FIX_4C_4_PHASE_B_2",
            "packet_period_s": 0.005,
        },
    )

    # ---------------------------------------------------------------
    # Feed exactly seven accepted records.
    # ---------------------------------------------------------------

    for i in range(N_RECORDS):
        q.put(make_record(i))

    print(
        f"[INFO] Queued {N_RECORDS} records."
    )

    # ---------------------------------------------------------------
    # Start logger thread.
    # ---------------------------------------------------------------

    logger_thread = threading.Thread(
        target=logger.run,
        name="fix4c4_logger_test",
    )

    logger_thread.start()

    # Tell logger to stop after queue drains.
    logger.stop()

    logger_thread.join(timeout=10.0)

    if logger_thread.is_alive():
        raise AssertionError(
            "Logger thread did not terminate within 10 seconds."
        )

    print("[PASS] Logger thread terminated cleanly.")

    if not h5_path.exists():
        raise AssertionError(
            "Expected HDF5 file was not created."
        )

    print("[PASS] Fresh HDF5 file exists.")


    # -----------------------------------------------------------------
    # Reopen actual generated HDF5 artifact.
    # -----------------------------------------------------------------

    with h5py.File(h5_path, "r") as f:

        print("\n" + "-" * 72)
        print("HDF5 STRUCTURE")
        print("-" * 72)

        if "sessions" not in f:
            raise AssertionError("Missing /sessions group.")

        if "session_001" not in f["sessions"]:
            raise AssertionError(
                "Missing /sessions/session_001."
            )

        session = f["sessions/session_001"]

        required_paths = [
            "timestamps",
            "abs_s",
            "seq",
            "cfo_hz",
            "csi",
            "csi/antenna0",
            "csi/sanitized",
            "csi/phase_sanitized",
        ]

        for path in required_paths:
            if path not in session:
                raise AssertionError(
                    f"Missing required dataset/group: {path}"
                )

            print(f"[PASS] {path}")


        # -------------------------------------------------------------
        # Dataset handles
        # -------------------------------------------------------------

        ds_ts = session["timestamps"]
        ds_abs = session["abs_s"]
        ds_seq = session["seq"]
        ds_cfo = session["cfo_hz"]

        ds_ant0 = session["csi/antenna0"]
        ds_san = session["csi/sanitized"]
        ds_phase = session["csi/phase_sanitized"]


        # -------------------------------------------------------------
        # Shape checks
        # -------------------------------------------------------------

        print("\n" + "-" * 72)
        print("SHAPE CHECKS")
        print("-" * 72)

        scalar_datasets = {
            "timestamps": ds_ts,
            "abs_s": ds_abs,
            "seq": ds_seq,
            "cfo_hz": ds_cfo,
        }

        for name, ds in scalar_datasets.items():

            expected_shape = (N_RECORDS,)

            if ds.shape != expected_shape:
                raise AssertionError(
                    f"{name}: expected shape {expected_shape}, "
                    f"got {ds.shape}"
                )

            print(
                f"[PASS] {name}: shape={ds.shape}"
            )


        csi_datasets = {
            "csi/antenna0": ds_ant0,
            "csi/sanitized": ds_san,
            "csi/phase_sanitized": ds_phase,
        }

        for name, ds in csi_datasets.items():

            expected_shape = (
                N_RECORDS,
                cfg.NUM_ACTIVE,
            )

            if ds.shape != expected_shape:
                raise AssertionError(
                    f"{name}: expected shape {expected_shape}, "
                    f"got {ds.shape}"
                )

            print(
                f"[PASS] {name}: shape={ds.shape}"
            )


        # -------------------------------------------------------------
        # Dtype checks
        # -------------------------------------------------------------

        print("\n" + "-" * 72)
        print("DTYPE CHECKS")
        print("-" * 72)

        expected_dtypes = {
            "timestamps": np.dtype(np.float64),
            "abs_s": np.dtype(np.int64),
            "seq": np.dtype(np.int64),
            "cfo_hz": np.dtype(np.float64),
            "csi/antenna0": np.dtype(np.complex64),
            "csi/sanitized": np.dtype(np.complex64),
            "csi/phase_sanitized": np.dtype(np.float64),
        }

        datasets = {
            "timestamps": ds_ts,
            "abs_s": ds_abs,
            "seq": ds_seq,
            "cfo_hz": ds_cfo,
            "csi/antenna0": ds_ant0,
            "csi/sanitized": ds_san,
            "csi/phase_sanitized": ds_phase,
        }

        for name, expected_dtype in expected_dtypes.items():

            actual = datasets[name].dtype

            if actual != expected_dtype:
                raise AssertionError(
                    f"{name}: expected dtype {expected_dtype}, "
                    f"got {actual}"
                )

            print(
                f"[PASS] {name}: dtype={actual}"
            )


        # -------------------------------------------------------------
        # Maxshape checks
        # -------------------------------------------------------------

        print("\n" + "-" * 72)
        print("MAXSHAPE CHECKS")
        print("-" * 72)

        for name, ds in datasets.items():

            if ds.maxshape[0] is not None:
                raise AssertionError(
                    f"{name}: first dimension is not unlimited. "
                    f"maxshape={ds.maxshape}"
                )

            print(
                f"[PASS] {name}: maxshape={ds.maxshape}"
            )


        # -------------------------------------------------------------
        # Chunk checks
        # -------------------------------------------------------------

        print("\n" + "-" * 72)
        print("CHUNK CHECKS")
        print("-" * 72)

        for name, ds in datasets.items():

            if ds.chunks is None:
                raise AssertionError(
                    f"{name}: dataset is not chunked."
                )

            print(
                f"[PASS] {name}: chunks={ds.chunks}"
            )


        # -------------------------------------------------------------
        # Record-count alignment
        # -------------------------------------------------------------

        print("\n" + "-" * 72)
        print("RECORD-COUNT ALIGNMENT")
        print("-" * 72)

        counts = {
            name: ds.shape[0]
            for name, ds in datasets.items()
        }

        for name, count in counts.items():
            print(f"{name}: {count}")

        if set(counts.values()) != {N_RECORDS}:
            raise AssertionError(
                f"Record-count mismatch: {counts}"
            )

        print(
            "[PASS] Every primary dataset contains exactly "
            f"{N_RECORDS} records."
        )


        # -------------------------------------------------------------
        # Read actual data
        # -------------------------------------------------------------

        timestamps = ds_ts[:]
        abs_s = ds_abs[:]
        seq = ds_seq[:]
        cfo = ds_cfo[:]

        ant0 = ds_ant0[:]
        san = ds_san[:]
        phase = ds_phase[:]


        # -------------------------------------------------------------
        # abs_s exact values
        # -------------------------------------------------------------

        print("\n" + "-" * 72)
        print("ABS_S VALUES")
        print("-" * 72)

        print(f"Expected: {EXPECTED_ABS_S}")
        print(f"Actual:   {abs_s}")

        if not np.array_equal(abs_s, EXPECTED_ABS_S):
            raise AssertionError(
                "abs_s values do not match expected values."
            )

        print("[PASS] abs_s values are exact.")


        # -------------------------------------------------------------
        # abs_s spacing
        # -------------------------------------------------------------

        abs_diffs = np.diff(abs_s)

        print(f"abs_s differences: {abs_diffs}")

        if not np.all(abs_diffs == PACKET_SPACING_BB):
            raise AssertionError(
                "abs_s packet spacing is not exactly 100000 BB samples."
            )

        print(
            "[PASS] abs_s spacing is exactly 100000 BB samples."
        )


        # -------------------------------------------------------------
        # Timestamp exact relationship
        # -------------------------------------------------------------

        expected_ts_from_abs = (
            T0 + abs_s.astype(np.float64) / float(cfg.FS_FFT)
        )

        print("\n" + "-" * 72)
        print("TIMESTAMP ↔ ABS_S")
        print("-" * 72)

        timestamp_error = np.abs(
            timestamps - expected_ts_from_abs
        )

        print(
            f"Maximum timestamp reconstruction error: "
            f"{np.max(timestamp_error):.3e} s"
        )

        if not np.allclose(
            timestamps,
            expected_ts_from_abs,
            rtol=0.0,
            atol=1e-10,
        ):
            raise AssertionError(
                "Timestamp / abs_s relationship failed."
            )

        print(
            "[PASS] timestamp = T0 + abs_s / FS_FFT."
        )


        # -------------------------------------------------------------
        # Timestamp spacing
        # -------------------------------------------------------------

        timestamp_diffs = np.diff(timestamps)

        print(
            f"Timestamp differences: {timestamp_diffs}"
        )

        if not np.allclose(
            timestamp_diffs,
            0.005,
            rtol=0.0,
            atol=1e-10,
        ):
            raise AssertionError(
                "Timestamp spacing is not 5 ms."
            )

        print(
            "[PASS] Packet timestamp spacing is exactly 5 ms "
            "within floating-point tolerance."
        )


        # -------------------------------------------------------------
        # Sequence values
        # -------------------------------------------------------------

        print("\n" + "-" * 72)
        print("SEQUENCE VALUES")
        print("-" * 72)

        print(f"Expected: {EXPECTED_SEQ}")
        print(f"Actual:   {seq}")

        if not np.array_equal(seq, EXPECTED_SEQ):
            raise AssertionError(
                "seq values do not match expected values."
            )

        print("[PASS] seq values are preserved.")


        # -------------------------------------------------------------
        # CFO values
        # -------------------------------------------------------------

        if not np.array_equal(cfo, EXPECTED_CFO):
            raise AssertionError(
                "CFO values do not match expected values."
            )

        print("[PASS] CFO values are preserved.")


        # -------------------------------------------------------------
        # CSI values
        # -------------------------------------------------------------

        for i in range(N_RECORDS):

            expected_ant0_value = complex(
                i + 1.0,
                -(i + 0.5),
            )

            expected_san_value = complex(
                i + 2.0,
                -(i + 0.25),
            )

            expected_phase_value = 0.01 * i

            if not np.all(
                ant0[i] == np.complex64(expected_ant0_value)
            ):
                raise AssertionError(
                    f"antenna0 CSI mismatch at record {i}."
                )

            if not np.all(
                san[i] == np.complex64(expected_san_value)
            ):
                raise AssertionError(
                    f"sanitized CSI mismatch at record {i}."
                )

            if not np.all(
                phase[i] == np.float64(expected_phase_value)
            ):
                raise AssertionError(
                    f"phase_sanitized mismatch at record {i}."
                )

        print(
            "[PASS] antenna0 CSI values are preserved."
        )

        print(
            "[PASS] sanitized CSI values are preserved."
        )

        print(
            "[PASS] phase_sanitized values are preserved."
        )


        # -------------------------------------------------------------
        # Cross-dataset row alignment
        # -------------------------------------------------------------

        for i in range(N_RECORDS):

            if abs_s[i] != EXPECTED_ABS_S[i]:
                raise AssertionError(
                    f"abs_s alignment failed at row {i}"
                )

            if seq[i] != EXPECTED_SEQ[i]:
                raise AssertionError(
                    f"seq alignment failed at row {i}"
                )

            if timestamps[i] != EXPECTED_TIMESTAMPS[i]:
                raise AssertionError(
                    f"timestamp alignment failed at row {i}"
                )

            if cfo[i] != EXPECTED_CFO[i]:
                raise AssertionError(
                    f"CFO alignment failed at row {i}"
                )

        print(
            "[PASS] CSI / timestamp / abs_s / seq / CFO "
            "row alignment is preserved."
        )


        # -------------------------------------------------------------
        # Session metadata
        # -------------------------------------------------------------

        print("\n" + "-" * 72)
        print("SESSION METADATA")
        print("-" * 72)

        attrs = session.attrs

        if attrs.get("test_name", None) != "FIX_4C_4_PHASE_B_2":
            raise AssertionError(
                "Session test_name metadata missing or incorrect."
            )

        if not np.isclose(
            float(attrs.get("packet_period_s")),
            0.005,
            rtol=0.0,
            atol=1e-15,
        ):
            raise AssertionError(
                "Session packet_period_s metadata incorrect."
            )

        print("[PASS] Session metadata preserved.")


print("\n" + "=" * 72)
print("FIX 4C-4 PHASE B-2 COMPLETE")
print("=" * 72)
print()
print("CURRENT logger.py successfully generated a fresh HDF5 artifact.")
print("Multiple flushes were exercised.")
print("Primary datasets remained aligned.")
print("No production file was modified.")
print("=" * 72)