from pathlib import Path
import h5py
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
H5_FILE = PROJECT_DIR / "csi_data.h5"

print("=" * 72)
print("FIX 4C-4 — EXISTING HDF5 ARTIFACT INSPECTION")
print("=" * 72)

print(f"[INFO] File: {H5_FILE}")

if not H5_FILE.exists():
    raise FileNotFoundError(H5_FILE)


def print_group(group, prefix=""):
    for name, obj in group.items():
        path = f"{prefix}/{name}" if prefix else name

        if isinstance(obj, h5py.Group):
            print(f"[GROUP]   {path}/")
            print_group(obj, path)

        elif isinstance(obj, h5py.Dataset):
            print(
                f"[DATASET] {path} | "
                f"shape={obj.shape} | "
                f"dtype={obj.dtype} | "
                f"maxshape={obj.maxshape} | "
                f"chunks={obj.chunks}"
            )


with h5py.File(H5_FILE, "r") as f:

    print("\n[1] HDF5 hierarchy")
    print("-" * 72)
    print_group(f)

    print("\n[2] Root attributes")
    print("-" * 72)

    if len(f.attrs) == 0:
        print("[INFO] No root attributes.")
    else:
        for key, value in f.attrs.items():
            print(f"{key} = {value}")

    print("\n[3] Sessions")
    print("-" * 72)

    if "sessions" not in f:
        print("[FAIL] No /sessions group found.")
        raise SystemExit(1)

    sessions = f["sessions"]

    if len(sessions) == 0:
        print("[FAIL] /sessions exists but contains no sessions.")
        raise SystemExit(1)

    print(f"[INFO] Session count: {len(sessions)}")

    for session_name, session in sessions.items():

        print(f"\nSESSION: {session_name}")
        print("-" * 72)

        print("[INFO] Session attributes:")
        for key, value in session.attrs.items():
            print(f"  {key} = {value}")

        print("\n[INFO] Direct datasets/groups:")

        for name, obj in session.items():

            if isinstance(obj, h5py.Dataset):
                print(
                    f"  {name}: "
                    f"shape={obj.shape}, "
                    f"dtype={obj.dtype}, "
                    f"maxshape={obj.maxshape}, "
                    f"chunks={obj.chunks}"
                )

            elif isinstance(obj, h5py.Group):
                print(f"  {name}/")

                for child_name, child in obj.items():
                    if isinstance(child, h5py.Dataset):
                        print(
                            f"    {name}/{child_name}: "
                            f"shape={child.shape}, "
                            f"dtype={child.dtype}, "
                            f"maxshape={child.maxshape}, "
                            f"chunks={child.chunks}"
                        )

        # -------------------------------------------------------------
        # Check expected current schema
        # -------------------------------------------------------------

        print("\n[4] Expected current schema check")

        required_paths = [
            "timestamps",
            "seq",
            "cfo_hz",
            "abs_s",
            "csi/antenna0",
            "csi/sanitized",
            "csi/phase_sanitized",
        ]

        for path in required_paths:
            if path in session:
                print(f"[PASS] {path}")
            else:
                print(f"[MISSING] {path}")

        # -------------------------------------------------------------
        # Record-count analysis
        # -------------------------------------------------------------

        print("\n[5] Record-count analysis")

        dataset_paths = [
            "timestamps",
            "seq",
            "cfo_hz",
            "abs_s",
            "csi/antenna0",
            "csi/sanitized",
            "csi/phase_sanitized",
        ]

        counts = {}

        for path in dataset_paths:
            if path in session:
                ds = session[path]
                counts[path] = ds.shape[0]
                print(f"{path}: {ds.shape[0]}")

        if counts:
            unique_counts = set(counts.values())

            if len(unique_counts) == 1:
                print("[PASS] All present datasets have identical record counts.")
            else:
                print("[WARNING] Record counts are NOT identical.")

        # -------------------------------------------------------------
        # Detailed CSI dimensions
        # -------------------------------------------------------------

        print("\n[6] CSI dimensionality")

        for path in [
            "csi/antenna0",
            "csi/sanitized",
            "csi/phase_sanitized",
        ]:
            if path not in session:
                continue

            ds = session[path]

            print(
                f"{path}: shape={ds.shape}, dtype={ds.dtype}"
            )

            if len(ds.shape) >= 2:
                print(
                    f"  records      = {ds.shape[0]}"
                )
                print(
                    f"  subcarriers  = {ds.shape[1]}"
                )

                if ds.shape[1] == 106:
                    print(
                        "  [PASS] 106 active subcarriers."
                    )
                else:
                    print(
                        f"  [INFO] Subcarrier dimension is "
                        f"{ds.shape[1]}, not 106."
                    )

        # -------------------------------------------------------------
        # Numeric metadata inspection
        # -------------------------------------------------------------

        print("\n[7] Metadata samples")

        for path in [
            "timestamps",
            "seq",
            "cfo_hz",
            "abs_s",
        ]:
            if path not in session:
                continue

            data = session[path][:]

            print(
                f"{path}: dtype={data.dtype}, "
                f"shape={data.shape}"
            )

            if len(data) > 0:
                n = min(5, len(data))
                print(f"  first {n}: {data[:n]}")

                if len(data) > n:
                    print(
                        f"  last  {n}: {data[-n:]}"
                    )

        print("\n" + "=" * 72)
        print(f"END SESSION: {session_name}")
        print("=" * 72)

print("\nEXISTING HDF5 INSPECTION COMPLETE.")
print("No files were modified.")