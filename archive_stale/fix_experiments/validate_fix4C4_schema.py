from pathlib import Path
import ast

PROJECT_DIR = Path(__file__).resolve().parent
LOGGER_FILE = PROJECT_DIR / "logger.py"
MAIN_FILE = PROJECT_DIR / "main_hardware.py"

print("=" * 72)
print("FIX 4C-4 — HDF5 SCHEMA / HARDWARE RECORD CONTRACT AUDIT")
print("=" * 72)

print(f"[INFO] Project directory : {PROJECT_DIR}")
print(f"[INFO] logger.py        : {LOGGER_FILE}")
print(f"[INFO] main_hardware.py : {MAIN_FILE}")

if not LOGGER_FILE.exists():
    raise FileNotFoundError(LOGGER_FILE)

if not MAIN_FILE.exists():
    raise FileNotFoundError(MAIN_FILE)


# ---------------------------------------------------------------------
# 1. Read current source files
# ---------------------------------------------------------------------

logger_text = LOGGER_FILE.read_text(encoding="utf-8")
main_text = MAIN_FILE.read_text(encoding="utf-8")

print("\n[1] Current source files loaded successfully.")


# ---------------------------------------------------------------------
# 2. Required hardware-record fields
# ---------------------------------------------------------------------

required_record_fields = {
    "H_hat",
    "H_sanitized",
    "phase_sanitized",
    "phase_slope",
    "phase_intercept",
    "rx_bits",
    "crc_ok",
    "cfo_hz",
    "timestamp",
    "abs_s",
    "seq",
    "dropped",
    "ring_occ",
}

print("\n[2] Checking current main_hardware.py record contract...")

main_tree = ast.parse(main_text)

record_dicts = []

for node in ast.walk(main_tree):
    if isinstance(node, ast.Dict):
        keys = []
        for key in node.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                keys.append(key.value)

        if "H_hat" in keys and "timestamp" in keys and "seq" in keys:
            record_dicts.append((node.lineno, set(keys)))

if not record_dicts:
    raise AssertionError(
        "Could not find the hardware record dictionary in main_hardware.py"
    )

for lineno, keys in record_dicts:
    print(f"\n[PASS] Candidate record dictionary found at line {lineno}")
    print("[INFO] Keys found:")
    for key in sorted(keys):
        print(f"       {key}")

    missing = required_record_fields - keys

    if missing:
        raise AssertionError(
            f"Hardware record is missing fields: {sorted(missing)}"
        )

print("\n[PASS] Required hardware-record fields are present.")


# ---------------------------------------------------------------------
# 3. Required HDF5 datasets
# ---------------------------------------------------------------------

required_hdf5_names = {
    "timestamps",
    "seq",
    "cfo_hz",
    "abs_s",
    "csi/antenna0",
    "csi/sanitized",
    "csi/phase_sanitized",
}

print("\n[3] Checking logger.py for required HDF5 datasets...")

expected_logger_tokens = {
    "timestamps": "timestamps",
    "seq": "seq",
    "cfo_hz": "cfo_hz",
    "abs_s": "abs_s",
    "csi/antenna0": "antenna0",
    "csi/sanitized": "sanitized",
    "csi/phase_sanitized": "phase_sanitized",
}

for logical_name, token in expected_logger_tokens.items():
    if token not in logger_text:
        raise AssertionError(
            f"Required logger token/dataset reference not found: "
            f"{logical_name!r} -> {token!r}"
        )

    print(f"[PASS] {logical_name}")


# ---------------------------------------------------------------------
# 4. Check append/flush paths for abs_s
# ---------------------------------------------------------------------

print("\n[4] Checking abs_s persistence path...")

required_abs_s_tokens = [
    "_buf_abs_s",
    "record['abs_s']",
    "abs_s",
]

for token in required_abs_s_tokens:
    if token not in logger_text:
        raise AssertionError(
            f"Required abs_s persistence token missing: {token!r}"
        )

    print(f"[PASS] Found {token!r}")


# ---------------------------------------------------------------------
# 5. Check that abs_s is stored as integer
# ---------------------------------------------------------------------

print("\n[5] Checking abs_s integer conversion...")

if "int(record['abs_s'])" not in logger_text:
    raise AssertionError(
        "logger.py does not explicitly convert record['abs_s'] to int "
        "before buffering."
    )

print("[PASS] abs_s is explicitly converted with int(record['abs_s']).")


# ---------------------------------------------------------------------
# 6. Check int64 HDF5 declaration
# ---------------------------------------------------------------------

print("\n[6] Checking HDF5 abs_s dtype...")

if "np.int64" not in logger_text:
    raise AssertionError(
        "logger.py does not contain np.int64. "
        "Cannot confirm int64 HDF5 storage."
    )

print("[PASS] np.int64 declaration/reference found.")


# ---------------------------------------------------------------------
# 7. Check timestamp relationship
# ---------------------------------------------------------------------

print("\n[7] Checking timestamp / abs_s relationship in hardware code...")

required_timestamp_expression = (
    "float(uhd_t0_s) + float(abs_s) / float(cfg.FS_FFT)"
)

if required_timestamp_expression not in main_text:
    raise AssertionError(
        "Expected UHD-derived timestamp expression was not found."
    )

print("[PASS] Timestamp is derived from:")
print("       uhd_t0_s + abs_s / FS_FFT")


# ---------------------------------------------------------------------
# 8. Check source-of-truth frequency
# ---------------------------------------------------------------------

print("\n[8] Checking FS_FFT reference...")

if "cfg.FS_FFT" not in main_text:
    raise AssertionError(
        "main_hardware.py does not reference cfg.FS_FFT for the "
        "absolute timestamp calculation."
    )

print("[PASS] main_hardware.py uses cfg.FS_FFT.")


# ---------------------------------------------------------------------
# 9. Check session organization
# ---------------------------------------------------------------------

print("\n[9] Checking session organization...")

session_tokens = [
    "sessions",
    "session_id",
]

for token in session_tokens:
    if token not in logger_text:
        raise AssertionError(
            f"Expected session-management token missing: {token!r}"
        )

    print(f"[PASS] Found {token!r}")


# ---------------------------------------------------------------------
# 10. Summary
# ---------------------------------------------------------------------

print("\n" + "=" * 72)
print("FIX 4C-4 SOURCE-LEVEL SCHEMA AUDIT COMPLETE")
print("=" * 72)
print()
print("This is a SOURCE audit only.")
print("No production file was modified.")
print()
print("If this test passes, the next step is to inspect an ACTUAL")
print("HDF5 file produced by the current logger and verify:")
print("  - dataset names")
print("  - shapes")
print("  - dtypes")
print("  - record counts")
print("  - CSI dimensions")
print("  - session structure")
print("  - abs_s / timestamp / seq alignment")
print()
print("Do NOT modify logger.py yet.")
print("=" * 72)