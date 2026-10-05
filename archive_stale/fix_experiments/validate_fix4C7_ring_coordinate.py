import numpy as np

from rx_sim import RingBuffer


def main():
    print("=" * 72)
    print("FIX 4C-7 — RING DROP / ABSOLUTE COORDINATE ANALYSIS")
    print("=" * 72)

    # Small deterministic ring.
    cap = 16
    ring = RingBuffer(capacity=cap)

    # ---------------------------------------------------------------
    # Establish a stream-coordinate sequence.
    #
    # First write:
    #   physical stream samples 0..11
    #
    # Then write:
    #   physical stream samples 12..19
    #
    # Ring capacity = 16.
    # Total physical samples written = 20.
    # Therefore 4 oldest samples must be discarded.
    # ---------------------------------------------------------------
    first = np.arange(12, dtype=np.float32).astype(np.complex64)
    second = np.arange(12, 20, dtype=np.float32).astype(np.complex64)

    ring.write(first)

    physical_before_second = 12

    assert ring.dropped == 0

    ring.write(second)

    physical_total = 20
    dropped = ring.dropped

    print(f"[INFO] Physical samples written : {physical_total}")
    print(f"[INFO] Ring capacity            : {cap}")
    print(f"[INFO] Ring dropped             : {dropped}")
    print(f"[INFO] Ring occupancy           : {ring.occupancy}")

    assert dropped == 4
    assert ring.occupancy == 1.0

    # ---------------------------------------------------------------
    # Read everything currently available.
    # ---------------------------------------------------------------
    out = ring.read(cap)

    expected_survivors = np.arange(4, 20, dtype=np.float32).astype(
        np.complex64
    )

    assert np.array_equal(out, expected_survivors)

    print("[PASS] Surviving samples are physical stream samples 4..19.")

    # ---------------------------------------------------------------
    # Coordinate distinction
    # ---------------------------------------------------------------
    processed_count = len(out)

    assert processed_count == 16
    assert physical_total == processed_count + dropped

    print()
    print("[INFO] Processed sample count :", processed_count)
    print("[INFO] Dropped sample count   :", dropped)
    print("[INFO] Physical total         :", physical_total)

    print()
    print("Coordinate interpretation:")
    print()
    print("  Processed coordinate:")
    print(f"      samples processed = {processed_count}")
    print()
    print("  Physical stream coordinate:")
    print(f"      samples generated = {physical_total}")
    print()
    print("  Difference:")
    print(f"      dropped samples = {dropped}")

    assert physical_total - processed_count == dropped

    print()
    print("[PASS] Ring drop creates a 4-sample gap between")
    print("       processed-sample count and physical stream count.")

    # ---------------------------------------------------------------
    # Timestamp consequence at 25 MS/s.
    # ---------------------------------------------------------------
    fs_hw = 25e6
    dropped_time_s = dropped / fs_hw

    print()
    print("At FS_HW = 25 MHz:")
    print(f"    dropped time = {dropped_time_s:.12e} s")
    print(f"                  = {dropped_time_s * 1e9:.3f} ns")

    # ---------------------------------------------------------------
    # Important conclusion:
    #
    # The ring itself cannot reconstruct physical coordinates merely
    # from the returned sample array. The cumulative dropped count is
    # required to reconstruct the stream coordinate.
    # ---------------------------------------------------------------
    print()
    print("=" * 72)
    print("FIX 4C-7 CHARACTERIZATION COMPLETE")
    print("=" * 72)
    print()
    print("No production code was changed.")
    print("The test demonstrates why ring-drop accounting must remain")
    print("separate from the processed sample coordinate.")


if __name__ == "__main__":
    main()