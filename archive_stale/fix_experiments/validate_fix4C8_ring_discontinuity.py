import numpy as np

from rx_sim import RingBuffer


def main():
    print("=" * 72)
    print("FIX 4C-8 — RING OVERFLOW DISCONTINUITY MODEL")
    print("=" * 72)

    FS_HW = 25e6
    FS_BB = 20e6
    RATIO = FS_BB / FS_HW

    capacity = 32
    ring = RingBuffer(capacity=capacity)

    # ------------------------------------------------------------------
    # Model a continuous UHD stream.
    #
    # Block 1:
    #   physical HW samples 0..23
    #
    # Block 2:
    #   physical HW samples 24..47
    #
    # The physical stream itself has NO gap.
    # ------------------------------------------------------------------
    block1 = np.arange(24, dtype=np.float32).astype(np.complex64)
    block2 = np.arange(24, 48, dtype=np.float32).astype(np.complex64)

    ring.write(block1)

    # Read only part of block 1, intentionally leaving samples in ring.
    first_read = ring.read(16)

    assert len(first_read) == 16
    assert ring.dropped == 0

    # ------------------------------------------------------------------
    # Now write block 2.
    #
    # Ring before block2:
    #   8 samples occupied
    #
    # Free space:
    #   32 - 8 = 24
    #
    # block2 contains 24 samples, so this exact write should NOT overflow.
    # ------------------------------------------------------------------
    ring.write(block2)

    assert ring.dropped == 0

    # ------------------------------------------------------------------
    # Force an actual overflow with another continuous physical block.
    #
    # block3:
    #   physical HW samples 48..79
    #
    # Ring currently contains 32 samples.
    # Therefore all 32 currently stored samples must be overwritten.
    # ------------------------------------------------------------------
    block3 = np.arange(48, 80, dtype=np.float32).astype(np.complex64)

    ring.write(block3)

    assert ring.dropped == 32

    print(f"[INFO] Ring capacity        : {capacity} HW samples")
    print(f"[INFO] Physical stream end  : 80 HW samples")
    print(f"[INFO] Ring dropped         : {ring.dropped} HW samples")
    print(f"[INFO] Ring occupancy       : {ring.occupancy:.3f}")

    # ------------------------------------------------------------------
    # The ring now contains physical samples 48..79.
    # ------------------------------------------------------------------
    survivors = ring.read(capacity)

    expected = np.arange(48, 80, dtype=np.float32).astype(np.complex64)

    assert np.array_equal(survivors, expected)

    print("[PASS] Ring contains the newest physical samples 48..79.")

    # ------------------------------------------------------------------
    # Coordinate comparison.
    #
    # Physical stream coordinate:
    #       80 HW samples have existed.
    #
    # Samples available for processing after overflow:
    #       32 HW samples.
    #
    # Difference:
    #       48 HW samples are no longer available to processing.
    #
    # The RingBuffer's cumulative dropped counter reports only the
    # samples discarded specifically by overflow, while samples already
    # consumed before the overflow are of course not "dropped".
    # ------------------------------------------------------------------
    physical_total_hw = 80
    processed_available_hw = len(survivors)

    consumed_before_overflow_hw = len(first_read)

    discarded_by_overflow_hw = ring.dropped

    assert (
        consumed_before_overflow_hw
        + discarded_by_overflow_hw
        + processed_available_hw
        == physical_total_hw
    )

    print()
    print("[INFO] Physical HW samples generated       : "
          f"{physical_total_hw}")
    print("[INFO] Already processed before overflow : "
          f"{consumed_before_overflow_hw}")
    print("[INFO] Samples discarded by ring          : "
          f"{discarded_by_overflow_hw}")
    print("[INFO] Samples remaining for processing  : "
          f"{processed_available_hw}")

    # ------------------------------------------------------------------
    # Convert the actual overflow discontinuity to time.
    # ------------------------------------------------------------------
    discontinuity_s = discarded_by_overflow_hw / FS_HW

    print()
    print("[INFO] FS_HW                         : "
          f"{FS_HW:.0f} samples/s")
    print("[INFO] FS_BB                         : "
          f"{FS_BB:.0f} samples/s")
    print("[INFO] BB/HW ratio                   : "
          f"{RATIO:.6f}")
    print("[INFO] Ring-overflow time discontinuity: "
          f"{discontinuity_s * 1e6:.3f} us")

    # ------------------------------------------------------------------
    # Important point:
    #
    # A simple multiplication by 0.8 gives a BB-equivalent duration,
    # but that does NOT repair the stateful resampler/detector state.
    # ------------------------------------------------------------------
    bb_equivalent = discarded_by_overflow_hw * RATIO

    print("[INFO] BB-equivalent dropped duration : "
          f"{bb_equivalent:.3f} samples")

    print()
    print("[PASS] Continuous physical stream can coexist with")
    print("       application-level ring overflow.")
    print("[PASS] Ring overflow is therefore distinct from UHD")
    print("       timestamp discontinuity.")

    print()
    print("=" * 72)
    print("FIX 4C-8 CHARACTERIZATION COMPLETE")
    print("=" * 72)
    print()
    print("No production code was changed.")
    print("A ring overflow must be handled as a processing")
    print("discontinuity, not merely as an abs_s arithmetic correction.")


if __name__ == "__main__":
    main()