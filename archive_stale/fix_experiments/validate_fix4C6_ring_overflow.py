import numpy as np

from rx_sim import RingBuffer


def main():
    print("=" * 72)
    print("FIX 4C-6 — RING BUFFER OVERFLOW INTEGRITY")
    print("=" * 72)

    capacity = 16
    ring = RingBuffer(capacity=capacity)

    # ---------------------------------------------------------------
    # Test 1: normal write/read
    # ---------------------------------------------------------------
    first = np.arange(8, dtype=np.float32).astype(np.complex64)

    ring.write(first)

    assert ring.occupancy == 8 / 16
    assert ring.dropped == 0

    out = ring.read(8)

    assert np.array_equal(out, first)
    assert ring.occupancy == 0.0
    assert ring.dropped == 0

    print("[PASS] Normal write/read preserves samples.")
    print("[PASS] No false drops during normal operation.")

    # ---------------------------------------------------------------
    # Test 2: deterministic overflow
    #
    # Fill 12 samples, then write 8 more.
    #
    # Existing free space = 4.
    # Therefore exactly 4 oldest samples must be discarded.
    # ---------------------------------------------------------------
    ring.write(np.arange(12, dtype=np.float32).astype(np.complex64))

    second = np.arange(100, 108, dtype=np.float32).astype(np.complex64)

    ring.write(second)

    expected_dropped = 4

    assert ring.dropped == expected_dropped, (
        f"Expected {expected_dropped} dropped samples, "
        f"got {ring.dropped}"
    )

    assert ring.occupancy == 1.0

    print("[PASS] Overflow drop count is exactly 4 samples.")
    print("[PASS] Occupancy is capped at 100%.")

    # ---------------------------------------------------------------
    # Test 3: verify exactly which samples survived
    #
    # Before second write:
    #   [0 ... 11]
    #
    # Free space:
    #   4
    #
    # Therefore oldest four:
    #   [0,1,2,3]
    #
    # are discarded.
    #
    # Remaining sequence:
    #   [4,5,...,11,100,...,107]
    # = 16 samples total.
    # ---------------------------------------------------------------
    expected = np.concatenate([
        np.arange(4, 12, dtype=np.float32),
        np.arange(100, 108, dtype=np.float32),
    ]).astype(np.complex64)

    actual = ring.read(16)

    assert np.array_equal(actual, expected), (
        "Ring overflow did not preserve the expected newest samples.\n"
        f"Expected: {expected}\n"
        f"Actual:   {actual}"
    )

    print("[PASS] Overflow preserves newest samples.")
    print("[PASS] Exactly the oldest 4 samples were discarded.")

    # ---------------------------------------------------------------
    # Test 4: post-overflow state
    # ---------------------------------------------------------------
    assert ring.occupancy == 0.0

    # dropped is cumulative by design
    assert ring.dropped == 4

    print("[PASS] Buffer is empty after complete read.")
    print("[PASS] Dropped counter remains cumulative.")

    print()
    print("=" * 72)
    print("FIX 4C-6 CHARACTERIZATION COMPLETE")
    print("=" * 72)
    print()
    print("IMPORTANT:")
    print("This test characterizes the CURRENT overflow behavior.")
    print("It does NOT claim that silent overwrite is acceptable")
    print("for hardware deployment.")


if __name__ == "__main__":
    main()