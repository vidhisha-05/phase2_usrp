"""
validate_fix2_detector_index.py

Fix 2 diagnostic:
Verify PacketDetector absolute indexing when fed realistic
4096-sample UHD blocks.

This script DOES NOT modify project files.
"""

import numpy as np

import config as cfg
import waveform
import scrambler as scrambler_mod
from detector import PacketDetector


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

HW_BLOCK = 4096

# 25 MS/s hardware -> 20 MS/s baseband
EXPECTED_BB = HW_BLOCK * cfg.FS_FFT / cfg.FS_HW


def make_packet(payload_bytes=64, seed=123):
    rng = np.random.default_rng(seed)

    payload_bits = rng.integers(
        0, 2,
        payload_bytes * 8,
        dtype=np.uint8
    )

    pkt_bb = waveform.assemble_packet(
        payload_bits,
        modulation="BPSK",
        scrambler_mod=scrambler_mod,
        encoder_mod=scrambler_mod,
        mapper_fn=scrambler_mod.map_bits_to_symbols,
        idle_samples=0
    )

    return pkt_bb.astype(np.complex64)


def print_detections(name, detections):
    print(f"\n{name}")
    print("-" * 60)

    if not detections:
        print("NO DETECTIONS")
        return

    for i, (idx, cfo) in enumerate(detections):
        print(
            f"  #{i}: abs_idx={int(idx):8d}   CFO={float(cfo):+10.2f} Hz"
        )


def run_one_shot(stream):
    det = PacketDetector()
    return det.process(stream)


def run_chunked(stream, chunk_size):
    det = PacketDetector()

    detections = []

    pos = 0

    while pos < len(stream):
        chunk = stream[pos:pos + chunk_size]

        print(
            f"[chunked] feeding samples "
            f"{pos} .. {pos + len(chunk) - 1} "
            f"(n={len(chunk)})"
        )

        out = det.process(chunk)

        if out:
            detections.extend(out)

        pos += len(chunk)

    print()
    print("[chunked] detector final state:")
    print(f"    _sample_idx = {det._sample_idx}")
    print(f"    len(_buf)   = {len(det._buf)}")
    print(f"    received    = {len(stream)}")

    return detections, det


def main():

    print("=" * 72)
    print("FIX 2 — DETECTOR ABSOLUTE INDEX DIAGNOSTIC")
    print("=" * 72)

    print()
    print("Configuration:")
    print(f"  FS_HW       = {cfg.FS_HW}")
    print(f"  FS_FFT      = {cfg.FS_FFT}")
    print(f"  HW block    = {HW_BLOCK}")
    print(f"  expected BB = {EXPECTED_BB:.3f}")

    # -------------------------------------------------------------
    # Create several packets separated by known silence.
    # -------------------------------------------------------------

    pkt1 = make_packet(64, seed=1)
    pkt2 = make_packet(64, seed=2)
    pkt3 = make_packet(64, seed=3)

    gap = 5000

    stream = np.concatenate([
        np.zeros(1000, dtype=np.complex64),
        pkt1,
        np.zeros(gap, dtype=np.complex64),
        pkt2,
        np.zeros(gap, dtype=np.complex64),
        pkt3,
        np.zeros(1000, dtype=np.complex64),
    ])

    print()
    print("Stream:")
    print(f"  total samples = {len(stream)}")
    print(f"  packet 1 nominal start = {1000}")
    print(f"  packet 2 nominal start = {1000 + len(pkt1) + gap}")
    print(f"  packet 3 nominal start = {1000 + len(pkt1) + gap + len(pkt2) + gap}")
    print(f"  packet length = {len(pkt1)}")

    # -------------------------------------------------------------
    # Test A: one-shot
    # -------------------------------------------------------------

    one_shot = run_one_shot(stream)

    print_detections(
        "TEST A — ONE-SHOT DETECTOR",
        one_shot
    )

    # -------------------------------------------------------------
    # Test B: realistic UHD block size
    # -------------------------------------------------------------

    chunked, det = run_chunked(
        stream,
        HW_BLOCK
    )

    print_detections(
        "TEST B — 4096 HW SAMPLE CHUNKS",
        chunked
    )

    # -------------------------------------------------------------
    # Compare detections
    # -------------------------------------------------------------

    print()
    print("=" * 72)
    print("COMPARISON")
    print("=" * 72)

    print(f"One-shot detections : {len(one_shot)}")
    print(f"Chunked detections  : {len(chunked)}")

    if len(one_shot) != len(chunked):
        print()
        print("FAIL: detection count differs.")
    else:
        print()
        print("Detection count matches.")

        for i, ((a, ca), (b, cb)) in enumerate(zip(one_shot, chunked)):

            idx_error = int(b) - int(a)
            cfo_error = float(cb) - float(ca)

            print(
                f"packet {i}: "
                f"one-shot={int(a)}, "
                f"chunked={int(b)}, "
                f"index_error={idx_error}, "
                f"CFO_error={cfo_error:+.2f} Hz"
            )

            if idx_error != 0:
                print(
                    "  WARNING: absolute detector index changed "
                    "with chunking."
                )

    # -------------------------------------------------------------
    # Critical internal consistency check
    # -------------------------------------------------------------

    print()
    print("=" * 72)
    print("INTERNAL INDEX CONSISTENCY")
    print("=" * 72)

    received = len(stream)

    internal_end = det._sample_idx + len(det._buf)

    print(f"Received stream samples       = {received}")
    print(f"Detector _sample_idx         = {det._sample_idx}")
    print(f"Detector remaining _buf      = {len(det._buf)}")
    print(f"_sample_idx + len(_buf)      = {internal_end}")
    print(f"Difference from received     = {internal_end - received}")

    if internal_end == received:
        print()
        print("PASS: detector coordinate system is consistent.")
    else:
        print()
        print("FAIL: detector coordinate system has drifted.")
        print()
        print(
            "This is the important Fix 2 result. "
            "Do NOT patch main_hardware.py yet."
        )


if __name__ == "__main__":
    main()