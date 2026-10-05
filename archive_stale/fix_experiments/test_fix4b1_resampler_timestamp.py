"""
test_fix4b1_resampler_timestamp.py

Fix 4B-1A:
Validate the absolute sample-coordinate convention of the persistent
25 MS/s -> 20 MS/s streaming resampler.

The streaming converter does not expose the finite-stream FIR tail until
flush(). Therefore this test compares:

    streaming process() + flush()

against the one-shot scipy reference.

IMPORTANT:
The internal pre_remove=7 is an implementation alignment quantity.
It is NOT a physical timestamp correction.
"""

import numpy as np
import waveform
import config as cfg


def run_streaming(x, chunks):
    """Process x through persistent streaming 25->20 converter."""
    rs = waveform.make_streaming_25to20()

    outputs = []
    pos = 0

    for n in chunks:
        if pos >= len(x):
            break

        end = min(pos + n, len(x))
        y = rs.process(x[pos:end])

        if len(y):
            outputs.append(y.copy())

        pos = end

    if pos < len(x):
        y = rs.process(x[pos:])
        if len(y):
            outputs.append(y.copy())

    # IMPORTANT:
    # Release the finite-stream FIR tail.
    y_tail = rs.flush()

    if len(y_tail):
        outputs.append(y_tail.copy())

    return (
        np.concatenate(outputs)
        if outputs
        else np.empty(0, dtype=np.complex64)
    )


def compare_case(input_index, n_input=4096):
    """
    Put one impulse at a known 25-MS/s input sample index and compare
    one-shot and streaming output.
    """

    x = np.zeros(n_input, dtype=np.complex64)
    x[input_index] = 1.0 + 0.0j

    # One-shot reference.
    y_ref = waveform.resample_25to20(x)

    # Streaming reference using deliberately irregular chunks.
    chunks = [1, 7, 31, 128, 160, 511, 1024, 4096]
    y_stream = run_streaming(x, chunks)

    print(
        f"input_index={input_index:4d} | "
        f"ref_len={len(y_ref):4d} | "
        f"stream_len={len(y_stream):4d}"
    )

    if len(y_ref) != len(y_stream):
        raise AssertionError(
            f"Output length mismatch: "
            f"reference={len(y_ref)}, streaming={len(y_stream)}"
        )

    err = np.max(np.abs(y_ref - y_stream))

    ref_peak = int(np.argmax(np.abs(y_ref)))
    stream_peak = int(np.argmax(np.abs(y_stream)))

    print(
        f"                 "
        f"ref_peak={ref_peak:4d} | "
        f"stream_peak={stream_peak:4d} | "
        f"max_error={err:.3e}"
    )

    if ref_peak != stream_peak:
        raise AssertionError(
            f"Peak index mismatch: "
            f"reference={ref_peak}, streaming={stream_peak}"
        )

    if err > 1e-5:
        raise AssertionError(
            f"Streaming/reference mismatch: {err:.3e}"
        )

    return ref_peak


def main():

    print("=" * 72)
    print("FIX 4B-1A — RESAMPLER TIMESTAMP COORDINATE TEST")
    print("=" * 72)

    print()
    print("Configuration:")
    print(f"  FS_HW  = {cfg.FS_HW:.0f} Hz")
    print(f"  FS_FFT = {cfg.FS_FFT:.0f} Hz")
    print(f"  ratio  = {cfg.RESAMP_INTERP}/{cfg.RESAMP_DECIM}")

    rs = waveform.make_streaming_25to20()

    print()
    print("Resampler internal parameters:")
    print(f"  FIR taps       = {len(rs.taps)}")
    print(f"  pre_pad        = {rs.pre_pad}")
    print(f"  pre_remove     = {rs.pre_remove}")
    print(f"  reported delay = {rs.latency_samples} output samples")

    assert len(rs.taps) == 64
    assert rs.pre_pad == 4
    assert rs.pre_remove == 7
    assert rs.latency_samples == 7

    print()
    print("Impulse-coordinate tests:")

    test_indices = [
        0,
        1,
        7,
        31,
        100,
        500,
        1000,
        2048,
    ]

    for idx in test_indices:
        compare_case(idx)

    print()
    print("Timestamp-coordinate sanity checks:")

    bb_period = 1.0 / cfg.FS_FFT

    print(
        f"  BB sample period = "
        f"{bb_period * 1e9:.3f} ns"
    )

    for n in [0, 1, 7, 997, 1000, 10000]:

        dt = n / cfg.FS_FFT

        print(
            f"  BB index {n:5d} -> "
            f"delta_t = {dt * 1e6:10.3f} us"
        )

    print()
    print("Coordinate rule:")
    print(
        "  t_BB(n) = t_UHD_0 + n / FS_FFT"
    )

    print()
    print("IMPORTANT:")
    print(
        "  pre_remove=7 is an internal resampler alignment quantity."
    )
    print(
        "  It is NOT added to or subtracted from the UHD timestamp."
    )

    print()
    print("RESULT: FIX 4B-1A PASS")


if __name__ == "__main__":
    main()