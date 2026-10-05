"""
validate_realtime_csi.py

Measures the current software CSI path separately from full payload decoding.

This is NOT a production-code change.
It answers whether the detector -> synchronization -> LTF -> CSI path
can plausibly support the intended 200-Hz CSI stream.

200 Hz => 5 ms/packet budget.

Measurements:
1. Packet generation/channel construction
2. Detector
3. sync_packet() / LTF timing / CFO / CSI
4. Full payload decode including Viterbi
"""

import time
import numpy as np

import config as cfg
import waveform
import scrambler
import demod as demod_mod

from detector import PacketDetector
from sync import sync_packet, apply_cfo_correction, sco_correct_symbol
from channel_bridge import ChannelModel


N_WARMUP = 10
N_TRIALS = 100
PAYLOAD_BYTES = 100
MODULATION = "BPSK"
NOISE = 0.005


def make_packet(seed: int):
    rng = np.random.default_rng(seed)
    bits = rng.integers(
        0, 2,
        PAYLOAD_BYTES * 8,
        dtype=np.uint8
    )

    pkt = waveform.assemble_packet(
        bits,
        MODULATION,
        scrambler,
        scrambler,
        scrambler.map_bits_to_symbols
    )

    return bits, pkt


def make_rx(seed: int):
    bits, pkt = make_packet(seed)

    ch = ChannelModel(
        noise_voltage=NOISE,
        taps=[1 + 0j],
        cfo_hz=0.0,
        sco_ppm=0.0,
        seed=seed
    )

    rx = ch.apply(pkt.copy()).astype(np.complex64)

    return bits, rx


def measure_detector(rx):
    det = PacketDetector()

    t0 = time.perf_counter()
    dets = det.process(rx)
    elapsed = time.perf_counter() - t0

    return elapsed, dets


def measure_csi(rx, det_cfo):
    """
    Measure the actual sensing path:

        synchronization
        -> CFO correction
        -> LTF CSI

    No payload demodulation / Viterbi here.
    """

    t0 = time.perf_counter()

    res = sync_packet(
        rx,
        coarse_cfo_hz=float(det_cfo),
        n_data_symbols=0
    )

    H_hat = res["H_hat"]

    if np.any(np.isnan(H_hat)):
        raise RuntimeError("CSI contains NaN")

    # These are part of the actual RX CSI path.
    _ = apply_cfo_correction(
        rx,
        res["total_cfo"],
        start_n=0
    )

    elapsed = time.perf_counter() - t0

    return elapsed, H_hat


def measure_full_decode(rx, det_cfo):
    """
    Full communication path:

        sync
        -> CFO correction
        -> SIGNAL
        -> DATA FFTs
        -> demodulation
        -> Viterbi
        -> CRC
    """

    t0 = time.perf_counter()

    res = sync_packet(
        rx,
        coarse_cfo_hz=float(det_cfo),
        n_data_symbols=0
    )

    H_hat = res["H_hat"]

    if np.any(np.isnan(H_hat)):
        return time.perf_counter() - t0, False

    lt = res["ltf_start"]
    rx_c = apply_cfo_correction(
        rx,
        res["total_cfo"],
        start_n=0
    )

    # -------------------------
    # SIGNAL
    # -------------------------

    sig_body_start = lt + cfg.LTF_LEN + cfg.CP_LEN
    sig_body_end = sig_body_start + cfg.FFT_SIZE

    Y_sig = np.fft.fft(
        rx_c[sig_body_start:sig_body_end],
        n=cfg.FFT_SIZE
    ).astype(np.complex64)

    sig = demod_mod.parse_signal_field(
        Y_sig,
        H_hat
    )

    if not sig["valid"]:
        return time.perf_counter() - t0, False

    n_bytes = int(sig["n_payload_bytes"])
    modulation = sig["modulation"]
    n_sym = int(sig["n_data_syms"])

    # -------------------------
    # DATA
    # -------------------------

    ds = lt + cfg.LTF_LEN + cfg.SIG_LEN

    ffts = []
    b = 0.0

    for m in range(n_sym):

        s = ds + m * cfg.SYMBOL_LEN + cfg.CP_LEN
        e = s + cfg.FFT_SIZE

        if e > len(rx_c):
            return time.perf_counter() - t0, False

        Y = np.fft.fft(
            rx_c[s:e],
            n=cfg.FFT_SIZE
        ).astype(np.complex64)

        Y, b, _ = sco_correct_symbol(
            Y,
            H_hat,
            symbol_idx=m,
            sco_b_accum=b
        )

        ffts.append(Y)

    if not ffts:
        return time.perf_counter() - t0, False

    _, crc_ok = demod_mod.demodulate_packet(
        ffts,
        H_hat,
        modulation=modulation,
        n_payload_bytes=n_bytes
    )

    elapsed = time.perf_counter() - t0

    return elapsed, bool(crc_ok)


def percentile_ms(values, p):
    return 1000.0 * float(np.percentile(values, p))


def summarize(name, values):
    values = np.asarray(values, dtype=np.float64)

    print(f"\n{name}")
    print("-" * 60)
    print(f"count       : {len(values)}")
    print(f"mean        : {1000*np.mean(values):.4f} ms")
    print(f"median      : {percentile_ms(values, 50):.4f} ms")
    print(f"p95         : {percentile_ms(values, 95):.4f} ms")
    print(f"p99         : {percentile_ms(values, 99):.4f} ms")
    print(f"max         : {1000*np.max(values):.4f} ms")


def main():

    print("=" * 72)
    print("REAL-TIME CSI PATH BENCHMARK")
    print("=" * 72)

    print(f"FFT size           : {cfg.FFT_SIZE}")
    print(f"BB sample rate     : {cfg.FS_FFT/1e6:.1f} MS/s")
    print(f"Target CSI rate    : {cfg.TX_PACKET_RATE_HZ:.1f} Hz")
    print(f"Packet period      : {1000*cfg.TX_PACKET_PERIOD_S:.3f} ms")
    print(f"CSI time budget    : {1000*cfg.TX_PACKET_PERIOD_S:.3f} ms")
    print(f"Trials             : {N_TRIALS}")
    print(f"Payload            : {PAYLOAD_BYTES} B {MODULATION}")

    # ---------------------------------------------------------
    # Warm-up
    # ---------------------------------------------------------

    print("\nWarming up ...")

    for i in range(N_WARMUP):

        _, rx = make_rx(10000 + i)

        elapsed, dets = measure_detector(rx)

        if not dets:
            raise RuntimeError(
                f"Warm-up detector failed at trial {i}"
            )

        _, det_cfo = dets[0]

        measure_csi(rx, det_cfo)
        measure_full_decode(rx, det_cfo)

    print("Warm-up complete.")

    # ---------------------------------------------------------
    # Measurements
    # ---------------------------------------------------------

    detector_times = []
    csi_times = []
    full_decode_times = []

    crc_ok = 0
    detected = 0

    for i in range(N_TRIALS):

        _, rx = make_rx(20000 + i)

        # Detector
        td, dets = measure_detector(rx)
        detector_times.append(td)

        if not dets:
            continue

        detected += 1

        det_cfo = dets[0][1]

        # CSI path
        tcsi, _ = measure_csi(
            rx,
            det_cfo
        )

        csi_times.append(tcsi)

        # Full decode
        tfull, ok = measure_full_decode(
            rx,
            det_cfo
        )

        full_decode_times.append(tfull)

        if ok:
            crc_ok += 1

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    summarize(
        "1. Packet detector",
        detector_times
    )

    summarize(
        "2. CSI path: sync + CFO + LTF CSI",
        csi_times
    )

    summarize(
        "3. Full communication decode",
        full_decode_times
    )

    budget = cfg.TX_PACKET_PERIOD_S

    print("\n" + "=" * 72)
    print("REAL-TIME DECISION")
    print("=" * 72)

    if csi_times:

        csi_p95 = float(np.percentile(csi_times, 95))
        csi_p99 = float(np.percentile(csi_times, 99))

        print(
            f"CSI p95 : {1000*csi_p95:.4f} ms "
            f"vs budget {1000*budget:.4f} ms"
        )

        print(
            f"CSI p99 : {1000*csi_p99:.4f} ms "
            f"vs budget {1000*budget:.4f} ms"
        )

        if csi_p95 < budget:
            print(
                "[PASS] CSI p95 is below the 5-ms target."
            )
        else:
            print(
                "[FAIL] CSI p95 exceeds the 5-ms target."
            )

    if full_decode_times:

        full_p95 = float(np.percentile(full_decode_times, 95))

        print(
            f"FULL decode p95 : {1000*full_p95:.4f} ms "
            f"vs budget {1000*budget:.4f} ms"
        )

        if full_p95 < budget:
            print(
                "[PASS] Full decode p95 is below the 5-ms target."
            )
        else:
            print(
                "[INFO] Full decode p95 exceeds the 5-ms target."
            )
            print(
                "       This does NOT automatically invalidate live CSI."
            )

    print("\nDetection rate:")
    print(
        f"  {detected}/{N_TRIALS} "
        f"({100*detected/N_TRIALS:.1f}%)"
    )

    print("CRC rate among detected packets:")
    print(
        f"  {crc_ok}/{detected if detected else 1} "
        f"({100*crc_ok/max(1, detected):.1f}%)"
    )

    print("\nNo production files were modified.")
    print("This benchmark only measures the current implementation.")


if __name__ == "__main__":
    main()