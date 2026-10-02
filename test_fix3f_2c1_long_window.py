import numpy as np
import waveform
import scrambler
import config as cfg
import detector
import sync
import demod


# ============================================================
# 3F-2C-1
# Same CFO + AWGN experiment as 3F-2C,
# but with a much longer receive window.
#
# PURPOSE:
# Determine whether "DATA symbol outside received buffer"
# is a real synchronization problem or merely a diagnostic
# window-length artifact.
#
# NO PRODUCTION FILES ARE MODIFIED.
# ============================================================

PAYLOAD_BYTES = 100
MODULATION = "BPSK"

PREFIX = 1000
SUFFIX = 12000

CFO_LIST_HZ = [
    -200_000,
    -100_000,
     -50_000,
          0,
      50_000,
     100_000,
     200_000,
]

SNR_LIST_DB = [
    30,
    20,
    15,
    10,
    5,
    0,
]

N_TRIALS = 5
BASE_SEED = 20261001

# IMPORTANT:
# The packet itself is 3328 samples.
# We now provide 12000 samples after the packet.
#
# This should be comfortably larger than any normal
# synchronization shift and therefore removes the previous
# 3840-sample diagnostic-window limitation.


def make_packet():
    payload_bits = (
        np.arange(PAYLOAD_BYTES * 8, dtype=np.uint8) & 1
    )

    packet = waveform.assemble_packet(
        payload_bits,
        modulation=MODULATION,
        scrambler_mod=scrambler,
        encoder_mod=scrambler,
        mapper_fn=scrambler.map_bits_to_symbols,
        idle_samples=0,
    )

    return payload_bits, packet.astype(np.complex64)


def apply_cfo(x, cfo_hz):
    n = np.arange(len(x), dtype=np.float64)

    phase = np.exp(
        1j * 2.0 * np.pi * cfo_hz * n / cfg.FS_FFT
    )

    return (
        x.astype(np.complex128) * phase
    ).astype(np.complex64)


def classify_case(
    noisy_stream,
    clean_packet,
    payload_bits,
    true_cfo,
):
    det = detector.PacketDetector()

    detections = det.process(noisy_stream)

    if len(detections) == 0:
        return {
            "status": "DETECTION_FAILURE",
            "det": None,
            "cfo_est": None,
            "crc": False,
            "bits": 0,
            "detail": "No detector output",
        }

    if len(detections) > 1:
        return {
            "status": "MULTIPLE_DETECTIONS",
            "det": detections[0][0],
            "cfo_est": detections[0][1],
            "crc": False,
            "bits": 0,
            "detail": f"{len(detections)} detections",
        }

    det_start, det_cfo = detections[0]

    # --------------------------------------------------------
    # IMPORTANT:
    # Start exactly at the detector output.
    # This is the same geometry used in validated 3F-2B.
    # --------------------------------------------------------

    start = int(det_start)

    # Give sync a very generous window.
    end = min(
        len(noisy_stream),
        start + len(clean_packet) + SUFFIX,
    )

    win = noisy_stream[start:end]

    # --------------------------------------------------------
    # Synchronization
    # --------------------------------------------------------

    try:
        res = sync.sync_packet(
            win,
            coarse_cfo_hz=det_cfo,
            n_data_symbols=0,
        )
    except Exception as exc:
        return {
            "status": "SYNC_EXCEPTION",
            "det": det_start,
            "cfo_est": det_cfo,
            "crc": False,
            "bits": 0,
            "detail": str(exc),
        }

    H_hat = res["H_hat"]
    ltf_start = int(res["ltf_start"])
    cfo_total = float(res["total_cfo"])

    # --------------------------------------------------------
    # Apply total CFO correction
    # --------------------------------------------------------

    try:
        rx_c = sync.apply_cfo_correction(
            win,
            cfo_total,
            start_n=0,
        )
    except Exception as exc:
        return {
            "status": "CFO_CORRECTION_EXCEPTION",
            "det": det_start,
            "cfo_est": det_cfo,
            "crc": False,
            "bits": 0,
            "detail": str(exc),
        }

    # --------------------------------------------------------
    # SIGNAL field
    # --------------------------------------------------------

    sig_body_start = (
        ltf_start
        + cfg.LTF_LEN
        + cfg.CP_LEN
    )

    sig_body_end = (
        sig_body_start
        + cfg.FFT_SIZE
    )

    if sig_body_end > len(rx_c):
        return {
            "status": "SIGNAL_WINDOW_FAILURE",
            "det": det_start,
            "cfo_est": det_cfo,
            "crc": False,
            "bits": 0,
            "detail": (
                f"SIGNAL end={sig_body_end}, "
                f"window={len(rx_c)}"
            ),
        }

    Y_sig = np.fft.fft(
        rx_c[
            sig_body_start:sig_body_end
        ],
        n=cfg.FFT_SIZE,
    ).astype(np.complex64)

    try:
        sig = demod.parse_signal_field(
            Y_sig,
            H_hat,
        )
    except Exception as exc:
        return {
            "status": "SIGNAL_EXCEPTION",
            "det": det_start,
            "cfo_est": det_cfo,
            "crc": False,
            "bits": 0,
            "detail": str(exc),
        }

    if not sig["valid"]:
        return {
            "status": "SIGNAL_PARSE_FAILURE",
            "det": det_start,
            "cfo_est": det_cfo,
            "crc": False,
            "bits": 0,
            "detail": "SIGNAL invalid",
        }

    n_bytes = int(sig["n_payload_bytes"])
    mod = sig["modulation"]
    n_sym = int(sig["n_data_syms"])

    # --------------------------------------------------------
    # DATA symbols
    # --------------------------------------------------------

    data_start = (
        ltf_start
        + cfg.LTF_LEN
        + cfg.SIG_LEN
    )

    ffts = []

    for m in range(n_sym):

        s = (
            data_start
            + m * cfg.SYMBOL_LEN
            + cfg.CP_LEN
        )

        e = s + cfg.FFT_SIZE

        if e > len(rx_c):
            return {
                "status": "DATA_WINDOW_FAILURE",
                "det": det_start,
                "cfo_est": det_cfo,
                "crc": False,
                "bits": 0,
                "detail": (
                    f"DATA symbol {m}: "
                    f"end={e}, "
                    f"window={len(rx_c)}, "
                    f"ltf_start={ltf_start}, "
                    f"n_sym={n_sym}"
                ),
            }

        ffts.append(
            np.fft.fft(
                rx_c[s:e],
                n=cfg.FFT_SIZE,
            ).astype(np.complex64)
        )

    # --------------------------------------------------------
    # DATA demodulation
    # --------------------------------------------------------

    try:
        rx_bits, crc_ok = demod.demodulate_packet(
            ffts,
            H_hat,
            modulation=mod,
            n_payload_bytes=n_bytes,
        )
    except Exception as exc:
        return {
            "status": "DEMOD_EXCEPTION",
            "det": det_start,
            "cfo_est": det_cfo,
            "crc": False,
            "bits": 0,
            "detail": str(exc),
        }

    rx_bits = np.asarray(rx_bits).astype(np.uint8)

    expected_bits = payload_bits

    payload_match = (
        len(rx_bits) == len(expected_bits)
        and np.array_equal(
            rx_bits,
            expected_bits,
        )
    )

    if bool(crc_ok) and payload_match:
        status = "PASS"
    elif not bool(crc_ok):
        status = "CRC_FAILURE"
    else:
        status = "PAYLOAD_MISMATCH"

    return {
        "status": status,
        "det": det_start,
        "cfo_est": det_cfo,
        "crc": bool(crc_ok),
        "bits": len(rx_bits),
        "detail": (
            f"mod={mod}, "
            f"bytes={n_bytes}, "
            f"symbols={n_sym}, "
            f"ltf_start={ltf_start}, "
            f"payload_match={payload_match}"
        ),
    }


# ============================================================
# Main experiment
# ============================================================

payload_bits, packet = make_packet()

signal_power = np.mean(
    np.abs(
        packet.astype(np.complex128)
    ) ** 2
)

clean_stream = np.concatenate(
    [
        np.zeros(
            PREFIX,
            dtype=np.complex64,
        ),
        packet,
        np.zeros(
            SUFFIX,
            dtype=np.complex64,
        ),
    ]
)

print("=" * 100)
print("3F-2C-1 LONG-WINDOW CFO + AWGN DIAGNOSTIC")
print("=" * 100)

print()
print("PACKET")
print("------")
print(f"Payload bytes:        {PAYLOAD_BYTES}")
print(f"Modulation:           {MODULATION}")
print(f"Packet length:        {len(packet)}")
print(f"Prefix:               {PREFIX}")
print(f"Suffix:               {SUFFIX}")
print(f"True packet start:    {PREFIX}")
print(f"FS_FFT:               {cfg.FS_FFT}")
print(f"Packet mean power:    {signal_power:.12f}")

print()
print("CFO sweep:")
for x in CFO_LIST_HZ:
    print(f"{x:>12} Hz")

print()
print("SNR sweep:")
for x in SNR_LIST_DB:
    print(f"{x:>12.1f} dB")

print()
print(f"Trials per condition: {N_TRIALS}")
print(f"Diagnostic suffix:    {SUFFIX}")
print("=" * 100)


results = []

for snr_db in SNR_LIST_DB:

    snr_linear = 10.0 ** (
        snr_db / 10.0
    )

    noise_power = (
        signal_power / snr_linear
    )

    sigma = np.sqrt(
        noise_power / 2.0
    )

    print()
    print("=" * 100)
    print(f"SNR = {snr_db:.1f} dB")
    print(
        f"Noise power = "
        f"{noise_power:.12e}"
    )
    print(
        f"Sigma       = "
        f"{sigma:.12e}"
    )
    print("=" * 100)

    for cfo_hz in CFO_LIST_HZ:

        passed = 0

        print()
        print(
            f"CFO = {cfo_hz:+d} Hz"
        )
        print("-" * 100)
        print(
            f"{'Trial':>7}"
            f"{'Det':>8}"
            f"{'CFO est':>15}"
            f"{'Status':>24}"
            f"{'CRC':>8}"
            f"{'Bits':>8}"
            f"  Detail"
        )

        for trial in range(N_TRIALS):

            seed = (
                BASE_SEED
                + int(round(snr_db * 100))
                + int(round(cfo_hz))
                + trial
            )

            rng = np.random.default_rng(
                seed
            )

            cfo_stream = apply_cfo(
                clean_stream,
                cfo_hz,
            )

            noise = (
                rng.standard_normal(
                    len(clean_stream)
                )
                + 1j
                * rng.standard_normal(
                    len(clean_stream)
                )
            ) * sigma

            noisy_stream = (
                cfo_stream.astype(
                    np.complex128
                )
                + noise
            ).astype(np.complex64)

            out = classify_case(
                noisy_stream,
                packet,
                payload_bits,
                cfo_hz,
            )

            if out["status"] == "PASS":
                passed += 1

            results.append(
                {
                    "snr": snr_db,
                    "cfo": cfo_hz,
                    "trial": trial,
                    **out,
                }
            )

            det_txt = (
                "-"
                if out["det"] is None
                else str(out["det"])
            )

            cfo_txt = (
                "-"
                if out["cfo_est"] is None
                else f"{out['cfo_est']:.2f}"
            )

            print(
                f"{trial:7d}"
                f"{det_txt:>8}"
                f"{cfo_txt:>15}"
                f"{out['status']:>24}"
                f"{str(out['crc']):>8}"
                f"{out['bits']:8d}"
                f"  {out['detail']}"
            )

        print(
            f"Condition result: "
            f"{passed}/{N_TRIALS} passed"
        )


# ============================================================
# Summary
# ============================================================

print()
print("=" * 110)
print("3F-2C-1 SUMMARY")
print("=" * 110)

print()
print(
    f"{'SNR':>7}"
    f"{'CFO':>12}"
    f"{'Pass':>10}"
    f"{'Total':>10}"
    f"{'Rate':>10}"
)

print("-" * 110)

for snr_db in SNR_LIST_DB:
    for cfo_hz in CFO_LIST_HZ:

        subset = [
            r
            for r in results
            if r["snr"] == snr_db
            and r["cfo"] == cfo_hz
        ]

        p = sum(
            r["status"] == "PASS"
            for r in subset
        )

        total = len(subset)

        print(
            f"{snr_db:7.0f}"
            f"{cfo_hz:12d}"
            f"{p:10d}"
            f"{total:10d}"
            f"{100.0 * p / total:9.1f}%"
        )


print()
print("=" * 110)
print("FAILURE CLASSIFICATION")
print("=" * 110)

from collections import Counter

counts = Counter(
    r["status"]
    for r in results
)

for status, count in counts.items():
    print(
        f"{status:30s}: {count}"
    )


print()
print("=" * 110)

total_pass = sum(
    r["status"] == "PASS"
    for r in results
)

total_cases = len(results)

print(
    f"Overall: "
    f"{total_pass}/{total_cases} "
    f"passed "
    f"({100.0 * total_pass / total_cases:.1f}%)"
)

print()
print("3F-2C-1 COMPLETE")
print("NO PRODUCTION FILES WERE MODIFIED.")
print("=" * 110)