import numpy as np

import config as cfg
import waveform
from detector import PacketDetector
from sync import sync_packet


# ============================================================
# 1. Generate one real packet using the CURRENT waveform API
# ============================================================

PAYLOAD_BYTES = 100
MODULATION = "BPSK"
PREFIX = 1000

# Deterministic 800-bit payload
payload_bits = (
    np.arange(PAYLOAD_BYTES * 8, dtype=np.uint8) & 1
)

import scrambler

pkt = waveform.assemble_packet(
    payload_bits,
    modulation=MODULATION,
    scrambler_mod=scrambler,
    encoder_mod=scrambler,
    mapper_fn=scrambler.map_bits_to_symbols,
    idle_samples=0,
)

pkt = np.asarray(pkt, dtype=np.complex64)

print("PACKET")
print("------")
print("Payload bytes:", PAYLOAD_BYTES)
print("Modulation:", MODULATION)
print("Packet length:", len(pkt))
print("STF length:", cfg.STF_LEN)
print("LTF length:", cfg.LTF_LEN)
print("SIGNAL length:", cfg.SIG_LEN)


# ============================================================
# 2. Put packet at a known absolute position
# ============================================================

stream = np.concatenate([
    np.zeros(PREFIX, dtype=np.complex64),
    pkt,
    np.zeros(6000, dtype=np.complex64),
])

TRUE_START = PREFIX

print()
print("TRUE PACKET GEOMETRY")
print("--------------------")
print("True packet start:", TRUE_START)


# ============================================================
# 3. Run packet detector
# ============================================================

det = PacketDetector()

dets = det.process(stream)

print()
print("DETECTOR")
print("--------")
print("Detections:", dets)

if len(dets) != 1:
    raise AssertionError(
        f"Expected exactly 1 detection, got {len(dets)}"
    )

DET_START, DET_CFO = dets[0]

print("Detected start:", DET_START)
print("Detection offset:", DET_START - TRUE_START)


# ============================================================
# 4. Construct the same maximum packet window used by hardware
# ============================================================

HEADROOM = (
    cfg.STF_LEN
    + cfg.LTF_LEN
    + cfg.SIG_LEN
    + cfg.MAX_DATA_SYMS * cfg.SYMBOL_LEN
)

if DET_START < 0:
    raise AssertionError("Detector returned a negative start index.")

if DET_START + HEADROOM > len(stream):
    raise AssertionError(
        "Not enough samples after detected start for HEADROOM."
    )

win = stream[
    DET_START:
    DET_START + HEADROOM
].astype(np.complex64)

print()
print("EXTRACTED WINDOW")
print("----------------")
print("Window length:", len(win))


# ============================================================
# 5. Determine true packet position inside extracted window
# ============================================================

TRUE_START_IN_WINDOW = TRUE_START - DET_START

print("True packet start inside window:", TRUE_START_IN_WINDOW)

if TRUE_START_IN_WINDOW < 0:
    raise AssertionError(
        "Detected start is after the true packet start."
    )


# ============================================================
# 6. Run the ACTUAL synchronizer
# ============================================================

res = sync_packet(
    win,
    coarse_cfo_hz=DET_CFO,
    n_data_symbols=0,
)

ltf_start = int(res["ltf_start"])

print()
print("SYNC RESULT")
print("-----------")
print("Returned ltf_start:", ltf_start)


# ============================================================
# 7. Calculate the TRUE LTF CP/body positions
# ============================================================

TRUE_LTF_CP_START = (
    TRUE_START_IN_WINDOW
    + cfg.STF_LEN
)

TRUE_LTF_BODY_START = (
    TRUE_LTF_CP_START
    + cfg.CP_LEN
)

print()
print("TRUE LTF GEOMETRY")
print("-----------------")
print("True LTF CP start:", TRUE_LTF_CP_START)
print("True LTF body start:", TRUE_LTF_BODY_START)


# ============================================================
# 8. Compare synchronizer result with true LTF CP boundary
# ============================================================

ltf_error = ltf_start - TRUE_LTF_CP_START

print()
print("TIMING ERRORS")
print("-------------")
print(
    "ltf_start - true LTF CP start =",
    ltf_error,
    "samples"
)


# ============================================================
# 9. Report result
# ============================================================

print()
print("RESULT")
print("------")

if abs(ltf_error) <= 4:
    print("3D downstream timing check: PASS")
    print(
        "Detector reference is early by",
        TRUE_START - DET_START,
        "samples."
    )
    print(
        "sync_packet() recovered the LTF CP boundary within",
        abs(ltf_error),
        "samples."
    )
else:
    print("3D downstream timing check: FAIL")
    print(
        "LTF timing error is",
        ltf_error,
        "samples."
    )

    raise AssertionError(
        f"LTF timing error too large: {ltf_error} samples"
    )