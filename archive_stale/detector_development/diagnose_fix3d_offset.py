import numpy as np
import config as cfg
import waveform


# ------------------------------------------------------------
# Known geometry
# ------------------------------------------------------------
PREFIX = 1000
STF = waveform.generate_stf()

# Put enough zeros BEFORE the STF and enough samples AFTER it
POST = 500

stream = np.concatenate([
    np.zeros(PREFIX, dtype=np.complex64),
    STF.astype(np.complex64),
    np.zeros(POST, dtype=np.complex64),
])

# ------------------------------------------------------------
# Same correlation geometry as detector.py
# ------------------------------------------------------------
L = 16

# Need enough samples around the STF for the correlation windows
buf = stream

N = len(buf) - L
if N <= 0:
    raise RuntimeError("Diagnostic buffer is too short.")

A = np.lib.stride_tricks.sliding_window_view(buf, L)
B = np.lib.stride_tricks.sliding_window_view(buf[L:], L)

# Make both arrays the same number of rows
Ncorr = min(len(A), len(B))
A = A[:Ncorr]
B = B[:Ncorr]

P = np.sum(A * np.conj(B), axis=1)
R = np.sum(np.abs(B) ** 2, axis=1)

metric = np.abs(P) ** 2 / (R ** 2 + 1e-12)

threshold = 0.65

# ------------------------------------------------------------
# Global threshold crossings
# ------------------------------------------------------------
crossings = np.where(metric >= threshold)[0]

print("Known STF insertion index:", PREFIX)

if len(crossings):
    first = int(crossings[0])
    print("First threshold crossing:", first)
    print("Offset:", first - PREFIX)
else:
    print("First threshold crossing: None")
    print("Offset: None")

# ------------------------------------------------------------
# Print metric around expected STF start
# ------------------------------------------------------------
print()
print("METRIC AROUND PACKET START")
print("--------------------------")

start = max(0, PREFIX - 20)
end = min(len(metric), PREFIX + 40)

for i in range(start, end):
    print(f"{i:5d}: {metric[i]:.9f}")

# ------------------------------------------------------------
# Threshold crossings near packet start
# ------------------------------------------------------------
print()
print("THRESHOLD CROSSINGS NEAR PACKET")
print("--------------------------------")

near = crossings[
    (crossings >= PREFIX - 100) &
    (crossings <= PREFIX + 200)
]

print(near.tolist())

# ------------------------------------------------------------
# Local maximum around packet start
# ------------------------------------------------------------
local_start = max(0, PREFIX - 100)
local_end = min(len(metric), PREFIX + 200)

local_metric = metric[local_start:local_end]

if len(local_metric) > 0:
    local_arg = int(np.argmax(local_metric))
    local_idx = local_start + local_arg

    print()
    print("LOCAL MAXIMUM")
    print("-------------")
    print("Index:", local_idx)
    print("Metric:", float(metric[local_idx]))
    print("Offset from known STF start:", local_idx - PREFIX)

# ------------------------------------------------------------
# Detector-style first threshold crossing
# ------------------------------------------------------------
print()
print("DETECTOR-STYLE RESULT")
print("---------------------")

if len(near):
    detector_style = int(near[0])
    print("First crossing near packet:", detector_style)
    print("Offset:", detector_style - PREFIX)
else:
    print("No threshold crossing near packet.")