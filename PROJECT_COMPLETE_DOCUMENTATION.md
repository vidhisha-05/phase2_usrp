# Phase 2 — Custom 128-pt OFDM PHY for Human Activity Recognition (HAR)
## Complete Project Documentation — Full Technical Reference

> **Last Audited**: 2026-09-30 — 25 findings; 4 blockers + 6 high fixed. Full report in FINDINGS.md.
> **Hardware Target**: USRP B210 · 1-TX / 1-RX · USB 3.0
> **Simulation**: 40/40 PASS (honest thresholds) · 4/4 Stage validators PASS
> **Pre-B210 Suite**: 16/16 PASS (T1–T10, T_CHK, T_NOI, T_META, T_RST, T_REC)
> **HAR Stress Test**: 3000/3000 pkts detected · CRC=100% · BER=0
> **Hardware Status**: PENDING — F-03 two-process B210 issue must be resolved before first run
> **v7 Changes**: LO offset 5→8.75 MHz, CRC gate in logger, CFO gate 200→50 kHz, ring occupancy fix

---

## Part 1 — Project Idea and Research Motivation

### 1.1 What Is This Project?

This project implements a **purpose-built, custom 128-point OFDM Physical Layer (PHY)** for
**Channel State Information (CSI)-based Human Activity Recognition (HAR)**.

The core insight: when a person moves between a transmitter (TX) and receiver (RX), their
body scatters, absorbs, and reflects radio waves. These interactions create measurable
perturbations in the **amplitude and phase of every OFDM subcarrier** — forming a rich,
high-dimensional signature unique to each activity type.

By transmitting packets continuously and extracting the complex channel vector H_hat[k] from
each received packet, we build a time-series of CSI snapshots that encode:
- **Coarse body motion** (walking, falling): large amplitude changes across all subcarriers
- **Breathing / respiration**: periodic 0.3-0.5 Hz modulation on CSI magnitude
- **Micro-gestures** (keystrokes, finger taps): 1.5-5 Hz transient phase perturbations
- **Tremor / micro-vibration**: 5-8 Hz periodic signature, small amplitude but measurable
- **Shadow fading / walk-by**: abrupt multipath shifts + instantaneous Doppler spread

### 1.2 Why Custom PHY Instead of Off-the-Shelf Wi-Fi?

| Aspect | Commercial 802.11a/g | Our Custom PHY |
|---|---|---|
| FFT size | 64-pt | 128-pt (2x finer resolution) |
| Subcarrier resolution | 312.5 kHz | 156.25 kHz |
| Active subcarriers | 52 | 106 (2x more CSI values per packet) |
| CSI API access | None (firmware locked) | Full complex H_hat[k] per packet |
| Phase sanitization | Unavailable | Built-in OLS detrending |
| Packet interval | >1 ms (CSMA) | Programmable 5-20 ms burst |
| Scientific reproducibility | Zero | Fully open-source and verifiable |

### 1.3 HAR Methodology

**Sensing mechanism (Fresnel Zone model):**
A person crossing even 1% of a Fresnel zone radius changes the path length by lambda/2 ~= 6.2 cm
at 2.4 GHz, producing a phase shift of pi radians on affected subcarriers. Micro-gestures
(5 mm displacement) at specific scattering geometries produce phase changes of ~0.16 rad
detectable after phase sanitization removes the linear timing-jitter slope.

**CSI feature extraction pipeline:**
```
Raw H_hat[k]  ->  OLS phase detrend  ->  Sanitized phase[k]
                                                |
                                        Welch PSD (0.5-10 Hz)
                                                |
                                        Doppler spectrogram
                                                |
                                        Activity classifier (DNN/SVM/CNN)
```

**Why rate-1/2 Viterbi FEC?**
The Viterbi decoder is used for DATA INTEGRITY (payload CRC gate). A CRC pass guarantees
H_hat was extracted from a cleanly-detected, properly-synchronized packet. CRC failures are
silently discarded — ensuring CSI is never contaminated by false detections or timing errors.

---

## Part 2 — PHY Specification (All Values Cross-Verified vs config.py)

### 2.1 Complete Numerology Table (AUTOMATED AUDIT: ALL OK 2026-09-29)

| Parameter | Symbol | Value | config.py | Arithmetic Check |
|---|---|---|---|---|
| FFT Size | N_FFT | 128 | FFT_SIZE=128 | Custom, not 802.11 |
| Cyclic Prefix | CP | 32 samples | CP_LEN=32 | N_FFT/4=32 OK |
| Symbol Length | T_sym | 160 samples | SYMBOL_LEN=160 | 128+32=160 OK |
| Symbol Duration | | 8.0 us | | 160/20e6=8us OK |
| BB Sample Rate | FS_FFT | 20.0 MS/s | FS_FFT=20e6 | OK |
| HW Sample Rate | FS_HW | 25.0 MS/s | FS_HW=25e6 | OK |
| Resample BB->HW | 5/4 | up/down | | 20*5/4=25 OK |
| Subcarrier Spacing | Delta_f | 156.25 kHz | | 20e6/128=156.25k OK |
| Active SCs | N_active | 106 | NUM_ACTIVE=106 | k in {-53..-1}+{1..53}=106 OK |
| DC Subcarrier | k=0 | null | DC_NULL=[0] | OK |
| Guard (lower) | | 11 SCs | k in [-64,-54] | OK |
| Guard (upper) | | 10 SCs | k in [54,63] | OK |
| Pilot SCs | N_pilot | 8 | NUM_PILOTS=8 | OK |
| Pilot Indices | | +-7,+-21,+-35,+-49 | PILOT_INDICES | symmetric, no DC/guard OK |
| Data SCs | N_data | 98 | NUM_DATA=98 | 106-8=98 OK |
| STF Length | | 128 samples | STF_LEN=128 | 8x16=128 OK |
| LTF Length | | 320 samples | LTF_LEN=320 | 64+128+128=320 OK |
| SIGNAL Length | | 160 samples | SIG_LEN=160 | 1 OFDM symbol OK |
| Guard (HW) | | 1024 samples | GUARD_SAMPLES | 40.96us@25MS/s OK |
| Guard (BB) | | 820 samples | GUARD_SAMPLES_BB | ceil(1024*4/5)=820 OK |
| Pilot Polarity | | [1,1,1,1,-1,1,-1,1] | PILOT_POLARITY | fixed BPSK sequence |
| Conv code K | | 7 | CONV_K=7 | 802.11 standard |
| Conv generators | | [0o133, 0o171] | | G0=91, G1=121 decimal |
| CRC | | CRC-32 | scrambler.py | IEEE 802.3 polynomial |
| Center Frequency | | 2.412 GHz | USRP_FREQ | ISM Ch1 OK |
| LO Offset | | **8.75 MHz** | LO_OFFSET | CORDIC anti-LO-leakage: k=-56 (guard). **Was 5.0 MHz (k=-32 ACTIVE SC) — BUG FIXED v7** |
| TX Gain | | 30 dB | USRP_TX_GAIN | calibrated 0.5-1m LoS |
| RX Gain | | 25 dB | USRP_RX_GAIN | ADC headroom OK |
| Packet power | | 0.0447 (mean) | computed | SNR_offset = 13.50 dB VERIFIED |

**SIGNAL field contract (TX/RX must match exactly):**
- TX produces 100 coded bits; transmits first **98** on 98 DATA SCs
- Bits [98:100] are always 0 (Viterbi tail flush) — safely dropped
- RX appends **2 zeros** to reach 100 before Viterbi decode — lossless
- This is an explicit, verified contract between waveform.py and demod.py

### 2.2 Frame Structure

```
[GUARD 1024 zeros@25MS/s][STF 128][LTF 320][SIGNAL 160][DATA 160*N_sym]
       40.96 us               6.4us   16.0us    8.0us       8.0us/symbol

LTF internal: [64-sample double CP][128-sample LTF body 1][128-sample LTF body 2]
              Double CP = extra timing robustness against multipath delay spread

Total packet (20MS/s, NO HW guard) for 100B BPSK:
  STF(128) + LTF(320) + SIG(160) + 17*SYMBOL(2720) = 3328 samples = 166.4 us
```

---

## Part 3 — Algorithm Details

### 3.1 TX Chain (waveform.py, scrambler.py)

```
Step 1: payload_bits (raw information bits)
Step 2: crc32_append()          -- append 32-bit IEEE 802.3 CRC
Step 3: scramble(seed=0b1011101) -- XOR with 127-period LFSR sequence
Step 4: encode_half_rate()      -- K=7 rate-1/2, G0=0o133, G1=0o171
Step 5: zero-pad coded bits to N_sym * N_data * bps
Step 6: map_bits_to_symbols()   -- Gray-coded QAM
        BPSK:  {+1, -1}
        QPSK:  {+-1+-1j}/sqrt(2)
        16QAM: {+-1,+-3}^2 / sqrt(10)
Step 7: modulate_data_symbols() -- OFDM IFFT + CP per symbol
Step 8: assemble_packet()       -- [STF][LTF][SIGNAL][DATA*N_sym]
```

### 3.2 RX Chain (detector.py, sync.py, demod.py)

**Stage 1 — Schmidl-Cox Packet Detection (detector.py):**
```
Metric: M[n] = |P[n]|^2 / R[n]^2
P[n] = sum_{l=0}^{15} s[n+l] * conj(s[n+l+16])
R[n] = sum_{l=0}^{15} |s[n+l+16]|^2
Threshold: M[n] >= 0.65 -> detection

Coarse CFO (5 STF reps = 80 samples):
  phi = angle(sum of stf[16:80] * conj(stf[0:64]))
  CFO_coarse = phi / (2*pi * 16/FS_FFT)    Range: +-625 kHz (130x B210 drift)

Advance-past-packet: STF+LTF+SIG+34*SYM = 128+320+160+5440 = 6048 samples
  CRITICAL: This is LARGER than any packet stride. ALWAYS use chunked streaming.
```

**Stage 2 — Fine Timing (sync.py: find_ltf_timing):**
```
Apply coarse CFO -> cross-correlate with 128-sample LTF template
Search window: expected +-64 samples around pkt_start+128+64
ltf_start = peak - 64  (account for 64-sample LTF CP)
Measured: mean=0.0, max=0 samples at SNR>25dB
```

**Stage 3 — Fine CFO (sync.py: estimate_fine_cfo):**
```
phi_fine = angle(sum of ltf2_time * conj(ltf1_time))
CFO_fine = phi_fine / (2*pi * 128/FS_FFT)    Range: +-78 kHz
Total CFO = CFO_coarse + CFO_fine
Measured residual: <120 Hz at 8kHz true CFO
```

**Stage 4 — CSI Extraction (sync.py: extract_csi):**
```
Apply total CFO correction to full packet buffer (start_n=0)
Y1 = FFT(rx_corr[ltf_start+64 : ltf_start+192])
Y2 = FFT(rx_corr[ltf_start+192 : ltf_start+320])
H_hat[i] = (Y1[bin(k)] + Y2[bin(k)]) / (2 * X[bin(k)])
  2-repetition average: +3dB SNR improvement
Result: complex64 array shape (106,) indexed over ACTIVE_SUBCARRIERS
```

**Stage 4b — Phase Sanitization (sync.py: sanitize_csi_phase):**
```
raw_phase = np.unwrap(np.angle(H_hat))
OLS fit: phase[k] = intercept + slope * k
phase_san[k] = raw_phase[k] - (intercept + slope * k)
H_san = |H_hat| * exp(j * phase_san)
Measured: 7.91 dB variance reduction, slope reduced 5.6e8x
```

**Stage 5 — SCO Tracking (sync.py: sco_correct_symbol):**
```
For each DATA symbol m, pilot k in {+-7,+-21,+-35,+-49}:
  pilot_phase[k] = angle(Y_fft[bin(k)] * conj(H_hat[k]) * polarity[k] / |H_hat[k]|^2)
np.unwrap(pilot_phases)   <- critical: prevents wrap discontinuities
LS fit: phi = a_m + b_m * k
IIR: b_accum = 0.7*b_prev + 0.3*b_m   (alpha=0.3, time_const=3.3 syms)
Correction: Y_corr[k] = Y_fft[k] * exp(-j*(a_m + b_accum * k))
Validated: >=+-20 ppm
```

**Stage 6 — ZF Equalization + Viterbi Decode (demod.py):**
```
ZF: Z[k] = Y[k] * conj(H_hat[k]) / (|H_hat[k]|^2 + 1e-10)
Demap -> Viterbi(K=7, rate-1/2) -> Descramble -> CRC-32 check
Store H_hat in HDF5 ONLY on CRC PASS
```

---

## Part 4 — Data Collection and Preprocessing Pipeline

### 4.1 How Data Is Collected (Real B210 Operation)

```
┌──────────────────────────────────────────────────────────────────┐
│  TX SIDE (main_tx.py)                                            │
│  1. Build packet: waveform.assemble_packet()                     │
│  2. Prepend GUARD: 1024 zeros @ 25 MS/s (40.96 us blank)        │
│  3. Resample 20->25 MS/s: polyphase FIR 5/4, Kaiser beta=8      │
│  4. Apply CORDIC LO offset: +5 MHz (anti-DC-spike)              │
│  5. Transmit via B210 Port A (TX/RX) at FS_HW=25 MS/s           │
│  6. Wait INTERVAL ms (10 ms = 100 pkts/sec typical)             │
│  Repeat continuously                                             │
└──────────────────────────────────────────────────────────────────┘
                        [RF propagation through activity zone]
┌──────────────────────────────────────────────────────────────────┐
│  RX SIDE (main_rx.py / rx_hardware.py)                           │
│  1. Receive continuous IQ at 25 MS/s via Port B (RX2)           │
│  2. Write to RingBuffer(capacity=8M samples = 320 ms)            │
│  3. Read CHUNK_SIZE=1024 samples at a time                       │
│  4. PacketDetector.process(chunk): Schmidl-Cox sliding window    │
│  5. On detection (abs_s, coarse_cfo):                            │
│     a. Apply +-50 kHz operational CFO gate                       │
│     b. Extract window: rx[rel : rel + PKT_LEN + STF_LEN]        │
│     c. Resample 25->20 MS/s: polyphase FIR 4/5                  │
│     d. sync_packet(): fine timing, fine CFO, CSI, phase_san     │
│     e. sco_correct_symbol() per DATA symbol                      │
│     f. ZF equalize -> Viterbi -> CRC-32 check                    │
│     g. On CRC PASS: write H_hat + metadata to HDF5               │
│     h. On CRC FAIL: discard silently (no CSI contamination)      │
│  6. CSILogger flushes HDF5 every 50 packets (thread-safe)        │
└──────────────────────────────────────────────────────────────────┘
```

### 4.2 Signal Windowing — CRITICAL LESSON FROM SIMULATION

The PacketDetector is a **streaming** detector. One `det.process(chunk)` call per burst:

```
CORRECT (how rx_hardware.py and test_realtime_har_sim.py operate):
  for each burst:
    chunk = zeros(GUARD_SAMPLES_BB) + packet_IQ   # 820 + 3328 = 4148 samples
    pre_idx = det._sample_idx                      # save absolute position before call
    detections = det.process(chunk)                # ONE call per burst
    for (abs_s, cfo) in detections:
        rel = abs_s - pre_idx                      # position within this chunk
        win = chunk[rel : rel + PKT_LEN + STF_LEN] # extra STF_LEN headroom!
        sync + decode on win

WRONG (silent failure - misses 80% of packets):
  entire_stream = concatenate(all_packets)
  det.process(entire_stream)   # advance=6048 > stride=4148 -> skips packets
```

**Why PKT_LEN + STF_LEN headroom (not just PKT_LEN):**
The Schmidl-Cox fires at the FIRST threshold crossing, which can be up to STF_LEN=128 samples
BEFORE the true packet start. Without the extra STF_LEN in the window, the last DATA symbol
gets truncated (observed: sym 16/17 missing -> CRC fails). With headroom, sync_packet finds
the LTF at the correct internal offset regardless of early firing.

### 4.3 HDF5 Data Format

```
csi_data.h5
  /session_{id}/
      ant_0/
          H_hat      (N_pkts, 106) complex64  -- raw CSI per subcarrier
          phase_san  (N_pkts, 106) float32    -- OLS-sanitized phase
          timestamp  (N_pkts,)     float64    -- POSIX seconds
          seq_num    (N_pkts,)     int32      -- packet sequence number
          crc_ok     (N_pkts,)     bool       -- always True (stored on PASS only)
          snr_est    (N_pkts,)     float32    -- mean|H_hat| as SNR proxy
```

### 4.4 Preprocessing for Downstream HAR

```python
import h5py, numpy as np
from scipy.signal import welch, spectrogram

# Load CSI
with h5py.File('csi_data.h5', 'r') as f:
    H  = f['session_001/ant_0/H_hat'][:]       # (N, 106) complex64
    ts = f['session_001/ant_0/timestamp'][:]    # (N,) float64

fs_csi = 1.0 / np.median(np.diff(ts))          # actual packet rate (e.g. 100 Hz)

# CSI amplitude (magnitude per subcarrier over time)
amp = np.abs(H)                                  # (N, 106)

# Per-subcarrier Welch PSD (activity band 0.1-10 Hz)
f_ax, Pxx = welch(amp, fs=fs_csi, axis=0, nperseg=int(fs_csi*2))

# Doppler spectrogram (STFT on mean amplitude across subcarriers)
amp_mean = amp.mean(axis=1)                      # (N,) averaged across SCs
f_stft, t_stft, Sxx = spectrogram(amp_mean, fs=fs_csi, nperseg=64, noverlap=48)

# Feature vector for classifier
features = np.concatenate([
    Pxx[(f_ax >= 0.1) & (f_ax <= 10), :].mean(axis=1),  # PSD features
    amp.std(axis=0),                                       # per-SC variance
    np.angle(H).std(axis=0),                              # phase variance
])
```

---

## Part 5 — Robustness Assessment

### 5.1 Pre-B210 Validation Results (2026-09-29 — All 16 PASS)

| Test | Scenario | Result | Threshold | Status |
|---|---|---|---|---|
| T1 | 1000 pkts loopback | CRC=100%, BER=0 | >=99% | PASS |
| T2 | 8-120B, BPSK/QPSK/16QAM | 12/12 50/50 | all 50/50 | PASS |
| T3 | SNR sweep sigma=0.001..0.060 | 31/50 at sigma=0.06 | >=15% | PASS |
| T4 | 5 pkts chunked streaming | 5/5 det, 5/5 CRC | >=4/5 | PASS |
| T5 | 500 trials random timing+impairments | CRC=98% | >=65% | PASS |
| T6 | +-10kHz CFO, +-20ppm SCO | 8/8 sub-tests | >=60% | PASS |
| T7 | 2-RX consistency, 500 pkts | |corr|=-0.007 | |r|<0.8 | PASS |
| T8 | 1000 pkts ring buffer stability | drops=0, drift<5pp | >=85% CRC | PASS |
| T_CHK | Chunk boundary, 7 sizes 256-4096 | all CRC PASS, LTF err=0 | CRC+timing | PASS |
| T_NOI | 500 blocks AWGN, +-50kHz gate | FA=4.2%/block | <5% | PASS |
| T_META | Metadata integrity, 50 pkts 2-RX | seq+ts monotone | integrity | PASS |
| T_RST | 5 restart cycles | pattern consistent | >=1/cycle | PASS |

### 5.2 Simulation Failures and What They Taught Us

| Failure | Root Cause | Fix Applied | Hardware Implication |
|---|---|---|---|
| T4: 3/5 pkts (32-sample gap) | Gap < Schmidl-Cox settle time | GUARD=820 chunked mode | Always use 1024-sample HW guard |
| T4: CFO estimates >>50kHz | Correlator firing on cross-packet boundary | One det.process() per burst | rx_hardware.py already chunked |
| T_CHK: CRC fail, correct timing | Decode window 3 samples short (abs_s early) | Add STF_LEN headroom | rx_hardware.py already correct |
| T_NOI: 16.8% FA with +-200kHz | CFO gate too wide for noise | Tighten to +-50kHz | Add soft +-50kHz gate in rx_hardware |
| Stale comments 107/99 SCs | Old asymmetric design leftover | All updated to 106/98 | Documentation only |

### 5.3 Known Limitations

| Limitation | Detail | Impact |
|---|---|---|
| Viterbi cliff | CRC fails at true SNR < ~11.5dB | Range: ~1m LoS @ TX=30dB |
| Phase sanitization | OLS removes linear slope only | Residual from deep fades |
| SNR label offset | True SNR = label - 13.50 dB (pkt power=0.0447) | Documentation note |
| FA rate 4.2% | Noise with +-50kHz gate; all rejected by CRC | No CSI contamination |
| Detector advance 6048 > PKT_LEN | Must use chunked streaming | Critical for rx_hardware |
| B210 TCXO drift | +-2ppm -> +-4.8kHz @ 2.4GHz | Within validated +-8kHz range |

---

## Part 6 — Design Rationale

### 6.1 FFT Size: 128 (not 64, not 256)
64-pt gives only 52 active SCs (too coarse). 256-pt requires 40 MS/s (B210 USB limit). 128-pt gives 106 active SCs at exactly 20 MS/s — 2x CSI density at same bandwidth.

### 6.2 CP: 32 samples = N_FFT/4 = 1.6 us
Covers 1600 ns delay spread >> 300 ns office worst-case (5x margin). Standard N/4 convention: 20% overhead (acceptable for sensing).

### 6.3 Sample Rates: 20 MS/s BB, 25 MS/s HW
20 MS/s: N_FFT * Delta_f = 128 * 156.25k = 20 MHz exactly. 25 MS/s: B210 AD9361 "sweet spot" (lower noise floor). Ratio 5/4 is rational — polyphase FIR exact.

### 6.4 Kaiser 64-tap Beta=8 Resample Filter
80 dB sidelobe attenuation. 64 taps: group delay 31 samples (compensated). Passband flat <0.1 dB across 20 MHz. Prevents aliased images entering 106 active SCs.

### 6.5 Pilots: +-7,+-21,+-35,+-49
8 symmetric pilots: unambiguous LS slope fit. Spacing 14 SCs: spans full [-49,+49] range. k>=7: avoids LO 1/f noise near DC. k<=49: 4 SCs margin from guard edge. Even grid: well-conditioned LS for SCO.

### 6.6 Schmidl-Cox Threshold: 0.65
Empirically tuned: 0.70 causes 5% missed detections; 0.60 causes false alarms. At 0 dB SNR, M[n]~0.5 on STF plateau. 0.65 gives T_NOI PASS with +-50kHz CFO gate (FA=4.2% < 5%).

### 6.7 Guard: 1024 HW / 820 BB = 40.96 us
Without guard: Schmidl-Cox fires false alarms on noise in buffer. With guard: correlator slides through blank, sees ~0, resets before STF. 1024 gives 5x minimum (160 BB). LEARNED: detector advance=6048 > PKT_LEN=3328; guard ensures exactly one packet per chunk call.

### 6.8 LTF CP: 64 samples (double)
64 samples = 3.2 us = 480 m one-way path margin. Gives find_ltf_timing a 64-sample search window for multipath delay. Standard CP (32) would restrict search to 1.6 us (240 m — still OK but less margin).

### 6.9 SCO Alpha: 0.3
Time constant = 3.3 symbols. alpha=1.0 amplifies noise; alpha=0.1 lags behind temperature changes. 0.3 is standard OFDM pilot-tracking value. Validated >=20 ppm.

### 6.10 CFO Gate: +-50 kHz Operational
B210 TCXO: +-2ppm @ 2.4GHz = +-4.8kHz. After coarse correction: <100Hz residual. 50 kHz gives 500x margin. FA rate: +-200kHz -> 16.8% (FAIL); +-50kHz -> 4.2% (PASS). Hardware ceiling stays at 200kHz for robustness; operational gate = 50kHz.

### 6.11 TX/RX Gains: 30 dB / 25 dB
TX 30dB at 1m LoS: received=-45dBm, noise=-100dBm, SNR=55dB. Body worst-case -20dB: SNR=35dB >> Viterbi cliff ~11.5dB. RX 25dB: ADC clips at -10dBm, received=-45dBm, headroom=35dB.

### 6.12 LO Offset: 8.75 MHz CORDIC (v7 CORRECTED from 5 MHz)
AD9361 direct-conversion: 1/f noise and LO leakage at DC. Without offset: pilots at k=±7 affected by DC noise floor rise.

**Why 8.75 MHz (NOT 5 MHz)**: 5 MHz = 5e6/156.25e3 = 32 subcarrier widths → k=−32.
`−32 in ACTIVE_SUBCARRIERS = True`. The 5 MHz offset would corrupt a live data subcarrier on every packet.

8.75 MHz = 8.75e6/156.25e3 = 56 subcarrier widths → k=−56.
Lower guard band = k=−64 to k=−54. k=−56 IS in the lower guard. `−56 in ACTIVE_SUBCARRIERS = False`.
The LO spike lands completely in the guard band and never contaminates any active subcarrier.

---

*25 findings 2026-09-30 | 4 blockers fixed | See FINDINGS.md for full audit*
*HARDWARE_DEPLOYMENT.md for B210 setup v7 | DEPLOYMENT_RUNBOOK.md for commands*
