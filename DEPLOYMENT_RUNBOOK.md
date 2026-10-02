# DEPLOYMENT RUNBOOK — Phase 2 Custom OFDM PHY for CSI-HAR
## Step-by-Step Operator Guide | USRP B210 | All Commands Validated

> This runbook is the SINGLE DOCUMENT an operator needs to go from zero to
> capturing HAR data with the B210. Every command is verified against the code.
>
> **Last updated**: 2026-09-30 v7 — 25 audit findings; 4 blockers fixed. See FINDINGS.md.
> **Simulation status**: 40/40 benchmark PASS | 4/4 stage validators PASS | 3000/3000 HAR stress PASS
> **BREAKING v7**: LO offset changed 5 MHz → 8.75 MHz (5 MHz was corrupting data SC k=-32)

---

## QUICK START (30 minutes, first-time setup)

```
Step 1:  pip install numpy scipy h5py pyzmq simpleaudio matplotlib
Step 2:  conda install -c conda-forge uhd       (or build UHD from source)
Step 3:  uhd_images_downloader                  (download B210 firmware)
Step 4:  python hw_readiness_check.py           (7/7 PASS expected)
Step 5:  Connect B210: Port A -> TX antenna, Port B -> RX antenna
Step 6:  Place antennas 0.7m apart at 0.9m height
Step 7:  python run_all_validators.py           (4/4 PASS expected)
Step 8:  python spectrum_scan.py --band 2.4     (choose clear channel)
Step 9:  python main_tx.py --mode hardware      (Terminal 1)
Step 10: python main_rx.py --mode hardware      (Terminal 2)
```

---

## Phase 0 — Simulation-First Validation (MANDATORY Before Hardware)

### 0.1 Unit Tests (PHY Primitives)

```bash
python -m pytest test_waveform.py -v
```
Expected: **16/16 PASS**. If any FAIL: do NOT proceed.

### 0.2 Core Stage Validators

```bash
python run_all_validators.py
```

Expected:
```
[Stage 1] PHY Primitives .............. PASS
[Stage 2] Impairment Tolerance ........ PASS
[Stage 3] E2E Pipeline + HDF5 ......... PASS
[Stage 4] Resampler Chain ............. PASS
4/4 PASS
```
If any stage FAILS: DO NOT proceed to hardware.

### 0.3 40-Scenario Quantitative Benchmark

```bash
python run_sim_results.py
```
Expected: **40/40 PASS** -> saved to sim_results.csv
Honest thresholds: CFO>=70% CRC, SCO>=70% CRC, Combined>=50% CRC

### 0.4 Comprehensive Pre-B210 Suite (16 Tests)

```bash
# Quick mode (14 tests, ~7 min) — skip long T8+TC
python validate_pre_b210.py --quick

# Full mode (16 tests, ~3 min with corrected T8=1k pkts)
python validate_pre_b210.py
```

Expected: **16/16 PASS**. Critical tests:

| Test | What It Validates | Expected |
|---|---|---|
| T1: Full loopback | CRC=100% over 1000 pkts | CRC>=99% |
| T4: Chunked streaming | 5 pkts, GUARD=820, one det.process/burst | 5/5 CRC |
| T5: Random timing+impairments | 500 trials | CRC>=65% |
| T6: Worst-case CFO+SCO | +-10kHz, +-20ppm | detect>=70% |
| T7: Two-RX consistency | 500 pkts, channels uncorrelated | OK |
| T8: Ring buffer stability | 1000 pkts, drops=0, drift<5pp | OK |
| T_CHK: Chunk boundary | 7 chunk sizes, 256-4096 | All CRC PASS |
| T_NOI: False-alarm gate | AWGN, +-50kHz CFO gate | FA<5%/block |

> **LESSON FROM SIMULATION**: T4 originally failed with 32-sample gap (gap < settle time).
> Fixed: always use GUARD_SAMPLES_BB=820 and ONE det.process() call per burst.
> Decode window needs PKT_LEN + STF_LEN headroom (not just PKT_LEN) — Schmidl-Cox
> can fire up to STF_LEN=128 samples before true packet start.

### 0.5 HAR Stress Test (Verify Activity Detection)

```bash
# Multi-activity stress: 3000 pkts, changing HAR channel models
python test_realtime_har_sim.py --scenario multi_activity --n_packets 3000 \
    --interval 0.010 --hdf5 csi_data.h5 --session_id session_001

# Individual scenarios (1000 pkts each, optional)
python test_realtime_har_sim.py --scenario transient_gesture --n_packets 1000 --hdf5 csi_data.h5 --session_id s_gesture
python test_realtime_har_sim.py --scenario fast_tremor      --n_packets 1000 --hdf5 csi_data.h5 --session_id s_tremor
python test_realtime_har_sim.py --scenario abrupt_walkby    --n_packets 1000 --hdf5 csi_data.h5 --session_id s_walkby
```

Expected (multi_activity):
```
Total detected:     3000/3000 (100%)
CRC passed:         3000/3000 (100%)
Phase var:          0.0366 rad^2 (<0.50 threshold)
Variance reduction: 7.91 dB   (>6.0 dB threshold)
Per-segment:        Micro-gestures | Tremor 5.5Hz | Walk-by -- all PASS
```

### 0.6 HDF5 Verification and Plots

```bash
# Verify HDF5 output from stress test
python verify_live_har.py --hdf5 csi_data.h5 --session session_001 \
    --fmin 0.1 --fmax 10.0 --nperseg 128 --outdir plots

# Quick plots (PSD, SCO, CSI heatmap, detector)
python plot_project.py --figs 3 5 6 7

# Core performance plots (BER, constellations, CFO)
python plot_project.py --figs 1 2 4

# Full diagnostic suite (30 plots, ~25 min)
python plot_diagnostics.py
```

---

## Phase 1 — Hardware Connection and Verification

### 1.1 Detect B210

```bash
uhd_find_devices
```

Success:
```
-- UHD Device 0
   serial: XXXXXXXX   product: B210   type: b200
```

Failure responses:
| Message | Fix |
|---|---|
| No devices found | Try different USB port (blue/SS port); check cable |
| Error: no compatible FPGA image | Run: uhd_images_downloader |
| USB 2.0 detected | Use blue USB 3.0 port; never use USB hub |

### 1.2 Full Device Probe

```bash
uhd_usrp_probe --args type=b200
```

Check the output for:
- "B210" product
- Frequency range: 70 MHz to 6 GHz
- RX gain range includes 25 dB
- TX gain range includes 30 dB

### 1.3 Verify Channel Mapping

```bash
python -c "import uhd; u=uhd.usrp.MultiUSRP(); print(u.get_pp_string())"
```

Verify Ch 0 maps to Front-End A. If different, edit config.py: USRP_SUBDEV_SPEC

### 1.4 LO Lock Test

```bash
python -c "
import uhd, time
u = uhd.usrp.MultiUSRP()
u.set_tx_rate(25e6); u.set_tx_freq(2.412e9, 0); u.set_tx_gain(30, 0)
time.sleep(0.5)
locked = u.get_tx_sensor('lo_locked', 0).to_bool()
print(f'TX LO locked: {locked}')
u.set_rx_rate(25e6); u.set_rx_freq(2.412e9, 0); u.set_rx_gain(25, 0)
time.sleep(0.5)
locked = u.get_rx_sensor('lo_locked', 0).to_bool()
print(f'RX LO locked: {locked}')
"
```

Both must print True. If False after 2 seconds: replug USB and retry.

### 1.5 Hardware Readiness Check

```bash
python hw_readiness_check.py
```

Expected: 7/7 PASS
Checks: UHD version, all .py files present, config.py compiles, constants correct.

---

## Phase 2 — Physical Setup

### 2.1 Antenna Connection

1. Connect TX antenna to Port A (labelled TX/RX) via 1m SMA extension cable
2. Connect RX antenna to Port B (labelled RX2) via SMA pigtail or directly
3. NEVER connect Port C or D
4. Confirm: Port A = TX, Port B = RX (NOT the other way around!)

### 2.2 Antenna Placement (Exact Values)

```
Antenna height:   0.8 to 1.2 m above floor (chest height, default 0.9 m)
Separation:       0.5 to 1.0 m (default 0.7 m for HAR)
Polarization:     Both vertical (antennas standing upright)
Orientation:      TX element facing RX, RX element facing TX
Subject position: Standing/walking directly between TX and RX antennas
Room clearance:   At least 0.5 m from walls on all sides
```

For extended range (>1.0 m):
```python
# Edit config.py:
USRP_TX_GAIN = 40.0    # was 30.0; safe up to 45.0 for B210
USRP_RX_GAIN = 30.0    # was 25.0; increase if signal weak
```

### 2.3 Channel Selection

```bash
python spectrum_scan.py --band 2.4
```

Output example:
```
Scanning 2.400 - 2.485 GHz...
Ch1  2.412 GHz: -72 dBm (CLEAR)    <- recommended
Ch6  2.437 GHz: -68 dBm (MODERATE)
Ch11 2.462 GHz: -74 dBm (CLEAR)
Recommendation: Use Ch1 (2.412 GHz)
```

If recommended channel is not Ch1, update config.py:
```python
USRP_CENTER_FREQ = 2.437e9    # example: Ch6
```

---

## Phase 3 — Running a HAR Data Collection Session

### 3.1 Open Terminal 1 (TX)

```bash
cd d:\phase2
python main_tx.py --mode hardware --mod BPSK --payload 100 --interval 0.005
```

Wait for:
```
[tx_hw] TX configured: 2412.000000 MHz  rate=25.000 MS/s  gain=30.0 dB
[tx_hw] TX pkt #1  pkt_len=4160  guard=1024  burst=5184
```

If you see "TX LO did not lock": check USB and wait up to 2 seconds.

### 3.2 Open Terminal 2 (RX + Logger)

```bash
cd d:\phase2
python main_rx.py --mode hardware --hdf5 csi_data.h5 --session_id session_001
```

Wait for:
```
[main_rx] Pre-flight simulation gate PASS (4/4 stages)
[rx_hw] RX Channel 0 configured: 2412.000000 MHz  rate=25.000 MS/s  gain=25.0 dB
[rx_hw] Pkt #0001 | CRC=PASS | CFO=+214.3 Hz | Slope=-0.0002 | Drops=0
[rx_hw] Pkt #0002 | CRC=PASS | CFO=+217.1 Hz | Slope=-0.0002 | Drops=0
```

### 3.3 Open Terminal 3 (Optional: Session Controller with Audio Cues)

```bash
cd d:\phase2
python session_controller.py
```

Plays audio cues for activities: idle -> walking -> sitting -> standing -> falling
Injects labeled trial markers into HDF5 for supervised learning.

### 3.4 Monitoring During Collection

Watch the RX console for:

| Indicator | Good | Investigate If |
|---|---|---|
| CRC=PASS rate | >=70% | <50% (alignment issue) |
| CFO value | +-5000 Hz range | >8000 Hz (LO issue) |
| Phase slope | +-0.001 rad/SC | >0.01 (SCO issue) |
| Drops count | 0 | >0 (USB issue) |
| Ring occupancy | <50% | >80% (CPU overloaded) |

---

## Phase 4 — OTA Bring-Up Validation

Run this BEFORE a full data collection session:

```bash
python validate_stage5_hardware.py --n_packets 50
```

Expected output:
```
[validate_stage5] Test: OTA CRC Gate
  Transmitted: 50 packets
  CRC PASS:    48/50 (96.0%)
  LO locked:   TX=True, RX=True
  Drops:       0
[validate_stage5] RESULT: PASS
```

Accept if: CRC >= 70%, Drops = 0, LO = True.
Retry if CRC < 70%: check antenna position, reduce separation, increase TX gain.

---

## Phase 5 — Post-Capture Verification and Analysis

### 5.1 Verify HDF5 Data Integrity

```bash
python verify_live_har.py --hdf5 csi_data.h5 --session session_001
```

Output includes:
```
HDF5 structure check:    PASS
Total packets logged:    N
CSI shape:               (N, 106) complex64
NaN count:               0 (0.00%)
Phase sanitization var:  0.041 rad^2  (< 0.50 threshold)  PASS
Doppler peak:            detected
Trial markers:           5 trials logged
```

### 5.2 Generate Figures

```bash
# 10 publication-quality figures -> plots/fig1.png to fig10.png
python plot_project.py

# 30 diagnostic figures -> plots/
python plot_diagnostics.py
```

Key figures:
- fig1.png: BER vs SNR curves (BPSK/QPSK/16QAM)
- fig2.png: Constellation diagrams before and after equalization
- fig3.png: Power spectral density of transmitted waveform
- fig4.png: CFO estimation accuracy vs true CFO
- fig5.png: SCO tracking convergence across 17 symbols
- fig9.png: CSI magnitude heatmap (subcarrier x time)
- fig10.png: Full system dashboard

---

## Phase 6 — Complete Parameter Reference

### Adjustable Parameters (config.py)

| Parameter | Default | When to Change |
|---|---|---|
| USRP_CENTER_FREQ | 2.412e9 (Ch1) | If Ch1 is busy (spectrum scan first) |
| USRP_TX_GAIN | 30 dB | Increase to 40-45 for range >1m |
| USRP_RX_GAIN | 25 dB | Increase to 35-38 for long range; watch clipping |
| GUARD_SAMPLES | 1024 | Increase to 2048 in very reflective rooms |
| HDF5_FLUSH_INTERVAL | 50 | Reduce to 20 for high-rate; increase for lower I/O |

### Fixed Parameters (Never Change Without Understanding Impact)

| Parameter | Value | Reason It Cannot Change |
|---|---|---|
| FFT_SIZE | 128 | All modules calibrated to 128-pt FFT |
| CP_LEN | 32 | Must be FFT_SIZE/4 for CP validity |
| FS_HW | 25.0 MS/s | B210 clock; resample ratio tied to this |
| FS_FFT | 20.0 MS/s | Must maintain 4/5 ratio with FS_HW |
| PILOT_INDICES | +/-7,+/-21,+/-35,+/-49 | SCO tracking anchored to these positions |
| CONV_GEN | [0o133, 0o171] | Viterbi decoder uses same polynomials |
| DETECT_THRESH | 0.65 | Tuned to minimize false detections vs misses |

---

## Phase 7 — Emergency Procedures

### Restart After UHD Overflow

If you see "[rx_hw] WARNING: UHD Overflow":
1. Note the drop count
2. Press Ctrl+C in both terminals
3. Wait 3 seconds
4. Restart TX first, then RX
5. Drops will reset to 0 at start

### Recover from Corrupted HDF5

```bash
python -c "
import h5py
try:
    f = h5py.File('csi_data.h5', 'r')
    print('HDF5 OK, sessions:', list(f.keys()))
    f.close()
except Exception as e:
    print('HDF5 corrupted:', e)
    print('Rename and start fresh: rename csi_data.h5 csi_data_backup.h5')
"
```

### If CRC Rate Drops Mid-Session

Symptoms: was 90%, dropped to 20% suddenly
Causes: Someone moved the antennas, large obstruction moved between TX/RX, USB overload
Fix: Check Drops counter. If 0 -> positional fix needed. If >0 -> USB issue.

---

## Quick Reference Card

```
UNIT TESTS:     python -m pytest test_waveform.py -v          (16/16 PASS)
SIM VALIDATE:   python run_all_validators.py                   (4/4 PASS)
BENCHMARK:      python run_sim_results.py                      (40/40 PASS)
PRE-B210 FULL:  python validate_pre_b210.py                    (16/16 PASS)
HAR STRESS:     python test_realtime_har_sim.py --scenario multi_activity --n_packets 3000
HDF5 VERIFY:    python verify_live_har.py --hdf5 csi_data.h5 --session session_001
PLOTS QUICK:    python plot_project.py --figs 3 5 6 7

HARDWARE:
DETECT B210:    uhd_find_devices
PROBE B210:     uhd_usrp_probe --args type=b200
HW READINESS:   python hw_readiness_check.py                   (7/7 PASS)
SPECTRUM SCAN:  python spectrum_scan.py --band 2.4
TX (Terminal1): python main_tx.py --mode hardware --mod BPSK --payload 100 --interval 0.005
RX (Terminal2): python main_rx.py --mode hardware --hdf5 csi_data.h5 --session_id session_001
SESSION CTRL:   python session_controller.py                   (Terminal3, optional)
OTA VALIDATE:   python validate_stage5_hardware.py --n_packets 50   (CRC>=70%)
```

```
ANTENNAS:   Port A = TX, Port B = RX, separation 0.7m, height 0.9m, both vertical
USB:        Blue USB 3.0 port, no hub, disable power management
GAINS:      TX=30dB, RX=25dB (increase for longer range)
GUARD:      ALWAYS 1024 zeros (40.96us) prepended to every burst
CFO GATE:   +-50 kHz operational (+-200 kHz hardware max) - tighter = fewer false alarms
CHUNKING:   ONE det.process(chunk) call per burst - NEVER process full stream at once
WINDOW:     Decode window = PKT_LEN + STF_LEN (3328+128=3456) — extra headroom required
```

### Robustness Checklist Before Hardware

- [ ] test_waveform.py: 16/16 PASS
- [ ] run_all_validators.py: 4/4 PASS
- [ ] run_sim_results.py: 40/40 PASS (CSV saved)
- [ ] validate_pre_b210.py: 16/16 PASS (all T1-T10, T_CHK, T_NOI, T_META, T_RST)
- [ ] test_realtime_har_sim.py multi_activity: 3000/3000 CRC, var<0.50, reduction>6dB
- [ ] verify_live_har.py: HDF5 shape OK, NaN=0, phase_san var<0.50
- [ ] uhd_find_devices: B210 detected
- [ ] hw_readiness_check.py: ALL PASS (50+ checks)
- [ ] spectrum_scan.py: clear channel identified
- [ ] validate_stage5_hardware.py: CRC>=70% OTA

---

## Appendix A — v7 Breaking Changes and Audit Corrections

### A.1 LO Offset Changed: 5 MHz → 8.75 MHz (CRITICAL)

The previous 5 MHz value mapped to subcarrier k=-32, which is inside the active band.
This would corrupt data SC k=-32 on every OTA packet without any visible error.

```python
# config.py (v7 — CORRECTED)
USRP_LO_OFFSET = 8.75e6  # 8.75 MHz → k=-56 (inside lower guard k=-64..-54)
# DO NOT change back to 5.0e6 (5 MHz maps to k=-32, an active data subcarrier)
```

### A.2 CRC Gate Added to logger.py

Previously logger.py wrote ALL packets to HDF5 regardless of CRC result.
Fixed: only CRC-passing packets are stored.

### A.3 CFO Gate Corrected: 200 kHz → 50 kHz

rx_hardware.py _CFO_LIMIT was 200,000 Hz; T_NOI showed FA=16.8% at 200 kHz (fail threshold).
Fixed to 50,000 Hz; FA=4.2% (pass).

### A.4 Ring Buffer Occupancy Fixed

rx_sim.py and rx_hardware.py were dividing ring.occupancy (already 0-1) by _cap (8M).
Result was ~1e-11, not a useful 0-1 monitoring value. Fixed.

### A.5 Remaining Open Items Before First OTA Run

1. **Two-process B210** (F-03): tx_hardware.py and rx_hardware.py as separate processes
   both call uhd.usrp.MultiUSRP() with no args. UHD will deny the second process.
   FIX: use main_tx.py/main_rx.py which manage the device correctly, OR merge TX+RX.

2. **IQ imbalance not modelled** (F-22): Run cable loopback test with 30-40 dB attenuator
   before first OTA deployment. Target CRC >= 90% in loopback.

3. **Link budget unverified** (F-12): TX power at 30 dB gain needs power meter.
   Expected -5 to 0 dBm (conservative); verify before increasing gain for range extension.

---

*v7 2026-09-30 | Phase 2 | Custom 128-pt OFDM PHY | USRP B210 | CSI-HAR*
*Pre-B210 suite: 16/16 PASS | Simulation: 40/40 PASS | HAR stress: 3000/3000 PASS*
*25 audit findings: 4 blockers fixed, 6 high fixed, 10 med/low documented in FINDINGS.md*
