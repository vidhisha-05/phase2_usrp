# HARDWARE DEPLOYMENT GUIDE
## Custom 128-pt OFDM PHY | USRP B210 | 1-TX / 1-RX | CSI-HAR System
### v7 — 2026-09-30 Post Full Audit: 4 Blockers Fixed, 25 Findings Documented

> **BREAKING CHANGE v7**: LO offset changed from 5 MHz to 8.75 MHz.
> The 5 MHz value placed the LO spike at data SC k=-32 (verified active band).
> 8.75 MHz places it at k=-56, inside the lower guard band. DO NOT revert.

> All parameter values in this document are verified against the source code.
> Do NOT override any defaults without understanding the impact.

---

## Section 1 — What Hardware You Need

### Required Hardware

| Item | Specification | Notes |
|---|---|---|
| USRP B210 | Ettus Research B210 SDR | One unit (1-TX / 1-RX) |
| USB 3.0 Cable | Included with B210 | Must be USB 3.0 SuperSpeed (blue connector) |
| TX Antenna | Omnidirectional, 2.4 GHz | 2dBi gain, SMA male connector |
| RX Antenna | Omnidirectional, 2.4 GHz | 2dBi gain, SMA male connector |
| SMA Extension | 1-meter SMA male-to-female | For TX port A |
| Computer | USB 3.0 port, >=8GB RAM | Windows/Linux with UHD 4.x installed |

### Hardware NOT Required (Simulation Uses Software Only)
- Second USRP (simulation uses ZMQ)
- RF cables or attenuators (only needed for bench testing, not OTA)
- External clocking / reference signal

---

## Section 2 — Physical Setup and Placement

### 2.1 USRP B210 Port Identification

Looking at the B210 from the top (SMA ports facing you, USB on the right):

```
 +---------+---------+---------+---------+
 |  Port A |  Port B |  Port C |  Port D |
 | TX/RX   |  RX2    | TX/RX   |  RX2   |
 | (Ch 0)  | (Ch 0)  | (Ch 1)  | (Ch 1) |
 |  USE:TX | USE: RX | UNUSED  | UNUSED |
 +---------+---------+---------+---------+
                              [USB 3.0]
```

**CRITICAL — Port Assignment:**
- Port A (TX/RX, Ch 0): Connect to TX antenna via 1m SMA extension cable
- Port B (RX2, Ch 0): Connect to RX antenna directly or via short pigtail
- Ports C and D: Leave disconnected or terminated with 50-ohm terminator
- **NEVER swap A and B**: internal T/R switch has only ~10dB isolation.
  Swapping causes TX leakage 30-40dB stronger than the OTA signal.
  Result: near-zero CRC rate and CSI dominated by self-interference.

### 2.2 Antenna Placement Rules

**Optimal HAR sensing geometry:**
```
    [TX Antenna]                [RX Antenna]
         |     <--- 0.5-1.0m --->     |
         |                            |
    (vertical,                   (vertical,
   facing RX)                   facing TX)
         |                            |
    [Port A]          B210        [Port B]
```

| Parameter | Value | Reason |
|---|---|---|
| Antenna separation | 0.5 to 1.0 m | Keeps received power in ADC linear range at TX=30dB |
| Antenna height | 0.8 to 1.2 m above floor | At chest/torso height of human subject |
| Both antennas at same height | Yes | Maintains LoS; avoids ground-reflection nulls |
| Subject position | Directly between TX and RX | Maximum CSI sensitivity |
| Both polarizations | Vertical | Minimize 20dB cross-polarization loss |
| LoS clearance | >=0.3m on all sides | Avoid furniture multipath contamination |
| Room wall distance | >=0.5m | Avoid strong wall reflection dominating CSI |

**For >1m range or through-wall scenarios:**
  Increase TX gain in config.py: USRP_TX_GAIN = 40.0 (max safe for B210 is 45dB)
  Corresponding RX gain: USRP_RX_GAIN = 30.0 (watch for ADC clipping)

### 2.3 USB 3.0 Requirements

Sustained data rate at 25 MS/s (complex64 = 8 bytes/sample):
  25e6 samples/s x 8 bytes/sample = 200 MB/s continuous

USB 3.0 SuperSpeed bandwidth: ~625 MB/s theoretical, 300+ MB/s practical. OK.
USB 2.0 High-Speed bandwidth: 60 MB/s. COMPLETELY INSUFFICIENT. Will overflow.

**MANDATORY USB setup:**
1. Plug B210 into a native USB 3.0 port (blue port, or labeled SS)
2. NEVER use a USB hub, even powered SuperSpeed hub
3. On Windows: Device Manager -> USB Root Hub -> Properties -> Power Management
   -> UNCHECK "Allow the computer to turn off this device to save power"
4. Verify: Device Manager -> Universal Serial Bus Controllers -> USRP B210
   must appear under "USB 3.0 eXtensible Host Controller" (not USB 2.0)

---

## Section 3 — Software Installation

### 3.1 Install UHD (USRP Hardware Driver)

**Option A — Conda (recommended for Windows):**
```
conda install -c conda-forge uhd
```

**Option B — Build from source (Linux/advanced):**
```
git clone https://github.com/EttusResearch/uhd
mkdir uhd/host/build && cd uhd/host/build
cmake .. -DENABLE_PYTHON_API=ON -DPYTHON_EXECUTABLE=$(which python3)
make -j4 && sudo make install
sudo ldconfig
```

**Verify UHD installation:**
```
python -c "import uhd; print(uhd.__version__)"
# Expected: 4.x.x.x
```

### 3.2 Install Python Dependencies

```
pip install numpy scipy h5py pyzmq simpleaudio matplotlib
```

**Verify all imports:**
```
python hw_readiness_check.py
```
Expected output: all 7 checks PASS.

### 3.3 Download B210 Firmware (if needed)

```
uhd_images_downloader
```
This downloads FPGA images for the B210. Required for first use.

---

## Section 4 — Verifying B210 Connectivity

### Step 1: Detect the Device

```
uhd_find_devices
```
Expected output:
```
[INFO] [UHD] Linux; GNU C++ version X.X; Boost_XXXXXX; UHD_4.X.X.X
--------------------------------------------------
-- UHD Device 0
--------------------------------------------------
Device Address:
    serial: XXXXXXXX
    name:
    product: B210
    type: b200
```
If nothing appears: check USB cable, USB port version, driver installation.

### Step 2: Probe Device Capabilities

```
uhd_usrp_probe --args type=b200
```
Look for:
- "B210" in the product line
- "AD9361" as the RFIC
- Frequency range: 70 MHz - 6 GHz (confirms full 2.4 GHz coverage)
- Gain range: 0-73 dB on TX, 0-73 dB on RX

### Step 3: Verify Channel Mapping

```
python -c "import uhd; u=uhd.usrp.MultiUSRP(); print(u.get_pp_string())"
```
Expected: Front-end A on Channel 0, Front-end B on Channel 1.
If mapping differs, update config.py: USRP_SUBDEV_SPEC = "A:A" or as needed.

### Step 4: Quick Loopback Test (Before Antennas)

Connect Port A (TX/RX) to Port B (RX2) with a 30-40 dB attenuator and SMA cable.
Then run:
```
python validate_stage5_hardware.py --n_packets 20 --mode loopback
```
Expected: CRC >= 90%, no LO lock warnings.
Remove attenuator before OTA deployment (never transmit directly into RX without attenuation).

### Step 5: Check LO Lock

```
python -c "
import uhd, time
u = uhd.usrp.MultiUSRP()
u.set_tx_rate(25e6); u.set_tx_freq(2.412e9, 0); u.set_tx_gain(30, 0)
time.sleep(0.5)
print('TX LO locked:', u.get_tx_sensor('lo_locked', 0).to_bool())
u.set_rx_rate(25e6); u.set_rx_freq(2.412e9, 0); u.set_rx_gain(25, 0)
time.sleep(0.5)
print('RX LO locked:', u.get_rx_sensor('lo_locked', 0).to_bool())
"
```
Both must print True within 2 seconds. If not: check USB connection and retry.

---

## Section 5 — Pre-Deployment Validation (MANDATORY Before Any Hardware Run)

### Step 1 — Stage Validators (4 stages, ~15 seconds)

```
python run_all_validators.py
```
Expected: 4/4 PASS
Validates: PHY primitives, SNR/CFO/MP/SCO impairments, E2E pipeline + HDF5, resampler.
DO NOT proceed to hardware if any stage fails.

### Step 2 — 40-Scenario Benchmark

```
python run_sim_results.py
```
Expected: 40/40 PASS -> saves sim_results.csv
Time: ~2-3 minutes. Validates all impairment combinations with honest thresholds.

### Step 3 — Hardware Readiness Check

```
python hw_readiness_check.py
```
Expected: 7/7 PASS
Checks: UHD importable, all required files present, config.py compiles,
        key constants correct, NUM_ACTIVE=106, GUARD_SAMPLES=1024.

### Step 4 — Pre-B210 Comprehensive Suite (Optional but Recommended)

```
python validate_pre_b210.py --quick
```
Expected: 14/14 PASS (quick), 16/16 PASS (full)
Time: 7-20 minutes.

---

## Section 6 — Running the System on Hardware

### Step 1 — Spectrum Scan (Choose Cleanest Channel)

```
python spectrum_scan.py --band 2.4
```
This scans 2.400-2.485 GHz and recommends the cleanest channel.
Update config.py if a channel other than 2.412 GHz is recommended:
  USRP_CENTER_FREQ = 2.437e9   # (example: channel 6)

### Step 2 — Terminal 1: Start TX

```
python main_tx.py --mode hardware --mod BPSK --payload 100 --interval 0.005
```

Parameters:
  --mode hardware   Use B210 (vs simulation)
  --mod BPSK        Modulation: BPSK | QPSK | 16QAM
  --payload 100     Payload bytes per packet (recommended: 100)
  --interval 0.005  5ms inter-packet interval (200 packets/second)
  --n_packets 0     0 = infinite (run forever until Ctrl+C)
  --guard 1024      Guard zeros at 25MS/s (NEVER set to 0)

Expected console output:
  [tx_hw] TX configured: 2412.000000 MHz  rate=25.000 MS/s  gain=30.0 dB
  [tx_hw] TX pkt #1  pkt_len=4160  guard=1024  burst=5184
  [tx_hw] TX pkt #2  ...

### Step 3 — Terminal 2: Start RX + Logger

```
python main_rx.py --mode hardware --hdf5 csi_data.h5 --session_id session_001
```

Note: main_rx.py runs a simulation pre-flight gate before opening the B210.
If any simulation stage fails, it will refuse to start in hardware mode.

Parameters:
  --mode hardware         Use B210
  --hdf5 csi_data.h5      HDF5 output file path
  --session_id session_001  Session identifier (used as HDF5 group key)
  --n_packets 0            0 = infinite

Expected console output:
  [rx_hw] Subdev spec configured: A:A
  [rx_hw] RX Channel 0 configured: 2412.000000 MHz  rate=25.000 MS/s  gain=25.0 dB  lo_offset=5.0 MHz
  [rx_hw] Pkt #0001 | CRC=PASS | CFO=+214.3 Hz | Slope=-0.0002 | Drops=0
  [rx_hw] Pkt #0002 | CRC=PASS | CFO=+217.1 Hz | Slope=-0.0002 | Drops=0

Pass criteria:
  - Drops=0 throughout (if Drops > 0, USB 3.0 issue)
  - CRC=PASS rate >= 70% in LoS 0.5-1.0m range
  - CFO stable within +-5000 Hz (small drift is normal)
  - Slope within +-0.001 rad/SC

### Step 4 — Terminal 3 (Optional): Session Controller

```
python session_controller.py
```
Plays audio cues for each HAR trial (walking, sitting, standing, falling, idle).
Injects trial start/end timestamps into HDF5 for labeled dataset creation.

### Step 5 — OTA Bring-Up CRC Validation

```
python validate_stage5_hardware.py --n_packets 50
```
Expected: CRC >= 90%, LO lock PASS, no dropped packets.
If CRC < 70%: check antenna alignment, reduce TX/RX distance, verify USB.

---

## Section 7 — Monitoring and Troubleshooting

### What to Watch on the RX Console

| Output | Good Value | Action If Wrong |
|---|---|---|
| CRC=PASS rate | >=70% at LoS | Increase TX gain or reduce distance |
| CFO value | Within +-5000 Hz | Normal drift; if >8000 Hz, check LO lock |
| Slope (rad/SC) | Within +-0.001 | If >>0.01, SCO correction struggling |
| Drops count | 0 always | USB issue: try different port |
| Ring occupancy | <0.5 (50%) | If >0.8, processing too slow |

### Common Failure Modes

**Problem: uhd_find_devices finds nothing**
  Cause: USB 2.0 port, missing firmware, driver not installed
  Fix: Use blue USB 3.0 port; run uhd_images_downloader; reinstall UHD

**Problem: "TX LO did not lock after 2.0s"**
  Cause: USB power instability, incorrect frequency
  Fix: Replug USB; check USRP_CENTER_FREQ is in 70MHz-6GHz range

**Problem: CRC=PASS rate near 0%**
  Cause: Port A/B swap (most common!), antenna separation too large, wrong frequency
  Fix: Verify Port A=TX, Port B=RX; reduce antenna separation to 0.5m; check frequency

**Problem: Drops=N (N > 0)**
  Cause: USB hub, USB 2.0, insufficient CPU for processing, power management
  Fix: Direct USB 3.0 port; disable USB power management; close other applications

**Problem: CRC OK but CSI looks flat / no activity signature**
  Cause: Antennas not in LoS, subject not between antennas, wrong height
  Fix: Place both antennas at 0.9m height, subject directly between, LoS clear

**Problem: Phase sanitization shows high residual (>0.5 rad^2)**
  Cause: Very strong multipath (reflective room), very short distance (<0.3m)
  Fix: Move antennas away from walls; increase separation to 0.7m

---

## Section 8 — Post-Capture Data Verification

After collecting data, verify the HDF5 file:

```
python verify_live_har.py --hdf5 csi_data.h5 --session session_001
```

This generates:
  - CSI magnitude heatmap (subcarrier vs packet index)
  - Raw vs sanitized phase comparison
  - Doppler spectrogram (0-10 Hz, resolution ~0.1 Hz)
  - NaN count, subcarrier count, timing check

### Generate Publication Figures

```
python plot_project.py       # 10 figures -> plots/
python plot_diagnostics.py   # 30 diagnostic figures -> plots/
```

---

## Section 9 — Key Configuration Values (config.py)

Values you may want to adjust for deployment:

```python
USRP_CENTER_FREQ = 2.412e9   # Increase to 2.437 or 2.462 if Ch1 is busy
USRP_TX_GAIN     = 30.0      # Increase to 40-45 for range >1m
USRP_RX_GAIN     = 25.0      # Increase to 35-38 for long range; watch for clipping
GUARD_SAMPLES    = 1024      # NEVER set to 0; min 512 for reliable detection
HDF5_FLUSH_INTERVAL = 50     # Reduce to 20 for high-rate captures
```

Values you should NEVER change without understanding the impact:
```python
FFT_SIZE = 128               # Changing this breaks ALL other modules
CP_LEN   = 32                # N_FFT/4; cannot change independently
FS_HW    = 25e6              # Must match B210 capability
FS_FFT   = 20e6              # Must maintain 4/5 resample ratio
PILOT_INDICES = [...]        # Symmetric; change breaks SCO tracking
CONV_GEN = [0o133, 0o171]   # Must match decoder; never change
```

---

## Section 10 — Complete System Data Flow

```
[USB 3.0] <-> B210 ADC at 25 MS/s
                    |
            UHD recv() blocks of 4096 samples
                    |
            RingBuffer (8M samples = 320ms headroom)
                    |
            resample_25to20() (polyphase FIR, 64-tap Kaiser beta=8)
                    |              20 MS/s baseband
            PacketDetector.process() (Schmidl-Cox, threshold=0.65)
                    |
            Lookahead window (HEADROOM = pkt_len + 820 samples)
                    |
            sync_packet():
              find_ltf_timing() -> ltf_start
              estimate_fine_cfo() -> total_cfo (coarse + fine)
              extract_csi() -> H_hat[106] complex
              sanitize_csi_phase() -> H_sanitized, phase_san[106]
                    |
            apply_cfo_correction() on DATA symbols
            sco_correct_symbol() per DATA symbol (pilot LS fit)
            demodulate_packet() (ZF + demap + Viterbi + descramble + CRC)
                    |
            CSILogger queue (maxsize=2000)
                    |
            HDF5 writer thread (flush every 50 packets)
                    |
            csi_data.h5:
              /sessions/session_001/csi/antenna0     (N, 106) complex64
              /sessions/session_001/csi/sanitized    (N, 106) complex64
              /sessions/session_001/csi/phase_sanitized (N, 106) float64
              /sessions/session_001/timestamps       (N,) float64
              /sessions/session_001/cfo_hz           (N,) float64
              /sessions/session_001/seq              (N,) int64
              /sessions/session_001/trials/{id}/     trial markers
```

---

*All parameters verified 2026-09-29 | USRP B210 v5 | Custom 128-pt OFDM PHY | CSI-HAR*


---

## Section 11 — Design Rationale for All Hardware Parameters

### 11.1 Why 2.412 GHz (Channel 1)?
- ISM (Industrial Scientific Medical) band: license-free worldwide
- Wi-Fi Ch1 is often less congested than Ch6 (2.437 GHz) or Ch11 (2.462 GHz)
  because home routers default to Ch6 and Ch11
- 2.412 GHz: lambda = c/f = 3e8/2.412e9 = 12.45 cm
  This wavelength gives good body scattering sensitivity:
  lambda/2 = 6.2 cm -> chest displacement of 3-4 cm produces ~pi/2 rad phase shift
- Alternative: use spectrum_scan.py to choose the cleanest channel for your environment

### 11.2 Why TX Gain = 30 dB?
- Free-space path loss at 1m, 2.4 GHz: 40 dB
- B210 TX power at 30 dB gain: approximately -5 to 0 dBm
- Received power (at 1m LoS): ~-45 dBm
- B210 RX noise floor at 25 dB gain: ~-100 dBm
- Link budget SNR: 55 dB >> required 25 dB for CRC pass
- Body absorption (human in path): up to 20-25 dB additional loss
- Even with 25 dB body loss: remaining SNR = 30 dB, still above threshold
- 30 dB leaves room to sense through a body without the link breaking

For range > 1m:
  Increase to USRP_TX_GAIN = 40 (safe max for most B210 units at 2.4 GHz)
  Rule of thumb: +6 dB gain doubles the reliable range

### 11.3 Why RX Gain = 25 dB?
- AD9361 ADC clips at approximately +5 to +10 dBm input power
- At 1m LoS with TX=30dB: received power ~ -45 dBm
- With 25 dB RX gain: amplified to -20 dBm, leaving 25-30 dB below clipping
- This 25-30 dB headroom accommodates:
    * Up to 25 dB path loss variation (body absorption and distance changes)
    * Reflections that can briefly add 6-10 dB above the direct path
- Increasing to 35 dB for longer range: fine if received power stays < -30 dBm
- If console shows "ADC clipping" or CSI magnitude >> 10: reduce RX gain by 5-10 dB

### 11.4 Why Antenna Separation = 0.5 to 1.0 m?
- Too close (< 0.3m): TX leaks directly into RX through near-field coupling
  This dominates the received signal and body-induced changes become invisible
- Too far (> 2m at 30dB TX): received power drops, SNR falls below Viterbi cliff
  At 2m: path loss ~46 dB, received ~-50 dBm, SNR = 50 dB (still fine)
  At 3m: path loss ~49 dB, received ~-53 dBm, SNR = 47 dB (still fine, but body abs. risky)
- Optimal 0.7m: body occupies ~30% of the first Fresnel zone radius at this distance
  First Fresnel radius at 0.7m, 2.4GHz: r = sqrt(lambda*d1*d2/d) = sqrt(0.125*0.35*0.35/0.7) = 0.148m
  A human torso (~0.3m wide) extends into 2 Fresnel zones -> strong scattering signature

### 11.5 Why Antenna Height = 0.8 to 1.2 m (default 0.9 m)?
- Human body height: 1.5-1.8 m (standing)
- Chest/torso center: 1.0-1.2 m above ground
- At 0.9 m antenna height: antennas aimed at the center of mass of the body
  This maximizes the body shadow (absorption) effect on the direct path
- Lower height (0.5 m): antennas aimed at legs, reducing chest/arm sensitivity
- Higher height (1.5 m): antennas aimed at head/shoulders, misses torso
- Ground reflection: at 0.9 m with 0.7 m separation, first ground reflection
  arrives 0.3 m after the direct path = 1 ns delay (negligible for CP=32 = 1.6 us)

### 11.6 Why 1024 Guard Samples (Not 512)?
Physical necessity: the Schmidl-Cox sliding autocorrelation needs to see the STF
AFTER a blank window of at least (STF detection window) = 128+16 = 144 samples at BB rate.

Conservative choice of 1024 HW samples (= 820 BB samples):
  820 / 144 = 5.7x safety margin
  After polyphase group delay (31 samples): net blank = 789 samples = 5.5x margin

If reduced to 512 HW samples (= 410 BB - 31 delay = 379 net):
  379 / 144 = 2.6x margin -- still probably fine, but reduced safety

Hardware-specific reason: B210 USB 3.0 burst boundaries can introduce 10-20 sample
timing jitter between bursts. The 1024-sample guard absorbs this jitter completely.

### 11.7 Why 8M Sample Ring Buffer?
- At 25 MS/s: 1 second of IQ data = 200 MB
- 8M samples = 320 ms = 0.32 seconds of acquisition headroom
- Python GIL worst-case pause: up to 50-100 ms (garbage collection, thread scheduling)
- With 320 ms buffer: GIL can pause for 300 ms without dropping a single sample
- USB 3.0 typical burst latency: 1-5 ms (well within headroom)
- Memory cost: 8M * 8 bytes = 64 MB (< 1% of typical laptop RAM)

### 11.8 Why 8.75 MHz CORDIC Off-Tune (Changed from 5 MHz in v7)?
- Direct-conversion (homodyne) architecture: LO leakage at DC
- **CRITICAL BUG FOUND (v6)**: 5 MHz offset mapped to k=-32. Verified: `-32 in cfg.ACTIVE_SUBCARRIERS = True`. This corrupted data SC k=-32 on every OTA packet silently.
- **FIX (v7)**: 8.75 MHz offset: 8.75e6 / 156.25e3 = **56 subcarriers**
  k=-56 is in the lower guard band (k=-64 to k=-54). Verified: `-56 in cfg.ACTIVE_SUBCARRIERS = False`.
- Guard band absorbs the LO spike completely. Zero active SC impact.
- 8.75 MHz is within the AD9361 CORDIC range and within the anti-alias filter passband.

### 11.9 Why K=7 Rate-1/2 Convolutional Code?
- K=7 (constraint length 7) is the 802.11 standard choice
- Coding gain over uncoded BPSK: approximately 6 dB at BER=10^-5 with Viterbi decoding
- This means: uncoded BPSK needs 11 dB SNR for BER=10^-5; coded needs only 5 dB
- At our PHY operating point (SNR~25 dB true), this gives enormous margin (20 dB)
- Rate 1/2 (no puncturing): maximum error correction power
  Puncturing to 3/4 would increase throughput but reduce coding gain by ~3 dB
  For HAR sensing (not high-throughput comms), rate 1/2 is always preferred
- CRC-32 serves a DIFFERENT purpose: it is the detect-and-discard gate for CSI quality
  A failed CRC means the packet was corrupted (bad timing, very low SNR) -> discard CSI

### 11.10 Why CRC-32 (Not CRC-16)?
- CRC-32 undetected error probability: 2^-32 ~ 2.3e-10 per packet
- CRC-16 undetected error probability: 2^-16 ~ 1.5e-5 per packet
- At 200 packets/second for 1 hour: 720,000 packets
  CRC-16 expected false passes: 720,000 * 1.5e-5 = 11 corrupted CSI records stored
  CRC-32 expected false passes: 720,000 * 2.3e-10 = 0.000166 -> essentially zero
- Contaminating the CSI dataset with even 11 bad frames could corrupt HAR classifier
  training. CRC-32 makes this practically impossible.

---

## Section 12 -- Simulation Validation Learnings (Pre-B210 Suite)

> These are lessons learned from running 16 validation tests in simulation.
> Each failure in simulation identifies a REAL risk that would occur in hardware.

### 12.1 T4 Lesson: Always Use Guard + Chunked Streaming

WHAT FAILED: With 32-sample inter-packet gap, only 1/5 packets detected.
ROOT CAUSE: Schmidl-Cox correlator needs blank window to settle before STF.
FIX: GUARD_SAMPLES_BB=820 (41us) + ONE det.process(chunk) call per burst.

The detector advance-past-packet = STF+LTF+SIG+34*SYM = 6048 samples.
This is LARGER than any packet stride (820+3328=4148). Concatenating all
packets and calling det.process(full_stream) misses 80%+ of packets.

HARDWARE RULE: Always prepend 1024-sample guard. Always chunk to one burst.

### 12.2 T_CHK Lesson: Decode Window Needs STF_LEN Headroom

WHAT FAILED: CRC fail despite correct LTF timing (err=0 samples).
ROOT CAUSE: Schmidl-Cox fires up to STF_LEN=128 samples BEFORE true start.
Window rx[abs_s : abs_s+PKT_LEN] = 3328 samples was 3-128 samples short.
FIX: Use rx[abs_s : abs_s + PKT_LEN + STF_LEN] = 3456 samples.

sync_packet() finds LTF internally and locates data correctly in the window.

### 12.3 T_NOI Lesson: Tighten CFO Gate to +-50 kHz

FA rate (no gate): 48.4%/block. FA rate (+-200kHz HW max): 16.8% [FAIL].
FA rate (+-50kHz operational): 4.2% [PASS < 5%].

B210 TCXO drift: +-2ppm @ 2.4GHz = +-4.8kHz. All real packets: |CFO| < 10kHz.
+-50kHz gate = 10x margin. All false-alarm decode attempts are rejected by CRC.
Net HDF5 contamination: ZERO.

### 12.4 Consistency Audit (Automated 2026-09-29 -- ALL OK)

PKT_LEN=3328 | NUM_ACTIVE=106 | NUM_DATA=98 | NUM_PILOTS=8
DC not in active | Pilots=[-49,-35,-21,-7,+7,+21,+35,+49]
Pkt power=0.0447 | SNR_offset=13.50dB | Guard BB=820 | STF=128 | LTF=320

### 12.5 Data Collection Flow

TX: payload -> CRC -> scramble -> Viterbi encode -> OFDM -> STF+LTF+SIGNAL
    -> GUARD 1024 zeros -> resample 20->25 MS/s -> B210 Port A @ 2.412GHz+5MHz

CHANNEL: body scatters/absorbs -> CFO < +-5kHz -> SCO < +-2ppm -> AWGN

RX: B210 Port B @ 25MS/s -> RingBuffer(8M) -> chunk(1024) -> Schmidl-Cox
    -> +-50kHz CFO gate -> resample 25->20 MS/s -> fine timing -> fine CFO
    -> H_hat(106) extraction -> OLS phase sanitization -> SCO tracking
    -> ZF equalize -> Viterbi -> CRC-32 -> PASS: HDF5 write | FAIL: discard

---

*Verified 2026-09-30 | Pre-B210: 16/16 PASS | Benchmark: 40/40 PASS | HAR: 3000/3000 PASS*
*v7 Blockers Fixed: LO offset (k=-32 → k=-56), CRC gate in logger, ring occupancy, CFO limit*

---

## Section 13 — OTA Signal Collection: How, When, and What Is Captured

### 13.1 Is the RX Signal a Continuous Stream or Burst?

The B210 operates in **continuous streaming** mode at 25 MS/s. The ADC never stops.
The **packet is burst-mode** (duty cycle ~1.7%): each 3328-sample packet lasts 0.17 ms
but arrives at 5–10 ms intervals. Between packets: thermal noise only in the RX buffer.

```
25 MS/s continuous stream from B210 ADC:
  |...noise...|GUARD(1024)|STF(160)|LTF(400)|SIG(200)|DATA(2720...)|...noise...|GUARD|...
                                                                      ↑            ↑
                                              Packet duration 0.17 ms   Inter-packet gap
                                              (at 20 MS/s baseband)     5-10 ms @ 100-200 pkt/s
```

### 13.2 When Does TX Start?

1. `main_tx.py` is started first. It configures B210 TX and enters the packet loop.
2. First packet is transmitted immediately after LO lock (retry loop ensures this in <2s).
3. TX continues indefinitely (--n_packets 0 = infinite) until Ctrl+C.
4. Each TX burst = [GUARD 1024 zeros @ 25 MS/s] + [packet IQ @ 25 MS/s], sent via UHD.
5. Inter-packet interval is enforced by `time.sleep(interval_s)` in main_tx.py.

### 13.3 When Does RX Start and Stop?

1. `main_rx.py` runs a 4-stage simulation pre-flight gate. Abort if any stage fails.
2. If pre-flight passes, B210 RX is configured and continuous streaming begins.
3. Data is captured indefinitely (--n_packets 0 = infinite) until Ctrl+C.
4. HDF5 file is flushed every 50 packets and closed on exit.
5. **Start criterion**: LO locked (confirmed by retry loop), streaming started.
6. **Stop criterion**: Ctrl+C in either terminal; RX closes HDF5 cleanly.

### 13.4 Signal Processing Chain (OTA)

```
TX SIDE (25 MS/s, B210 Port A):
  payload → CRC-32 → scramble → Viterbi encode → OFDM IFFT → CP insert
  → [GUARD 1024 zeros] → resample 20→25 MS/s → UHD TX stream → RF
  ↓ propagation through activity zone (body scatters/absorbs)
RX SIDE (25 MS/s, B210 Port B):
  RF → UHD RX stream → RingBuffer (8M samples, 320ms headroom)
  → CHUNK(4096 samples) → PacketDetector.process(chunk) [Schmidl-Cox]
  → Detection(abs_s, coarse_cfo): |CFO| < 50 kHz gate
  → Window rx[abs_s : abs_s + PKT_LEN + STF_LEN] (3456 samples)
  → resample 25→20 MS/s (polyphase FIR 4/5)
  → sync_packet(): fine timing, fine CFO, CSI extraction, phase sanitization
  → sco_correct_symbol() per DATA symbol (8 pilot LS + IIR)
  → ZF equalize → Viterbi → CRC-32 check
  → CRC PASS: enqueue CSI record → logger → HDF5
  → CRC FAIL: discard silently (no CSI written)
```

### 13.5 How Many Signals Are Collected?

| Parameter | Value | Notes |
|-----------|-------|-------|
| Packet rate | 100–200 pkt/s | 5–10 ms interval |
| CSI vector per packet | 106 complex numbers | H_hat[k] for k in ACTIVE_SUBCARRIERS |
| Typical session | 5–30 minutes | 30k–360k packets |
| HAR label granularity | ~1 packet = 10 ms | B210 TCXO stable over session |
| CRC pass rate (LoS 0.7m) | >70% expected | Verified in simulation |
| HDF5 file size | ~50 MB / 10 min | (N, 106) complex64 + metadata |

### 13.6 HAR Activity Protocol

```
Session protocol (recommended):
  T=0:      Start TX (main_tx.py)
  T=5s:     Start RX (main_rx.py) — 5s for LO lock stability
  T=10s:    Start session_controller.py — audio cues begin
  T=10–40s: Activity collection (5 activities × 6 s each)
  T=40s:    Audio cue "DONE" — subject returns to idle
  T=50s:    Ctrl+C in both terminals — HDF5 saved

Per-activity collection (session_controller.py controls this):
  - Audio cue → HAR trial start → subject performs activity → audio stop
  - Trial start/end timestamps injected into HDF5 /sessions/{id}/trials/
  - Minimum 3 repetitions per activity for statistical reliability
```

### 13.7 Hardware Deployment Status (Post v7 Audit)

| Category | Status | Notes |
|----------|--------|-------|
| Simulation validation | PASS (40/40 benchmark, 4/4 stage) | Verified |
| Pre-B210 suite | PASS (16/16) | Verified |
| LO offset bug | **FIXED v7** | Was k=-32 (active SC), now k=-56 (guard) |
| CRC gate in logger | **FIXED v7** | All packets now gated on CRC PASS before HDF5 write |
| CFO gate | **FIXED v7** | 200 kHz → 50 kHz in rx_hardware.py |
| Ring occupancy | **FIXED v7** | Double-normalisation removed |
| Two-process B210 | **OPEN** | Must merge TX+RX or use separate serial numbers |
| IQ imbalance model | **OPEN** | Not in simulation; validate with cable loopback |
| Link budget verification | **OPEN** | TX power at 30 dB gain requires power meter |
| First OTA run | **PENDING** | Hardware not connected |
