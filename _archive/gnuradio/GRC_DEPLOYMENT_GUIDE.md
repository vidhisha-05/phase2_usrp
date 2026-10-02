# GNU Radio Deployment Guide
## Custom 128-pt OFDM PHY — Python Blocks in GNU Radio Companion

---

## FILE STRUCTURE

```
d:\phase2\
  gnuradio\
    gr_block_tx.py              <- BLOCK 1: OFDM packet generator (source)
    gr_block_channel.py         <- BLOCK 2: AWGN+CFO+SCO+multipath
    gr_block_detector.py        <- BLOCK 3: STF Schmidl&Cox detector (tagged)
    gr_block_csi_extractor.py   <- BLOCK 4: LTF sync + 2-stage CFO + H_hat
    gr_block_csi_logger.py      <- BLOCK 5: HDF5 writer (background thread)
    gr_block_csi_monitor.py     <- BLOCK 6: Live ASCII CSI display
    gr_flowgraph.py             <- MASTER: wires all 6 blocks, runnable directly
    ofdm_csi_flowgraph.grc      <- GRC YAML: open in GNU Radio Companion GUI
```

---

## FLOWGRAPH DIAGRAM

```
+------------------+     complex64 IQ (20 MS/s)
|  OFDM TX Block   |----->+----------------------+
|  (gr_block_tx)   |      | Channel Impairment   |
|  STF+LTF+SIG+DAT |      | AWGN + CFO + SCO     |
+------------------+      | + Multipath          |
                          +----------+-----------+
                                     |
                                     | complex64 (impaired)
                                     v
                          +----------+-----------+
                          | STF Packet Detector  |
                          | Schmidl & Cox        |
                          | Attaches stream tags:|
                          | key="pkt_start"      |
                          | val=coarse_cfo (Hz)  |
                          +----------+-----------+
                                     |
                                     | complex64 TAGGED stream
                                     v
                          +----------+-----------+
                          | CSI Extractor        |
                          | Read pkt_start tags  |
                          | LTF cross-corr timing|
                          | 2-stage CFO correct  |
                          | H_hat[107] via FFT   |
                          | SCO pilot tracking   |
                          +----+------------+----+
                               |            |
                     PMT dict  |            | PMT dict
                     csi_out   |            | csi_out
                               |            |
                    +----------+--+    +----+------------+
                    | HDF5 Logger |    | CSI Monitor     |
                    | Background  |    | Terminal display |
                    | thread write|    | ASCII |H_hat|   |
                    | sessions/   |    | bar chart       |
                    | .../csi/    |    | per 5 packets   |
                    | antenna0    |    +------------------+
                    | shape(N,107)|
                    +-------------+
```

---

## METHOD 1 — Run Directly as Python (No GRC GUI Needed)

### Step 1: Install GNU Radio

**Linux (Ubuntu 20.04+)**:
```bash
sudo apt install gnuradio python3-gnuradio
# Verify:
python3 -c "import gnuradio; print(gnuradio.__version__)"
# Expected: 3.10.x.x
```

**Windows (Radioconda)**:
```powershell
# Download from https://github.com/ryanvolz/radioconda
# Install, then open Radioconda terminal:
conda activate radioconda
python -c "import gnuradio; print(gnuradio.__version__)"
```

### Step 2: Run the flowgraph

```bash
cd d:\phase2\gnuradio

# Zero-impairment baseline:
python gr_flowgraph.py

# With CFO + moderate AWGN:
python gr_flowgraph.py --noise 0.02 --cfo 1500 --sco 2.0

# Two-tap multipath:
python gr_flowgraph.py --taps "1+0j,0.3-0.1j" --noise 0.01

# Stop after 200 packets:
python gr_flowgraph.py --n_packets 200 --hdf5 test_200.h5

# Hardware mode (B210):
python gr_flowgraph.py --mode hardware --freq 2.462e9 --rx_gain 38
```

### What to expect on terminal:

```
============================================================
  Custom 128-pt OFDM PHY -- GNU Radio Flowgraph
============================================================
  Mode         : simulation
  Noise voltage: 0.005
  CFO inject   : 500.0 Hz
  SCO inject   : 1.0 ppm
  HDF5 output  : gr_csi_output.h5
  Session ID   : gr_session_001
============================================================

  FLOWGRAPH:
  [TX] -> [Channel] -> [Detector] -> [CSI Extractor]
                                           |
                              +------------+------------+
                              |                         |
                        [HDF5 Logger]           [CSI Monitor]
                        gr_csi_output.h5          (terminal)

  Starting flowgraph. Press Ctrl+C to stop.

[HDF5 Logger] Open: gr_csi_output.h5  session=gr_session_001
  pkt#    5  cfo=   537.2Hz  |H|=[.:-=+#@$...::-=++##....]  mean=0.997  rate=98.3/s
  pkt#   10  cfo=   537.2Hz  |H|=[.:-=+#@$...::-=++##....]  mean=1.002  rate=99.1/s
[HDF5 Logger] Flushed 50 pkts  total=50
  pkt#   15  cfo=   537.2Hz  |H|=[.:-=+#@$...::-=++##....]  mean=0.999  rate=99.4/s
  ...
```

---

## METHOD 2 — Open in GNU Radio Companion (GRC) GUI

### Step 1: Launch GRC
```bash
gnuradio-companion
# OR on Windows Radioconda:
gnuradio-companion.exe
```

### Step 2: Open the flowgraph
```
File -> Open -> d:\phase2\gnuradio\ofdm_csi_flowgraph.grc
```

### Step 3: How blocks appear in GRC

```
+---------------------------+
|  Custom OFDM TX (128-pt)  |  <- BLOCK 1: Orange source block
|  mod_scheme: BPSK         |     Right-click -> Properties to change
|  payload_bytes: 60        |     mod/interval/n_packets
|  packet_interval_s: 0.01  |
+------------+--------------+
             | out0 (complex64)
             v
+---------------------------+
|  Channel Impairment       |  <- BLOCK 2: Blue through-block
|  noise_voltage: 0.005     |     Edit noise/cfo/sco/taps in Properties
|  cfo_hz: 500.0            |
|  sco_ppm: 1.0             |
|  tap_string: "1+0j"       |
+------------+--------------+
             | out0 (complex64)
             v
+---------------------------+
|  STF Packet Detector      |  <- BLOCK 3: Blue through-block
|  detect_threshold: 0.65   |     Tagged stream: adds "pkt_start" tags
|  corr_window: 16          |
+------------+--------------+
             | out0 (complex64 tagged)
             v
+---------------------------+
|  CSI Extractor (128-pt)   |  <- BLOCK 4: Purple sink + message source
|  n_data_symbols: 0        |     Message output: "csi_out" (PMT dict)
+-----+-------------------+-+
      |                   |
      | csi_out    csi_out|
      v                   v
+-------------+    +--------------+
| HDF5 Logger |    | CSI Monitor  |  <- BLOCKS 5 & 6: Green sinks
|  csi_in     |    |  csi_in      |
+-------------+    +--------------+
```

### Step 4: Edit parameters in GRC

Double-click any block to open Properties. The key variables at the top of the flowgraph can also be edited directly:

| Variable | Default | What it changes |
|---|---|---|
| `noise_voltage` | 0.005 | AWGN power; 0.001 = quiet, 0.1 = very noisy |
| `cfo_hz` | 500.0 | Injected frequency offset (Hz) |
| `sco_ppm` | 1.0 | Clock offset (ppm) |
| `tap_string` | "1+0j" | Multipath taps (complex, comma-separated) |
| `hdf5_path` | "gr_csi_output.h5" | Output file |
| `session_id` | "gr_session_001" | HDF5 group name |
| `n_packets` | 0 | Stop count (0 = run forever) |

### Step 5: Click RUN (or F6)

Watch the terminal below GRC for live CSI output.

---

## METHOD 3 — Use Each Block as an Embedded Python Block in GRC

If you want to wire blocks yourself in GRC from scratch:

### How to add an Embedded Python Block

1. In GRC: **Blocks menu -> Misc -> Python Block**
2. Double-click the block -> **Open in Editor**
3. **Paste the contents** of the corresponding `gr_block_*.py` file
4. Set the block parameters to match

### Block-by-block instructions:

#### BLOCK 1 — TX Source

```
In GRC: Add -> Python Block (Source)
Paste: contents of gr_block_tx.py
Block settings:
  Name: "Custom OFDM TX"
  Output ports: 1 x complex float32
  Parameters to expose in GRC UI:
    mod_scheme        (string, default "BPSK")
    payload_bytes     (int, default 60)
    packet_interval_s (float, default 0.01)
    n_packets         (int, default 0)
```

#### BLOCK 2 — Channel Impairment

```
In GRC: Add -> Python Block (Through)
Paste: contents of gr_block_channel.py
Block settings:
  Name: "Channel Impairment"
  Input ports:  1 x complex float32
  Output ports: 1 x complex float32
  Parameters:
    noise_voltage (float, default 0.005)
    cfo_hz        (float, default 0.0)
    sco_ppm       (float, default 0.0)
    tap_string    (string, default "1+0j")
```

#### BLOCK 3 — STF Detector

```
In GRC: Add -> Python Block (Through)
Paste: contents of gr_block_detector.py
Block settings:
  Name: "STF Packet Detector"
  Input ports:  1 x complex float32
  Output ports: 1 x complex float32 (tagged)
  Parameters:
    detect_threshold (float, default 0.65)
    corr_window      (int, default 16)
```

#### BLOCK 4 — CSI Extractor

```
In GRC: Add -> Python Block (Sink + Message Source)
Paste: contents of gr_block_csi_extractor.py
Block settings:
  Name: "CSI Extractor (128-pt OFDM)"
  Input ports:  1 x complex float32
  Output ports: none (sample stream)
  Message ports OUT: "csi_out"
  Parameters:
    n_data_symbols (int, default 0)
```

#### BLOCK 5 — HDF5 Logger

```
In GRC: Add -> Python Block (Message Sink)
Paste: contents of gr_block_csi_logger.py
Block settings:
  Name: "HDF5 CSI Logger"
  Input ports:  none
  Message ports IN: "csi_in"
  Parameters:
    hdf5_path      (string, default "gr_csi_output.h5")
    session_id     (string, default "gr_session_001")
    n_rx_channels  (int, default 1)
    flush_interval (int, default 50)
```

#### BLOCK 6 — CSI Monitor

```
In GRC: Add -> Python Block (Message Sink)
Paste: contents of gr_block_csi_monitor.py
Block settings:
  Name: "CSI Monitor (Live)"
  Input ports:  none
  Message ports IN: "csi_in"
  Parameters:
    update_every_n (int, default 5)
```

### Wiring in GRC

After placing all blocks:

1. **Sample stream wires** (grey lines in GRC):
   - TX `out0` -> Channel `in0`
   - Channel `out0` -> Detector `in0`
   - Detector `out0` -> CSI Extractor `in0`

2. **Message wires** (orange lines in GRC):
   - CSI Extractor `csi_out` -> Logger `csi_in`
   - CSI Extractor `csi_out` -> Monitor `csi_in`

---

## READING THE HDF5 OUTPUT

After the flowgraph runs:

```python
import h5py, numpy as np

with h5py.File("gr_csi_output.h5", "r") as f:
    H = f["/sessions/gr_session_001/csi/antenna0"][:]

print(f"Shape:       {H.shape}")         # (N_packets, 107)
print(f"Mean |H|:    {np.mean(np.abs(H)):.4f}")
print(f"Phase std:   {np.mean(np.std(np.angle(H), axis=0)):.4f} rad")

# Plot amplitude spectrum (one packet)
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))
plt.plot(np.abs(H[0]), marker='.')
plt.xlabel("Subcarrier index (0-106)")
plt.ylabel("|H_hat|")
plt.title("CSI Amplitude — Packet 0")
plt.grid(True)
plt.tight_layout()
plt.savefig("csi_spectrum.png")
plt.show()
```

---

## PARALLEL VALIDATION WHILE FLOWGRAPH IS RUNNING

In a second terminal while `gr_flowgraph.py` is running:

```bash
# Watch HDF5 file grow in real time:
python -c "
import h5py, time, os
path = 'gr_csi_output.h5'
while True:
    if os.path.exists(path):
        try:
            with h5py.File(path, 'r') as f:
                s = f['/sessions/gr_session_001/csi/antenna0'].shape
                print(f'  Logged: {s[0]} packets  shape={s}', end='\r')
        except: pass
    time.sleep(1)
"
```

---

## TROUBLESHOOTING

| Problem | Cause | Fix |
|---|---|---|
| `ImportError: gnuradio` | GNU Radio not installed | `sudo apt install gnuradio` |
| `ImportError: pmt` | Old GNU Radio (<3.8) | Upgrade to GRC 3.10 |
| No CSI records produced | Detector threshold too high | Lower `detect_threshold` to 0.55 |
| `pmt.is_c32vector` error | PMT API changed | Use `pmt.is_uniform_vector` + `pmt.f32vector_elements` |
| HDF5 file never grows | Logger queue blocked | Increase `flush_interval` to 20 |
| GRC `.grc` YAML fails to load | GRC version mismatch | Run `python gr_flowgraph.py` instead |
| Hardware: no UHD source in GRC | gr-uhd not installed | `sudo apt install gr-uhd` |
| Hardware: LO not locked | Frequency out of range | B210 range: 70 MHz - 6 GHz |

---

## IMPORTANT REMINDER

> `gr-ieee802-11` is **NEVER used** anywhere in this system.
> All blocks use custom 128-FFT logic from the project's own Python modules.
> The waveform is a custom OFDM PHY, NOT IEEE 802.11a/g.
