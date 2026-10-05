128-Point OFDM Integrated Sensing and Communication (ISAC)

Project README / Technical Handover Document

This project implements a custom 128-point OFDM-based Integrated Sensing and Communication (ISAC) system using a Python PHY and an NI/Ettus USRP B210.

Project Overview

This project implements a custom 128-point OFDM-based Integrated Sensing and Communication (ISAC) system using a Python PHY and an NI/Ettus USRP B210.

The central idea is to transmit a custom OFDM packet, receive it, recover the communication data, and simultaneously estimate the wireless channel from the known OFDM training field so that changes in the channel can subsequently be used for human sensing.

The same RF waveform therefore provides two functions:

Communication: packet detection, synchronization, CFO estimation/correction, SIGNAL decoding, DATA demodulation, FEC decoding, and CRC verification.

Sensing: LTF-based channel estimation followed by CSI processing and human-activity analysis.

The fundamental channel-estimation equation is

$$
\hat{H}[k]=\frac{Y[k]}{X[k]},
$$

where $X[k]$ is the known transmitted LTF subcarrier and $Y[k]$ is the received LTF subcarrier.

What This Project Is Not

This is not an IEEE 802.11/Wi-Fi PHY implementation.

Although the waveform is Wi-Fi-like in some design choices, the implementation is a custom OFDM PHY with a 128-point FFT, custom subcarrier allocation, custom pilot placement, custom packet structure, custom synchronization, a custom SIGNAL field, and custom CSI extraction.

Use the terminology:

Custom 128-point OFDM PHY

or

Wi-Fi-like custom OFDM waveform

rather than ``IEEE 802.11 implementation.''

Main Project Objective

The project is intended to build a complete experimental pipeline in which:

a programmable OFDM communication waveform is generated;

it is transmitted using a USRP B210;

the receiver detects and synchronizes the packet;

the receiver recovers the communication payload;

the receiver simultaneously extracts CSI from the LTF;

CSI is timestamped and stored;

human-induced CSI variations can subsequently be analyzed for activity sensing.

The overall direction is:

Robust OFDM-based ISAC where communication data recovery and human-channel sensing are performed simultaneously from the same packet stream.

Overall System Architecture

config.py
                       |
                       v
              +-----------------+
              |   waveform.py   |
              |    TX PHY       |
              +--------+--------+
                       |
                       v
                Custom OFDM packet
                       |
                       v
              +-----------------+
              |     B210 TX     |
              +--------+--------+
                       |
                       v
                  RF CHANNEL
                       |
                       v
              +-----------------+
              |     B210 RX     |
              +--------+--------+
                       |
                       v
              Streaming resampler
                       |
                       v
              +-----------------+
              |   detector.py   |
              | Packet detection|
              +--------+--------+
                       |
                       v
              +-----------------+
              |     sync.py     |
              | Timing + CFO    |
              | CSI estimation  |
              +--------+--------+
                       |
                       v
              +-----------------+
              |    demod.py     |
              | SIGNAL + DATA   |
              | FEC + CRC       |
              +--------+--------+
                       |
              +--------+--------+
              |                 |
              v                 v
       Communication           CSI
          result                |
                                v
                         Phase processing
                                |
                                v
                          HDF5 logger
                                |
                                v
                         HAR / ISAC analysis

Hardware Architecture

The final hardware architecture is based on one B210 device and one MultiUSRP handle:

ONE MultiUSRP()
                       |
              +--------+--------+
              |                 |
         TX streamer       RX streamer
              |                 |
              v                 v
           B210 TX           B210 RX

main_hardware.py is the main hardware entry point and owns both the TX and RX streamers.

Experimental Modes

Mode A: Conducted Communication Test

A controlled RF/cable/attenuator path can initially be used:

B210 TX
   |
controlled attenuation / RF path
   |
B210 RX

This validates transmission, detection, synchronization, CFO estimation, channel estimation, demodulation, CRC, and logging.

A direct conducted connection does not demonstrate human sensing.

Mode B: Actual ISAC / Human Sensing

For sensing, the human must influence the RF propagation environment:

Human
                   |
                   | movement
                   v
B210 TX ---- RF propagation ---- B210 RX
                    |
                    v
                 CSI change

Current OFDM Numerology

Parameter

Current value

FFT size

128

CP length

32 samples

OFDM symbol length

160 samples

Baseband sample rate

20 MS/s

Hardware sample rate

25 MS/s

TX resampling

20 $arrow$ 25 MS/s

RX resampling

25 $arrow$ 20 MS/s

Subcarrier spacing

156.25 kHz

Active subcarriers

106

Pilot subcarriers

8

Data subcarriers

98

DC

nulled

The subcarrier spacing is

$$
\Delta f=\frac{20\times10^6}{128}=156.25~\mathrm{kHz}.
$$

An OFDM symbol contains

$$
128+32=160
$$

samples, giving

$$
T_{\mathrm{sym}}=\frac{160}{20\times10^6}=8~\mu\mathrm{s}.
$$

Subcarrier Allocation

The centered FFT indices are

$$
k\in{-64,...,63}.
$$

The active subcarriers are

$$
{-53,...,-1}\cup{1,...,53}.
$$

Therefore there are

$$
53+53=106
$$

active subcarriers.

The DC subcarrier $k=0$ is unused.

The eight pilot subcarriers are

$$
{-49,-35,-21,-7,7,21,35,49}.
$$

Therefore:

$$
106-8=98
$$

data subcarriers.

Source-of-truth rule: the current config.py defines 106 active and 98 data subcarriers. If older documents contain different values, use the current implementation.

Packet Structure

+------------+
|    STF     |
+------------+
|    LTF     |
+------------+
|   SIGNAL   |
+------------+
|    DATA    |
| DATA x N   |
+------------+

Field

Length

STF

128 samples

LTF

320 samples

SIGNAL

160 samples

DATA

$160N_{\mathrm{sym}}$ samples

The total baseband packet length is

$$
N_{\mathrm{pkt}}=128+320+160+160N_{\mathrm{sym}}.
$$

STF

The Short Training Field is used primarily for packet detection and coarse synchronization.

The detector operates on a continuous stream rather than isolated packet buffers. It therefore maintains streaming state and uses bounded packet processing and suppression after detection.

LTF

The Long Training Field supports both communication and sensing.

The current LTF is:

64-sample CP
+
128-sample LTF
+
128-sample LTF
=
320 samples

Therefore:

$$
T_{\mathrm{LTF}}=\frac{320}{20\times10^6}=16~\mu\mathrm{s}.
$$

The receiver knows the transmitted LTF, allowing

$$
\hat H[k]=\frac{Y[k]}{X[k]}.
$$

The LTF is therefore the bridge between communication channel equalization and sensing.

SIGNAL Field

The SIGNAL field tells the receiver how to interpret the DATA field.

The current implementation dynamically determines modulation, payload length, and the required number of DATA symbols.

Supported modulations:

BPSK

QPSK

16QAM

The current modulation mapping is:

Modulation

SIGNAL rate bits

BPSK

1011

QPSK

0101

16QAM

1101

The receiver must not assume a fixed payload length or modulation.

DATA Field

Modulation

Bits/subcarrier

BPSK

1

QPSK

2

16QAM

4

There are 98 data subcarriers and the coding rate is

$$
R=\frac12.
$$

The communication chain includes scrambling, convolutional coding, modulation, OFDM mapping, and the corresponding inverse operations at the receiver.

CRC

A decoded packet is considered valid only after the communication decoding chain and CRC verification succeed.

Detection
   |
Synchronization
   |
CFO correction
   |
SIGNAL decoding
   |
DATA equalization
   |
Demapping
   |
Viterbi
   |
Descrambling
   |
CRC
   |
Valid packet

Payload Capacity

With 98 data subcarriers and a maximum of 34 DATA symbols:

Modulation

Maximum payload

BPSK

204 bytes

QPSK

412 bytes

16QAM

829 bytes

The number of required DATA symbols is

\lceil
\frac{2(N_{\mathrm{bytes}}+4)8}{98 b}
\rceil,
$$

where $b$ is the number of bits per subcarrier and the 4-byte term represents CRC overhead.

Example: 64-Byte BPSK Packet

For a 64-byte payload:

$$
64\times8+32=544
$$

wire bits are produced.

After rate-$1/2$ coding:

$$
544\times2=1088
$$

coded bits are required.

Therefore:

$$
N_{\mathrm{sym}}=\lceil\frac{1088}{98}\rceil=12.
$$

The packet length is

$$
128+320+160+(12\times160)=2528
$$

baseband samples.

At 20 MS/s:

$$
T_{\mathrm{pkt}}=\frac{2528}{20\times10^6}=126.4~\mu\mathrm{s}.
$$

At the 25 MS/s hardware rate:

$$
2528\times\frac54=3160
$$

hardware samples.

Hardware/Baseband Resampling

The B210 hardware operates at 25 MS/s while the internal PHY operates at 20 MS/s.

RX:

$$
25arrow20
$$

using $\frac45$.

TX:

$$
20arrow25
$$

using $\frac54$.

The implementation uses a stateful streaming rational resampler. FIR history and rational phase persist between blocks.

Packet Timing

The target packet start-to-start period is

$$
T_{\mathrm{packet}}=5~\mathrm{ms}.
$$

Therefore:

$$
f_{\mathrm{packet}}=\frac1{0.005}=200~\mathrm{Hz}.
$$

At 20 MS/s:

$$
20\times10^6\times0.005=100000
$$

baseband samples occur per 5-ms interval.

Thus the target CSI sampling rate is approximately 200 Hz and the corresponding Nyquist frequency is approximately 100 Hz.

Packet Scheduling

The hardware TX uses UHD timed transmission rather than simply sleeping after each transmission.

T0
|
+-- packet 1
|
+-------- 5 ms --------> packet 2
|
+-------- 5 ms --------> packet 3
|
+-------- 5 ms --------> packet 4

Absolute Sample Coordinate

Every decoded hardware packet has an absolute baseband sample coordinate abs_s.

For example:

packet 1 -> abs_s = 997
packet 2 -> abs_s = 100997
packet 3 -> abs_s = 200997

The spacing is

$$
100997-997=100000
$$

BB samples, corresponding to

$$
\frac{100000}{20\times10^6}=5~\mathrm{ms}.
$$

Timestamp Definition

The packet timestamp is based on the UHD timestamp of the first received hardware block:

t_0+\frac{abs_s}{F_{S,\mathrm{BB}}},
$$

where

$$
F_{S,\mathrm{BB}}=20~\mathrm{MHz}.
$$

This uses the SDR sample-time basis rather than Python processing time.

\section{seq Versus abs_s}

seq is the local RX packet record index:

seq = 0
seq = 1
seq = 2
seq = 3

It is not a transmitter sequence number.

abs_s is the absolute BB sample coordinate.

The timestamp represents the corresponding UHD-derived sample time.

Thus:

seq       -> record index
abs_s     -> sample-stream position
timestamp -> RF sample time

Packet Detection

The receiver continuously processes samples and does not know packet boundaries in advance.

The detector uses the STF correlation metric to identify candidate packet starts.

The current detector includes persistent streaming state, bounded packet-length advancement, packet suppression, and a defined maximum number of DATA symbols.

Look-Ahead Processing

RX data arrives in chunks and a packet may cross a chunk boundary.

The receiver therefore maintains a look-ahead region.

An index inside a temporary buffer is not automatically an absolute sample coordinate. The implementation explicitly accounts for look-ahead when converting buffer indices into absolute BB coordinates.

Synchronization Chain

STF detection
     |
coarse timing / CFO information
     |
LTF timing
     |
fine CFO
     |
CFO correction
     |
LTF channel estimation
     |
SIGNAL decoding
     |
DATA decoding

The receiver also contains sampling-clock-offset correction at the symbol level.

CFO

Carrier Frequency Offset is estimated from known packet structure.

The hardware path currently uses an acceptance gate around

$$
\pm50~\mathrm{kHz}.
$$

This is a runtime acceptance criterion, not a statement that the physical hardware can never experience a larger offset.

CSI Extraction

For every valid packet, the receiver obtains a complex channel estimate:

$$
\hat{\mathbf H}\in\mathbb C^{106}.
$$

Conceptually:

H_hat =
[
 H[-53], ..., H[-1],
 H[1], ..., H[53]
]

with DC omitted.

Why CSI Can Be Used for Sensing

A simplified multipath model is

$$
H(f,t)=\sum_i a_i(t)e^{-j2\pi f\tau_i(t)}.
$$

Human movement can change path amplitudes and delays. Consequently, $H[k,t]$ contains temporal information about changes in the propagation environment.

Phase Processing

Raw CSI phase can contain unwanted components associated with synchronization and hardware effects.

The project stores raw CSI and processed phase information:

csi/antenna0
csi/sanitized
csi/phase_sanitized

The phase-sanitization processing removes an estimated linear phase trend across subcarriers.

Communication and Sensing from One Waveform

Same OFDM packet
                        |
              +---------+---------+
              |                   |
              v                   v
       Communication          Sensing
              |                   |
       Payload recovery       LTF -> CSI
              |                   |
              v                   v
             CRC             phase/amplitude
                                  |
                                  v
                          activity information

Current Hardware Parameters

Parameter

Current value

Device

NI/Ettus B210

Center frequency

2.412 GHz

TX sample rate

25 MS/s

RX sample rate

25 MS/s

Internal BB rate

20 MS/s

TX gain

30 dB candidate

RX gain

25 dB candidate

LO offset

8.75 MHz candidate

TX/RX architecture

One MultiUSRP

Packet period

5 ms target

Packet rate

200 Hz target

The RF frequency, gain and LO-offset values are initial deployment candidates and still require physical hardware characterization.

LO Offset

The current LO offset is 8.75 MHz. Relative to the subcarrier spacing:

$$
\frac{8.75~\mathrm{MHz}}{156.25~\mathrm{kHz}}=56.
$$

Thus the offset corresponds to approximately 56 subcarrier spacings. The current active region ends at $\pm53$, so the nominal offset lies outside the active region.

HDF5 Data Organization

The logger stores data under:

/sessions/<session_id>/

The important datasets are:

/sessions/session_001/
    csi/
        antenna0
        sanitized
        phase_sanitized
    timestamps
    abs_s
    seq
    cfo_hz

The row-alignment rule is:

CSI[i]
timestamp[i]
abs_s[i]
seq[i]
cfo_hz[i]

which all refer to the same accepted packet record.

Meaning of the Main HDF5 Datasets

csi/antenna0 — Raw complex channel estimate. For the current one-RX configuration its shape is $(N_{\mathrm{packets}},106)$.

csi/sanitized — Processed/sanitized complex CSI.

csi/phase_sanitized — Sanitized/detrended CSI phase in radians.

timestamps — UHD-derived packet timestamps.

abs_s — Absolute BB sample coordinate, in 20-MS/s baseband samples.

seq — RX-side packet record number.

cfo_hz — Estimated carrier-frequency offset in Hz.

Main Source-of-Truth Files

A new team member should focus on these files:

config.py

waveform.py

detector.py

sync.py

demod.py

rx_hardware.py

main_hardware.py

logger.py

\subsection{config.py --- First File}

This is the primary source of truth for implemented PHY numerology and major runtime parameters, including FFT size, CP, sample rates, subcarriers, pilots, payload limits, hardware settings, and packet timing.

\subsection{waveform.py --- TX}

This answers: ``How exactly is the OFDM waveform generated?''

Conceptually:

payload
 |
CRC
 |
scrambling
 |
FEC
 |
interleaving
 |
modulation
 |
OFDM mapping
 |
pilots
 |
IFFT
 |
CP
 |
STF/LTF/SIGNAL/DATA
 |
complete packet

\subsection{demod.py --- RX Data Recovery}

This handles SIGNAL decoding, modulation interpretation, payload length, DATA equalization, demapping, Viterbi decoding, descrambling, and CRC.

\subsection{scrambler.py}

Implements packet bit scrambling. Read together with waveform.py and demod.py.

\subsection{encoder.py}

Contains convolutional coding functionality used by the communication chain.

\subsection{sync.py}

One of the most important RX files. It covers synchronization, CFO estimation/correction, LTF processing, channel estimation, CSI extraction, and sampling-clock-offset correction.

\subsection{detector.py}

Handles packet detection in the continuous RX stream.

\subsection{main_hardware.py --- Final Hardware Entry Point}

This is the primary hardware deployment file. It combines the B210, TX/RX streamers, TX scheduling, RX acquisition, detection, synchronization, demodulation, CSI extraction, and logging.

\subsection{rx_hardware.py}

Contains hardware-side RX support including ring buffering, hardware block handling, streaming support, and packet-window sizing.

\subsection{logger.py}

The data-storage layer. It receives valid packet/CSI records and writes them to HDF5.

\subsection{channel_bridge.py}

Primarily for software channel simulation. It should not be confused with the physical B210 RF channel.

Recommended File Reading Order

For a new team member:

README.md

config.py

waveform.py

scrambler.py

encoder.py

detector.py

sync.py

demod.py

rx_hardware.py

main_hardware.py

logger.py

There is no need to start by reading every validation script.

Simulation Versus Hardware

Software simulation:

waveform
   |
channel_bridge
   |
detector
   |
sync
   |
demod
   |
CSI

Hardware:

waveform
   |
B210 TX
   |
RF channel
   |
B210 RX
   |
streaming resampler
   |
detector
   |
sync
   |
demod
   |
CSI
   |
HDF5

PHY Versus HAR

The project has two layers.

Layer 1: PHY/ISAC Acquisition

This includes OFDM generation, transmission, reception, synchronization, communication decoding, CSI extraction, timestamping, and storage.

Layer 2: Human Activity Recognition

This includes CSI preprocessing, feature extraction, temporal analysis, and activity classification.

The PHY acquisition layer must be reliable before the HAR layer can be trusted.

Potential Sensing Pipeline

Raw CSI
   |
H_hat[k,n]
   |
Phase extraction
   |
Phase sanitization / detrending
   |
Temporal signal
   |
+-------------------------------+
| amplitude features            |
| phase features                |
| spectral features             |
| Doppler-related features      |
| temporal statistics           |
+-------------------------------+
               |
               v
        Activity model

The exact HAR classifier is a downstream research component separate from the core OFDM PHY.

Validation and Hardware Status

The PHY has undergone substantial software validation covering waveform generation, streaming resampling, packet detection, detector indexing, packet suppression, dynamic payload length, dynamic modulation, packet boundaries, back-to-back packets, CFO estimation, noise stress testing, packet timing, UHD timestamp mapping, absolute sample coordinates, and HDF5 abs_s persistence.

However:

Software validation is not the same as successful RF or human-sensing hardware validation.

The B210 hardware path still requires physical characterization and experiments. Human activity sensing should not be claimed as experimentally demonstrated until the corresponding hardware experiment has actually been performed.

Hardware Deployment Ladder

Software PHY

Streaming simulation

B210 conducted communication

Static OTA

Human stationary

Gross human movement

Micro-activity / micro-motion

Important Numerical Values to Know

FFT                = 128
CP                  = 32
OFDM symbol        = 160 samples
BB sample rate     = 20 MS/s
HW sample rate     = 25 MS/s

Subcarrier spacing = 156.25 kHz

Active SCs         = 106
Pilots             = 8
Data SCs           = 98
DC                 = 0

STF                = 128 samples
LTF                = 320 samples
SIGNAL             = 160 samples

Supported modulation:
    BPSK
    QPSK
    16QAM

Coding rate        = 1/2

Maximum DATA symbols = 34

Maximum payload:
    BPSK   = 204 B
    QPSK   = 412 B
    16QAM  = 829 B

Target packet period = 5 ms
Target packet rate   = 200 Hz

BB samples / 5 ms   = 100000

B210 center freq    = 2.412 GHz
LO offset           = 8.75 MHz
TX gain candidate   = 30 dB
RX gain candidate   = 25 dB

Core Conceptual Picture

CUSTOM 128-POINT OFDM
                                  |
                                  v
                         +-----------------+
                         |   STF / LTF     |
                         | SIGNAL / DATA   |
                         +--------+--------+
                                  |
                                  v
                              B210 TX
                                  |
                                  v
                           RF PROPAGATION
                                  |
                         +--------+--------+
                         |                 |
                    communication      human/environment
                    information          interaction
                         |                 |
                         v                 v
                      B210 RX          CSI changes
                         |                 |
                         +--------+--------+
                                  v
                         Packet processing
                                  |
              +-------------------+-------------------+
              |                                       |
              v                                       v
       DATA recovery                               CSI
              |                                       |
              v                                       v
            CRC                              phase/amplitude
                                                      |
                                                      v
                                            activity information

One-Sentence Project Description

We are building a custom 128-point OFDM-based ISAC system on a USRP B210 in which the same OFDM packets provide reliable communication and per-subcarrier CSI, allowing the received channel variations to be subsequently analyzed for human activity sensing.

Short Project Introduction

Our project is a custom 128-point OFDM-based ISAC system. We generate our own OFDM waveform instead of using standard Wi-Fi. A B210 transmits and receives the packets, and the receiver performs packet detection, synchronization, CFO correction, SIGNAL/DATA decoding and CRC verification. At the same time, the known LTF allows us to estimate the complex channel response for every active subcarrier. Because human movement changes the wireless propagation channel, those CSI variations can be processed to detect human activities. Thus communication and sensing are obtained simultaneously from the same RF waveform.

Source-of-Truth Hierarchy

When files disagree, use this hierarchy:

config.py --- numerical PHY configuration.

waveform.py, detector.py, sync.py, demod.py --- actual PHY behavior.

main_hardware.py, rx_hardware.py, logger.py --- hardware and data-acquisition behavior.

Documentation --- describes the implementation and should be updated when implementation changes.

Do not change PHY numbers in random files. Check the source-of-truth configuration first and update dependent components consistently.

Final Handover Summary

The core chain is:

config.py
      |
      v
waveform.py
      |
      v
detector.py / sync.py / demod.py
      |
      v
main_hardware.py
      |
      v
logger.py

Understanding this chain is sufficient for a new team member to understand the principal architecture without reading the entire repository.
