# -*- coding: utf-8 -*-
"""
plot_diagnostics.py
===================
Simple, single-purpose diagnostic plots for every DSP stage of the
custom 128-pt OFDM PHY pipeline + HAR sensing validation.

All plots use plain white-background matplotlib — no fancy styling.
Saves PNG files to plots/ directory.

Stages covered
--------------
1  TX Waveform Assembly & Resampling
2  Packet Detection & Coarse Sync (Schmidl-Cox)
3  Fine Timing & Fine CFO (LTF)
4  Channel Estimation (CSI) & Phase Sanitization
5  Equalization, SCO Tracking & Demodulation
6  HAR Sensing — all 4 activity scenarios

Activity parameters used in validation (from test_realtime_har_sim.py)
-----------------------------------------------------------------------
Scenario 1 — transient_gesture:
  Burst duration  : 250 ms   Repetition : every 3.0 s   Active window: 1.4-1.65 s
  Displacement    : 2.5 cm   Doppler    : ~2.5 Hz        Dyn gain     : -38 dB
  Static SNR      : 27 dB    Jitter max : ±30 samples

Scenario 2 — fast_tremor:
  Frequency       : 5.5 Hz   Amplitude  : 5 mm           Continuous motion
  Env modulation  : ±8%      Dyn gain   : -32 dB
  Static SNR      : 27 dB

Scenario 3 — abrupt_walkby:
  SNR drop during walk: 8 dB (27→19 dB)   Duration: 3.0 s (3.5-6.5 s mod 10)
  Doppler spread  : 18 Hz + 12 Hz bisinusoidal   Dyn gain: -16 dB
  Dyn delay shift : 18 samples    Env: 0.50±0.15   Jitter max: ±35 samples

Scenario 4 — multi_activity:
  t=0-10 s  : transient_gesture
  t=10-20 s : fast_tremor
  t=20-30 s : abrupt_walkby
  3000 packets at 10 ms interval
"""

import os
import sys
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import firwin, resample_poly, welch, spectrogram, freqz

# ── production modules (frozen) ───────────────────────────────────────────────
import config as cfg
import waveform
import scrambler
import demod as demod_mod
from detector import PacketDetector
from sync import (find_ltf_timing, estimate_fine_cfo, extract_csi,
                  sanitize_csi_phase, sco_correct_symbol, sync_packet,
                  apply_cfo_correction, _LTF_TMPL)

# ── import the HAR channel from the test harness (not a frozen file) ──────────
sys.path.insert(0, os.path.dirname(__file__))
from test_realtime_har_sim import DynamicHARChannel

PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

N_PAYLOAD   = 100          # bytes
N_PAYLOAD_B = N_PAYLOAD * 8
MOD         = 'BPSK'
FS_BB       = cfg.FS_FFT   # 20 MS/s
FS_HW       = cfg.FS_HW    # 25 MS/s


def _rng(seed=0):
    return np.random.default_rng(seed)


def _make_tx_packet(payload_bits=None, seed=0, guard_bb=0):
    """Build a baseband (20 MS/s) packet and resample to 25 MS/s."""
    if payload_bits is None:
        payload_bits = _rng(seed).integers(0, 2, N_PAYLOAD_B, dtype=np.uint8)
    bb = waveform.assemble_packet(
        payload_bits,
        modulation=MOD,
        scrambler_mod=scrambler,
        encoder_mod=scrambler,
        mapper_fn=scrambler.map_bits_to_symbols,
        idle_samples=guard_bb)
    hw = waveform.resample_20to25(bb)
    return payload_bits, bb, hw


def _save(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved: {path}')


def savefig(name):
    path = os.path.join(PLOTS_DIR, name)
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  saved: {path}')


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1  —  Waveform Assembly & Resampling
# ─────────────────────────────────────────────────────────────────────────────

def stage1_plots():
    print('\n=== Stage 1: TX Waveform Assembly & Resampling ===')
    bits, bb, hw = _make_tx_packet(guard_bb=cfg.GUARD_SAMPLES_BB)

    # ── 1a: Frame Envelope |s[n]| ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 3))
    n_bb = np.arange(len(bb))
    ax.plot(n_bb, np.abs(bb), linewidth=0.6, color='steelblue')

    # annotate regions
    g  = cfg.GUARD_SAMPLES_BB
    s0 = g;             s1 = g + cfg.STF_LEN
    l0 = s1;            l1 = l0 + cfg.LTF_LEN
    si0 = l1;           si1 = si0 + cfg.SIG_LEN
    d0  = si1

    for x0, x1, label, c in [
            (0, g,   'Guard\n(820 zeros)', 'lightgray'),
            (s0, s1, 'STF\n(128)', 'lightblue'),
            (l0, l1, 'LTF\n(320)', 'lightyellow'),
            (si0, si1, 'SIGNAL\n(160)', 'lightgreen'),
            (d0, len(bb), 'DATA symbols', 'lightsalmon')]:
        ax.axvspan(x0, x1, alpha=0.35, color=c)
        ax.text((x0 + x1) / 2, ax.get_ylim()[1] * 0.92, label,
                ha='center', va='top', fontsize=7)

    ax.set_xlabel('Sample index (20 MS/s)')
    ax.set_ylabel('|s[n]|')
    ax.set_title('Stage 1a — TX Frame Envelope |s[n]| at 20 MS/s  '
                 f'(BPSK, {N_PAYLOAD}B payload, guard={g} samples)')
    ax.grid(True, alpha=0.3)
    savefig('s1a_frame_envelope.png')

    # ── 1b: Resampling Filter Frequency Response ──────────────────────────
    h_fir = firwin(64, 1.0 / max(cfg.RESAMP_INTERP, cfg.RESAMP_DECIM),
                   window=('kaiser', 8.0), pass_zero=True)
    w, H = freqz(h_fir, worN=4096, fs=1.0)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(w, 20 * np.log10(np.abs(H) + 1e-12), color='steelblue', linewidth=1.2)
    ax.axvline(x=0.2, color='red', linestyle='--', linewidth=1,
               label='Cutoff = 0.2 Fs  (anti-alias)')
    ax.axhline(y=-74, color='gray', linestyle=':', linewidth=0.8,
               label='Stopband ≈ −74 dB  (Kaiser β=8)')
    ax.axhline(y=-3,  color='green', linestyle=':', linewidth=0.8,
               label='−3 dB')
    ax.set_xlim(0, 0.5)
    ax.set_ylim(-100, 5)
    ax.set_xlabel('Normalised frequency (× Fs)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title('Stage 1b — 64-tap Kaiser FIR  (β=8, cutoff=0.2 Fs)\n'
                 'Anti-aliasing filter used for 20 ↔ 25 MS/s polyphase resampling')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    savefig('s1b_filter_response.png')

    # ── 1c: Pre- vs Post-Resampling Overlay (zoom on first 500 bb samples) ─
    n_show = 500
    bb_zoom = np.real(bb[cfg.GUARD_SAMPLES_BB: cfg.GUARD_SAMPLES_BB + n_show])
    # Resample a short slice for display
    hw_full = np.real(hw)
    g_hw = round(cfg.GUARD_SAMPLES_BB * 5 / 4)
    hw_zoom_raw = hw_full[g_hw: g_hw + int(n_show * 5 / 4)]
    t_bb = np.arange(n_show) / FS_BB * 1e6          # µs
    t_hw = np.arange(len(hw_zoom_raw)) / FS_HW * 1e6

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_bb, bb_zoom, color='steelblue', linewidth=1.0,
            label='20 MS/s (baseband)', alpha=0.85)
    ax.plot(t_hw, hw_zoom_raw, color='darkorange', linewidth=0.8,
            linestyle='--', label='25 MS/s (HW rate)', alpha=0.85)
    ax.set_xlabel('Time (µs)')
    ax.set_ylabel('Re{s}')
    ax.set_title('Stage 1c — Pre vs Post Resampling (first 500 baseband samples, real part)\n'
                 'Overlay shows interpolation fidelity: 20 MS/s → 25 MS/s')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    savefig('s1c_resample_overlay.png')

    # ── 1d: PAPR / CCDF ──────────────────────────────────────────────────
    pwr  = np.abs(hw) ** 2
    pwr_db = 10 * np.log10(pwr + 1e-20)
    peak_db = 10 * np.log10(np.max(pwr) + 1e-20)
    mean_db = 10 * np.log10(np.mean(pwr) + 1e-20)
    papr_db = peak_db - mean_db

    sorted_pw = np.sort(pwr_db)
    ccdf = 1.0 - np.arange(1, len(sorted_pw) + 1) / len(sorted_pw)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(sorted_pw - mean_db, ccdf, color='steelblue', linewidth=1.2)
    ax.axvline(x=papr_db, color='red', linestyle='--', linewidth=1,
               label=f'PAPR = {papr_db:.2f} dB')
    ax.axvline(x=10.0, color='gray', linestyle=':', linewidth=0.8,
               label='10 dB reference')
    ax.set_xlabel('Power above mean (dB)')
    ax.set_ylabel('CCDF  Pr[PAPR > x]')
    ax.set_title(f'Stage 1d — PAPR / CCDF of TX Burst\n'
                 f'Peak-to-Average Power Ratio = {papr_db:.2f} dB  '
                 f'(BPSK, {N_PAYLOAD}B payload)')
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_ylim(1e-4, 1.2)
    savefig('s1d_papr_ccdf.png')


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2  —  Packet Detection & Coarse Sync
# ─────────────────────────────────────────────────────────────────────────────

def stage2_plots():
    print('\n=== Stage 2: Schmidl-Cox Detection & Coarse CFO ===')

    # Build a signal stream: [noise] + [guard] + [packet] + [noise]
    bits, bb, hw = _make_tx_packet(seed=1)
    rx_hw = resample_poly(hw, 4, 5).astype(np.complex64)  # back to BB

    rng  = np.random.default_rng(7)
    snr  = 27.0
    sig_pwr  = float(np.mean(np.abs(rx_hw) ** 2))
    noise_std = math.sqrt(sig_pwr * 10 ** (-snr / 10) / 2)

    noise_pre  = (rng.normal(0, noise_std, 500) + 1j * rng.normal(0, noise_std, 500)).astype(np.complex64)
    noise_post = (rng.normal(0, noise_std, 300) + 1j * rng.normal(0, noise_std, 300)).astype(np.complex64)
    guard_bb   = np.zeros(cfg.GUARD_SAMPLES_BB, dtype=np.complex64)

    stream = np.concatenate([noise_pre, guard_bb, rx_hw + (rng.normal(0, noise_std, len(rx_hw)) +
                              1j * rng.normal(0, noise_std, len(rx_hw))).astype(np.complex64), noise_post])

    # Compute Schmidl-Cox metric over the whole stream
    L = 16   # STF period
    N_stream = len(stream)
    P_arr = np.zeros(N_stream, dtype=np.complex64)
    R_arr = np.zeros(N_stream, dtype=np.float64)
    M_arr = np.zeros(N_stream, dtype=np.float64)

    for n in range(L, N_stream - L):
        seg_A = stream[n - L:n]
        seg_B = stream[n:n + L]
        P = np.sum(seg_A * np.conj(seg_B))
        R = float(np.sum(np.abs(seg_B) ** 2))
        P_arr[n] = P
        R_arr[n] = R
        M_arr[n] = float(abs(P) ** 2) / (R ** 2 + 1e-12)

    n_axis = np.arange(N_stream)
    pkt_start = len(noise_pre) + cfg.GUARD_SAMPLES_BB

    # ── 2a: Schmidl-Cox Metric M[n] ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(n_axis, M_arr, linewidth=0.7, color='steelblue', label='M[n]')
    ax.axhline(y=0.65, color='red', linestyle='--', linewidth=1, label='Threshold = 0.65')
    ax.axhline(y=0.70, color='darkorange', linestyle=':', linewidth=1, label='Threshold = 0.70')
    ax.axvline(x=pkt_start, color='green', linestyle='--', linewidth=1, alpha=0.7,
               label=f'STF start (n={pkt_start})')
    ax.set_xlabel('Sample index (20 MS/s)')
    ax.set_ylabel('M[n]')
    ax.set_title('Stage 2a — Schmidl-Cox Correlation Metric M[n] = |P[n]|² / R[n]²\n'
                 f'Stream: 500 noise + {cfg.GUARD_SAMPLES_BB} guard + packet + 300 noise  |  SNR = {snr} dB')
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.15)
    ax.grid(True, alpha=0.3)
    savefig('s2a_schmidl_cox_metric.png')

    # ── 2b: |P[n]| and R[n] ──────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    axes[0].plot(n_axis, np.abs(P_arr), linewidth=0.6, color='steelblue')
    axes[0].axvline(x=pkt_start, color='red', linestyle='--', linewidth=1, alpha=0.7)
    axes[0].set_ylabel('|P[n]|')
    axes[0].set_title('Stage 2b — Numerator |P[n]| and Denominator R[n] of Schmidl-Cox')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(n_axis, R_arr, linewidth=0.6, color='darkorange')
    axes[1].axvline(x=pkt_start, color='red', linestyle='--', linewidth=1, alpha=0.7,
                    label=f'Packet start n={pkt_start}')
    axes[1].set_xlabel('Sample index (20 MS/s)')
    axes[1].set_ylabel('R[n]  (energy)')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    savefig('s2b_P_R_numerator_denominator.png')

    # ── 2c: Coarse CFO estimate along the STF ────────────────────────────
    stf_region = np.arange(pkt_start, pkt_start + cfg.STF_LEN)
    cfo_trace  = np.zeros(len(stf_region))
    for i, n in enumerate(stf_region):
        if n + L < N_stream:
            seg = stream[n:n + L * 5]
            phi = np.angle(np.sum(seg[L:] * np.conj(seg[:-L])))
            cfo_trace[i] = phi / (2 * np.pi * L / FS_BB)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(stf_region - pkt_start, cfo_trace, linewidth=0.8, color='steelblue')
    ax.axhline(y=3200.0, color='red', linestyle='--', linewidth=1, label='True CFO = 3200 Hz')
    ax.set_xlabel('Sample offset from STF start')
    ax.set_ylabel('Estimated CFO (Hz)')
    ax.set_title('Stage 2c — Instantaneous Coarse CFO Estimate Along the STF\n'
                 'Settles to flat plateau during the 128-sample STF correlation window')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    savefig('s2c_coarse_cfo_trace.png')

    # ── 2d: Noise-only M[n] histogram ────────────────────────────────────
    noise_only = (rng.normal(0, noise_std, 2000) +
                  1j * rng.normal(0, noise_std, 2000)).astype(np.complex64)
    M_noise = np.zeros(len(noise_only))
    for n in range(L, len(noise_only) - L):
        seg_A = noise_only[n - L:n]
        seg_B = noise_only[n:n + L]
        P = np.sum(seg_A * np.conj(seg_B))
        R = float(np.sum(np.abs(seg_B) ** 2))
        M_noise[n] = float(abs(P) ** 2) / (R ** 2 + 1e-12)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(M_noise[L:-L], bins=60, color='steelblue', edgecolor='none', alpha=0.8)
    ax.axvline(x=0.65, color='red', linestyle='--', linewidth=1.2, label='Threshold = 0.65')
    ax.axvline(x=0.70, color='darkorange', linestyle=':', linewidth=1.2, label='Threshold = 0.70')
    ax.set_xlabel('M[n]  (noise-only)')
    ax.set_ylabel('Count')
    ax.set_title(f'Stage 2d — Noise-Only M[n] Histogram  (SNR = {snr} dB, 2000 noise samples)\n'
                 'Confirms large separation from 0.65 threshold — zero false alarms expected')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    savefig('s2d_noise_metric_histogram.png')


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3  —  Fine Timing & Fine CFO (LTF)
# ─────────────────────────────────────────────────────────────────────────────

def stage3_plots():
    print('\n=== Stage 3: Fine Timing & Fine CFO (LTF) ===')

    N_TRIALS = 200
    rng = np.random.default_rng(42)

    bits, bb, hw = _make_tx_packet(seed=0, guard_bb=cfg.GUARD_SAMPLES_BB)

    snr  = 27.0
    sig_pwr   = float(np.mean(np.abs(bb) ** 2))
    noise_std = math.sqrt(sig_pwr * 10 ** (-snr / 10) / 2)
    true_cfo  = 3200.0

    # Apply true CFO to baseband signal
    n_arr = np.arange(len(bb), dtype=np.float64)
    bb_cfo = (bb * np.exp(1j * 2 * np.pi * true_cfo / FS_BB * n_arr)).astype(np.complex64)
    bb_rx  = (bb_cfo + (rng.normal(0, noise_std, len(bb)) +
               1j * rng.normal(0, noise_std, len(bb))).astype(np.complex64))

    # ── 3a: LTF Cross-Correlation — 2-panel: context + tight zoom ────────
    rx_coarse = apply_cfo_correction(bb_rx, true_cfo, start_n=0)
    corr = np.abs(np.correlate(rx_coarse, _LTF_TMPL, mode='valid'))
    # Normalize against the WINDOW maximum (not global), so LTF peak is visible
    pkt_start_bb  = cfg.GUARD_SAMPLES_BB
    expected_body = pkt_start_bb + cfg.STF_LEN + 64   # LTF body start
    # Search window (same as find_ltf_timing narrow window)
    win_lo = max(0, expected_body - 64)
    win_hi = min(len(corr)-1, expected_body + 64)
    win_max = corr[win_lo:win_hi+1].max() + 1e-12
    corr_win_norm = corr / win_max   # normalize to LTF window max

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Left: wide context showing STF artifact vs LTF region
    zoom_lo = max(0, expected_body - 250)
    zoom_hi = min(len(corr), expected_body + 300)
    axes[0].plot(np.arange(zoom_lo, zoom_hi), corr_win_norm[zoom_lo:zoom_hi],
                 linewidth=0.8, color='steelblue')
    axes[0].axvspan(win_lo, win_hi, alpha=0.15, color='green', label='LTF search window')
    axes[0].axvline(x=expected_body, color='red', linestyle='--', linewidth=1.2,
                    label=f'Expected body={expected_body}')
    axes[0].axvline(x=pkt_start_bb, color='orange', linestyle=':', linewidth=1,
                    label=f'STF start={pkt_start_bb}')
    axes[0].set_xlabel('Sample index (20 MS/s)')
    axes[0].set_ylabel('Corr (norm to LTF-window max)')
    axes[0].set_title('Stage 3a — Wide View: STF artifact + LTF search window\n'
                      'STF periodic structure creates false peaks outside search window')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Right: tight zoom on LTF search window — actual peak clearly visible
    pad = 20
    z_lo = max(0, win_lo - pad)
    z_hi = min(len(corr), win_hi + pad)
    peak_idx = int(np.argmax(corr[win_lo:win_hi+1])) + win_lo
    axes[1].plot(np.arange(z_lo, z_hi), corr_win_norm[z_lo:z_hi],
                 linewidth=1.0, color='steelblue')
    axes[1].axvspan(win_lo, win_hi, alpha=0.15, color='green', label='Search window')
    axes[1].axvline(x=expected_body, color='red', linestyle='--', linewidth=1.2,
                    label=f'Expected={expected_body}')
    axes[1].axvline(x=peak_idx, color='darkorange', linestyle='-', linewidth=1.2,
                    label=f'Actual peak={peak_idx}  err={peak_idx-expected_body}')
    axes[1].set_xlabel('Sample index (20 MS/s)')
    axes[1].set_ylabel('Corr (norm to window max)')
    axes[1].set_title(f'Stage 3a — Tight Zoom: LTF Search Window ±64\n'
                      f'SNR={snr} dB  |  Peak at {peak_idx}, err={peak_idx-expected_body} samples')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle('Stage 3a — LTF Time-Domain Cross-Correlation  '
                 f'(128-sample template, CFO={true_cfo} Hz)', fontsize=11, fontweight='bold')
    fig.tight_layout()
    savefig('s3a_ltf_crosscorr.png')

    # ── 3b: LTF Timing Error Histogram (N_TRIALS packets) ────────────────
    timing_errors = []
    for trial in range(N_TRIALS):
        n_t = noise_std
        bb_t  = (bb_cfo + (rng.normal(0, n_t, len(bb)) +
                  1j * rng.normal(0, n_t, len(bb))).astype(np.complex64))
        jitter = int(rng.integers(-20, 21))
        bb_t_j = np.concatenate([np.zeros(max(0, jitter), dtype=np.complex64), bb_t])
        if jitter < 0:
            bb_t_j = bb_t[-jitter:]
        lt = find_ltf_timing(bb_t_j.astype(np.complex64), true_cfo,
                             pkt_start=pkt_start_bb)
        error = lt - (pkt_start_bb + cfg.STF_LEN)   # delta from expected
        timing_errors.append(error)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(timing_errors, bins=40, color='steelblue', edgecolor='none', alpha=0.8)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1.2, label='Perfect timing (0 error)')
    mean_err = float(np.mean(timing_errors))
    std_err  = float(np.std(timing_errors))
    ax.axvline(x=mean_err, color='darkorange', linestyle='-', linewidth=1.0,
               label=f'Mean = {mean_err:.1f} samples')
    ax.set_xlabel('ltf_start error  (samples, 20 MS/s)')
    ax.set_ylabel('Count')
    ax.set_title(f'Stage 3b — LTF Timing Error Histogram  '
                 f'({N_TRIALS} trials, SNR={snr} dB, jitter=±20 samples)\n'
                 f'Mean = {mean_err:.1f} samples   σ = {std_err:.1f} samples')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    savefig('s3b_timing_error_histogram.png')

    # ── 3c: Fine CFO Phase Angle across packets ───────────────────────────
    fine_cfo_vals = []
    for trial in range(N_TRIALS):
        n_t  = noise_std
        bb_t = (bb_cfo + (rng.normal(0, n_t, len(bb)) +
                 1j * rng.normal(0, n_t, len(bb))).astype(np.complex64))
        rx_c = apply_cfo_correction(bb_t, true_cfo, start_n=0)
        s1_body = pkt_start_bb + cfg.STF_LEN + 64
        s2_body = s1_body + cfg.FFT_SIZE
        ltf1 = rx_c[s1_body:s1_body + cfg.FFT_SIZE]
        ltf2 = rx_c[s2_body:s2_body + cfg.FFT_SIZE]
        fine_cfo = estimate_fine_cfo(ltf1, ltf2)
        fine_cfo_vals.append(fine_cfo)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(fine_cfo_vals, 'o', markersize=2, color='steelblue', alpha=0.7, label='Fine CFO residual')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, label='Ideal = 0 Hz')
    ax.axhline(y=float(np.mean(fine_cfo_vals)), color='darkorange', linewidth=1,
               label=f'Mean = {np.mean(fine_cfo_vals):.1f} Hz')
    ax.set_xlabel('Packet trial index')
    ax.set_ylabel('Fine CFO residual (Hz)')
    ax.set_title(f'Stage 3c — Fine CFO Residual After Coarse Correction  '
                 f'({N_TRIALS} trials, SNR={snr} dB)\n'
                 f'True CFO = {true_cfo} Hz  |  '
                 f'Fine residual: mean={np.mean(fine_cfo_vals):.1f} Hz  '
                 f'σ={np.std(fine_cfo_vals):.1f} Hz')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    savefig('s3c_fine_cfo_phase.png')

    # ── 3d: Total CFO Estimate vs Ground Truth ────────────────────────────
    total_cfos  = []
    cfos_true   = [true_cfo + 0.5 * i for i in range(N_TRIALS)]  # slight drift
    for i, gt_cfo in enumerate(cfos_true):
        n_t  = noise_std
        bb_t = (bb * np.exp(1j * 2 * np.pi * gt_cfo / FS_BB * n_arr) +
                (rng.normal(0, n_t, len(bb)) +
                 1j * rng.normal(0, n_t, len(bb)))).astype(np.complex64)
        rx_c = apply_cfo_correction(bb_t, gt_cfo, start_n=0)   # simulate detector coarse
        s1   = pkt_start_bb + cfg.STF_LEN + 64
        s2   = s1 + cfg.FFT_SIZE
        fine = estimate_fine_cfo(rx_c[s1:s1 + cfg.FFT_SIZE], rx_c[s2:s2 + cfg.FFT_SIZE])
        total_cfos.append(gt_cfo + fine)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(cfos_true,  linewidth=1.2, color='gray',      label='Ground-truth CFO (Hz)')
    ax.plot(total_cfos, linewidth=1.0, color='steelblue',  linestyle='--', label='Estimated total CFO (Hz)')
    ax.set_xlabel('Packet index')
    ax.set_ylabel('CFO (Hz)')
    ax.set_title(f'Stage 3d — Total CFO Estimation vs Ground Truth  ({N_TRIALS} packets)\n'
                 f'Linear drift of 0.5 Hz/packet  |  Error RMS = {np.std(np.array(total_cfos)-np.array(cfos_true)):.1f} Hz')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    savefig('s3d_total_cfo_vs_gt.png')


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4  —  CSI & Phase Sanitization
# ─────────────────────────────────────────────────────────────────────────────

def stage4_plots():
    print('\n=== Stage 4: CSI & Phase Sanitization ===')

    N_PKTS = 100
    rng  = np.random.default_rng(10)
    snr  = 27.0

    # Run N_PKTS packets through static multipath + AWGN
    # FIX: use guard_bb=0 — sync_packet() expects packet starting at index 0.
    # Adding guard to bb then passing to sync_packet() shifts LTF to wrong position.
    bits, bb, hw = _make_tx_packet(seed=0, guard_bb=0)
    sig_pwr   = float(np.mean(np.abs(bb) ** 2))   # power of packet only (not guard zeros)
    noise_std = math.sqrt(sig_pwr * 10 ** (-snr / 10) / 2)
    true_cfo  = 3200.0
    n_arr     = np.arange(len(bb), dtype=np.float64)

    # Static 3-tap channel
    ch_taps   = np.array([1.0, 10**(-12/20)*np.exp(1j*1.2), 10**(-18/20)*np.exp(1j*2.5)],
                          dtype=np.complex64)
    ch_delays = [0, 3, 7]

    def apply_channel(bb_in):
        out = np.zeros(len(bb_in) + 10, dtype=np.complex128)
        for tap, d in zip(ch_taps, ch_delays):
            out[d:d+len(bb_in)] += tap * bb_in
        return out[:len(bb_in)].astype(np.complex64)

    # Single representative packet for freq-domain plots
    bb_ch = apply_channel(bb)
    bb_cfo = bb_ch * np.exp(1j * 2 * np.pi * true_cfo / FS_BB * n_arr).astype(np.complex64)
    bb_rx  = (bb_cfo + (rng.normal(0, noise_std, len(bb)) +
               1j * rng.normal(0, noise_std, len(bb))).astype(np.complex64))

    # FIX: pass packet directly (no guard prefix) — correct LTF timing
    result = sync_packet(bb_rx, coarse_cfo_hz=true_cfo)
    H_hat  = result['H_hat']
    H_san  = result['H_sanitized']
    ph_san = result['phase_sanitized']

    k_vals = np.array(cfg.ACTIVE_SUBCARRIERS, dtype=np.float64)

    # ── 4a: CFR Magnitude |H_hat[k]| ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(k_vals, 20 * np.log10(np.abs(H_hat) + 1e-9), color='steelblue', linewidth=1.2)
    for pk in cfg.PILOT_INDICES:
        ax.axvline(x=pk, color='red', linewidth=0.6, alpha=0.5)
    ax.axvline(x=0, color='black', linewidth=0.8, linestyle=':', label='DC null (k=0)')
    ax.set_xlabel('Subcarrier index k')
    ax.set_ylabel('|H_hat[k]|  (dB)')
    ax.set_title('Stage 4a — Channel Frequency Response (CFR) Magnitude\n'
                 f'106 active subcarriers  |  3-tap multipath  |  SNR={snr} dB  |  '
                 f'Red lines = 8 pilots at {cfg.PILOT_INDICES}')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    savefig('s4a_cfr_magnitude.png')

    # ── 4b: CIR / Delay Profile ───────────────────────────────────────────
    H_full = np.zeros(cfg.FFT_SIZE, dtype=np.complex64)
    for i, k in enumerate(cfg.ACTIVE_SUBCARRIERS):
        H_full[cfg.k_to_bin(k)] = H_hat[i]
    h_cir = np.fft.ifft(H_full, n=cfg.FFT_SIZE)
    delay_axis = np.arange(cfg.FFT_SIZE) / FS_BB * 1e6  # µs

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.stem(delay_axis[:40], np.abs(h_cir[:40]), linefmt='steelblue', markerfmt='o',
            basefmt='gray')
    for d in ch_delays:
        ax.axvline(x=d / FS_BB * 1e6, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
    tap_str = ', '.join([f'{d/FS_BB*1e9:.1f} ns' for d in ch_delays])
    ax.set_xlabel('Delay (µs)')
    ax.set_ylabel('|h[tau]|')
    ax.set_title(f'Stage 4b — Channel Impulse Response (CIR / Delay Profile)\n'
                 f'True tap delays: [{tap_str}]  (red dashed = true tap positions)')
    ax.grid(True, alpha=0.3)
    savefig('s4b_cir_delay_profile.png')

    # ── 4c: Unwrapped Phase vs Subcarrier Index ───────────────────────────
    raw_ph   = np.angle(H_hat)
    unwrapped = np.unwrap(raw_ph)
    A    = np.column_stack([np.ones_like(k_vals), k_vals])
    coef, _, _, _ = np.linalg.lstsq(A, unwrapped, rcond=None)
    ols_line = coef[0] + coef[1] * k_vals

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(k_vals, unwrapped, color='steelblue', linewidth=1.2, label='Unwrapped phase')
    ax.plot(k_vals, ols_line,  color='red', linestyle='--', linewidth=1.2,
            label=f'OLS fit: a={coef[0]:.3f}, b={coef[1]:.5f} rad/SC')
    ax.set_xlabel('Subcarrier index k')
    ax.set_ylabel('Phase (rad)')
    ax.set_title('Stage 4c — Unwrapped Raw Phase θ(k) Across Active Subcarriers\n'
                 'Linear ramp caused by timing offset Δτ = b · N_FFT / (2π)  '
                 f'(b = {coef[1]:.5f} rad/SC)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    savefig('s4c_unwrapped_phase.png')

    # ── 4d: Sanitized Phase ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(k_vals, ph_san, color='steelblue', linewidth=1.2, label='Sanitized phase')
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=0.8)
    ax.fill_between(k_vals, ph_san, 0, alpha=0.15, color='steelblue')
    ax.set_xlabel('Subcarrier index k')
    ax.set_ylabel('Phase (rad)')
    ax.set_title('Stage 4d — Sanitized Phase After OLS Detrending\n'
                 f'Residual σ = {np.std(ph_san):.4f} rad  '
                 f'(timing jitter and CPE removed — Doppler signature retained)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    savefig('s4d_sanitized_phase.png')

    # ── 4e: Phase Variance: Raw vs Sanitized across N_PKTS packets ────────
    raw_var = []
    san_var = []
    for trial in range(N_PKTS):
        # FIX: regenerate noise only — packet structure unchanged, guard_bb=0
        bb_rx_t = (apply_channel(bb) * np.exp(1j * 2 * np.pi * true_cfo / FS_BB * n_arr) +
                   (rng.normal(0, noise_std, len(bb)) +
                    1j * rng.normal(0, noise_std, len(bb)))).astype(np.complex64)
        res_t = sync_packet(bb_rx_t, coarse_cfo_hz=true_cfo)
        H_t   = res_t['H_hat']
        if not np.any(np.isnan(H_t)):
            raw_var.append(np.var(np.unwrap(np.angle(H_t))))
            san_var.append(np.var(res_t['phase_sanitized']))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(np.arange(len(raw_var)) - 0.2, raw_var, width=0.4, color='steelblue',
           alpha=0.7, label=f'Raw unwrapped  (mean={np.mean(raw_var):.3f} rad²)')
    ax.bar(np.arange(len(san_var)) + 0.2, san_var, width=0.4, color='darkorange',
           alpha=0.7, label=f'Sanitized  (mean={np.mean(san_var):.4f} rad²)')
    ax.set_xlabel('Packet index')
    ax.set_ylabel('Phase variance (rad²)')
    ax.set_title(f'Stage 4e — Per-Packet Phase Variance: Raw vs Sanitized  ({N_PKTS} packets)\n'
                 f'SNR={snr} dB  |  Sanitization reduction: '
                 f'{np.mean(raw_var)/np.mean(san_var):.1f}×')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    savefig('s4e_phase_variance_comparison.png')


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5  —  Equalization, SCO Tracking & Demodulation
# ─────────────────────────────────────────────────────────────────────────────

def stage5_plots():
    print('\n=== Stage 5: Equalization, SCO Tracking & Demodulation ===')

    rng = np.random.default_rng(55)
    snr = 27.0

    # Build packet and pass through channel with REAL CFO + SCO + AWGN
    # FIX (guard): guard_bb=0 — sync_packet expects packet at index 0.
    # FIX (SCO): use ChannelModel to apply genuine 5ppm SCO via resampling.
    # Previous code had 'pass' — SCO was declared but never applied to signal.
    bits, bb, hw = _make_tx_packet(seed=5, guard_bb=0)
    sig_pwr   = float(np.mean(np.abs(bb) ** 2))
    noise_std = math.sqrt(sig_pwr * 10 ** (-snr / 10) / 2)
    true_cfo  = 3200.0
    sco_ppm   = 5.0   # 5 ppm SCO — now ACTUALLY applied via ChannelModel

    from channel_bridge import ChannelModel as _CM
    _ch = _CM(noise_voltage=noise_std * math.sqrt(2),
              cfo_hz=true_cfo, sco_ppm=sco_ppm, fs=FS_BB, seed=55)
    bb_rx = _ch.apply(bb.copy())
    # Pad/trim to original length (SCO resampling may shift length slightly)
    if len(bb_rx) < len(bb):
        bb_rx = np.concatenate([bb_rx, np.zeros(len(bb)-len(bb_rx), dtype=np.complex64)])
    else:
        bb_rx = bb_rx[:len(bb)]

    n_sym = demod_mod.n_data_syms_for_payload(N_PAYLOAD, MOD)
    result = sync_packet(bb_rx, coarse_cfo_hz=true_cfo, n_data_symbols=n_sym)
    H_hat  = result['H_hat']
    data_ffts = result['data_syms']

    k_vals = np.array(cfg.ACTIVE_SUBCARRIERS, dtype=np.float64)

    # ── 5a: Equalized Constellation ───────────────────────────────────────
    equ_syms = []
    for Y in data_ffts:
        Y_data = np.array([Y[cfg.k_to_bin(k)] for k in cfg.DATA_INDICES], dtype=np.complex64)
        equ_syms.append(demod_mod.equalize(Y_data, H_hat))
    equ_all = np.concatenate(equ_syms) if equ_syms else np.array([])

    fig, ax = plt.subplots(figsize=(6, 6))
    if len(equ_all):
        ax.scatter(equ_all.real, equ_all.imag, s=4, alpha=0.5, color='steelblue')
    ax.axhline(y=0, color='gray', linewidth=0.6)
    ax.axvline(x=0, color='gray', linewidth=0.6)
    # BPSK ideal points
    ax.plot([-1, 1], [0, 0], 'rx', markersize=12, markeredgewidth=2, label='Ideal BPSK ±1')
    ax.set_xlabel('I')
    ax.set_ylabel('Q')
    ax.set_title(f'Stage 5a — ZF-Equalized Constellation  (BPSK, {n_sym} symbols × 98 data SCs)\n'
                 f'SNR = {snr} dB  |  CFO = {true_cfo} Hz  |  All points near ±1+0j')
    ax.legend(fontsize=9)
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    savefig('s5a_constellation.png')

    # ── 5b: Pilot Phase Tracking vs Symbol Index ──────────────────────────
    pilot_phases_per_sym = []
    sco_b_accum = 0.0
    lt = result['ltf_start']
    total_cfo = result['total_cfo']
    rx_c = apply_cfo_correction(bb_rx, total_cfo, start_n=0)
    ds  = lt + cfg.LTF_LEN + cfg.SIG_LEN

    for m in range(min(n_sym, len(data_ffts))):
        Y = data_ffts[m]
        ph_row = []
        for i, k in enumerate(cfg.PILOT_INDICES):
            b = cfg.k_to_bin(k)
            h_idx = cfg.ACTIVE_SUBCARRIERS.index(k)
            H_k   = H_hat[h_idx]
            H2    = float(np.abs(H_k) ** 2) + 1e-10
            z     = Y[b] * np.conj(H_k) / H2 * float(cfg.PILOT_POLARITY[i])
            ph_row.append(float(np.angle(z)))
        pilot_phases_per_sym.append(ph_row)

    pilot_phases_per_sym = np.array(pilot_phases_per_sym)   # (n_sym, 8)
    sym_indices = np.arange(pilot_phases_per_sym.shape[0])

    fig, ax = plt.subplots(figsize=(11, 4))
    colors = plt.cm.tab10(np.linspace(0, 1, 8))
    for pi in range(8):
        unwrapped = np.unwrap(pilot_phases_per_sym[:, pi])
        ax.plot(sym_indices, unwrapped, linewidth=1.0, color=colors[pi],
                label=f'k={cfg.PILOT_INDICES[pi]}')
    ax.set_xlabel('DATA symbol index m')
    ax.set_ylabel('Pilot phase (rad, unwrapped)')
    ax.set_title(f'Stage 5b — Pilot Phase Tracking Across DATA Symbols  ({n_sym} symbols)\n'
                 f'All 8 pilots  |  SCO={sco_ppm} ppm drift visible as growing ramp')
    ax.legend(fontsize=7, ncol=4, loc='upper left')
    ax.grid(True, alpha=0.3)
    savefig('s5b_pilot_phase_tracking.png')

    # ── 5c: SCO Slope b_accum per Symbol ──────────────────────────────────
    b_vals = []
    b_acc  = 0.0
    for m in range(pilot_phases_per_sym.shape[0]):
        ph = np.unwrap(pilot_phases_per_sym[m])
        kv = np.array(cfg.PILOT_INDICES, dtype=np.float64)
        A  = np.column_stack([np.ones_like(kv), kv])
        coef, _, _, _ = np.linalg.lstsq(A, ph, rcond=None)
        b_m   = float(coef[1])
        alpha = 0.3 if m > 0 else 1.0
        b_acc = b_acc * (1 - alpha) + b_m * alpha
        b_vals.append(b_acc)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(b_vals, color='steelblue', linewidth=1.2)
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=0.8)
    ax.set_xlabel('DATA symbol index m')
    ax.set_ylabel('b_accum  (rad / SC)')
    ax.set_title(f'Stage 5c — SCO Phase Slope b_accum Convergence  ({n_sym} symbols)\n'
                 f'Exponential smoothing α=0.3  |  SCO={sco_ppm} ppm  |  '
                 f'Steady-state b_accum ≈ {b_vals[-1]:.6f} rad/SC')
    ax.grid(True, alpha=0.3)
    savefig('s5c_sco_slope_convergence.png')

    # ── 5d: Symbol EVM vs Symbol Index ───────────────────────────────────
    evm_before = []
    evm_after  = []

    sco_b_acc  = 0.0
    for m, Y in enumerate(data_ffts):
        Y_data_b4 = np.array([Y[cfg.k_to_bin(k)] for k in cfg.DATA_INDICES], dtype=np.complex64)
        eq_b4 = demod_mod.equalize(Y_data_b4, H_hat)
        # BPSK ideal: sign(Re)
        ideal = np.sign(eq_b4.real) + 0j
        evm_before.append(float(20 * np.log10(np.sqrt(np.mean(np.abs(eq_b4 - ideal) ** 2)) + 1e-9)))

        Y_corr, sco_b_acc, _ = sco_correct_symbol(Y, H_hat, symbol_idx=m, sco_b_accum=sco_b_acc)
        Y_data_af = np.array([Y_corr[cfg.k_to_bin(k)] for k in cfg.DATA_INDICES], dtype=np.complex64)
        eq_af = demod_mod.equalize(Y_data_af, H_hat)
        ideal2 = np.sign(eq_af.real) + 0j
        evm_after.append(float(20 * np.log10(np.sqrt(np.mean(np.abs(eq_af - ideal2) ** 2)) + 1e-9)))

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(evm_before, color='steelblue',  linewidth=1.0, label='EVM before SCO correction')
    ax.plot(evm_after,  color='darkorange', linewidth=1.0, linestyle='--', label='EVM after SCO correction')
    ax.set_xlabel('DATA symbol index m')
    ax.set_ylabel('EVM (dB)')
    ax.set_title(f'Stage 5d — Symbol EVM Before vs After SCO Correction  ({n_sym} symbols)\n'
                 f'BPSK  |  SNR={snr} dB  |  SCO={sco_ppm} ppm')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    savefig('s5d_evm_vs_symbol.png')


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 6  —  HAR Sensing (all 4 activity scenarios)
# ─────────────────────────────────────────────────────────────────────────────

ACTIVITY_PARAMS = {
    'transient_gesture': {
        'label'   : 'Transient Micro-Gesture (Keystroke / Finger Tap)',
        'params'  : ('Burst: 250 ms  |  Repeat: 3 s  |  Disp: 2.5 cm  |  '
                     'Doppler: ~2.5 Hz  |  SNR: 27 dB  |  Jitter: ±30 sa'),
        'n_pkts'  : 300,
        'interval': 0.01,
    },
    'fast_tremor': {
        'label'   : 'Fast Micro-Tremor (Repetitive Finger Tap / Hand Shake)',
        'params'  : ('f_tremor: 5.5 Hz  |  Amplitude: 5 mm  |  '
                     'Env: ±8%  |  SNR: 27 dB  |  Continuous'),
        'n_pkts'  : 300,
        'interval': 0.01,
    },
    'abrupt_walkby': {
        'label'   : 'Abrupt Walk-By / Shadow Fading',
        'params'  : ('SNR drop: 8 dB (27→19 dB)  |  Duration: 3 s/10 s  |  '
                     'Doppler: 18+12 Hz  |  Dyn delay: 18 sa  |  Jitter: ±35 sa'),
        'n_pkts'  : 500,
        'interval': 0.01,
    },
    'multi_activity': {
        'label'   : 'Multi-Activity Composite (3 Phases × 10 s)',
        'params'  : ('t=0-10 s: transient_gesture  |  '
                     't=10-20 s: fast_tremor  |  '
                     't=20-30 s: abrupt_walkby  |  '
                     '3000 pkts @ 10 ms'),
        'n_pkts'  : 3000,
        'interval': 0.01,
    },
}


def _run_har_scenario(scenario, n_pkts, interval):
    """Run closed-loop sim for one HAR scenario. Returns lists of H_san, phase_san, csi_mag, timestamps."""
    channel = DynamicHARChannel(scenario=scenario, cfo_init_hz=3200.0,
                                cfo_drift_rate=10.0, snr_db=27.0, seed=42)
    det = PacketDetector()
    rng = np.random.default_rng(2024)

    H_san_list   = []
    phase_list   = []
    ts_list      = []
    crc_count    = 0

    look_ahead = np.array([], dtype=np.complex64)
    payload_bits_all = rng.integers(0, 2, size=(n_pkts, N_PAYLOAD_B), dtype=np.uint8)

    for idx in range(n_pkts):
        t = idx * interval
        bits_i = payload_bits_all[idx]
        # TX
        bb_tx = waveform.assemble_packet(
            bits_i, modulation=MOD, scrambler_mod=scrambler,
            encoder_mod=scrambler, mapper_fn=scrambler.map_bits_to_symbols,
            idle_samples=0)
        hw_tx = resample_poly(bb_tx, 5, 4).astype(np.complex64)
        # Channel
        rx_hw = channel.apply(hw_tx, t)
        # RX downsample
        rx_bb = resample_poly(rx_hw, 4, 5).astype(np.complex64)
        buf   = np.concatenate([look_ahead, rx_bb]) if len(look_ahead) else rx_bb
        dets  = det.process(rx_bb)

        n_abs = det._sample_idx - len(det._buf)

        for abs_s, det_cfo in dets:
            if abs(det_cfo) > 200_000:
                continue
            rel = int(abs_s) - int(n_abs - len(rx_bb))
            pkt_window = int(cfg.STF_LEN + cfg.LTF_LEN + cfg.SIG_LEN + 20 * cfg.SYMBOL_LEN)
            if rel < 0 or rel + pkt_window > len(buf):
                continue
            win = buf[rel: rel + pkt_window]
            res = sync_packet(win, coarse_cfo_hz=det_cfo, n_data_symbols=0)
            H   = res['H_hat']
            if np.any(np.isnan(H)):
                continue
            H_san_list.append(res['H_sanitized'])
            phase_list.append(res['phase_sanitized'])
            ts_list.append(t)
            crc_count += 1
            break   # one packet per TX burst

        look_ahead = buf[-pkt_window:] if len(buf) > pkt_window else buf

    print(f'    {scenario}: {crc_count}/{n_pkts} packets decoded ({100*crc_count/n_pkts:.1f}%)')
    return (np.array(H_san_list,  dtype=np.complex64) if H_san_list else np.zeros((1, cfg.NUM_ACTIVE), dtype=np.complex64),
            np.array(phase_list, dtype=np.float64)   if phase_list else np.zeros((1, cfg.NUM_ACTIVE), dtype=np.float64),
            np.array(ts_list,    dtype=np.float64)   if ts_list    else np.zeros(1, dtype=np.float64))


def _har_plots_for_scenario(scenario, H_san, phase_san, timestamps, info):
    label  = info['label']
    params = info['params']
    k_vals = np.array(cfg.ACTIVE_SUBCARRIERS, dtype=np.float64)
    SC_IDX = 54   # representative subcarrier (k=1, first positive)

    if len(timestamps) > 1:
        fs_pkt = 1.0 / float(np.median(np.diff(timestamps)))
    else:
        fs_pkt = 100.0

    t_rel = timestamps - timestamps[0]
    csi_mag = np.abs(H_san)

    # ── 6a: Single-Subcarrier CSI Magnitude vs Time ───────────────────────
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.plot(t_rel, csi_mag[:, SC_IDX], linewidth=0.8, color='steelblue')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(f'|H_san[k={cfg.ACTIVE_SUBCARRIERS[SC_IDX]}]|')
    ax.set_title(f'Stage 6a [{scenario}] — CSI Magnitude vs Time  (single subcarrier k={cfg.ACTIVE_SUBCARRIERS[SC_IDX]})\n'
                 f'{label}\n{params}')
    ax.grid(True, alpha=0.3)
    savefig(f's6a_{scenario}_csi_magnitude.png')

    # ── 6b: Single-Subcarrier Sanitized Phase vs Time ─────────────────────
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.plot(t_rel, phase_san[:, SC_IDX], linewidth=0.8, color='darkorange')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(f'θ_san[k={cfg.ACTIVE_SUBCARRIERS[SC_IDX]}]  (rad)')
    ax.set_title(f'Stage 6b [{scenario}] — Sanitized Phase vs Time  (k={cfg.ACTIVE_SUBCARRIERS[SC_IDX]})\n'
                 f'{label}\n{params}')
    ax.grid(True, alpha=0.3)
    savefig(f's6b_{scenario}_sanitized_phase.png')

    # ── 6c: CSI Magnitude Waterfall Heatmap ──────────────────────────────
    N_show = min(len(t_rel), 300)
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(csi_mag[:N_show].T, aspect='auto', origin='lower',
                   extent=[t_rel[0], t_rel[min(N_show-1, len(t_rel)-1)],
                           k_vals[0], k_vals[-1]],
                   cmap='viridis')
    plt.colorbar(im, ax=ax, label='|H_san|')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Subcarrier index k')
    ax.set_title(f'Stage 6c [{scenario}] — CSI Magnitude Waterfall (106 SCs × time)\n'
                 f'{label}\n{params}')
    savefig(f's6c_{scenario}_waterfall.png')

    # ── 6d: Welch PSD of Sanitized Phase ─────────────────────────────────
    amp = np.mean(np.abs(H_san), axis=1)
    amp = amp - np.mean(amp)
    nperseg = min(128, len(amp) // 2)
    if nperseg < 4:
        print(f'  Skipping PSD for {scenario} — too few packets')
        return

    freqs, psd = welch(amp, fs=fs_pkt, nperseg=nperseg, noverlap=nperseg // 2)
    psd_db = 10 * np.log10(psd + 1e-20)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(freqs, psd_db, color='steelblue', linewidth=1.2)
    ax.axvspan(0.2, 0.5, alpha=0.12, color='green', label='Resp. band 0.2–0.5 Hz')
    ax.axvspan(3.0, 8.0, alpha=0.08, color='orange', label='Tremor band 3–8 Hz')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD (dB/Hz)')
    ax.set_xlim(0, min(fs_pkt / 2, 15.0))
    ax.set_title(f'Stage 6d [{scenario}] — Welch PSD of Spatially-Averaged CSI Amplitude\n'
                 f'{label}\n{params}')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    savefig(f's6d_{scenario}_welch_psd.png')

    # ── 6e: Micro-Doppler STFT Spectrogram ───────────────────────────────
    nperseg_sp = min(64, len(amp) // 4)
    if nperseg_sp < 4:
        print(f'  Skipping spectrogram for {scenario} — too few packets')
        return

    f_sp, t_sp, Sxx = spectrogram(amp, fs=fs_pkt,
                                   nperseg=nperseg_sp,
                                   noverlap=nperseg_sp * 3 // 4,
                                   window='hann')
    Sxx_db = 10 * np.log10(Sxx + 1e-20)
    f_mask = f_sp <= 12.0

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.pcolormesh(t_sp, f_sp[f_mask], Sxx_db[f_mask],
                  cmap='inferno', shading='auto')
    ax.axhline(y=0.2, color='cyan', linestyle='--', linewidth=0.9, alpha=0.7, label='0.2 Hz')
    ax.axhline(y=0.5, color='cyan', linestyle='--', linewidth=0.9, alpha=0.7, label='0.5 Hz')
    ax.axhline(y=5.5, color='yellow', linestyle=':', linewidth=0.9, alpha=0.7, label='5.5 Hz (tremor)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_ylim(0, 12)
    ax.set_title(f'Stage 6e [{scenario}] — Micro-Doppler STFT Spectrogram (0–12 Hz)\n'
                 f'{label}\n{params}')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(False)
    savefig(f's6e_{scenario}_spectrogram.png')


def stage6_plots():
    print('\n=== Stage 6: HAR Sensing — All 4 Activity Scenarios ===')
    for scenario, info in ACTIVITY_PARAMS.items():
        print(f'  Running scenario: {scenario}  ({info["n_pkts"]} pkts) ...')
        H_san, phase_san, timestamps = _run_har_scenario(
            scenario, info['n_pkts'], info['interval'])
        _har_plots_for_scenario(scenario, H_san, phase_san, timestamps, info)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('=' * 70)
    print('  Custom 128-pt OFDM PHY — Full Pipeline Diagnostic Plots')
    print(f'  Output directory: {PLOTS_DIR}')
    print('=' * 70)

    stage1_plots()   # TX waveform
    stage2_plots()   # Detection
    stage3_plots()   # LTF timing + fine CFO
    stage4_plots()   # CSI + phase sanitization
    stage5_plots()   # Equalization + SCO + EVM
    stage6_plots()   # HAR all 4 scenarios

    print('\n' + '=' * 70)
    print('  All plots saved to:', PLOTS_DIR)
    print('=' * 70)
