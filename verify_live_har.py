# -*- coding: utf-8 -*-
"""
verify_live_har.py — Post-capture HDF5 validation and Doppler spectrogram extraction.

Loads a recorded HAR session from HDF5, performs:
  1. Schema integrity check (array shapes, packet count, CRC pass rate)
  2. CSI magnitude stability analysis (cross-subcarrier variance)
  3. Per-packet phase re-sanitization validation (computes residual variance)
  4. Welch PSD computation for Doppler activity detection (0.1 – 10.0 Hz)
  5. Short-time Fourier Transform (STFT) Doppler spectrogram generation
  6. Three publication-quality output figures:
       - har_verification_<session_id>_magnitude.png   : CSI magnitude heatmap
       - har_verification_<session_id>_phase.png       : Raw vs sanitized phase
       - har_verification_<session_id>_spectrogram.png : Doppler spectrogram

Usage:
    python verify_live_har.py --hdf5 csi_data.h5 --session har_session_001
    python verify_live_har.py --hdf5 csi_data.h5 --session har_session_001 \\
        --fmin 0.1 --fmax 4.0 --nperseg 256 --subcarrier 54

Reference: HARDWARE_DEPLOYMENT.md Part 9, SYSTEM_SPEC.md Section 15
"""

import argparse
import sys
import os
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')   # Non-interactive — safe for headless / server execution
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import welch, spectrogram, get_window
from typing import Optional, Tuple

import config as cfg


# ─────────────────────────────────────────────────────────────────────────────
# 1. HDF5 Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_session(hdf5_path: str, session_id: str) -> dict:
    """
    Load all datasets from a logged HDF5 session.

    FIX (Bug 10): previously raised KeyError if the literal session_id string
    was not an exact key under /sessions/.  The logger stores sessions under
    keys like 'session_001' while callers may pass 'har_session_001'.
    Resolution order:
      1. Exact match at /sessions/<session_id>
      2. Any key under /sessions/ that contains session_id as a substring
      3. If exactly one session exists, use it (and warn)
      4. Raise KeyError listing all available sessions

    Expected schema under /sessions/<session_key>/:
        csi/antenna0        : complex64 (N_pkts, NUM_ACTIVE)
        csi/sanitized       : complex64 (N_pkts, NUM_ACTIVE)
        csi/phase_sanitized : float64   (N_pkts, NUM_ACTIVE)
        timestamps          : float64   (N_pkts,)
        seq                 : int64     (N_pkts,)
        cfo_hz              : float64   (N_pkts,)

    Returns:
        dict with keys: H_raw, H_san, phase_san, timestamps, seq, cfo_hz, attrs
    """
    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    with h5py.File(hdf5_path, 'r') as f:
        available = list(f.get('/sessions', {}).keys())

        # Resolution step 1: exact match
        resolved_key = None
        exact_base = f"/sessions/{session_id}"
        if exact_base in f:
            resolved_key = session_id

        # Resolution step 2: substring match
        if resolved_key is None:
            matches = [k for k in available if session_id in k or k in session_id]
            if len(matches) == 1:
                resolved_key = matches[0]
                print(f"[load_session] '{session_id}' not found exactly; "
                      f"auto-resolved to '{resolved_key}'")
            elif len(matches) > 1:
                raise KeyError(
                    f"Ambiguous session_id '{session_id}' matches {matches} "
                    f"in {hdf5_path}. Pass a more specific --session argument.")

        # Resolution step 3: single session fallback
        if resolved_key is None and len(available) == 1:
            resolved_key = available[0]
            print(f"[load_session] Warning: session '{session_id}' not found; "
                  f"only one session exists, using '{resolved_key}'")

        if resolved_key is None:
            raise KeyError(
                f"Session '{session_id}' not found in {hdf5_path}. "
                f"Available sessions: {available}"
            )

        base  = f"/sessions/{resolved_key}"
        grp   = f[base]
        attrs = dict(grp.attrs)
        csi_g = grp['csi']

        H_raw      = csi_g['antenna0'][:]
        H_san      = csi_g['sanitized'][:]
        phase_san  = csi_g['phase_sanitized'][:]
        timestamps = grp['timestamps'][:]
        seq        = grp['seq'][:]
        cfo_hz     = grp['cfo_hz'][:]

    return {
        'H_raw':      H_raw,
        'H_san':      H_san,
        'phase_san':  phase_san,
        'timestamps': timestamps,
        'seq':        seq,
        'cfo_hz':     cfo_hz,
        'attrs':      attrs,
        'n_pkts':     H_raw.shape[0],
        'n_sc':       H_raw.shape[1],
        'session_key': resolved_key,
    }



# ─────────────────────────────────────────────────────────────────────────────
# 2. Schema Integrity Checks
# ─────────────────────────────────────────────────────────────────────────────

def check_schema(data: dict) -> list:
    """
    Verify array shapes, monotonicity, and basic sanity.

    Returns list of (level, message) tuples — level in {'PASS', 'WARN', 'FAIL'}.
    """
    results = []
    N   = data['n_pkts']
    NSC = data['n_sc']

    def chk(cond, msg_pass, msg_fail, level='FAIL'):
        results.append(('PASS' if cond else level, msg_pass if cond else msg_fail))

    # Shape checks
    chk(NSC == cfg.NUM_ACTIVE,
        f"Shape OK: H_raw[{N}, {NSC}] — correct NUM_ACTIVE={cfg.NUM_ACTIVE}",
        f"Shape MISMATCH: H_raw has {NSC} subcarriers, expected {cfg.NUM_ACTIVE}")

    chk(data['H_san'].shape  == (N, NSC),
        f"Shape OK: H_sanitized[{N}, {NSC}]",
        f"Shape MISMATCH: H_sanitized shape {data['H_san'].shape}")

    chk(data['phase_san'].shape == (N, NSC),
        f"Shape OK: phase_sanitized[{N}, {NSC}]",
        f"Shape MISMATCH: phase_sanitized shape {data['phase_san'].shape}")

    chk(len(data['timestamps']) == N,
        f"Shape OK: timestamps[{N}]",
        f"Shape MISMATCH: timestamps has {len(data['timestamps'])} entries, expected {N}")

    # Packet count
    chk(N >= 10,
        f"Packet count: {N} (≥ 10 required for meaningful analysis)",
        f"Too few packets: {N} — re-run with longer capture",
        level='WARN')

    # Timestamp monotonicity
    if N > 1:
        diffs = np.diff(data['timestamps'])
        chk(np.all(diffs > 0),
            f"Timestamps: monotonically increasing (min_gap={diffs.min()*1000:.2f} ms)",
            f"Timestamps: NON-MONOTONE detected ({(diffs <= 0).sum()} violations)")

    # Sequence monotonicity
    if N > 1:
        diffs_seq = np.diff(data['seq'])
        chk(np.all(diffs_seq > 0),
            f"Seq numbers: monotonically increasing",
            f"Seq numbers: NON-MONOTONE detected ({(diffs_seq <= 0).sum()} violations)")

    # NaN/Inf check on CSI
    nan_raw = np.sum(np.isnan(data['H_raw']))
    chk(nan_raw == 0,
        f"H_raw: no NaN values",
        f"H_raw: {nan_raw} NaN values detected (synchronization failures)",
        level='WARN')

    nan_san = np.sum(np.isnan(data['H_san']))
    chk(nan_san == 0,
        f"H_sanitized: no NaN values",
        f"H_sanitized: {nan_san} NaN values detected",
        level='WARN')

    # CFO range check (B210 TCXO ±2 ppm at 2.4 GHz → ±4.8 kHz expected)
    cfo_max = float(np.max(np.abs(data['cfo_hz'])))
    chk(cfo_max < 20_000,
        f"CFO range: max={cfo_max:.1f} Hz (within ±20 kHz nominal range)",
        f"CFO range: max={cfo_max:.1f} Hz EXCEEDS ±20 kHz — check LO, SNR",
        level='WARN')

    # Phase sanitization quality: residual variance should be small
    valid_mask = ~np.any(np.isnan(data['phase_san']), axis=1)
    if valid_mask.sum() > 1:
        var_per_pkt = np.var(data['phase_san'][valid_mask], axis=1)
        mean_var    = float(np.mean(var_per_pkt))
        chk(mean_var < 0.5,
            f"Phase sanitization: mean residual variance = {mean_var:.4f} rad² (< 0.5 target)",
            f"Phase sanitization: mean residual variance = {mean_var:.4f} rad² EXCEEDS 0.5 — "
            f"check SNR, phase detrending",
            level='WARN')

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3. HAR Doppler Analysis
# ─────────────────────────────────────────────────────────────────────────────

def compute_har_psd(data: dict,
                    fmin: float = 0.1,
                    fmax: float = 10.0,
                    nperseg: int = 128,
                    subcarrier_idx: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Compute Welch PSD of the sanitized CSI amplitude over time.

    The CSI amplitude |H_sanitized[t, k]| tracks the channel envelope.
    For human respiration (0.2-0.5 Hz) and micro-gestures (1-10 Hz), this
    modulates with the body's movement Doppler signature.

    Args:
        data:           Loaded session data dict.
        fmin, fmax:     Frequency band of interest (Hz).
        nperseg:        Welch segment length (samples). Shorter = better
                        time resolution; longer = better freq resolution.
        subcarrier_idx: If given, analyse only this subcarrier index (0–106).
                        If None, average PSD across all 106 subcarriers.

    Returns:
        freqs:   Frequency axis (Hz).
        psd_db:  PSD in dB/Hz, shape (len(freqs),).
        peaks:   List of (freq_hz, power_db) for significant peaks in band.
    """
    # Packet rate estimate from timestamps
    if len(data['timestamps']) > 1:
        fs_pkt = 1.0 / float(np.median(np.diff(data['timestamps'])))
    else:
        fs_pkt = 100.0   # Fallback: assume 100 pkt/s (10 ms interval)

    H_san = data['H_san']  # (N, 106) complex64
    valid  = ~np.any(np.isnan(H_san), axis=1)
    H_san  = H_san[valid]

    if H_san.shape[0] < nperseg:
        nperseg = max(8, H_san.shape[0] // 2)

    # Select subcarrier(s)
    if subcarrier_idx is not None:
        amp = np.abs(H_san[:, subcarrier_idx])   # (N,)
    else:
        # Spatially average CSI amplitude across all subcarriers
        # (reduces uncorrelated noise; preserves common Doppler modulation)
        amp = np.mean(np.abs(H_san), axis=1)    # (N,)

    # Remove mean (center the time series around 0)
    amp = amp - np.mean(amp)

    freqs, psd = welch(amp, fs=fs_pkt, nperseg=nperseg, noverlap=nperseg // 2)

    # Mask to [fmin, fmax] band
    band = (freqs >= fmin) & (freqs <= fmax)
    f_b  = freqs[band]
    p_b  = psd[band]

    psd_db = 10 * np.log10(psd + 1e-20)

    # Find peaks: local maxima above mean+1std in the band
    if len(p_b) > 3:
        threshold = np.mean(p_b) + np.std(p_b)
        is_peak   = np.zeros(len(p_b), dtype=bool)
        for i in range(1, len(p_b) - 1):
            if p_b[i] > p_b[i - 1] and p_b[i] > p_b[i + 1] and p_b[i] > threshold:
                is_peak[i] = True
        peaks = [(float(f_b[i]), float(10 * np.log10(p_b[i] + 1e-20)))
                 for i in np.where(is_peak)[0]]
        peaks.sort(key=lambda x: x[1], reverse=True)   # sort by power descending
    else:
        peaks = []

    return freqs, psd_db, peaks


def compute_har_spectrogram(data: dict,
                             nperseg: int = 64,
                             noverlap_frac: float = 0.75,
                             subcarrier_idx: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute STFT spectrogram of the sanitized CSI amplitude time-series.

    Returns:
        t_axis   : Time axis (seconds from start of session).
        f_axis   : Frequency axis (Hz).
        Sxx_db   : Spectrogram power in dB, shape (n_freqs, n_times).
    """
    if len(data['timestamps']) > 1:
        fs_pkt = 1.0 / float(np.median(np.diff(data['timestamps'])))
    else:
        fs_pkt = 100.0

    H_san  = data['H_san']
    valid  = ~np.any(np.isnan(H_san), axis=1)
    H_san  = H_san[valid]
    ts_v   = data['timestamps'][valid]

    if subcarrier_idx is not None:
        amp = np.abs(H_san[:, subcarrier_idx])
    else:
        amp = np.mean(np.abs(H_san), axis=1)

    amp = amp - np.mean(amp)
    noverlap = int(nperseg * noverlap_frac)

    f_axis, t_stft, Sxx = spectrogram(amp, fs=fs_pkt,
                                       nperseg=min(nperseg, len(amp) // 2),
                                       noverlap=noverlap,
                                       window='hann')
    # Shift time axis to absolute timestamps
    t0     = float(ts_v[0])
    t_axis = t_stft + t0

    Sxx_db = 10 * np.log10(Sxx + 1e-20)
    return t_axis, f_axis, Sxx_db


# ─────────────────────────────────────────────────────────────────────────────
# 4. Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_magnitude(data: dict, session_id: str, out_dir: str):
    """CSI magnitude heatmap + temporal mean per subcarrier."""
    H_raw = data['H_raw']
    H_san = data['H_san']
    valid  = ~np.any(np.isnan(H_raw), axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), facecolor='#0e1117')
    fig.suptitle(f'CSI Magnitude Analysis — Session: {session_id}',
                 color='white', fontsize=13, fontweight='bold', y=0.98)

    k_vals = np.array(cfg.ACTIVE_SUBCARRIERS)

    # Top: magnitude heatmap over time (raw)
    ax = axes[0]
    ax.set_facecolor('#151820')
    mag_raw = 20 * np.log10(np.abs(H_raw[valid]) + 1e-9)
    im = ax.imshow(mag_raw.T, aspect='auto', origin='lower',
                   extent=[0, valid.sum(), k_vals[0], k_vals[-1]],
                   cmap='viridis', vmin=-30, vmax=10)
    plt.colorbar(im, ax=ax, label='Magnitude (dB)')
    ax.set_xlabel('Packet index', color='white')
    ax.set_ylabel('Subcarrier index k', color='white')
    ax.set_title('Raw H_hat magnitude (dB)', color='#a0c4ff', fontsize=11)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')

    # Bottom: mean magnitude spectrum (raw vs sanitized)
    ax = axes[1]
    ax.set_facecolor('#151820')
    mean_raw = np.mean(20 * np.log10(np.abs(H_raw[valid]) + 1e-9), axis=0)
    mean_san = np.mean(20 * np.log10(np.abs(H_san[valid]) + 1e-9), axis=0)
    ax.plot(k_vals, mean_raw, color='#4dabf7', linewidth=1.2, label='Raw H_hat (mean dB)', alpha=0.8)
    ax.plot(k_vals, mean_san, color='#ffa94d', linewidth=1.5, label='Sanitized (mean dB)', linestyle='--')
    ax.axhline(y=-3, color='red', linestyle=':', linewidth=0.8, alpha=0.5, label='-3 dB reference')
    for k in cfg.PILOT_INDICES:
        ax.axvline(x=k, color='lime', linewidth=0.6, alpha=0.3)
    ax.set_xlabel('Subcarrier index k', color='white')
    ax.set_ylabel('Mean magnitude (dB)', color='white')
    ax.set_title(f'Mean magnitude spectrum | Pilots at {cfg.PILOT_INDICES}',
                 color='#a0c4ff', fontsize=11)
    ax.legend(facecolor='#1e2130', edgecolor='#444', labelcolor='white', fontsize=9)
    ax.tick_params(colors='white')
    ax.set_xlim(k_vals[0], k_vals[-1])
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')

    plt.tight_layout()
    out_path = os.path.join(out_dir, f'har_verification_{session_id}_magnitude.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0e1117')
    plt.close()
    print(f'[verify] Saved magnitude plot: {out_path}')


def plot_phase(data: dict, session_id: str, out_dir: str):
    """Raw vs sanitized phase across subcarriers for a representative packet."""
    H_raw  = data['H_raw']
    phase_san = data['phase_san']
    valid  = ~np.any(np.isnan(H_raw), axis=1)

    # Pick middle packet as representative
    valid_idx = np.where(valid)[0]
    if len(valid_idx) == 0:
        print('[verify] WARNING: no valid packets for phase plot')
        return

    rep = valid_idx[len(valid_idx) // 2]
    k_vals = np.array(cfg.ACTIVE_SUBCARRIERS)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), facecolor='#0e1117')
    fig.suptitle(f'Phase Analysis — Session: {session_id}  (Packet #{rep})',
                 color='white', fontsize=13, fontweight='bold')

    # Top: raw unwrapped phase
    ax = axes[0]
    ax.set_facecolor('#151820')
    raw_phase     = np.angle(H_raw[rep])
    unwrapped_raw = np.unwrap(raw_phase)
    ax.plot(k_vals, unwrapped_raw, color='#4dabf7', linewidth=1.5, label='Unwrapped raw phase')
    # Overlay OLS fit line
    A    = np.column_stack([np.ones_like(k_vals), k_vals])
    coef, _, _, _ = np.linalg.lstsq(A, unwrapped_raw, rcond=None)
    fitted = coef[0] + coef[1] * k_vals
    ax.plot(k_vals, fitted, color='#ff6b6b', linewidth=1.5, linestyle='--',
            label=f'OLS fit (slope={coef[1]:+.5f} rad/SC)')
    ax.set_ylabel('Phase (rad)', color='white')
    ax.set_title('Unwrapped raw phase: θ_raw(k) and OLS detrend line', color='#a0c4ff', fontsize=11)
    ax.legend(facecolor='#1e2130', edgecolor='#444', labelcolor='white', fontsize=9)
    ax.tick_params(colors='white')
    ax.set_xlim(k_vals[0], k_vals[-1])
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')

    # Bottom: sanitized phase (after OLS detrend)
    ax = axes[1]
    ax.set_facecolor('#151820')
    psan = phase_san[rep]
    ax.plot(k_vals, psan, color='#ffa94d', linewidth=1.5, label='Sanitized phase (detrended)')
    ax.axhline(y=0, color='white', linewidth=0.7, alpha=0.4, linestyle=':')
    ax.fill_between(k_vals, psan, 0, alpha=0.15, color='#ffa94d')
    residual_std = float(np.std(psan))
    ax.set_title(f'Sanitized phase: θ_san(k) = θ_raw - (a + b·k) '
                 f'| Residual σ = {residual_std:.4f} rad',
                 color='#a0c4ff', fontsize=11)
    ax.set_xlabel('Subcarrier index k', color='white')
    ax.set_ylabel('Phase (rad)', color='white')
    ax.legend(facecolor='#1e2130', edgecolor='#444', labelcolor='white', fontsize=9)
    ax.tick_params(colors='white')
    ax.set_xlim(k_vals[0], k_vals[-1])
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')

    plt.tight_layout()
    out_path = os.path.join(out_dir, f'har_verification_{session_id}_phase.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0e1117')
    plt.close()
    print(f'[verify] Saved phase plot: {out_path}')


def plot_spectrogram(data: dict, session_id: str, out_dir: str,
                     fmin: float, fmax: float, nperseg: int,
                     subcarrier_idx: Optional[int] = None):
    """Doppler spectrogram + Welch PSD side-by-side."""
    freqs, psd_db, peaks = compute_har_psd(data, fmin, fmax, nperseg, subcarrier_idx)
    t_axis, f_axis, Sxx_db = compute_har_spectrogram(data, nperseg, subcarrier_idx=subcarrier_idx)

    sc_label = f'SC {subcarrier_idx} (k={cfg.ACTIVE_SUBCARRIERS[subcarrier_idx]})' \
        if subcarrier_idx is not None else 'All SCs (spatially averaged)'

    fig = plt.figure(figsize=(16, 9), facecolor='#0e1117')
    fig.suptitle(f'HAR Doppler Analysis — Session: {session_id}\n{sc_label}',
                 color='white', fontsize=13, fontweight='bold')

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.35)

    # ── Top-left: Spectrogram ──────────────────────────────────────────────
    ax_sg = fig.add_subplot(gs[0, :])
    ax_sg.set_facecolor('#0a0d14')

    # Clip to [fmin, fmax+2] for display
    f_mask = f_axis <= (fmax + 1.0)
    Sxx_plot = Sxx_db[f_mask, :]

    vmin_sg = float(np.percentile(Sxx_plot, 10))
    vmax_sg = float(np.percentile(Sxx_plot, 98))

    t_rel = t_axis - t_axis[0]
    im = ax_sg.pcolormesh(t_rel, f_axis[f_mask], Sxx_plot,
                          cmap='inferno', vmin=vmin_sg, vmax=vmax_sg,
                          shading='auto')
    plt.colorbar(im, ax=ax_sg, label='Power (dB/Hz)')

    # Mark respiratory band
    ax_sg.axhline(y=0.2, color='cyan', linewidth=1.0, linestyle='--', alpha=0.6)
    ax_sg.axhline(y=0.5, color='cyan', linewidth=1.0, linestyle='--', alpha=0.6)
    ax_sg.text(t_rel[-1] * 0.01, 0.35, 'Resp. band', color='cyan', fontsize=8, alpha=0.8)

    ax_sg.set_xlabel('Time from session start (s)', color='white')
    ax_sg.set_ylabel('Doppler frequency (Hz)', color='white')
    ax_sg.set_title('STFT Doppler Spectrogram of CSI Amplitude |H_sanitized|',
                    color='#a0c4ff', fontsize=11)
    ax_sg.set_ylim(0, fmax + 0.5)
    ax_sg.tick_params(colors='white')
    for spine in ax_sg.spines.values():
        spine.set_edgecolor('#444')

    # ── Bottom-left: Welch PSD in band ─────────────────────────────────────
    ax_psd = fig.add_subplot(gs[1, 0])
    ax_psd.set_facecolor('#151820')

    band_mask = (freqs >= fmin) & (freqs <= fmax)
    ax_psd.plot(freqs[band_mask], psd_db[band_mask],
                color='#69db7c', linewidth=1.5, label='Welch PSD')
    ax_psd.axvspan(0.2, 0.5, alpha=0.12, color='cyan', label='Respiratory band')

    for peak_f, peak_p in peaks[:5]:   # top 5 peaks
        ax_psd.axvline(x=peak_f, color='#ffa94d', linewidth=1.0, linestyle='--', alpha=0.7)
        ax_psd.text(peak_f, peak_p + 0.5, f'{peak_f:.2f} Hz',
                    color='#ffa94d', fontsize=7, ha='center')

    ax_psd.set_xlabel('Frequency (Hz)', color='white')
    ax_psd.set_ylabel('PSD (dB/Hz)', color='white')
    ax_psd.set_title(f'Welch PSD [{fmin:.1f}–{fmax:.1f} Hz]', color='#a0c4ff', fontsize=11)
    ax_psd.legend(facecolor='#1e2130', edgecolor='#444', labelcolor='white', fontsize=8)
    ax_psd.tick_params(colors='white')
    for spine in ax_psd.spines.values():
        spine.set_edgecolor('#444')

    # ── Bottom-right: summary text ─────────────────────────────────────────
    ax_txt = fig.add_subplot(gs[1, 1])
    ax_txt.set_facecolor('#151820')
    ax_txt.axis('off')

    N     = data['n_pkts']
    cfo   = data['cfo_hz']
    ts    = data['timestamps']
    dur   = float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0
    prate = (N - 1) / dur if dur > 0 else 0.0

    # CFO stats
    cfo_mean = float(np.mean(cfo))
    cfo_std  = float(np.std(cfo))
    cfo_max  = float(np.max(np.abs(cfo)))

    summary_lines = [
        f"Session Summary",
        f"───────────────",
        f"Packets logged : {N}",
        f"Duration       : {dur:.1f} s",
        f"Packet rate    : {prate:.1f} pkt/s",
        f"",
        f"CFO mean : {cfo_mean:+.1f} Hz",
        f"CFO σ    : {cfo_std:.1f} Hz",
        f"CFO max  : {cfo_max:.1f} Hz",
        f"",
        f"Detected HAR peaks:",
    ]
    for i, (pf, pp) in enumerate(peaks[:5]):
        band_str = ""
        if 0.2 <= pf <= 0.5:
            band_str = " ← RESPIRATION"
        elif 0.5 < pf <= 2.0:
            band_str = " ← SLOW MOTION"
        elif 2.0 < pf <= 8.0:
            band_str = " ← TREMOR/TAP"
        elif pf > 8.0:
            band_str = " ← GESTURE"
        summary_lines.append(f"  [{i+1}] {pf:.2f} Hz ({pp:.1f} dBHz){band_str}")

    if not peaks:
        summary_lines.append("  No significant peaks detected")
        summary_lines.append("  (increase capture duration or check SNR)")

    text = "\n".join(summary_lines)
    ax_txt.text(0.05, 0.95, text, transform=ax_txt.transAxes,
                fontsize=9, color='white', verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#1e2130', alpha=0.8, edgecolor='#444'))

    out_path = os.path.join(out_dir, f'har_verification_{session_id}_spectrogram.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0e1117')
    plt.close()
    print(f'[verify] Saved spectrogram: {out_path}')
    return peaks


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='Post-capture HDF5 verification and HAR Doppler analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python verify_live_har.py --hdf5 csi_data.h5 --session har_session_001
  python verify_live_har.py --hdf5 csi_data.h5 --session har_session_001 --fmax 8.0
  python verify_live_har.py --hdf5 csi_data.h5 --session har_session_001 --subcarrier 54
        """
    )
    p.add_argument('--hdf5',        type=str,   default='csi_data.h5',
                   help='Path to HDF5 file (default: csi_data.h5)')
    p.add_argument('--session',     type=str,   default='har_session_001',
                   help='Session ID to load (default: har_session_001)')
    p.add_argument('--fmin',        type=float, default=0.1,
                   help='Min frequency for HAR analysis in Hz (default: 0.1)')
    p.add_argument('--fmax',        type=float, default=10.0,
                   help='Max frequency for HAR analysis in Hz (default: 10.0)')
    p.add_argument('--nperseg',     type=int,   default=128,
                   help='Welch/STFT segment length in packets (default: 128)')
    p.add_argument('--subcarrier',  type=int,   default=None,
                   help='Subcarrier index 0–106 for per-SC analysis (default: spatial average)')
    p.add_argument('--outdir',      type=str,   default='.',
                   help='Output directory for plots (default: current dir)')
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print('=' * 72)
    print('  HAR Session Verification — Custom 128-pt OFDM PHY')
    print(f'  HDF5  : {args.hdf5}')
    print(f'  Session: {args.session}')
    print('=' * 72)

    # ── Load data ─────────────────────────────────────────────────────────
    print('\n[1/5] Loading HDF5 session...')
    try:
        data = load_session(args.hdf5, args.session)
    except (FileNotFoundError, KeyError) as e:
        print(f'\nERROR: {e}')
        sys.exit(1)

    print(f'      Loaded: {data["n_pkts"]} packets × {data["n_sc"]} subcarriers')
    if data['attrs']:
        print(f'      Metadata: {data["attrs"]}')

    # ── Schema checks ──────────────────────────────────────────────────────
    print('\n[2/5] Running schema integrity checks...')
    results = check_schema(data)
    n_pass = n_warn = n_fail = 0
    for level, msg in results:
        icon = {'PASS': '  ✅', 'WARN': '  ⚠️', 'FAIL': '  ❌'}[level]
        print(f'{icon} {msg}')
        if level == 'PASS': n_pass += 1
        elif level == 'WARN': n_warn += 1
        else: n_fail += 1

    print(f'\n      Results: {n_pass} PASS  {n_warn} WARN  {n_fail} FAIL')
    if n_fail > 0:
        print('      CRITICAL: Fix FAIL items before interpreting HAR results.')

    # ── PSD analysis ───────────────────────────────────────────────────────
    print(f'\n[3/5] Computing Welch PSD ({args.fmin:.1f}–{args.fmax:.1f} Hz)...')
    freqs, psd_db, peaks = compute_har_psd(
        data, args.fmin, args.fmax, args.nperseg, args.subcarrier)

    if peaks:
        print(f'      Detected {len(peaks)} significant peaks:')
        for i, (pf, pp) in enumerate(peaks[:10]):
            band_str = ''
            if 0.2 <= pf <= 0.5:   band_str = ' [RESPIRATION]'
            elif 0.5 < pf <= 2.0:  band_str = ' [SLOW MOTION]'
            elif 2.0 < pf <= 8.0:  band_str = ' [TREMOR/TAP]'
            elif pf > 8.0:          band_str = ' [GESTURE]'
            print(f'        Peak {i+1}: {pf:.3f} Hz  ({pp:.1f} dB/Hz){band_str}')

        resp_peaks = [f for f, _ in peaks if 0.2 <= f <= 0.5]
        if resp_peaks:
            print(f'\n  ✅  RESPIRATION DETECTED: {resp_peaks[0]:.3f} Hz '
                  f'(range: 12–30 breaths/min = 0.2–0.5 Hz)')
        else:
            print(f'\n  ℹ️  No respiration peak (0.2–0.5 Hz) detected.')
            print(f'      Possible causes: subject not present, insufficient capture duration,')
            print(f'      or SNR too low for micro-motion sensing.')
    else:
        print('      No significant peaks found in HAR band.')

    # ── Generate plots ──────────────────────────────────────────────────────
    print(f'\n[4/5] Generating diagnostic plots...')
    plot_magnitude(data, args.session, args.outdir)
    plot_phase(data, args.session, args.outdir)
    plot_spectrogram(data, args.session, args.outdir,
                     args.fmin, args.fmax, args.nperseg, args.subcarrier)

    # ── Final report ───────────────────────────────────────────────────────
    print(f'\n[5/5] Final summary:')
    print(f'      Packets captured : {data["n_pkts"]}')
    N   = data['n_pkts']
    ts  = data['timestamps']
    dur = float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0
    print(f'      Session duration : {dur:.1f} s  ({N/dur if dur>0 else 0:.1f} pkt/s)')
    cfo = data['cfo_hz']
    print(f'      CFO: mean={np.mean(cfo):+.1f} Hz  std={np.std(cfo):.1f} Hz  max={np.max(np.abs(cfo)):.1f} Hz')

    valid_mask = ~np.any(np.isnan(data['H_san']), axis=1)
    print(f'      Valid (non-NaN) CSI records: {valid_mask.sum()} / {N}')

    if valid_mask.sum() > 1:
        var_per = np.var(data['phase_san'][valid_mask], axis=1)
        print(f'      Sanitized phase residual variance: mean={np.mean(var_per):.4f} rad²  '
              f'max={np.max(var_per):.4f} rad²')

    schema_ok = n_fail == 0
    print(f'\n  {"✅ Schema: PASS" if schema_ok else "❌ Schema: FAIL — investigate above errors"}')
    print(f'  {"✅ " if peaks else "⚠️  "}HAR peaks: {len(peaks)} detected')
    print(f'\n  Output plots written to: {os.path.abspath(args.outdir)}')
    print('=' * 72)


if __name__ == '__main__':
    main()
