"""
sync.py — LTF-based fine timing, two-stage CFO correction,
          CSI extraction (H_hat[k]), and pilot-based SCO tracking.

Fixes applied (v4 — all audit issues):
  Bug 1:  Fine CFO uses coarse-corrected LTF windows (no double-count).
  Bug 5:  SCO physical model corrected: b_accum stored in rad/SC, applied
          as phi(k,m) = a_m + (m+1)*b_accum*k (no _SCO_PHASE_SCALE factor).
  Bug 6:  np.unwrap() applied to 8-pilot phase array before LS fit.
  Bug 7:  LTF search window widened to ±64; unconstrained fallback if
          narrow-window peak is below minimum metric.
  Bug 8:  sync_packet() fallback corrected to ltf_start = len(buf) - LTF_LEN.

Physical model for SCO phase ramp (reference):
  phi(k, m) = 2*pi*eps*(m+1)*(Ncp+Nfft)/Nfft * k/Nfft
  The LS-fit slope b_m has units rad/subcarrier-index. It is NOT the same
  as eps (dimensionless ~ppm*1e-6).  We store b_accum in rad/SC and apply
  the correction directly without any extra scale factor.

Reference: phase2 (2).md — Sections 7, 7.1, 7.2, 7.3, 15
"""

import numpy as np
import config as cfg
from waveform import LTF_FREQ

# ─────────────────────────────────────────────────────────────────────────────
# Pre-computed helpers
# ─────────────────────────────────────────────────────────────────────────────

# Active-subcarrier index for each pilot (used in ZF normalization)
_PILOT_ACTIVE_IDX = [cfg.ACTIVE_SUBCARRIERS.index(k) for k in cfg.PILOT_INDICES]

# Known LTF time-domain template (128 samples, no CP)
_LTF_TMPL = np.fft.ifft(LTF_FREQ, n=cfg.FFT_SIZE).astype(np.complex64)

# Minimum cross-correlation metric to accept a narrow-window LTF peak
_LTF_MIN_METRIC = 0.3   # below this → widen search or use fallback


# ─────────────────────────────────────────────────────────────────────────────
# 1. Fine Timing via LTF Cross-Correlation  (Issue 7 fix)
# ─────────────────────────────────────────────────────────────────────────────

def find_ltf_timing(rx: np.ndarray,
                    coarse_cfo_hz: float = 0.0,
                    fs: float = cfg.FS_FFT,
                    pkt_start: int = 0) -> int:
    """
    Locate the start of the LTF CP by cross-correlation with the 128-sample
    LTF body template after coarse-CFO correction.

    Issue 7 fix — STF-aware anchored search:
      1. Narrow window: LTF body expected at pkt_start+STF_LEN+64 ± 64.
      2. If peak metric < _LTF_MIN_METRIC → widen, but floor search at
         pkt_start+STF_LEN//2 to block false STF autocorrelation peaks.
      3. ltf_start = peak_of_body - 64  (CP precedes body by 64 samples).

    Args:
        rx:            complex64 received samples.
        coarse_cfo_hz: coarse CFO estimate from the detector (Hz).
        fs:            sample rate (Hz).
        pkt_start:     index in rx[] where the STF begins. Default 0 = rx_buf
                       already starts at the packet (rx_sim / sync_packet usage).
                       Pass lead-in length when rx includes pre-packet noise.
    Returns:
        ltf_start: index in rx[] of the first sample of the LTF CP.
    """
    n    = np.arange(len(rx))
    rx_c = (rx * np.exp(-1j * 2 * np.pi * coarse_cfo_hz / fs * n)
            ).astype(np.complex64)

    if len(rx_c) < len(_LTF_TMPL):
        return pkt_start

    corr = np.abs(np.correlate(rx_c, _LTF_TMPL, mode='valid'))
    if len(corr) == 0:
        return pkt_start

    # Normalize cross-correlation to [0, 1] for thresholding
    corr_norm = corr / (corr.max() + 1e-12)

    # Narrow window: LTF body expected at pkt_start+STF_LEN+64, allow ±64
    expected  = pkt_start + cfg.STF_LEN + 64
    lo_narrow = max(0, expected - 64)
    hi_narrow = min(len(corr) - 1, expected + 64)

    peak_narrow = int(np.argmax(corr[lo_narrow:hi_narrow + 1])) + lo_narrow
    metric_narrow = float(corr_norm[peak_narrow])

    if metric_narrow >= _LTF_MIN_METRIC:
        peak = peak_narrow
    else:
        # Widen search but floor at pkt_start+STF_LEN//2.
        # This blocks false peaks from the STF's periodic autocorrelation
        # partially matching the LTF template in the pre-LTF region.
        lo_wide = max(0, pkt_start + cfg.STF_LEN // 2)
        peak = int(np.argmax(corr[lo_wide:])) + lo_wide

    # peak is index of LTF body start; LTF CP precedes it by 64 samples
    ltf_start = max(pkt_start, peak - 64)
    return ltf_start


# ─────────────────────────────────────────────────────────────────────────────
# 2. Fine CFO Estimation from LTF (Section 7.1)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_fine_cfo(ltf1: np.ndarray,
                      ltf2: np.ndarray,
                      fs: float = cfg.FS_FFT) -> float:
    """
    Fine CFO from phase rotation between the two LTF repetitions.

    Inputs must already have coarse CFO removed so this measures only the
    residual fine component.  The two LTF symbols are identical at TX; any
    phase rotation between them is caused by the residual frequency offset.

    Returns:
        fine_cfo_hz (float) — residual fine CFO in Hz.
    """
    n = min(len(ltf1), len(ltf2), cfg.FFT_SIZE)
    if n < 8:
        return 0.0
    l1, l2 = ltf1[:n], ltf2[:n]
    phi = float(np.angle(np.sum(l2 * np.conj(l1))))
    T   = cfg.FFT_SIZE / fs        # one OFDM symbol duration (s)
    return phi / (2 * np.pi * T)


# ─────────────────────────────────────────────────────────────────────────────
# 3. CFO Correction
# ─────────────────────────────────────────────────────────────────────────────

def apply_cfo_correction(samples: np.ndarray,
                         cfo_hz: float,
                         start_n: int = 0,
                         fs: float = cfg.FS_FFT) -> np.ndarray:
    """
    De-rotate samples by cfo_hz, referenced from absolute sample index start_n.

    The correction is exp(-j*2*pi*cfo_hz/fs * n) for n = start_n … start_n+N-1.
    Use start_n=0 when correcting a packet buffer from its own first sample.
    Use start_n=sym_start when correcting a DATA symbol slice taken from a
    larger buffer, so the phase ramp is continuous across symbols.
    """
    n = np.arange(start_n, start_n + len(samples), dtype=np.float64)
    return (samples * np.exp(-1j * 2 * np.pi * cfo_hz / fs * n)).astype(np.complex64)


# ─────────────────────────────────────────────────────────────────────────────
# 4. CSI Extraction  (Section 7)
#    H_hat[k] = (Y_1[k] + Y_2[k]) / (2 * X[k])
# ─────────────────────────────────────────────────────────────────────────────

def extract_csi(rx_packet: np.ndarray,
                ltf_start: int,
                total_cfo_hz: float = 0.0,
                fs: float = cfg.FS_FFT) -> np.ndarray:
    """
    Extract complex channel estimate H_hat from one received packet.

    H_hat[i] = (Y1[k] + Y2[k]) / (2 * X[k])  for k = ACTIVE_SUBCARRIERS[i]

    CFO correction uses start_n=0 (absolute sample 0 of the packet buffer).
    ltf_start points to the first sample of the LTF CP (64-sample guard).

    Args:
        rx_packet:    complex64 received samples.
        ltf_start:    index of first LTF CP sample in rx_packet.
        total_cfo_hz: coarse + fine CFO in Hz.
    Returns:
        H_hat: complex64 (NUM_ACTIVE,).  All-NaN if buffer too short.
    """
    rx_corr = apply_cfo_correction(rx_packet, total_cfo_hz, start_n=0)

    start1 = ltf_start + 64          # skip 64-sample LTF CP
    start2 = start1 + cfg.FFT_SIZE   # second LTF body

    ltf1_t = rx_corr[start1:start1 + cfg.FFT_SIZE]
    ltf2_t = rx_corr[start2:start2 + cfg.FFT_SIZE]

    if len(ltf1_t) < cfg.FFT_SIZE or len(ltf2_t) < cfg.FFT_SIZE:
        return np.full(cfg.NUM_ACTIVE, np.nan, dtype=np.complex64)

    Y1 = np.fft.fft(ltf1_t, n=cfg.FFT_SIZE).astype(np.complex64)
    Y2 = np.fft.fft(ltf2_t, n=cfg.FFT_SIZE).astype(np.complex64)
    X  = LTF_FREQ   # ±1 BPSK on active subcarriers

    H_hat = np.empty(cfg.NUM_ACTIVE, dtype=np.complex64)
    for i, k in enumerate(cfg.ACTIVE_SUBCARRIERS):
        b  = cfg.k_to_bin(k)
        Xk = X[b]
        H_hat[i] = (Y1[b] + Y2[b]) / (2.0 * Xk) if abs(Xk) > 1e-9 else 0j

    return H_hat


# ─────────────────────────────────────────────────────────────────────────────
# 4b. Single-Antenna CSI Phase Sanitization / Linear Detrending
# ─────────────────────────────────────────────────────────────────────────────

def sanitize_csi_phase(H_hat: np.ndarray) -> tuple:
    """
    Linear regression phase detrending / phase sanitization for single-antenna CSI.
    
    Removes timing-jitter induced phase slope (-2*pi * k/N_FFT * delta_tau) and 
    carrier phase offset (phi_0) across active subcarriers k in {-53...53}, k!=0.

    Args:
        H_hat: complex64 array of shape (NUM_ACTIVE,) (106 subcarriers).
    Returns:
        H_sanitized: complex64 array (NUM_ACTIVE,) with linear phase ramp detrended.
        phase_sanitized: float64 array (NUM_ACTIVE,) containing detrended phase in radians.
        slope: OLS slope b (rad / subcarrier index).
        intercept: OLS intercept a (radians).
    """
    if np.any(np.isnan(H_hat)):
        return (np.full(cfg.NUM_ACTIVE, np.nan, dtype=np.complex64),
                np.full(cfg.NUM_ACTIVE, np.nan, dtype=np.float64),
                0.0, 0.0)

    raw_phase = np.angle(H_hat)
    unwrapped_phase = np.unwrap(raw_phase)
    
    k_vals = np.array(cfg.ACTIVE_SUBCARRIERS, dtype=np.float64)
    A = np.column_stack([np.ones_like(k_vals), k_vals])
    coefs, _, _, _ = np.linalg.lstsq(A, unwrapped_phase, rcond=None)
    intercept, slope = float(coefs[0]), float(coefs[1])
    
    phase_sanitized = unwrapped_phase - (intercept + slope * k_vals)
    mag = np.abs(H_hat)
    H_sanitized = (mag * np.exp(1j * phase_sanitized)).astype(np.complex64)
    
    return H_sanitized, phase_sanitized, slope, intercept


# ─────────────────────────────────────────────────────────────────────────────
# 5. Pilot-based SCO Tracking per DATA Symbol  (Section 7.2)
#
# Physical model:
#   phi(k, m) = 2*pi*eps*(m+1)*(Ncp+Nfft)/Nfft * k/Nfft
#   where eps = fractional clock error (dimensionless).
#
# The LS fit gives slope b_m in rad / subcarrier-index (not rad/sample or ppm).
# b_m relates to eps as:  b_m = 2*pi*eps*(Ncp+Nfft)/Nfft / Nfft
#
# Issue 5 fix: store b_accum in rad/SC.  Apply correction:
#   phi_corr(k) = a_m  +  (m+1) * b_accum * k
#   — a_m  removes residual CPE (intercept)
#   — (m+1)*b_accum*k removes the cumulative SCO slope
#   No _SCO_PHASE_SCALE factor: the scale is already embedded in b_m.
#
# Issue 6 fix: np.unwrap() applied to pilot phases before LS fit to avoid
#   wrapping discontinuities when the slope exceeds ±π.
# ─────────────────────────────────────────────────────────────────────────────

def sco_correct_symbol(Y_fft: np.ndarray,
                       H_hat: np.ndarray,
                       symbol_idx: int = 0,
                       sco_b_accum: float = 0.0) -> tuple:
    """
    Estimate and correct SCO-induced phase ramp and residual CPE from pilots.

    Issue 5 fix — correct physical model:
      Fit:       phi_pilot[i] = a_m + b_m * k[i]    (rad, rad/SC)
      Smooth:    b_accum = (1-alpha)*b_accum_prev + alpha*b_m
      Correct:   phi(k)  = a_m + (symbol_idx+1) * b_accum * k

    Issue 6 fix — phase unwrapping:
      np.unwrap() is applied to the 8-pilot phase array before LS fit.

    ZF normalization (Bug 5b from previous round, retained):
      z = Y[b] * conj(H[k]) / (|H[k]|^2 + eps) * P[i]
      angle(z) = SCO_phase(k) + CPE   (channel phase removed)

    Args:
        Y_fft:       complex64 (FFT_SIZE,) — one FFT'd DATA symbol.
        H_hat:       complex64 (NUM_ACTIVE,) — channel estimate from LTF.
        symbol_idx:  0-based DATA symbol index m within the packet.
        sco_b_accum: running slope estimate b_accum (rad/SC) from previous symbols.
    Returns:
        (Y_corrected, new_b_accum, a_m)
        Y_corrected:  complex64 (FFT_SIZE,) with SCO + CPE removed.
        new_b_accum:  updated running slope (pass to next symbol call).
        a_m:          residual CPE intercept for this symbol (radians).
    """
    k_vals       = np.array(cfg.PILOT_INDICES, dtype=np.float64)
    pilot_phases = np.empty(cfg.NUM_PILOTS, dtype=np.float64)

    for i, k in enumerate(cfg.PILOT_INDICES):
        b      = cfg.k_to_bin(k)
        h_idx  = _PILOT_ACTIVE_IDX[i]
        H_k    = H_hat[h_idx]
        H_mag2 = float(np.abs(H_k) ** 2) + 1e-10
        # ZF: remove channel phase + pilot polarity → pure SCO+CPE phase
        z      = Y_fft[b] * np.conj(H_k) / H_mag2 * float(cfg.PILOT_POLARITY[i])
        pilot_phases[i] = float(np.angle(z))

    # Issue 6 fix: unwrap across the 8 pilots before LS fit
    pilot_phases = np.unwrap(pilot_phases)

    # Least-squares linear fit: phi = a_m + b_m * k
    A              = np.column_stack([np.ones_like(k_vals), k_vals])
    coefs, _, _, _ = np.linalg.lstsq(A, pilot_phases, rcond=None)
    a_m, b_m = float(coefs[0]), float(coefs[1])

    # Issue 5 fix: update running slope (exponential smoothing, rad/SC)
    # b_m is the LS slope measured directly from pilot phases at this symbol.
    # It already reflects the cumulative SCO phase at symbol m.
    # We apply the smoothed estimate directly (no (m+1) multiplier needed —
    # the accumulation is already captured in b_m growing across symbols).
    alpha       = 0.3 if symbol_idx > 0 else 1.0
    new_b_accum = sco_b_accum * (1.0 - alpha) + b_m * alpha

    # Apply correction to all active subcarriers (use float64 to avoid drift)
    # phi(k) = a_m + new_b_accum * k
    # a_m  → removes residual CPE / inter-symbol phase drift
    # new_b_accum * k → removes the frequency-linear SCO phase ramp
    Y_corr = Y_fft.astype(np.complex128).copy()
    for k in cfg.ACTIVE_SUBCARRIERS:
        b   = cfg.k_to_bin(k)
        phi = a_m + new_b_accum * k   # Issue 5: b_accum directly, no (m+1) factor
        Y_corr[b] = Y_fft[b] * np.exp(-1j * phi)

    return Y_corr.astype(np.complex64), new_b_accum, a_m


# ─────────────────────────────────────────────────────────────────────────────
# 6. Full Synchronization Pipeline for One Packet
# ─────────────────────────────────────────────────────────────────────────────

def sync_packet(rx_buf: np.ndarray,
                coarse_cfo_hz: float,
                n_data_symbols: int = 0) -> dict:
    """
    Full synchronization and CSI extraction for one detected packet.

    Steps:
      1. Fine timing: find_ltf_timing() with widened search window.
      2. Fine CFO:    estimate from coarse-corrected LTF windows (no double-count).
      3. CSI:         extract_csi() with total (coarse+fine) CFO.
      4. SCO:         sco_correct_symbol() for each DATA symbol (if requested).

    Issue 8 fix — fallback formula:
      When the buffer is too short for the found ltf_start, the correct fallback
      is ltf_start = len(rx_buf) - cfg.LTF_LEN  (LTF_LEN = 320 = CP64 + 2*FFT128).
      Previous code subtracted an extra CP_LEN+FFT_SIZE = 160 = one symbol too many.

    Args:
        rx_buf:         raw received samples starting from ~packet start.
        coarse_cfo_hz:  coarse CFO from detector.py (Hz).
        n_data_symbols: DATA symbols to SCO-track and return (0 = CSI only).
    Returns:
        dict:
            'H_hat'     : complex64 (NUM_ACTIVE,)
            'ltf_start' : int
            'total_cfo' : float (Hz)
            'data_syms' : list of SCO-corrected FFT arrays (complex64)
    """
    result = {}

    # ── Step 1: Fine timing ───────────────────────────────────────────────────
    ltf_start = find_ltf_timing(rx_buf, coarse_cfo_hz)

    # Issue 8 fix: correct fallback when buffer too short for found ltf_start.
    # Need at least ltf_start + LTF_LEN samples to extract both LTF bodies.
    s1_body = ltf_start + 64                   # first LTF body start
    s2_body = s1_body + cfg.FFT_SIZE           # second LTF body start
    if s2_body + cfg.FFT_SIZE > len(rx_buf):
        # Fallback: place LTF at the end of the available buffer
        ltf_start = max(0, len(rx_buf) - cfg.LTF_LEN)   # was wrong before
        s1_body   = ltf_start + 64
        s2_body   = s1_body + cfg.FFT_SIZE

    result['ltf_start'] = ltf_start

    # ── Step 2: Fine CFO from coarse-corrected LTF windows (Bug 1 fix) ───────
    # Apply coarse correction first so estimate_fine_cfo sees only residual.
    rx_coarse = apply_cfo_correction(rx_buf, coarse_cfo_hz, start_n=0)
    ltf1      = rx_coarse[s1_body:s1_body + cfg.FFT_SIZE].astype(np.complex64)
    ltf2      = rx_coarse[s2_body:s2_body + cfg.FFT_SIZE].astype(np.complex64)
    fine_cfo  = estimate_fine_cfo(ltf1, ltf2)
    total_cfo = coarse_cfo_hz + fine_cfo
    result['total_cfo'] = total_cfo

    # ── Step 3: CSI extraction & phase sanitization ───────────────────────────
    H_hat = extract_csi(rx_buf, ltf_start, total_cfo)
    H_sanitized, phase_sanitized, slope, intercept = sanitize_csi_phase(H_hat)
    result['H_hat']           = H_hat
    result['H_sanitized']     = H_sanitized
    result['phase_sanitized'] = phase_sanitized
    result['phase_slope']     = slope
    result['phase_intercept'] = intercept

    # ── Step 4: SCO tracking on DATA symbols (optional) ──────────────────────
    data_syms   = []
    sco_b_accum = 0.0
    if n_data_symbols > 0 and not np.any(np.isnan(H_hat)):
        # DATA symbols start after preamble: LTF_end + SIGNAL
        # ltf_start points to LTF CP; LTF field = 320 samples; SIGNAL = 160
        sig_end = ltf_start + cfg.LTF_LEN + cfg.SIG_LEN
        for m in range(n_data_symbols):
            sym_cp_start  = sig_end + m * cfg.SYMBOL_LEN
            sym_body_start = sym_cp_start + cfg.CP_LEN
            sym_body_end   = sym_body_start + cfg.FFT_SIZE
            if sym_body_end > len(rx_buf):
                break
            sym_t = rx_buf[sym_body_start:sym_body_end].astype(np.complex64)
            # CFO correction referenced from absolute sample sym_body_start
            sym_t = apply_cfo_correction(sym_t, total_cfo, start_n=sym_body_start)
            Y_fft = np.fft.fft(sym_t, n=cfg.FFT_SIZE).astype(np.complex64)
            # SCO correction (Issues 5+6 fixed)
            Y_corr, sco_b_accum, _ = sco_correct_symbol(
                Y_fft, H_hat, symbol_idx=m, sco_b_accum=sco_b_accum)
            data_syms.append(Y_corr)

    result['data_syms'] = data_syms
    return result
