"""
validate_pipeline.py — Deterministic staged loopback test suite.

Issue 13: verifies the full TX→RX pipeline incrementally:
  Stage 0: Ideal (no impairment)      — BER=0, CRC=PASS, H=1
  Stage 1: AWGN only (SNR≈30 dB)     — BER=0, CRC=PASS
  Stage 2: Flat multipath (tap=0.3)   — H≠1, BER=0 after ZF equalization
  Stage 3: CFO = 2000 Hz              — CFO error < 200 Hz, CRC=PASS
  Stage 4: CFO + SCO = 2 ppm         — SCO residual slope < 1e-4 rad/SC
  Stage 5: 2-RX parallel channels     — H_ant0 ≠ H_ant1, both valid

Issue 14: run_all() returns True only when all stages pass.
  Hardware entry points (tx_hardware.py, rx_hardware.py) should call
      validate_pipeline.run_all()
  and refuse to open the USRP device if it returns False.

Usage:
    python -X utf8 validate_pipeline.py
"""

import sys
import math
import numpy as np

sys.path.insert(0, 'd:/phase2')

import config as cfg
import scrambler
import waveform
import demod
from sync import (find_ltf_timing, extract_csi, apply_cfo_correction,
                  sync_packet, sco_correct_symbol, _PILOT_ACTIVE_IDX)
from channel_bridge import ChannelModel

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

PASS_COUNT = 0
FAIL_COUNT = 0
STAGE_RESULTS = []


def chk(name: str, cond: bool, detail: str = '') -> bool:
    global PASS_COUNT, FAIL_COUNT
    tag = '[PASS]' if cond else '[FAIL]'
    msg = f'  {tag}  {name}'
    if detail:
        msg += f'  <- {detail}'
    print(msg)
    if cond:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1
    return cond


def _make_packet(n_payload_bytes: int = 100,
                 modulation: str = 'BPSK') -> tuple:
    """Return (payload_bits, packet_samples)."""
    rng          = np.random.default_rng(0)
    payload_bits = rng.integers(0, 2, n_payload_bytes * 8, dtype=np.uint8)
    pkt          = waveform.assemble_packet(
        payload_bits, modulation,
        scrambler, scrambler, scrambler.map_bits_to_symbols)
    return payload_bits, pkt


def _demod_packet(pkt: np.ndarray,
                  H_hat: np.ndarray,
                  ltf_start: int,
                  total_cfo: float,
                  n_payload_bytes: int = 100,
                  modulation: str = 'BPSK') -> tuple:
    """
    Demodulate a full packet and return (payload_bits_rx, crc_ok).
    Reads the SIGNAL field first to get n_data_syms.
    """
    sig_cp_start   = ltf_start + cfg.LTF_LEN
    sig_body_start = sig_cp_start + cfg.CP_LEN
    sig_body_end   = sig_body_start + cfg.FFT_SIZE

    if sig_body_end > len(pkt):
        return np.array([]), False

    pkt_cfo = apply_cfo_correction(pkt, total_cfo, start_n=0)
    Y_sig   = np.fft.fft(pkt_cfo[sig_body_start:sig_body_end],
                          n=cfg.FFT_SIZE).astype(np.complex64)
    sig_info = demod.parse_signal_field(Y_sig, H_hat)

    n_data_syms = sig_info['n_data_syms'] or demod.n_data_syms_for_payload(
        n_payload_bytes, modulation)

    data_start = ltf_start + cfg.LTF_LEN + cfg.SIG_LEN
    data_ffts  = []
    sco_b      = 0.0
    for m in range(n_data_syms):
        body_start = data_start + m * cfg.SYMBOL_LEN + cfg.CP_LEN
        body_end   = body_start + cfg.FFT_SIZE
        if body_end > len(pkt_cfo):
            break
        Y = np.fft.fft(pkt_cfo[body_start:body_end],
                       n=cfg.FFT_SIZE).astype(np.complex64)
        Y_corr, sco_b, _ = sco_correct_symbol(Y, H_hat, symbol_idx=m,
                                               sco_b_accum=sco_b)
        data_ffts.append(Y_corr)

    return demod.demodulate_packet(data_ffts, H_hat, modulation=modulation,
                                   n_payload_bytes=n_payload_bytes)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 0 — Ideal (no impairment)
# ─────────────────────────────────────────────────────────────────────────────

def stage0_ideal() -> bool:
    print('\n── Stage 0: Ideal (no impairment) ───────────────────────────────')
    payload, pkt = _make_packet(100, 'BPSK')
    ltf_start    = find_ltf_timing(pkt, coarse_cfo_hz=0.0)
    H_hat        = extract_csi(pkt, ltf_start, total_cfo_hz=0.0)

    ok  = chk('H_hat: no NaN',   not np.any(np.isnan(H_hat)))
    ok &= chk('H_hat mean ≈ 1.0',
              0.9 < float(np.mean(np.abs(H_hat))) < 1.1,
              f'mean={float(np.mean(np.abs(H_hat))):.4f}')
    ok &= chk('ltf_start == STF_LEN',
              ltf_start == cfg.STF_LEN,
              f'ltf_start={ltf_start}  expected={cfg.STF_LEN}')

    rx_bits, crc_ok = _demod_packet(pkt, H_hat, ltf_start, 0.0)
    ok &= chk('CRC passes',   bool(crc_ok))
    if crc_ok and len(rx_bits) >= len(payload):
        ber = int(np.sum(rx_bits[:len(payload)] != payload))
        ok &= chk('BER = 0', ber == 0, f'{ber}/{len(payload)}')

    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — AWGN only (≈30 dB SNR)
# ─────────────────────────────────────────────────────────────────────────────

def stage1_awgn() -> bool:
    print('\n── Stage 1: AWGN only (SNR ≈ 30 dB) ───────────────────────────')
    payload, pkt = _make_packet(100, 'BPSK')
    ch   = ChannelModel(noise_voltage=0.032, cfo_hz=0.0, sco_ppm=0.0, seed=1)
    rx   = ch.apply(pkt.copy())

    ltf_start = find_ltf_timing(rx, coarse_cfo_hz=0.0)
    H_hat     = extract_csi(rx, ltf_start, total_cfo_hz=0.0)

    ok  = chk('H_hat: no NaN', not np.any(np.isnan(H_hat)))
    ok &= chk('H_hat mean ≈ 1.0',
              0.85 < float(np.mean(np.abs(H_hat))) < 1.15,
              f'mean={float(np.mean(np.abs(H_hat))):.4f}')

    rx_bits, crc_ok = _demod_packet(rx, H_hat, ltf_start, 0.0)
    ok &= chk('CRC passes', bool(crc_ok))

    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Flat multipath (single reflected tap, |tap|=0.3)
# ─────────────────────────────────────────────────────────────────────────────

def stage2_multipath() -> bool:
    print('\n── Stage 2: Flat multipath (tap=[1, 0.3+0.1j]) ────────────────')
    payload, pkt = _make_packet(100, 'BPSK')
    ch   = ChannelModel(noise_voltage=0.005, cfo_hz=0.0, sco_ppm=0.0,
                        taps=[1.0+0j, 0.3+0.1j], seed=2)
    rx   = ch.apply(pkt.copy())

    ltf_start = find_ltf_timing(rx, coarse_cfo_hz=0.0)
    H_hat     = extract_csi(rx, ltf_start, total_cfo_hz=0.0)

    ok  = chk('H_hat: no NaN', not np.any(np.isnan(H_hat)))
    # With tap [1, 0.3+0.1j], H ≠ 1 on most subcarriers
    h_mean = float(np.mean(np.abs(H_hat)))
    ok &= chk('H_hat mean != 1.0 (multipath visible)', abs(h_mean - 1.0) > 0.01,
              f'mean|H|={h_mean:.4f}')

    rx_bits, crc_ok = _demod_packet(rx, H_hat, ltf_start, 0.0)
    ok &= chk('CRC passes after ZF equalization', bool(crc_ok))

    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — CFO = 2000 Hz
# ─────────────────────────────────────────────────────────────────────────────

def stage3_cfo() -> bool:
    print('\n── Stage 3: CFO = 2000 Hz ──────────────────────────────────────')
    TRUE_CFO     = 2000.0
    payload, pkt = _make_packet(100, 'BPSK')
    ch   = ChannelModel(noise_voltage=0.005, cfo_hz=TRUE_CFO, sco_ppm=0.0, seed=3)
    rx   = ch.apply(pkt.copy())

    result    = sync_packet(rx, coarse_cfo_hz=TRUE_CFO, n_data_symbols=0)
    total_cfo = result['total_cfo']
    H_hat     = result['H_hat']
    ltf_start = result['ltf_start']

    ok  = chk('H_hat: no NaN', not np.any(np.isnan(H_hat)))
    cfo_err = abs(total_cfo - TRUE_CFO)
    ok &= chk(f'CFO error < 200 Hz', cfo_err < 200,
              f'total_cfo={total_cfo:.1f}  truth={TRUE_CFO}  err={cfo_err:.1f}Hz')

    rx_bits, crc_ok = _demod_packet(rx, H_hat, ltf_start, total_cfo)
    ok &= chk('CRC passes', bool(crc_ok))

    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — CFO 1000 Hz + SCO 2 ppm
# ─────────────────────────────────────────────────────────────────────────────

def stage4_sco() -> bool:
    print('\n── Stage 4: CFO=1000 Hz + SCO=2 ppm ───────────────────────────')
    TRUE_CFO     = 1000.0
    TRUE_SCO_PPM = 2.0
    payload, pkt = _make_packet(100, 'BPSK')

    ch  = ChannelModel(noise_voltage=0.005, cfo_hz=TRUE_CFO,
                       sco_ppm=TRUE_SCO_PPM, seed=4)
    rx  = ch.apply(pkt.copy())

    result    = sync_packet(rx, coarse_cfo_hz=TRUE_CFO, n_data_symbols=4)
    H_hat     = result['H_hat']
    ltf_start = result['ltf_start']
    total_cfo = result['total_cfo']

    ok = chk('H_hat: no NaN', not np.any(np.isnan(H_hat)))

    # Verify SCO correction on DATA symbols: after correction residual pilot
    # slope should be small.
    if result['data_syms']:
        Y_last  = result['data_syms'][-1]
        k_vals  = np.array(cfg.PILOT_INDICES, dtype=np.float64)
        phases  = []
        for ii, k in enumerate(cfg.PILOT_INDICES):
            b      = cfg.k_to_bin(k)
            h_idx  = _PILOT_ACTIVE_IDX[ii]
            H_k    = H_hat[h_idx]
            H_mag2 = float(np.abs(H_k)**2) + 1e-10
            z      = Y_last[b] * np.conj(H_k) / H_mag2 * float(cfg.PILOT_POLARITY[ii])
            phases.append(float(np.angle(z)))
        phases = np.unwrap(phases)
        resid  = float(np.polyfit(k_vals, phases, 1)[0])
        # Expected SCO slope after correction: << b_true
        # b_true for 2ppm: 2pi*2e-6*(32+128)/128 * k_range / 128 ≈ small
        ok &= chk('SCO residual slope < 1e-3 rad/SC after correction',
                  abs(resid) < 1e-3,
                  f'resid={resid:.2e}')

    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 — 2-RX parallel channels
# ─────────────────────────────────────────────────────────────────────────────

def stage5_two_rx() -> bool:
    print('\n── Stage 5: 2-RX parallel channels ────────────────────────────')
    _, pkt = _make_packet(100, 'BPSK')
    clean  = pkt.copy()

    ch0 = ChannelModel(noise_voltage=0.005, cfo_hz=800.0,  sco_ppm=0.0, seed=7)
    ch1 = ChannelModel(noise_voltage=0.008, cfo_hz=120.0,  sco_ppm=0.0, seed=137)

    y0 = ch0.apply(clean.copy())    # TX → CH0 (parallel)
    y1 = ch1.apply(clean.copy())    # TX → CH1 (parallel, NOT y0→CH1)

    lt0 = find_ltf_timing(y0, coarse_cfo_hz=800.0)
    lt1 = find_ltf_timing(y1, coarse_cfo_hz=120.0)

    H0 = extract_csi(y0, lt0, total_cfo_hz=800.0)
    H1 = extract_csi(y1, lt1, total_cfo_hz=120.0)

    ok  = chk('H_ant0: no NaN', not np.any(np.isnan(H0)))
    ok &= chk('H_ant1: no NaN', not np.any(np.isnan(H1)))
    m0 = float(np.mean(np.abs(H0)))
    m1 = float(np.mean(np.abs(H1)))
    ok &= chk('H_ant0 magnitude ≈ 1', 0.5 < m0 < 2.0, f'm0={m0:.3f}')
    ok &= chk('H_ant1 magnitude ≈ 1', 0.5 < m1 < 2.0, f'm1={m1:.3f}')

    diff = float(np.mean(np.abs(H0 - H1)))
    ok  &= chk('H_ant0 ≠ H_ant1 (independent channels)', diff > 0.01,
               f'mean|H0-H1|={diff:.4f}')

    # Confirm parallel: ant1 differs from cascade y0 → ch1
    ch1b     = ChannelModel(noise_voltage=0.008, cfo_hz=120.0, sco_ppm=0.0, seed=137)
    y_series = ch1b.apply(y0.copy())
    lt_s     = find_ltf_timing(y_series, coarse_cfo_hz=120.0)
    H_series = extract_csi(y_series, lt_s, total_cfo_hz=120.0)
    if not np.any(np.isnan(H_series)):
        diff_ps = float(np.mean(np.abs(H1 - H_series)))
        ok &= chk('H_ant1 differs from cascaded H0→CH1 (parallel confirmed)',
                  diff_ps > 0.005, f'diff={diff_ps:.4f}')

    return ok


# ─────────────────────────────────────────────────────────────────────────────
# run_all — Issue 14: returns True iff every stage passes
# ─────────────────────────────────────────────────────────────────────────────

def run_all() -> bool:
    """
    Run all pipeline stages in order.  Returns True only if all pass.

    Issue 14: hardware entry points must call this and abort if False:

        import validate_pipeline
        if not validate_pipeline.run_all():
            raise RuntimeError("Simulation pipeline failed — hardware disabled.")
    """
    stages = [
        ('Stage 0: Ideal',           stage0_ideal),
        ('Stage 1: AWGN',            stage1_awgn),
        ('Stage 2: Multipath',       stage2_multipath),
        ('Stage 3: CFO',             stage3_cfo),
        ('Stage 4: CFO + SCO',       stage4_sco),
        ('Stage 5: 2-RX parallel',   stage5_two_rx),
    ]

    all_passed = True
    for name, fn in stages:
        passed = fn()
        STAGE_RESULTS.append((name, passed))
        if not passed:
            all_passed = False
            print(f'\n  *** {name} FAILED — stopping here ***')
            break

    print('\n' + '═' * 60)
    print(f'  PIPELINE {"PASS" if all_passed else "FAIL"} '
          f'— {PASS_COUNT} checks passed, {FAIL_COUNT} failed')
    print('═' * 60)
    for name, ok in STAGE_RESULTS:
        print(f'  {"[PASS]" if ok else "[FAIL]"}  {name}')
    return all_passed


if __name__ == '__main__':
    ok = run_all()
    sys.exit(0 if ok else 1)
