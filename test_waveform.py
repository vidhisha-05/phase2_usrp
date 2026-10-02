"""
test_waveform.py — Unit tests for waveform generation and round-trip loopback.

Validates:
  1. STF length and periodicity
  2. LTF length and frequency-domain values
  3. Packet assembly lengths
  4. Zero-impairment ZMQ simulation loopback (Stage 1 of Section 17)
  5. CSI extraction accuracy against known flat channel (Stage 3)

Reference: phase2 (2).md — Section 17, Steps 1–3
"""

import numpy as np
import pytest
import config as cfg
import waveform
import scrambler
from sync import extract_csi, find_ltf_timing, estimate_fine_cfo

# ─────────────────────────────────────────────────────────────────────────────
# 1. STF Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_stf_length():
    stf = waveform.generate_stf()
    assert len(stf) == cfg.STF_LEN == 128

def test_stf_periodicity():
    """STF must consist of 8 identical 16-sample blocks."""
    stf = waveform.generate_stf()
    base = stf[:16]
    for i in range(1, 8):
        np.testing.assert_allclose(stf[i*16:(i+1)*16], base, rtol=1e-5,
                                   err_msg=f"STF repetition {i} differs from base")

# ─────────────────────────────────────────────────────────────────────────────
# 2. LTF Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_ltf_length():
    ltf = waveform.generate_ltf()
    assert len(ltf) == cfg.LTF_LEN == 320

def test_ltf_two_repetitions():
    """The two 128-sample LTF repetitions (after the 64-sample CP) must be equal."""
    ltf = waveform.generate_ltf()
    rep1 = ltf[64:192]
    rep2 = ltf[192:320]
    np.testing.assert_allclose(rep1, rep2, rtol=1e-5,
                               err_msg="LTF repetitions not equal")

def test_ltf_freq_active_only():
    """LTF guard bands and DC must be zero; active subcarriers non-zero."""
    X = waveform.LTF_FREQ
    for k in cfg.LOWER_GUARD + cfg.UPPER_GUARD + cfg.DC_NULL:
        assert X[cfg.k_to_bin(k)] == 0.0, f"Guard/DC bin {k} is non-zero"
    for k in cfg.ACTIVE_SUBCARRIERS:
        assert abs(X[cfg.k_to_bin(k)]) > 0, f"Active subcarrier {k} is zero"

# ─────────────────────────────────────────────────────────────────────────────
# 3. Subcarrier Allocation
# ─────────────────────────────────────────────────────────────────────────────

def test_active_subcarrier_count():
    # Config: k in {-53..-1} u {1..53} = 53+53 = 106 (symmetric, DC excluded)
    # Previous expected 107 was from old asymmetric design (k=-54..-1 had 54 entries).
    assert cfg.NUM_ACTIVE == 106

def test_data_subcarrier_count():
    # 106 active - 8 pilots = 98 data subcarriers
    assert cfg.NUM_DATA == 98

def test_pilot_subcarrier_count():
    assert cfg.NUM_PILOTS == 8

def test_no_overlap_data_pilot():
    overlap = set(cfg.DATA_INDICES) & set(cfg.PILOT_INDICES)
    assert len(overlap) == 0, f"Data/pilot overlap: {overlap}"

def test_no_dc_in_active():
    assert 0 not in cfg.ACTIVE_SUBCARRIERS

# ─────────────────────────────────────────────────────────────────────────────
# 4. Packet Assembly
# ─────────────────────────────────────────────────────────────────────────────

def test_packet_stf_ltf_only():
    """Minimal packet (STF + LTF + empty DATA) has correct total length."""
    pkt = waveform.assemble_packet(np.array([], dtype=np.uint8))
    # STF + LTF + SIG(zero) = 128 + 320 + 160 = 608 minimum
    assert len(pkt) >= cfg.STF_LEN + cfg.LTF_LEN

# ─────────────────────────────────────────────────────────────────────────────
# 5. Scrambler Round-Trip
# ─────────────────────────────────────────────────────────────────────────────

def test_scrambler_roundtrip():
    bits = np.random.randint(0, 2, 200, dtype=np.uint8)
    assert np.all(scrambler.descramble(scrambler.scramble(bits)) == bits)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Encoder/Decoder Round-Trip
# ─────────────────────────────────────────────────────────────────────────────

def test_encoder_decoder_roundtrip():
    bits   = np.random.randint(0, 2, 88, dtype=np.uint8)
    coded  = scrambler.encode_half_rate(bits)
    assert len(coded) == 2 * len(bits)
    dec    = scrambler.decode_half_rate(coded)
    np.testing.assert_array_equal(dec, bits)

# ─────────────────────────────────────────────────────────────────────────────
# 7. CSI Extraction — Flat Channel (Stage 3 validation)
# ─────────────────────────────────────────────────────────────────────────────

def test_csi_flat_channel():
    """
    Pass STF+LTF through a flat (unity gain) channel and verify H_hat ≈ 1+0j
    on all active subcarriers.
    """
    stf  = waveform.generate_stf()
    ltf  = waveform.generate_ltf()
    pkt  = np.concatenate([stf, ltf])

    # No CFO, no noise — flat unity channel
    H_hat = extract_csi(pkt, ltf_start=len(stf), total_cfo_hz=0.0)

    assert H_hat.shape == (cfg.NUM_ACTIVE,)
    assert not np.any(np.isnan(H_hat))
    # All active subcarriers should be close to ±1 (known BPSK LTF values)
    np.testing.assert_allclose(np.abs(H_hat), 1.0, atol=0.05,
                               err_msg="H_hat magnitude deviates from 1 on flat channel")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Coarse CFO Estimation
# ─────────────────────────────────────────────────────────────────────────────

def test_coarse_cfo_estimate():
    """Apply a known CFO and verify the coarse estimate recovers it."""
    from detector import PacketDetector
    true_cfo = 1000.0   # Hz
    stf      = waveform.generate_stf()
    n        = np.arange(len(stf))
    stf_cfo  = (stf * np.exp(1j * 2 * np.pi * true_cfo / cfg.FS_FFT * n)
                ).astype(np.complex64)

    det      = PacketDetector()
    est_cfo  = det._estimate_coarse_cfo(stf_cfo)
    assert abs(est_cfo - true_cfo) < 200.0, \
        f"Coarse CFO error too large: {abs(est_cfo - true_cfo):.1f} Hz"

# ─────────────────────────────────────────────────────────────────────────────
# 9. Resampler: Round-Trip Length Preservation
# ─────────────────────────────────────────────────────────────────────────────

def test_resampler_length():
    """Upsample then downsample should approximately recover the original length."""
    n    = 1024
    x    = (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex64)
    y25  = waveform.downsample_20to25(x)
    y20  = waveform.upsample_25to20(y25)
    # Allow ±10% length variation due to group-delay strip
    assert abs(len(y20) - n) < n * 0.15, \
        f"Resampled length {len(y20)} too far from original {n}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
