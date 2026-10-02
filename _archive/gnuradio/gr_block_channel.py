"""
gr_block_channel.py  --  GNU Radio Python Block: Channel Impairment Model
==========================================================================
Applies configurable AWGN, CFO, SCO, and multipath to the IQ stream.
Works as a through-block (1 in, 1 out).

In GNU Radio Companion:
  Type       : Python Block (Embedded / OOT)
  In ports   : 1 x complex64
  Out ports  : 1 x complex64
  Parameters : noise_voltage, cfo_hz, sco_ppm, tap_string
               tap_string format: "1+0j,0.3-0.1j"
"""

import numpy as np
import sys, os

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import config as cfg

try:
    import gnuradio.gr as gr
    _GR_AVAILABLE = True
except ImportError:
    _GR_AVAILABLE = False


# ---------------------------------------------------------------------------
# Core impairment engine (shared by GRC block and standalone tests)
# ---------------------------------------------------------------------------

class ChannelImpairment:
    """
    Stateful single-channel impairment model.
    Maintains phase accumulators for CFO and SCO.
    """
    def __init__(self,
                 noise_voltage: float = 0.005,
                 cfo_hz:        float = 0.0,
                 sco_ppm:       float = 0.0,
                 taps:          list  = None,
                 fs:            float = cfg.FS_FFT):
        self.noise_voltage = noise_voltage
        self.cfo_hz        = cfo_hz
        self.sco_ppm       = sco_ppm
        self.taps          = np.array(taps or [1.0 + 0j], dtype=np.complex64)
        self.fs            = fs
        self._sample_idx   = 0     # running sample counter for CFO phase

    def apply(self, samples: np.ndarray) -> np.ndarray:
        """Apply all impairments; maintains phase continuity across calls."""
        y = samples.astype(np.complex64).copy()

        # Multipath convolution
        if len(self.taps) > 1:
            y = np.convolve(y, self.taps)[:len(samples)]

        # CFO rotation (phase-continuous across block calls)
        n = np.arange(self._sample_idx,
                      self._sample_idx + len(y), dtype=np.float64)
        y = y * np.exp(1j * 2 * np.pi * self.cfo_hz / self.fs * n
                       ).astype(np.complex64)

        # SCO: scale sample index by (1 + sco_ppm*1e-6)
        # Implemented as a resampling phase ramp (simplified linear)
        if self.sco_ppm != 0:
            sco_factor = self.sco_ppm * 1e-6
            phase_ramp = 2 * np.pi * sco_factor * n / self.fs
            y = y * np.exp(1j * phase_ramp).astype(np.complex64)

        # AWGN
        noise = (self.noise_voltage / np.sqrt(2)) * (
            np.random.randn(len(y)).astype(np.float32) +
            1j * np.random.randn(len(y)).astype(np.float32))
        y = y + noise.astype(np.complex64)

        self._sample_idx += len(y)
        return y.astype(np.complex64)

    @staticmethod
    def parse_taps(tap_string: str) -> list:
        """
        Parse a comma-separated complex tap string.
        Example: "1+0j,0.3-0.1j" -> [1+0j, 0.3-0.1j]
        """
        result = []
        for t in tap_string.split(","):
            t = t.strip()
            if t:
                result.append(complex(t))
        return result if result else [1.0 + 0j]


# ---------------------------------------------------------------------------
# GNU Radio sync_block wrapper
# ---------------------------------------------------------------------------

if _GR_AVAILABLE:
    class blk(gr.sync_block):
        """
        GNU Radio Through Block: Custom OFDM Channel Impairment
        --------------------------------------------------------
        Ports:
          Input  0: complex64 stream (clean IQ)
          Output 0: complex64 stream (impaired IQ)
        """

        def __init__(self,
                     noise_voltage: float = 0.005,
                     cfo_hz:        float = 0.0,
                     sco_ppm:       float = 0.0,
                     tap_string:    str   = "1+0j"):
            gr.sync_block.__init__(
                self,
                name   = "Channel Impairment (AWGN+CFO+SCO+MP)",
                in_sig = [np.complex64],
                out_sig= [np.complex64],
            )
            taps = ChannelImpairment.parse_taps(tap_string)
            self._ch = ChannelImpairment(
                noise_voltage=noise_voltage,
                cfo_hz=cfo_hz,
                sco_ppm=sco_ppm,
                taps=taps,
            )

        def work(self, input_items, output_items):
            inp = input_items[0]
            output_items[0][:] = self._ch.apply(inp)
            return len(inp)
