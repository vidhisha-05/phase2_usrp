"""Hardware readiness check script."""
import sys, os, py_compile, inspect
sys.path.insert(0, 'd:/phase2')

results = []

# 1. All core files present
REQUIRED = ['config','waveform','scrambler','demod','sync','detector',
            'channel_bridge','tx_hardware','rx_hardware','rx_sim',
            'tx_sim','logger','validate_pre_b210','plot_project']
for name in REQUIRED:
    path = f'd:/phase2/{name}.py'
    ok   = os.path.exists(path)
    results.append((f'File {name}.py exists', ok))

# 2. All files compile
for name in REQUIRED:
    path = f'd:/phase2/{name}.py'
    try:
        py_compile.compile(path, doraise=True)
        results.append((f'{name}.py compiles', True))
    except Exception as e:
        results.append((f'{name}.py compiles', False, str(e)[:80]))

# 3. config.py has all required B210 constants
import config as cfg
b210_attrs = ['GUARD_SAMPLES','GUARD_SAMPLES_BB','USRP_CENTER_FREQ',
              'USRP_TX_GAIN','USRP_RX_GAIN','USRP_SUBDEV_SPEC',
              'FS_HW','FS_FFT','NUM_RX_CHANNELS','HDF5_FILE_PATH']
for a in b210_attrs:
    results.append((f'cfg.{a} defined', hasattr(cfg, a)))

# 4. Key constants sane
results.append(('GUARD_SAMPLES >= 512',   cfg.GUARD_SAMPLES >= 512))
results.append(('GUARD_SAMPLES = 1024',   cfg.GUARD_SAMPLES == 1024))
results.append(('GUARD_SAMPLES_BB = 820', cfg.GUARD_SAMPLES_BB == 820))
results.append(('FS_HW = 25 MS/s',        cfg.FS_HW == 25e6))
results.append(('FS_FFT = 20 MS/s',       cfg.FS_FFT == 20e6))
results.append(('FFT_SIZE = 128',         cfg.FFT_SIZE == 128))
results.append(('NUM_RX_CHANNELS = 1 (single-channel OTA)',  cfg.NUM_RX_CHANNELS == 1))
results.append(('NUM_DATA = 98 (106 active - 8 pilots)',       cfg.NUM_DATA == 98))
results.append(('STF_LEN = 128',          cfg.STF_LEN == 128))
results.append(('LTF_LEN = 320',          cfg.LTF_LEN == 320))
results.append(('CP_LEN = 32',            cfg.CP_LEN == 32))
results.append(('SYMBOL_LEN = 160',       cfg.SYMBOL_LEN == 160))

# 5. waveform.assemble_packet has idle_samples
import waveform
sig = inspect.signature(waveform.assemble_packet)
results.append(('assemble_packet(idle_samples) exists', 'idle_samples' in sig.parameters))

# 6. guard default correct in waveform
default_idle = sig.parameters['idle_samples'].default
results.append(('idle_samples default = 0 (backward compat)', default_idle == 0))

# 7. tx_hardware has guard_samples param
import tx_hardware
sig2 = inspect.signature(tx_hardware.run_tx_hardware)
results.append(('run_tx_hardware(guard_samples) exists', 'guard_samples' in sig2.parameters))
default_g = sig2.parameters['guard_samples'].default
results.append((f'guard_samples default = cfg.GUARD_SAMPLES ({cfg.GUARD_SAMPLES})',
                default_g == cfg.GUARD_SAMPLES))

# 8. rx_hardware checks
src_rx = open('d:/phase2/rx_hardware.py', encoding='utf-8').read()
results.append(('rx_hardware: _decode_window (CRC path)', '_decode_window' in src_rx))
results.append(('rx_hardware: lookahead buffer (look0_hw)', 'look0_hw' in src_rx))
results.append(('rx_hardware: abs sample tracking (buf_abs_s)', 'buf_abs_s' in src_rx))
results.append(('rx_hardware: overflow warning', 'overflow' in src_rx))
results.append(('rx_hardware: LO lock retry loop', 'lo_timeout_s' in src_rx))
results.append(('rx_hardware: ring buffer 8M samples', '1 << 23' in src_rx))
results.append(('rx_hardware: crc_ok in CSI record', "'crc_ok'" in src_rx))
results.append(('rx_hardware: cfo_hz in CSI record', "'cfo_hz'" in src_rx))

# 9. validate_pre_b210: oracle-free
vsrc = open('d:/phase2/validate_pre_b210.py', encoding='utf-8').read()
results.append(('validate: no coarse_cfo=cfo oracle', 'coarse_cfo=cfo' not in vsrc))
results.append(('validate: no pkt_start=lead oracle', 'pkt_start=lead' not in vsrc))
results.append(('validate: uses PacketDetector', 'PacketDetector()' in vsrc))
results.append(('validate: multi-candidate T5', 'trial_ok' in vsrc))

# 10. Docs exist
results.append(('DEPLOYMENT_RUNBOOK.md exists', os.path.exists('d:/phase2/DEPLOYMENT_RUNBOOK.md')))
# Note: SYSTEM_SPEC.md removed (file does not exist in this project)

# 11. Actually run simulation validation (F-06 fix: was hardcoded True)
import subprocess
try:
    ret = subprocess.run(
        [sys.executable, 'd:/phase2/run_all_validators.py'],
        capture_output=True, timeout=120, cwd='d:/phase2')
    sim_pass = (ret.returncode == 0)
except Exception as e:
    sim_pass = False
results.append(('Simulation validators 4/4 PASS (run_all_validators.py)', sim_pass))

# Print report
n_pass = sum(1 for r in results if r[1])
n_fail = sum(1 for r in results if not r[1])
print(f'Hardware Readiness Check  ({n_pass} PASS / {n_fail} FAIL)')
print('='*60)
for r in results:
    icon = 'PASS' if r[1] else 'FAIL'
    note = f'  <- {r[2]}' if len(r) > 2 else ''
    print(f'  [{icon}]  {r[0]}{note}')
print('='*60)
verdict = 'READY FOR B210 HARDWARE DEPLOYMENT' if n_fail == 0 else f'NOT READY - {n_fail} issues'
print(f'  VERDICT: {verdict}')
