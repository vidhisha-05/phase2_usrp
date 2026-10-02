"""
test_blocks_in_grc_python.py
============================
Finds GRC's Python interpreter and tests that all epy_blocks
load without errors (simulates exactly what GRC does).

Run from any Python:
    python test_blocks_in_grc_python.py
"""

import subprocess
import sys
import os
import glob

GRC_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "ofdm_csi_flowgraph.grc")

# ── Find all Python installs that might have gnuradio ─────────────────────
CANDIDATES = [
    sys.executable,
    r"C:\radioconda\python.exe",
    r"C:\radioconda\envs\base\python.exe",
    r"C:\ProgramData\radioconda\python.exe",
    r"C:\Program Files\GNURadio\gr-python311\python.exe",
    r"C:\Program Files\GNURadio\bin\python3.exe",
    r"C:\msys64\usr\bin\python3.exe",
    r"C:\msys64\mingw64\bin\python3.exe",
]

# Also search via glob
for pattern in [
    r"C:\radioconda*\python.exe",
    r"C:\*radio*\python*.exe",
    r"C:\*GNURadio*\**\python*.exe",
    r"C:\Program Files\**\gnuradio\**\python*.exe",
]:
    CANDIDATES += glob.glob(pattern, recursive=True)[:3]

TEST_CODE = r"""
import sys, traceback, yaml

GRC_FILE = r'__GRC_FILE__'

with open(GRC_FILE, encoding='utf-8') as f:
    doc = yaml.safe_load(f)

try:
    from gnuradio import gr, pmt
except ImportError as e:
    print(f'NOGNURADIO:{e}')
    sys.exit(1)

epy = [b for b in doc['blocks'] if b['id'] == 'epy_block']
errors = []
for b in epy:
    src = b['parameters']['_source_code']
    ns  = {'gr': gr, 'pmt': pmt}
    try:
        exec(src, ns)
        blk_cls = ns.get('blk')
        if blk_cls is None:
            errors.append(f"{b['name']}: no class blk")
            continue
        inst = blk_cls()
        print(f"OK:{b['name']}")
    except Exception as ex:
        errors.append(f"{b['name']}: {ex}")
        print(f"FAIL:{b['name']}:{ex}")

if errors:
    sys.exit(2)
sys.exit(0)
""".replace('__GRC_FILE__', GRC_FILE.replace('\\', '\\\\'))

print("=" * 60)
print("  GRC Block Loader Test")
print("=" * 60)
print()

found_grc_python = None
for py in CANDIDATES:
    if not os.path.isfile(py):
        continue
    result = subprocess.run(
        [py, "-c", TEST_CODE],
        capture_output=True, text=True, timeout=30
    )
    out    = result.stdout.strip()
    err    = result.stderr.strip()

    if "NOGNURADIO" in out:
        print(f"  SKIP  {py}  (no gnuradio)")
        continue

    if result.returncode == 0:
        print(f"  [OK]  {py}")
        print(f"        Blocks loaded: {out}")
        found_grc_python = py
        break
    else:
        print(f"  [!!]  {py}")
        if out:
            print(f"        stdout: {out[:200]}")
        if err:
            print(f"        stderr: {err[:200]}")

print()
if found_grc_python:
    print(f"GRC Python found: {found_grc_python}")
    print()
    print("Open the flowgraph with:")
    grc_dir = os.path.dirname(found_grc_python)
    grc_exe = os.path.join(grc_dir, "gnuradio-companion.exe")
    if not os.path.exists(grc_exe):
        grc_exe = os.path.join(grc_dir, "Scripts", "gnuradio-companion.exe")
    print(f"  {grc_exe}  {GRC_FILE}")
else:
    print("No Python with gnuradio found in standard locations.")
    print()
    print("To find your GRC Python, open GNU Radio Companion and go to:")
    print("  Tools -> Options -> Python Executable")
    print("  OR check  Tools -> More -> Python console  and type: import sys; sys.executable")
    print()
    print("Then re-run this test with that Python:")
    print("  <grc-python-path>  test_blocks_in_grc_python.py")
