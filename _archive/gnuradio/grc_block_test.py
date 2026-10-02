# grc_block_test.py -- Tests all epy_blocks with GRC's actual Python
# Run: C:/Users/Vidhisha/radioconda/python.exe grc_block_test.py

import yaml
import sys
import traceback
import os

GRC_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "ofdm_csi_flowgraph.grc")

with open(GRC_FILE, encoding="utf-8") as f:
    doc = yaml.safe_load(f)

from gnuradio import gr
import pmt
print(f"gnuradio {gr.version()}  python {sys.version.split()[0]}")
print()

epy = [b for b in doc["blocks"] if b["id"] == "epy_block"]
ok_count = 0
for b in epy:
    src = b["parameters"]["_source_code"]
    ns  = {"gr": gr, "pmt": pmt, "__name__": "__main__"}
    try:
        exec(src, ns)
        cls = ns.get("blk")
        if cls is None:
            print(f"[!!] {b['name']}: no class blk after exec")
            continue
        inst = cls()
        print(f"[OK] {b['name']}  in_sig={inst.in_sig}  out_sig={inst.out_sig}")
        ok_count += 1
    except Exception:
        print(f"[!!] {b['name']}:")
        traceback.print_exc()
    print()

print(f"Result: {ok_count}/{len(epy)} blocks loaded successfully")
if ok_count == len(epy):
    print("ALL BLOCKS OK -- the .grc file should show green blocks in GRC")
else:
    print("SOME BLOCKS FAILED -- see errors above")
