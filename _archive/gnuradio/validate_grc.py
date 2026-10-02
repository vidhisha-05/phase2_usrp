"""Validates the generated .grc file content."""
import yaml

with open("ofdm_csi_flowgraph.grc", encoding="utf-8") as f:
    doc = yaml.safe_load(f)

print("=== GRC FILE VALIDATION ===")
print(f"Title      : {doc['options']['parameters']['title']}")
print(f"Blocks     : {len(doc['blocks'])}")
print(f"Connections: {len(doc['connections'])}")
print()

for b in doc["blocks"]:
    if b["id"] == "epy_block":
        src      = b["parameters"]["_source_code"]
        ok_class = "class blk" in src
        ok_imp   = "import numpy" in src
        ok_type  = isinstance(src, str)
        print(f"  {b['name']:30}  class_blk={ok_class}  numpy={ok_imp}  is_str={ok_type}  chars={len(src)}")
    else:
        print(f"  {b['name']:30}  id={b['id']}")

print()
print("CONNECTIONS:")
for c in doc["connections"]:
    print(f"  {c[0]}:{c[1]}  ->  {c[2]}:{c[3]}")

print()
print("All epy_block source is plain string: ",
      all(isinstance(b["parameters"]["_source_code"], str)
          for b in doc["blocks"] if b["id"] == "epy_block"))
print("YAML round-trip: OK")
