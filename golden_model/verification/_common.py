import os, sys, json, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "results", "verification")
os.makedirs(OUT, exist_ok=True)


def read_hex_bytes(path):
    return bytes(int(t, 16) for t in open(path).read().split())


def save_report(name, lines, data):
    with open(os.path.join(OUT, name + ".txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(OUT, name + ".json"), "w") as f:
        json.dump(data, f, indent=1)
    print("\n".join(lines))
