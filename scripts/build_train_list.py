"""Rebuild the training list from the CURATED data/female_clips/ folder.
Transcripts are looked up (by filename) from the ASR .list files.
Run after the user finishes deleting bad clips from female_clips/.

  python scripts/build_train_list.py   (any python with stdlib is fine)
"""
import os, glob

MORNYE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEMALE_DIR = os.path.join(MORNYE, "data", "female_clips")
OUT = os.path.join(MORNYE, "data", "asr", "sliced_female.list")
# all ASR .list files as transcript sources, except generated *_female.list / the output
SOURCES = [p for p in glob.glob(os.path.join(MORNYE, "data", "asr", "*.list"))
           if not os.path.basename(p).endswith("_female.list") and os.path.abspath(p) != os.path.abspath(OUT)]

# basename -> transcript
tx = {}
for src in SOURCES:
    if not os.path.exists(src):
        continue
    with open(src, encoding="utf8") as f:
        for line in f.read().splitlines():
            if not line.strip():
                continue
            parts = line.split("|")
            tx[os.path.basename(parts[0])] = parts[-1].strip()

lines, missing = [], []
for wav in sorted(glob.glob(os.path.join(FEMALE_DIR, "*.wav"))):
    name = os.path.basename(wav)
    text = tx.get(name)
    if not text:
        missing.append(name)
        continue
    lines.append(f"{os.path.abspath(wav)}|mornye|KO|{text}")

with open(OUT, "w", encoding="utf8") as f:
    f.write("\n".join(lines) + "\n")

print(f"curated clips in female_clips: {len(glob.glob(os.path.join(FEMALE_DIR, '*.wav')))}")
print(f"written to list: {len(lines)}")
print(f"wrote: {OUT}")
if missing:
    print(f"WARNING: {len(missing)} clips had no transcript (skipped):")
    for m in missing[:10]:
        print("  ", m)
