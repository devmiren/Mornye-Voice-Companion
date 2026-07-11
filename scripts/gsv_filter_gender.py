"""Gender filter by median F0. Keep female (Mornye) clips, drop male speakers.
Run with package runtime python (has librosa).

Usage:
  runtime/python.exe -s scripts/gsv_filter_gender.py            # analyze + report distribution
  runtime/python.exe -s scripts/gsv_filter_gender.py --thr 165  # also write filtered list
"""
import os, sys, argparse
import numpy as np
import librosa

MORNYE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def median_f0(path):
    try:
        y, sr = librosa.load(path, sr=16000, mono=True)
        f0, voiced, _ = librosa.pyin(y, fmin=65, fmax=500, sr=sr, frame_length=1024)
        f0v = f0[~np.isnan(f0)]
        if len(f0v) < 3:
            return None
        return float(np.median(f0v))
    except Exception as e:
        print("ERR", os.path.basename(path), e)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default=os.path.join(MORNYE, "data", "asr", "sliced.list"), help="input ASR .list")
    ap.add_argument("--out", default=None, help="output kept list (default <list>_female.list)")
    ap.add_argument("--thr", type=float, default=None, help="F0 threshold Hz; >= keeps as female. If set, writes list.")
    ap.add_argument("--copyto", default=None, help="folder to copy kept (female) wavs into")
    a = ap.parse_args()

    LIST = a.list
    OUT_KEEP = a.out or (os.path.splitext(LIST)[0] + "_female.list")

    rows = []
    with open(LIST, encoding="utf8") as f:
        lines = [l for l in f.read().splitlines() if l.strip()]
    for i, line in enumerate(lines):
        parts = line.split("|")
        path, text = parts[0], parts[-1]
        f0 = median_f0(path)
        rows.append((f0, path, text, line))
        if (i + 1) % 25 == 0:
            print(f"...{i+1}/{len(lines)}", flush=True)

    valid = [r for r in rows if r[0] is not None]
    f0s = sorted(r[0] for r in valid)
    print(f"\ntotal={len(rows)} valid_f0={len(valid)} none={len(rows)-len(valid)}")
    if f0s:
        import statistics
        print(f"F0 min={f0s[0]:.0f} p25={f0s[len(f0s)//4]:.0f} median={statistics.median(f0s):.0f} "
              f"p75={f0s[3*len(f0s)//4]:.0f} max={f0s[-1]:.0f}")
        # histogram buckets
        buckets = [(0,120),(120,150),(150,165),(165,180),(180,200),(200,240),(240,1000)]
        print("F0 histogram (Hz):")
        for lo, hi in buckets:
            c = sum(1 for x in f0s if lo <= x < hi)
            print(f"  {lo:>3}-{hi:<4}: {'#'*c} {c}")

    if a.thr is not None:
        keep = [r for r in rows if r[0] is not None and r[0] >= a.thr]
        drop = [r for r in rows if r[0] is None or r[0] < a.thr]
        with open(OUT_KEEP, "w", encoding="utf8") as f:
            f.write("\n".join(r[3] for r in keep) + "\n")
        print(f"\nthr={a.thr}Hz -> KEEP(female)={len(keep)}  DROP(male/none)={len(drop)}")
        print("wrote:", OUT_KEEP)
        if a.copyto:
            import shutil
            os.makedirs(a.copyto, exist_ok=True)
            nc = 0
            for r in keep:
                try:
                    shutil.copy(r[1], a.copyto); nc += 1
                except Exception:
                    pass
            print(f"copied {nc} female wavs -> {a.copyto}")
        print("\n--- sample DROPPED (should be male / noise) ---")
        for r in sorted(drop, key=lambda x: (x[0] is not None, x[0] if x[0] else 0))[:12]:
            print(f"  F0={('%.0f'%r[0]) if r[0] else 'NA':>4}  {r[2][:40]}")
        print("\n--- sample KEPT (should be Mornye) ---")
        for r in sorted(keep, key=lambda x: x[0])[:8]:
            print(f"  F0={r[0]:.0f}  {r[2][:40]}")


if __name__ == "__main__":
    main()
