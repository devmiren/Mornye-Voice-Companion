import os, subprocess, torch
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

B = os.path.expanduser("~/Desktop/Mornye")
SRC = os.path.join(B, "ref", "mornye_source.wav")
OUT = os.path.join(B, "ref")
SR = 16000

print("loading audio for VAD...")
wav = read_audio(SRC, sampling_rate=SR)
model = load_silero_vad()
ts = get_speech_timestamps(wav, model, sampling_rate=SR)
print(f"raw speech chunks: {len(ts)}")

# merge chunks separated by small gaps (<0.6s)
merged = []
gap = int(0.6 * SR)
for c in ts:
    if merged and c["start"] - merged[-1]["end"] <= gap:
        merged[-1]["end"] = c["end"]
    else:
        merged.append(dict(c))

# keep segments >=6s; if longer, trim to ~12s
MIN, TRIM = 6*SR, 12*SR
cands = []
for m in merged:
    dur = m["end"] - m["start"]
    if dur >= MIN:
        end = m["start"] + min(dur, TRIM)
        cands.append((m["start"], end, dur))

print(f"candidate segments (>=6s): {len(cands)}")

# pick the 8 longest (best continuous speech)
cands.sort(key=lambda x: -x[2])
cands = [(s, e) for s, e, _ in cands[:8]]
cands.sort()

for i, (s, e) in enumerate(cands, 1):
    st, en = s / SR, e / SR
    dst = os.path.join(OUT, f"cand_{i:02d}.wav")
    # extract from original 48k, downmix mono, 44.1k
    subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-i", SRC,
        "-ss", f"{st:.2f}", "-to", f"{en:.2f}",
        "-ac", "1", "-ar", "44100", dst
    ], check=True)
    print(f"cand_{i:02d}.wav  <- {st:6.1f}s ~ {en:6.1f}s  ({en-st:.1f}s)")

print("done")
