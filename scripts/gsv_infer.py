"""Inference with the fine-tuned Mornye GPT-SoVITS model (Korean).
Run with package runtime python, cwd = package root.

  runtime/python.exe -s scripts/gsv_infer.py --text "합성할 문장" [--out out.wav]
"""
import os, sys, argparse, soundfile as sf
import numpy as np, librosa

MORNYE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(MORNYE, "GPT-SoVITS-v2pro-20250604")
LIST = os.path.join(MORNYE, "data", "asr", "sliced_female.list")

GPT = "GPT_weights_v2Pro/mornye-e15.ckpt"
SOVITS = "SoVITS_weights_v2Pro/mornye_e8_s224.pth"


def load_list():
    with open(LIST, encoding="utf8") as f:
        return [l for l in f.read().splitlines() if l.strip()]


def text_for(path):
    base = os.path.basename(path)
    for line in load_list():
        parts = line.split("|")
        if os.path.basename(parts[0]) == base:
            return parts[-1].strip()
    return None


def pick_reference():
    """choose a clean 3-9s Mornye clip + its transcript as the reference prompt."""
    best = None
    for line in load_list():
        parts = line.split("|")
        path, text = parts[0], parts[-1].strip()
        if len(text) < 10:
            continue
        try:
            info = sf.info(path)
            dur = info.frames / info.samplerate
        except Exception:
            continue
        if 3.0 <= dur <= 9.0:
            # prefer the longest transcript in range (more phonetic content)
            if best is None or len(text) > len(best[1]):
                best = (path, text, dur)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True, help="Korean text to synthesize")
    ap.add_argument("--out", default=os.path.join(MORNYE, "out", "mornye_test.wav"))
    ap.add_argument("--speed", type=float, default=0.85, help="<1 = slower/calmer")
    ap.add_argument("--speeds", default=None, help="comma list of speeds; writes one file per speed")
    ap.add_argument("--ref", default=None, help="explicit reference wav (else auto-pick)")
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--top_p", type=float, default=0.6)
    ap.add_argument("--temperature", type=float, default=0.6)
    a = ap.parse_args()

    speeds = [float(s) for s in a.speeds.split(",")] if a.speeds else [a.speed]

    os.chdir(PKG)  # inference_webui reads relative model paths (bert/cnhubert) against CWD
    from GPT_SoVITS import inference_webui as iw

    # change_sovits_weights is a GENERATOR — iterating runs the load; its final UI yield
    # throws (UnboundLocalError) when prompt_language is None, after the model is loaded → swallow.
    def _apply(ret):
        try:
            for _ in (ret or []):
                pass
        except Exception:
            pass

    _apply(iw.change_gpt_weights(gpt_path=GPT))
    _apply(iw.change_sovits_weights(sovits_path=SOVITS))

    ko = [k for k, v in iw.dict_language.items() if v == "all_ko"][0]

    if a.ref:
        ref_path = a.ref
        ref_text = text_for(ref_path) or ""
        dur = sf.info(ref_path).frames / sf.info(ref_path).samplerate
    else:
        ref = pick_reference()
        if not ref:
            raise SystemExit("no suitable reference clip found")
        ref_path, ref_text, dur = ref
    print(f"[ref] {os.path.basename(ref_path)} ({dur:.1f}s)  text: {ref_text}")
    print(f"[tgt] {a.text}  | speeds={speeds}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    for sp in speeds:
        result = list(iw.get_tts_wav(
            ref_wav_path=ref_path, prompt_text=ref_text, prompt_language=ko,
            text=a.text, text_language=ko,
            top_k=a.top_k, top_p=a.top_p, temperature=a.temperature, speed=sp,
        ))
        if not result:
            print(f"[warn] speed {sp}: nothing"); continue
        sr, audio = result[-1]
        audio = np.asarray(audio)
        dtype = audio.dtype
        fl = audio.astype(np.float32)
        if fl.size and np.max(np.abs(fl)) > 0:
            # high-pass ~70Hz to remove low-frequency rumble/onset noise
            try:
                from scipy.signal import butter, sosfilt
                sos = butter(4, 70, btype="highpass", fs=sr, output="sos")
                fl = sosfilt(sos, fl).astype(np.float32)
            except Exception:
                pass
            peak = np.max(np.abs(fl)) or 1.0
            fl_n = fl / peak
            # aggressive leading/trailing silence removal (no head margin)
            _, idx = librosa.effects.trim(fl_n, top_db=40)
            fl = fl[idx[0]:idx[1]]
            # long fade in (60ms) / out (40ms) to kill onset clicks / residual front noise
            fin = min(int(0.06 * sr), fl.size // 3)
            fout = min(int(0.04 * sr), fl.size // 3)
            if fin > 0:
                fl[:fin] *= np.linspace(0.0, 1.0, fin, dtype=np.float32)
            if fout > 0:
                fl[-fout:] *= np.linspace(1.0, 0.0, fout, dtype=np.float32)
            audio = fl.astype(dtype)
        out = a.out if len(speeds) == 1 else a.out.replace(".wav", f"_spd{sp:.2f}.wav")
        sf.write(out, audio, sr)
        print(f"[out] {out}  (sr={sr}, {len(audio)/sr:.1f}s)")


if __name__ == "__main__":
    main()
