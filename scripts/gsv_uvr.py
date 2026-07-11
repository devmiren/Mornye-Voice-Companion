"""WebUI-free UVR5 vocal separation wrapper.
Replicates tools/uvr5/webui.py:uvr() core logic as a CLI.
Run with the package runtime python; cwd must be the package root.

Usage:
  runtime/python.exe -s <this> --in <input_file_or_dir> --out <vocal_out_dir> [--model HP5_only_main_vocal] [--agg 10]
"""
import os, sys, argparse, traceback

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # will be overridden; see main
# When invoked, cwd is the GPT-SoVITS package root. Add tools/uvr5 to path like webui does.

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--model", default="HP5_only_main_vocal")
    ap.add_argument("--agg", type=int, default=10)
    ap.add_argument("--half", default="True")
    a = ap.parse_args()

    pkg = os.getcwd()  # expected: package root
    sys.path.insert(0, os.path.join(pkg, "tools", "uvr5"))
    import torch, ffmpeg
    is_half = (a.half.lower() == "true") and torch.cuda.is_available()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight_root = "tools/uvr5/uvr5_weights"

    from vr import AudioPre, AudioPreDeEcho  # noqa

    model_name = a.model
    func = AudioPre if "DeEcho" not in model_name else AudioPreDeEcho
    pre_fun = func(agg=int(a.agg),
                   model_path=os.path.join(weight_root, model_name + ".pth"),
                   device=device, is_half=is_half)

    os.makedirs(a.out, exist_ok=True)
    ins_dir = a.out + "_ins"
    os.makedirs(ins_dir, exist_ok=True)

    inp = a.inp
    paths = [inp] if os.path.isfile(inp) else [os.path.join(inp, n) for n in os.listdir(inp)]
    tmp = os.environ.get("TEMP", os.path.join(pkg, "TEMP"))
    os.makedirs(tmp, exist_ok=True)

    for p in paths:
        if not os.path.isfile(p):
            continue
        inp_path = p
        need_reformat = 1
        try:
            info = ffmpeg.probe(inp_path, cmd="ffprobe")
            if info["streams"][0]["channels"] == 2 and info["streams"][0]["sample_rate"] == "44100":
                need_reformat = 0
        except Exception:
            traceback.print_exc()
        if need_reformat == 1:
            tmp_path = os.path.join(tmp, os.path.basename(inp_path) + ".reformatted.wav")
            os.system(f'ffmpeg -i "{inp_path}" -vn -acodec pcm_s16le -ac 2 -ar 44100 "{tmp_path}" -y')
            inp_path = tmp_path
        try:
            pre_fun._path_audio_(inp_path, ins_dir, a.out, "wav", False)
            print(f"[uvr] {os.path.basename(p)} -> Success")
        except Exception:
            print(f"[uvr] {os.path.basename(p)} -> FAIL")
            traceback.print_exc()

    try:
        del pre_fun.model, pre_fun
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[uvr] done. vocals in:", a.out)

if __name__ == "__main__":
    main()
