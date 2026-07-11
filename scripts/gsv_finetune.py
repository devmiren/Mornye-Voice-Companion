"""WebUI-free GPT-SoVITS v2Pro fine-tuning driver.
Replicates webui.py's pipeline via CLI subprocess calls.

Stages (in order):
  uvr     - BGM/vocal separation (optional)
  slice   - slice long audio into segments (32k)
  denoise - denoise sliced segments (optional)
  asr     - Korean ASR -> transcript .list
  1a      - text/phoneme feature extraction
  1b      - hubert ssl + speaker-verification (v2Pro) features
  1c      - semantic token extraction
  s2      - train SoVITS
  s1      - train GPT

Run all:      runtime/python.exe -s scripts/gsv_finetune.py --stages all
Run subset:   ... --stages slice,denoise,asr
Skip uvr:     ... --stages slice,denoise,asr,1a,1b,1c,s2,s1

All subprocesses run with cwd = package root and the runtime python.
"""
import os, sys, json, subprocess, argparse

# ---- paths ----
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
MORNYE = os.path.dirname(SCRIPTS)                    # ~/Desktop/Mornye
PKG = os.path.join(MORNYE, "GPT-SoVITS-v2pro-20250604")
PY = os.path.join(PKG, "runtime", "python.exe")

DATA = os.path.join(MORNYE, "data")
RAW = os.path.join(MORNYE, "ref", "mornye_source.wav")   # source audio (single file)
VOCAL_DIR = os.path.join(DATA, "vocals")
SLICE_DIR = os.path.join(DATA, "sliced")
DENOISE_DIR = os.path.join(DATA, "denoised")
ASR_DIR = os.path.join(DATA, "asr")

# ---- experiment config ----
EXP = "mornye"
VERSION = "v2Pro"
GPU = "0"
IS_HALF = "True"
PREC = "float16"
SOVITS_BATCH = 2   # reduced 6->2 for GTX 1060 6GB (dev was RTX 4080 16GB)
SOVITS_EPOCH = 8
GPT_BATCH = 2      # reduced 6->2 for GTX 1060 6GB
GPT_EPOCH = 15

# ---- pretrained (relative to PKG) ----
BERT = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
CNHUBERT = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
SV = "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"
S2G = "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth"
S2D = "GPT_SoVITS/pretrained_models/v2Pro/s2Dv2Pro.pth"
S1 = "GPT_SoVITS/pretrained_models/s1v3.ckpt"
S2CONFIG = "GPT_SoVITS/configs/s2v2Pro.json"

OPT_DIR = f"logs/{EXP}"                              # relative to PKG -> PKG/logs/mornye
ASR_LIST = None                                     # resolved after asr


def run(cmd, extra_env=None):
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = PKG + os.pathsep + env.get("PYTHONPATH", "")
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})
    print("\n>>>", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=PKG, env=env)
    if r.returncode != 0:
        raise SystemExit(f"stage failed (exit {r.returncode}): {cmd}")


FEMALE_DIR = os.path.join(DATA, "female_clips")


def wav_dir_for_features():
    """curated female_clips if present, else denoise output, else sliced."""
    if os.path.isdir(FEMALE_DIR) and os.listdir(FEMALE_DIR):
        return FEMALE_DIR
    return DENOISE_DIR if os.path.isdir(DENOISE_DIR) and os.listdir(DENOISE_DIR) else SLICE_DIR


def list_path():
    # prefer gender-filtered (Mornye-only) list if present
    filt = os.path.abspath(os.path.join(ASR_DIR, "sliced_female.list"))
    if os.path.exists(filt):
        return filt
    wd = wav_dir_for_features()
    return os.path.abspath(os.path.join(ASR_DIR, os.path.basename(wd.rstrip("/\\")) + ".list"))


# ---------- stages ----------
def st_uvr():
    os.makedirs(VOCAL_DIR, exist_ok=True)
    run([PY, "-s", os.path.join(SCRIPTS, "gsv_uvr.py"),
         "--in", RAW, "--out", VOCAL_DIR, "--model", "HP5_only_main_vocal", "--agg", "10", "--half", IS_HALF])


def st_slice():
    # if UVR ran, slice the vocals; else slice RAW
    inp = VOCAL_DIR if (os.path.isdir(VOCAL_DIR) and os.listdir(VOCAL_DIR)) else RAW
    os.makedirs(SLICE_DIR, exist_ok=True)
    # slice_audio.py inp opt threshold min_length min_interval hop_size max_sil_kept _max alpha i_part n_parts
    run([PY, "-s", "tools/slice_audio.py", inp, SLICE_DIR,
         "-34", "4000", "300", "10", "500", "0.9", "0.25", "0", "1"])


def st_denoise():
    os.makedirs(DENOISE_DIR, exist_ok=True)
    run([PY, "-s", "tools/cmd-denoise.py", "-i", SLICE_DIR, "-o", DENOISE_DIR, "-p", PREC])


def st_asr():
    wd = wav_dir_for_features()
    os.makedirs(ASR_DIR, exist_ok=True)
    run([PY, "-s", "tools/asr/fasterwhisper_asr.py",
         "-i", wd, "-o", ASR_DIR, "-s", "large-v3", "-l", "ko", "-p", PREC])
    print("[asr] list ->", list_path())


def st_1a():
    wd = wav_dir_for_features()
    env = {"inp_text": list_path(), "inp_wav_dir": wd, "exp_name": EXP,
           "opt_dir": OPT_DIR, "bert_pretrained_dir": BERT,
           "i_part": "0", "all_parts": "1", "_CUDA_VISIBLE_DEVICES": GPU, "is_half": IS_HALF}
    run([PY, "-s", "GPT_SoVITS/prepare_datasets/1-get-text.py"], env)
    # merge part -> 2-name2text.txt
    d = os.path.join(PKG, OPT_DIR)
    part = os.path.join(d, "2-name2text-0.txt")
    if os.path.exists(part):
        with open(part, encoding="utf8") as f:
            data = f.read()
        with open(os.path.join(d, "2-name2text.txt"), "w", encoding="utf8") as f:
            f.write(data if data.endswith("\n") else data + "\n")
        os.remove(part)
    print("[1a] wrote 2-name2text.txt")


def st_1b():
    wd = wav_dir_for_features()
    env = {"inp_text": list_path(), "inp_wav_dir": wd, "exp_name": EXP,
           "opt_dir": OPT_DIR, "cnhubert_base_dir": CNHUBERT, "sv_path": SV,
           "i_part": "0", "all_parts": "1", "_CUDA_VISIBLE_DEVICES": GPU, "is_half": IS_HALF}
    run([PY, "-s", "GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py"], env)
    # v2Pro also needs speaker-verification embeddings
    run([PY, "-s", "GPT_SoVITS/prepare_datasets/2-get-sv.py"], env)
    print("[1b] hubert + sv done")


def st_1c():
    env = {"inp_text": list_path(), "exp_name": EXP, "opt_dir": OPT_DIR,
           "pretrained_s2G": S2G, "s2config_path": S2CONFIG,
           "i_part": "0", "all_parts": "1", "_CUDA_VISIBLE_DEVICES": GPU, "is_half": IS_HALF}
    run([PY, "-s", "GPT_SoVITS/prepare_datasets/3-get-semantic.py"], env)
    d = os.path.join(PKG, OPT_DIR)
    part = os.path.join(d, "6-name2semantic-0.tsv")
    if os.path.exists(part):
        with open(part, encoding="utf8") as f:
            body = f.read().strip("\n")
        with open(os.path.join(d, "6-name2semantic.tsv"), "w", encoding="utf8") as f:
            f.write("item_name\tsemantic_audio\n" + body + "\n")
        os.remove(part)
    print("[1c] wrote 6-name2semantic.tsv")


def st_s2():
    os.makedirs(os.path.join(PKG, OPT_DIR, f"logs_s2_{VERSION}"), exist_ok=True)
    with open(os.path.join(PKG, S2CONFIG), encoding="utf8") as f:
        data = json.load(f)
    data["train"]["fp16_run"] = True
    data["train"]["batch_size"] = SOVITS_BATCH
    data["train"]["epochs"] = SOVITS_EPOCH
    data["train"]["text_low_lr_rate"] = 0.4
    data["train"]["pretrained_s2G"] = S2G
    data["train"]["pretrained_s2D"] = S2D
    data["train"]["if_save_latest"] = True
    data["train"]["if_save_every_weights"] = True
    data["train"]["save_every_epoch"] = 4
    data["train"]["gpu_numbers"] = GPU
    data["train"]["grad_ckpt"] = False
    data["train"]["lora_rank"] = 32
    data["model"]["version"] = VERSION
    data["data"]["exp_dir"] = data["s2_ckpt_dir"] = OPT_DIR
    data["save_weight_dir"] = "SoVITS_weights_v2Pro"
    data["name"] = EXP
    data["version"] = VERSION
    tmp = os.path.join(PKG, "TEMP", "tmp_s2.json")
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    with open(tmp, "w", encoding="utf8") as f:
        json.dump(data, f)
    run([PY, "-s", "GPT_SoVITS/s2_train.py", "--config", tmp])


def st_s1():
    import yaml
    os.makedirs(os.path.join(PKG, OPT_DIR, "logs_s1"), exist_ok=True)
    os.makedirs(os.path.join(PKG, OPT_DIR, f"logs_s1_{VERSION}"), exist_ok=True)
    with open(os.path.join(PKG, "GPT_SoVITS/configs/s1longer-v2.yaml"), encoding="utf8") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    data["train"]["precision"] = "16-mixed"
    data["train"]["batch_size"] = GPT_BATCH
    data["train"]["epochs"] = GPT_EPOCH
    data["pretrained_s1"] = S1
    data["train"]["save_every_n_epoch"] = 5
    data["train"]["if_save_every_weights"] = True
    data["train"]["if_save_latest"] = True
    data["train"]["if_dpo"] = False
    data["train"]["half_weights_save_dir"] = "GPT_weights_v2Pro"
    data["train"]["exp_name"] = EXP
    data["train_semantic_path"] = f"{OPT_DIR}/6-name2semantic.tsv"
    data["train_phoneme_path"] = f"{OPT_DIR}/2-name2text.txt"
    data["output_dir"] = f"{OPT_DIR}/logs_s1_{VERSION}"
    tmp = os.path.join(PKG, "TEMP", "tmp_s1.yaml")
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    with open(tmp, "w", encoding="utf8") as f:
        yaml.dump(data, f, default_flow_style=False)
    run([PY, "-s", "GPT_SoVITS/s1_train.py", "--config_file", tmp],
        {"_CUDA_VISIBLE_DEVICES": GPU, "hz": "25hz"})


STAGES = {"uvr": st_uvr, "slice": st_slice, "denoise": st_denoise, "asr": st_asr,
          "1a": st_1a, "1b": st_1b, "1c": st_1c, "s2": st_s2, "s1": st_s1}
ALL = ["uvr", "slice", "denoise", "asr", "1a", "1b", "1c", "s2", "s1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="all", help="comma list or 'all'")
    a = ap.parse_args()
    seq = ALL if a.stages == "all" else [s.strip() for s in a.stages.split(",") if s.strip()]
    print("PKG:", PKG)
    print("running stages:", seq)
    for s in seq:
        if s not in STAGES:
            raise SystemExit(f"unknown stage: {s}")
        print(f"\n===== STAGE: {s} =====")
        STAGES[s]()
    print("\nALL DONE:", seq)


if __name__ == "__main__":
    main()
