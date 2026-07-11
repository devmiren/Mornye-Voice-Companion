# Mornye TTS — Reproduction Guide

This document is written so that **any Claude session can reproduce the current state from scratch**.
Goal: fine-tune the voice of a specific Korean female speaker (the character "Mornye") and build a
pipeline that automatically plays the `🔊` summary at the end of every Claude Code response in that
voice. **The entire process is done via CLI, without any WebUI.**

> Persona (speech-style) rules are NOT included in this document. See the separate file
> `~/.claude/CLAUDE.md` (Section 9).
> The trained model and audio are for personal use only (copyright). No redistribution (Section 12).

---

## 0. Step Zero — Choose the persona language (do this first)

Before anything else, **decide which language the persona will speak in**, then wire up the matching
persona file. The pipeline itself (data, training, inference, hook) is **language-agnostic**; only the
persona file and the `🔊` summary language change.

- Persona files live next to this guide, one per language:
  - `CLAUDE.persona_kr.md` (Korean), `CLAUDE.persona_en.md` (English),
    `CLAUDE.persona_cn.md` (Chinese), `CLAUDE.persona_jp.md` (Japanese).
- Pick one, then copy its contents into `~/.claude/CLAUDE.md` (see Section 9).
- **The chosen language must match the fine-tuned voice.** The reference/dataset voice in this project
  is Korean, so the Korean persona is the natural default. If you train a different-language voice,
  choose the corresponding persona and set the ASR language (Section 4.4) and `🔊` summary language
  (Section 8) accordingly.

Everything below assumes this choice has been made.

---

## 1. Final Deliverable Summary
- Tool: **GPT-SoVITS v2Pro** (Windows integrated package, includes its own runtime)
- Training weights: `SoVITS_weights_v2Pro/mornye_e8_s224.pth`, `GPT_weights_v2Pro/mornye-e15.ckpt`
- Data: ~220 single-speaker (female) Korean clips (3 YouTube sources → voice separation → gender filter)
- Inference defaults: reference = v3 clip, `speed=0.97`, `top_k=20 top_p=0.6 temperature=0.6`
- Auto-playback: resident TTS server (:9899) + Claude Code `Stop` hook

---

## 2. Environment / Prerequisites
- OS: Windows, NVIDIA GPU (developed on RTX 4080 SUPER 16GB), CUDA available
- `ffmpeg`, `git` on PATH
- Python 3.11 (venv for tooling; the GPT-SoVITS package uses its own runtime)
- Work root: `~/Desktop/Mornye/`

```
Folder structure
Mornye/
├─ GPT-SoVITS-v2pro-20250604/   # third-party package (download yourself, GPL-3.0) + runtime + weights
├─ venv/                        # auxiliary venv (yt-dlp, py7zr, etc.)
├─ ref/                         # original downloaded audio (wav)
├─ data/
│  ├─ vocals*/  sliced*/        # UVR output / slices
│  ├─ asr/                      # ASR .list (transcripts)
│  └─ female_clips/             # ★ curated final training clip folder
├─ out/                         # synthesis output
├─ scripts/                     # all custom scripts (Section 11)
├─ start_tts_server.bat
├─ REPRODUCE.md  README.md  CLAUDE.persona_*.md
```

---

## 3. Tool Decision (why GPT-SoVITS v2Pro)
- Requirements: Korean support, local GPU, **fine-tuning** (not zero-shot), no WebUI, personal use.
- Rejected: Fish Speech/OpenAudio S1-mini (gated login wall), S2 (24GB requirement), IndexTTS2 (no Korean),
  CosyVoice2 (requires WSL2), XTTS (mostly zero-shot).
- Chosen: **GPT-SoVITS v2Pro** — Korean support, trainable on a consumer GPU, integrated package with
  slicing/ASR/training built in, each stage is a CLI script so it can run without a WebUI.

---

## 4. Dataset Construction

All commands run with the **package runtime Python**, with **cwd = package root**.
`PY = GPT-SoVITS-v2pro-20250604/runtime/python.exe`. The orchestrator `scripts/gsv_finetune.py`
handles cwd / env / PYTHONPATH.

### 4.1 Audio collection
- `venv/Scripts/yt-dlp.exe -f "bestaudio[ext=m4a]/bestaudio" -x --audio-format wav --audio-quality 0 --retries 20 -o "ref/<name>.%(ext)s" <youtube_url>`
- A single-speaker source (character lecture/monologue) is ideal. Multi-speaker (quest) sources can also
  be refined via the 4.5 filter.

### 4.2 UVR voice separation (BGM removal)
- UVR5 `webui.py` is Gradio-only → use the CLI wrapper `scripts/gsv_uvr.py` (VR `AudioPre`, model `HP5_only_main_vocal`).
- `PY -s scripts/gsv_uvr.py --in ref/<x>.wav --out data/vocals_<x> --model HP5_only_main_vocal --agg 10 --half True`

### 4.3 Slicing
- `PY -s tools/slice_audio.py <in_dir> <out_dir> -34 4000 300 10 500 0.9 0.25 0 1` (outputs 32k wav)

### 4.4 ASR (transcription)
- `PY -s tools/asr/fasterwhisper_asr.py -i <wav_dir> -o data/asr -s large-v3 -l ko -p float16`
  (set `-l` to the language of your training voice)
- Output: `data/asr/<wav_dir_basename>.list` (format `path|speaker|LANG|text`)
- large-v3 auto-downloads once on first run (~3GB, cached in `tools/asr/models`).
- ⚠️ On exit a `0xC0000409` (cuDNN cleanup) crash may occur, but it is harmless because it happens
  **after the `.list` is written**.
- ⚠️ `tools/cmd-denoise.py` (denoise) is **broken** due to a modelscope↔datasets version conflict in the
  runtime (`cannot import name 'LargeList'`) → **skip it**. UVR is sufficient.

### 4.5 F0 gender filter (single-speaker extraction)
- Key trick: if all other speakers are male, you can separate by gender using the **median F0**
  (female = target speaker).
- `scripts/gsv_filter_gender.py --list data/asr/<x>.list --thr 160 --copyto data/female_clips`
- In the dev data the F0 distribution was clearly bimodal (male 89–150Hz / empty gap 150–165 /
  female 165–499Hz, median ~280) → a **160Hz threshold** separated them cleanly.

### 4.6 Curation + training list
- `data/female_clips/` is the final training set. The user listens through and deletes bad clips
  (manual curation).
- When combining multiple sources: process each source through 4.1–4.5 and collect the female clips
  into `female_clips/`.
- Build the training list: `PY scripts/build_train_list.py`
  → matches filenames from `data/asr/*.list` (transcript sources) and writes `data/asr/sliced_female.list`
  based on the contents of `female_clips/`.

---

## 5. Feature Extraction & Training

Orchestrator: `scripts/gsv_finetune.py` (exp="mornye", version="v2Pro", GPU 0, is_half True).
Select stages with `--stages`. `wav_dir_for_features()` uses `female_clips` if present;
`list_path()` prefers `sliced_female.list`.

- Feature extraction: `PY -s scripts/gsv_finetune.py --stages 1a,1b,1c`
  - 1a `prepare_datasets/1-get-text.py` (phonemes/text) — parameters passed via **environment variables**.
  - 1b `2-get-hubert-wav32k.py` + (v2Pro) `2-get-sv.py` (SSL + speaker-verification embeddings)
  - 1c `3-get-semantic.py` (semantic tokens)
  - Output: in `logs/mornye/`: 2-name2text.txt, 5-wav32k, 4-cnhubert, 7-sv_cn, 6-name2semantic.tsv
  - `3-bert=0` is normal (Korean does not use the Chinese BERT).
- Training: `PY -s scripts/gsv_finetune.py --stages s2,s1`
  - s2 (SoVITS): tmp config based on `configs/s2v2Pro.json` → `s2_train.py`. batch 6, 8 epochs.
  - s1 (GPT): tmp based on `configs/s1longer-v2.yaml` → `s1_train.py`. batch 6, 15 epochs.
    env `_CUDA_VISIBLE_DEVICES`, `hz=25hz`.
  - ★ **Required fix**: create the log directories before training so checkpoint saving does not fail —
    `logs/mornye/logs_s2_v2Pro`, `logs/mornye/logs_s1_v2Pro`, `logs/mornye/logs_s1` (handled by gsv_finetune.py).
  - Weights: `SoVITS_weights_v2Pro/mornye_e8_s224.pth`, `GPT_weights_v2Pro/mornye-e15.ckpt`
- Pretrained paths (v2Pro): s2G `pretrained_models/v2Pro/s2Gv2Pro.pth`, s2D same folder s2Dv2Pro.pth,
  s1 `pretrained_models/s1v3.ckpt`, bert `chinese-roberta-wwm-ext-large`, cnhubert `chinese-hubert-base`,
  sv `pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt`.

---

## 6. Inference

Script `scripts/gsv_infer.py`. `PY -s scripts/gsv_infer.py --text "..." --ref <clip> --speed 0.97`.
- ★ **The most important bug**: `inference_webui.change_sovits_weights` is a **generator** (it has `yield`),
  so simply calling it does nothing → inference falls back to the **default weights (e4)** loaded at import.
  You must **exhaust it** for e8 to load. Furthermore, if `prompt_language=None`, the last UI yield raises
  `UnboundLocalError` (after the model is loaded) → **exhaust it inside try/except**.
  `change_gpt_weights` is a normal function, so a plain call is fine.
  ```python
  def _apply(ret):
      try:
          for _ in (ret or []): pass
      except Exception: pass
  _apply(iw.change_gpt_weights(gpt_path=GPT))
  _apply(iw.change_sovits_weights(sovits_path=SOVITS))
  ```
- Before inference, `os.chdir(PKG)` (inference_webui reads bert/cnhubert relative paths from cwd).
- Korean language key: use the key in `dict_language` whose value is `"all_ko"`.
- Sampling defaults **top_k=20, top_p=0.6, temperature=0.6** (WebUI defaults; leaving them at 1.0/1.0
  causes onset noise and large length variance).
- Long text: `how_to_cut=i18n("按标点符号切")` to split on punctuation.
- Post-processing (front-noise removal): high-pass 70Hz → `librosa.effects.trim(top_db=40)` →
  fade-in 60ms / fade-out 40ms.
- Reference clip: use a clean single-speaker v3 clip (low reverb). The reference only sets timbre/tone
  and is unrelated to the target text.

---

## 7. Auto-Playback Pipeline
- `scripts/tts_server.py`: resident HTTP server (127.0.0.1:9899). Loads the model once (applies Section 6
  rules), with fixed reference/speed/params/post-processing.
  `GET /tts?text=` → wav, `GET /ping` → ok.
- `scripts/tts_hook.py`: Claude Code `Stop` hook.
  - Reads stdin JSON as `utf-8-sig` (allows BOM).
  - Extracts the last assistant text from `transcript_path`.
  - Extracts **only the text after the `🔊` marker** (`tts_segment`); if absent, silence.
  - `clean()` removes code/tables/URLs, keeps text and punctuation (MAXLEN=None).
  - GETs the server (proxy-bypass opener), saves wav, then **plays in a separate process**
    (powershell SoundPlayer).
  - All errors are silently ignored (so the hook never blocks Claude).
- `start_tts_server.bat`: for keeping the server always running.

---

## 8. `🔊` Summary Rule (the pipeline's input contract)
- Claude appends a summary paragraph starting with `🔊` at the **very end** of every response, written in
  the persona's chosen language (no special symbols/foreign letters that TTS can't read; end every
  sentence with a period).
- The hook reads only this paragraph. Detailed rules are in `~/.claude/CLAUDE.md` Section 2 (see Section 9).

---

## 9. Global Configuration
- **Speech style + TTS rules**: `~/.claude/CLAUDE.md` (separate file). Described as behavior rules so it
  reproduces even without knowing the character.
  + Copy the contents of the chosen `CLAUDE.persona_<lang>.md` (see Section 0).
- **Auto-playback hook (global)**: `hooks.Stop` in `~/.claude/settings.json` (add while keeping existing keys):
  ```json
  "hooks": { "Stop": [ { "hooks": [ { "type": "command",
    "command": "\"<Mornye>\\GPT-SoVITS-v2pro-20250604\\runtime\\python.exe\" -s \"<Mornye>\\scripts\\tts_hook.py\"" } ] } ] }
  ```
- Hooks load at session start → restart Claude Code after changing. The server must be running for sound.
- Auto-start of the server on login is (by user choice) not configured.

---

## 10. Troubleshooting Summary (key pitfalls)
1. `._pth` ignores PYTHONPATH/script path → add the 4 lines like `..` (Section 3).
2. `denoise` broken (modelscope/datasets) → skip.
3. `0xC0000409` on ASR exit → harmless (after `.list` is written).
4. `change_sovits_weights` generator + `UnboundLocalError` → exhaust in try/except (Section 6).
   **If unhandled, inference runs with e4.**
5. s2/s1 log directories not created → checkpoint save fails → makedirs beforehand (Section 5).
6. Hook stdin BOM → `utf-8-sig`. localhost proxy → `ProxyHandler({})` opener.
7. Sampling 1.0/1.0 → front noise / length variance → 0.6/0.6/top_k20.

---

## 11. Script Inventory (`scripts/`)
- `gsv_uvr.py` — UVR5 voice separation CLI wrapper
- `gsv_filter_gender.py` — median-F0 gender filter (`--list/--out/--thr/--copyto`)
- `extract_clips.py` — silero-VAD candidate clip extraction (for reference candidates)
- `build_train_list.py` — build training list based on female_clips
- `gsv_finetune.py` — pipeline orchestrator (uvr, slice, denoise, asr, 1a, 1b, 1c, s2, s1)
- `gsv_infer.py` — one-shot synthesis (`--text/--ref/--speed/--speeds/--top_k/--top_p/--temperature`)
- `tts_server.py` — resident TTS HTTP server
- `tts_hook.py` — Claude Code Stop hook

---

## 12. License / Distribution Notice
- When publishing to GitHub, include **custom scripts only**. Exclude: package/runtime, training weights
  (`*.pth/*.ckpt`), audio (`data/ ref/ out/ *.wav`), venv, 7z/exe. (See `.gitignore`.)
- The trained voice/audio is derived from the work of the actual voice actor / game company
  (e.g. Kuro Games) → **personal / non-commercial only, no redistribution**.
- The glue scripts can be MIT, but since they import/run GPT-SoVITS (GPL-3.0), distributing them together
  makes GPL-3.0 apply.
