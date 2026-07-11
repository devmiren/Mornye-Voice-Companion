<div align="center">

<img src="./Mornye.png" alt="Mornye" width="420" />

# Mornye TTS

**A CLI-only voice fine-tuning + auto-play pipeline for a game-character voice — wired into Claude Code.**

Fine-tune a single-speaker voice with [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS), then let a
Claude Code `Stop` hook speak a short summary of every reply in that voice. No WebUI — every step is a
CLI script.

![Platform](https://img.shields.io/badge/platform-Windows%20%2B%20NVIDIA%20GPU-informational)
![Engine](https://img.shields.io/badge/engine-GPT--SoVITS%20v2Pro-blueviolet)
![Scripts](https://img.shields.io/badge/license-MIT-green)
![Bundled](https://img.shields.io/badge/GPT--SoVITS-GPL--3.0-orange)

</div>

---

> [!WARNING]
> **Personal / educational use only.** This repo contains **only my own orchestration scripts.** It does
> **not** include GPT-SoVITS, model weights, or any audio. A voice you train is derived from a copyrighted
> character / voice actor, and the source audio belongs to its rights holders (e.g. Kuro Games, the voice
> actor, YouTube uploaders). **Do not redistribute trained models or training/reference audio.** Respect
> all applicable ToS and rights.

## ✨ Overview

```
audio ──▶ separate ──▶ slice ──▶ ASR ──▶ features ──▶ train (s2 + s1) ──▶ voice model
                                                                              │
                              Claude reply ──▶ Stop hook ──▶ TTS server ──────┘──▶ 🔊 spoken summary
```

The whole thing runs headless from the command line and plugs into Claude Code: end each reply with a
`🔊 <summary.>` line, and the hook synthesizes just that line in your trained voice.

## 📂 What's here

| Script | Role |
| --- | --- |
| `scripts/gsv_uvr.py` | UVR5 vocal separation (CLI wrapper) |
| `scripts/gsv_filter_gender.py` | Keep one speaker via median-F0 (gender) filtering |
| `scripts/extract_clips.py` | VAD-based candidate clip extraction |
| `scripts/build_train_list.py` | Build a training list from a curated clip folder |
| `scripts/gsv_finetune.py` | Full pipeline driver: slice → ASR → features → train (s2 + s1) |
| `scripts/gsv_infer.py` | One-off synthesis |
| `scripts/tts_server.py` | Persistent HTTP TTS server (loads the model once) |
| `scripts/tts_hook.py` | Claude Code `Stop` hook — speaks the text after the last `🔊` marker |
| `start_tts_server.bat` | Keep the server running |

> 📖 For a full, reproduce-from-scratch walkthrough (every command, every pitfall), see **[`REPRODUCE.md`](./REPRODUCE.md)**.

## 🚀 Prerequisites (you provide these yourself)

1. Download the GPT-SoVITS Windows package (e.g. `GPT-SoVITS-v2pro-...`) and extract it here as
   `GPT-SoVITS-v2pro-20250604/` (it ships its own Python runtime + pretrained models).
2. An NVIDIA GPU with CUDA, and `ffmpeg` on your `PATH`.
3. Your **own** legally-obtained, single-speaker training audio → `ref/`.

## 🛠️ Usage (sketch)

```bash
# 1. data prep + training (edit paths/params at the top of the script)
runtime/python.exe -s scripts/gsv_finetune.py --stages uvr,slice,asr,1a,1b,1c,s2,s1

# 2. one-off synthesis
runtime/python.exe -s scripts/gsv_infer.py --text "..." --ref <clip> --speed 0.97

# 3. auto-play in Claude Code
#    run start_tts_server.bat, then add the Stop hook to .claude/settings.json
```

End Claude replies with a `🔊 <summary.>` line (in your chosen persona language) — only that line is spoken.

## 🎭 Persona (optional — Claude-only roleplay)

An optional **roleplay persona** makes Claude answer *in character* as "Mornye" — a calm, understated
professor who speaks in polite, cosmos-flavored language — and close each reply with the `🔊` summary the
hook reads. This is purely a **Claude prompt/roleplay layer**; it changes how Claude *writes*, and touches
nothing in the voice pipeline.

Persona guides are provided in four languages:

| File | Language |
| --- | --- |
| `CLAUDE.persona_kr.md` | Korean (한국어) |
| `CLAUDE.persona_en.md` | English |
| `CLAUDE.persona_cn.md` | Chinese (中文) |
| `CLAUDE.persona_jp.md` | Japanese (日本語) |

Each is tuned to the character's actual localized speech style. To use one, **just ask Claude to play the
persona** — point it at the file (*"follow `CLAUDE.persona_en.md`"*) or copy the file's contents into
`~/.claude/CLAUDE.md` — and Claude adopts the voice for the rest of the session.

## 📜 License

MIT — see [`LICENSE`](./LICENSE). Note: these scripts import/drive **GPT-SoVITS (GPL-3.0)**, which this
repo does not redistribute. If you bundle GPT-SoVITS, GPL-3.0 terms apply to the combined work.
