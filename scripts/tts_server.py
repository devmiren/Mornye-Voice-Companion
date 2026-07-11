"""Persistent Mornye TTS server. Loads the model ONCE, serves synthesis over HTTP.
Run with the package runtime python:
  runtime/python.exe -s scripts/tts_server.py
GET http://127.0.0.1:9899/tts?text=<urlencoded>  ->  audio/wav
GET http://127.0.0.1:9899/ping                    ->  "ok"
"""
import os, io, urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np, soundfile as sf, librosa

MORNYE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(MORNYE, "GPT-SoVITS-v2pro-20250604")
REF = os.path.join(MORNYE, "data", "female_clips",
                   "vocal_mornye3.wav_10.wav_0012465920_0012690240.wav")
GPT = "GPT_weights_v2Pro/mornye-e15.ckpt"
SOVITS = "SoVITS_weights_v2Pro/mornye_e8_s224.pth"
SPEED, TOPK, TOPP, TEMP, PORT = 0.97, 20, 0.6, 0.6, 9899

os.chdir(PKG)
from GPT_SoVITS import inference_webui as iw


def _apply(ret):
    # change_sovits_weights is a GENERATOR: iterating runs the model load, but its final
    # UI-update yield throws UnboundLocalError when prompt_language is None. Model is already
    # loaded by then, so swallow it.
    try:
        for _ in (ret or []):
            pass
    except Exception:
        pass


_apply(iw.change_gpt_weights(gpt_path=GPT))
_apply(iw.change_sovits_weights(sovits_path=SOVITS))
KO = [k for k, v in iw.dict_language.items() if v == "all_ko"][0]


def _ref_text():
    lst = os.path.join(MORNYE, "data", "asr", "sliced_female.list")
    base = os.path.basename(REF)
    for line in open(lst, encoding="utf8"):
        p = line.split("|")
        if os.path.basename(p[0]) == base:
            return p[-1].strip()
    return ""


REFTXT = _ref_text()


def synth(text):
    res = list(iw.get_tts_wav(ref_wav_path=REF, prompt_text=REFTXT, prompt_language=KO,
                              text=text, text_language=KO,
                              how_to_cut=iw.i18n("按标点符号切"),  # split long text by punctuation
                              top_k=TOPK, top_p=TOPP, temperature=TEMP, speed=SPEED))
    sr, audio = res[-1]
    audio = np.asarray(audio); dtype = audio.dtype; fl = audio.astype(np.float32)
    if fl.size and np.max(np.abs(fl)) > 0:
        try:
            from scipy.signal import butter, sosfilt
            fl = sosfilt(butter(4, 70, btype="highpass", fs=sr, output="sos"), fl).astype(np.float32)
        except Exception:
            pass
        peak = np.max(np.abs(fl)) or 1.0
        _, idx = librosa.effects.trim(fl / peak, top_db=40); fl = fl[idx[0]:idx[1]]
        fin = min(int(0.06 * sr), fl.size // 3); fout = min(int(0.04 * sr), fl.size // 3)
        if fin > 0: fl[:fin] *= np.linspace(0, 1, fin, dtype=np.float32)
        if fout > 0: fl[-fout:] *= np.linspace(1, 0, fout, dtype=np.float32)
        audio = fl.astype(dtype)
    buf = io.BytesIO(); sf.write(buf, audio, sr, format="WAV"); return buf.getvalue()


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        q = urllib.parse.urlparse(self.path)
        if q.path == "/ping":
            self.send_response(200); self.end_headers(); self.wfile.write(b"ok"); return
        if q.path != "/tts":
            self.send_response(404); self.end_headers(); return
        text = urllib.parse.parse_qs(q.query).get("text", [""])[0]
        if not text.strip():
            self.send_response(400); self.end_headers(); return
        try:
            wav = synth(text)
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(wav)))
            self.end_headers(); self.wfile.write(wav)
        except Exception as e:
            self.send_response(500); self.end_headers(); self.wfile.write(str(e).encode())

    def log_message(self, *a):
        pass


print(f"[tts_server] ready on 127.0.0.1:{PORT}  ref={os.path.basename(REF)}", flush=True)
HTTPServer(("127.0.0.1", PORT), H).serve_forever()
