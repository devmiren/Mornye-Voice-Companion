"""Claude Code Stop hook: speak the assistant's last reply in Mornye's voice.
Reads hook JSON on stdin, pulls the last assistant text from the transcript,
cleans it, asks the TTS server to synthesize, then plays it (detached).
Fails silently so it never blocks Claude Code.
"""
import sys, os, json, re, subprocess, tempfile, urllib.parse, urllib.request

PORT = 9899
MAXLEN = None  # None = no cap (read the whole reply)


def last_assistant_text(transcript_path):
    if not transcript_path or not os.path.exists(transcript_path):
        return ""
    lines = open(transcript_path, encoding="utf-8-sig", errors="ignore").read().splitlines()
    for line in reversed(lines):
        try:
            o = json.loads(line)
        except Exception:
            continue
        msg = o.get("message", o)
        if o.get("type") == "assistant" or msg.get("role") == "assistant":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
            txt = " ".join(p for p in parts if p).strip()
            if txt:
                return txt
    return ""


def tts_segment(raw):
    """Speak ONLY the author-written summary after the last 🔊 marker. No marker -> silent."""
    idx = raw.rfind("\U0001f50a")  # 🔊
    return raw[idx + 1:] if idx != -1 else ""


def clean(text):
    text = re.sub(r"```.*?```", " ", text, flags=re.S)          # code blocks -> drop
    text = re.sub(r"`[^`]*`", " ", text)                         # inline code -> drop
    text = re.sub(r"!\[.*?\]\([^)]*\)", " ", text)               # images -> drop
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)         # links -> keep label
    text = re.sub(r"https?://\S+", " ", text)                    # urls -> drop
    text = re.sub(r"^\s*\|.*$", " ", text, flags=re.M)           # table rows -> drop
    # strip leading list/quote/heading markers but KEEP the line content
    text = re.sub(r"^\s*(?:[-*+]|\d+\.|>|#{1,6})\s+", "", text, flags=re.M)
    text = re.sub(r"[*_#>`~|]", " ", text)                       # stray md symbols
    text = re.sub(r"[^가-힣ㄱ-ㆎ0-9A-Za-z.,!?~…\s]", " ", text)   # keep KR/alnum/basic punct
    text = re.sub(r"\s+", " ", text).strip()
    if MAXLEN and len(text) > MAXLEN:
        cut = text[:MAXLEN]
        m = re.search(r"[.!?…][^.!?…]*$", cut)
        text = cut[:m.start() + 1] if m else cut
    return text


def main():
    try:
        raw = sys.stdin.buffer.read()
        data = json.loads(raw.decode("utf-8-sig"))  # tolerate a BOM on stdin
    except Exception:
        return
    text = clean(tts_segment(last_assistant_text(data.get("transcript_path"))))
    if not text:
        return
    try:
        url = f"http://127.0.0.1:{PORT}/tts?text=" + urllib.parse.quote(text)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # bypass proxy for localhost
        wav = opener.open(url, timeout=60).read()
    except Exception:
        return  # server not up / error -> silent
    out = os.path.join(tempfile.gettempdir(), "mornye_say.wav")
    with open(out, "wb") as f:
        f.write(wav)
    # play detached so the hook returns immediately (no window)
    try:
        subprocess.Popen(
            ["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command",
             f"(New-Object Media.SoundPlayer '{out}').PlaySync()"],
            creationflags=0x08000000)  # CREATE_NO_WINDOW
    except Exception:
        pass


if __name__ == "__main__":
    main()
