"""Voice in, voice out.

Speech-to-text: microphone -> local faster-whisper or the OpenAI Whisper API.
Triggers: say "Hey DM ..." (always listening), press Enter in the terminal, or
press Space on the projector page. Typed terminal lines go straight to the DM.

Text-to-speech: ElevenLabs (streamed), Piper (local), or pyttsx3 (local).
Speech plays on its own worker thread so projector effects fire at the same
moment the DM starts talking.
"""

from __future__ import annotations

import io
import os
import queue
import re
import subprocess
import sys
import threading
import time
import wave
from typing import Callable

import numpy as np
import requests


def _sd():
    try:
        import sounddevice as sd
    except (ImportError, OSError) as e:  # OSError: PortAudio library missing
        raise RuntimeError(f"Audio needs the sounddevice package and PortAudio: {e}") from e
    return sd


# ---------------------------------------------------------------- hotword

_DM_FUZZ = r"(?:d\.?\s*m\.?|dee\s*em|d\s*n|the\s+dm|dm)"


def hotword_pattern(hotword: str) -> re.Pattern:
    words = hotword.lower().split()
    if words == ["hey", "dm"]:
        body = rf"(?:hey|hi|hay|a|okay|ok)[\s,]+{_DM_FUZZ}"
    else:
        body = r"[\s,.]+".join(re.escape(w) for w in words)
    return re.compile(rf"^\s*{body}(?![a-z])[\s,.!?:;-]*", re.I)


def strip_hotword(text: str, pattern: re.Pattern) -> str | None:
    """Command after the hotword, or None if the phrase didn't start with it."""
    m = pattern.match(text)
    if not m:
        return None
    return text[m.end():].strip()


# ---------------------------------------------------------------- audio helpers

def rms(block: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(block, dtype=np.float64)))) if block.size else 0.0


def to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


class VoiceService:
    def __init__(self, config: dict, on_command: Callable[[str], None]):
        self.cfg = config["voice"]
        self.on_command = on_command
        self.sample_rate = int(self.cfg["sample_rate"])
        self.pattern = hotword_pattern(self.cfg["hotword"])
        self._armed = threading.Event()      # next phrase is a command, no hotword needed
        self._speaking = threading.Event()   # mute the mic while the DM talks
        self._stop = threading.Event()
        self._tts_queue: queue.Queue[str | None] = queue.Queue()
        self._whisper = None
        self._threads: list[threading.Thread] = []

    # ---------------------------------------------------------------- lifecycle

    def start(self) -> None:
        self._spawn(self._tts_worker)
        try:
            _sd()
            self._spawn(self._listen_loop)
            print(f"[voice] listening. Say \"{self.cfg['hotword']} ...\" or press Space on the projector page.")
        except RuntimeError as e:
            print(f"[voice] microphone disabled: {e}")
        if self.cfg.get("keyboard_enabled", True) and sys.stdin and sys.stdin.isatty():
            self._spawn(self._keyboard_loop)
            print("[voice] terminal: press Enter to talk, or type a command and press Enter.")

    def stop(self) -> None:
        self._stop.set()
        self._tts_queue.put(None)

    def _spawn(self, fn) -> None:
        t = threading.Thread(target=fn, daemon=True)
        t.start()
        self._threads.append(t)

    def push_to_talk(self) -> None:
        """Treat the next spoken phrase as a command (no hotword needed)."""
        self._armed.set()
        print("[voice] listening for your command...")

    # ---------------------------------------------------------------- input loops

    def _keyboard_loop(self) -> None:
        for line in sys.stdin:
            if self._stop.is_set():
                return
            text = line.strip()
            if text:
                self.on_command(text)
            else:
                self.push_to_talk()

    def _listen_loop(self) -> None:
        while not self._stop.is_set():
            try:
                audio = self.record_phrase()
            except Exception as e:
                print(f"[voice] mic error: {e}; retrying in 2 s")
                time.sleep(2)
                continue
            if audio is None or self._speaking.is_set():
                continue
            armed = self._armed.is_set()
            if not armed and not self.cfg.get("hotword_enabled", True):
                continue  # nobody asked for this phrase; skip transcription entirely
            try:
                text = self.transcribe(audio).strip()
            except Exception as e:
                print(f"[voice] transcription failed: {e}")
                continue
            if not text:
                continue
            if armed:
                self._armed.clear()
                command = text
            else:
                command = strip_hotword(text, self.pattern)
                if command is None:
                    continue
                if not command:  # just "Hey DM" -> listen for the real command next
                    self._armed.set()
                    continue
            print(f"[voice] heard: {command}")
            self.on_command(command)

    def record_phrase(self) -> np.ndarray | None:
        """Block until someone speaks, return that phrase as float32 mono audio."""
        sd = _sd()
        block_s = 0.1
        block = int(self.sample_rate * block_s)
        threshold = float(self.cfg["silence_threshold"])
        silence_blocks = int(self.cfg["silence_duration_s"] / block_s)
        max_blocks = int(self.cfg["max_phrase_s"] / block_s)
        preroll: list[np.ndarray] = []
        frames: list[np.ndarray] = []
        quiet = 0
        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="float32", blocksize=block) as stream:
            while not self._stop.is_set():
                data, _ = stream.read(block)
                data = data[:, 0].copy()
                loud = rms(data) >= threshold
                if not frames:
                    preroll = (preroll + [data])[-3:]  # keep 0.3 s so the first syllable isn't clipped
                    if loud:
                        frames = preroll[:]
                    continue
                frames.append(data)
                quiet = 0 if loud else quiet + 1
                if quiet >= silence_blocks or len(frames) >= max_blocks:
                    break
        if len(frames) < 4:  # under ~0.4 s: a cough or a dice roll
            return None
        return np.concatenate(frames)

    # ---------------------------------------------------------------- speech to text

    def transcribe(self, audio: np.ndarray) -> str:
        if self.cfg["stt"] == "openai":
            from openai import OpenAI
            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                raise RuntimeError("Set OPENAI_API_KEY to use the Whisper API (or set voice.stt: local)")
            client = OpenAI(api_key=key)
            result = client.audio.transcriptions.create(
                model=self.cfg["openai_stt_model"],
                file=("speech.wav", to_wav_bytes(audio, self.sample_rate), "audio/wav"),
            )
            return result.text
        if self._whisper is None:
            from faster_whisper import WhisperModel
            print(f"[voice] loading local Whisper '{self.cfg['local_whisper_model']}' (first run downloads it)")
            self._whisper = WhisperModel(self.cfg["local_whisper_model"], device="auto", compute_type="int8")
        segments, _ = self._whisper.transcribe(audio, language="en", beam_size=1)
        return " ".join(s.text.strip() for s in segments)

    # ---------------------------------------------------------------- text to speech

    def speak(self, text: str) -> None:
        """Queue narration; returns immediately."""
        if text and self.cfg.get("tts", "none") != "none":
            self._tts_queue.put(text)

    def _tts_worker(self) -> None:
        engine = None
        while True:
            text = self._tts_queue.get()
            if text is None:
                return
            self._speaking.set()
            try:
                kind = self.cfg["tts"]
                if kind == "elevenlabs":
                    self._speak_elevenlabs(text)
                elif kind == "piper":
                    self._speak_piper(text)
                else:
                    if engine is None:  # pyttsx3 must stay on the thread that created it
                        import pyttsx3
                        engine = pyttsx3.init()
                    engine.say(text)
                    engine.runAndWait()
            except Exception as e:
                print(f"[voice] TTS failed ({self.cfg['tts']}): {e}")
            finally:
                time.sleep(0.3)  # let the room echo die before the mic listens again
                self._speaking.clear()

    def _play_pcm_stream(self, chunks, sample_rate: int) -> None:
        sd = _sd()
        leftover = b""
        with sd.RawOutputStream(samplerate=sample_rate, channels=1, dtype="int16") as out:
            for chunk in chunks:
                data = leftover + chunk
                cut = len(data) - (len(data) % 2)  # int16 = 2 bytes; keep samples whole
                if cut:
                    out.write(data[:cut])
                leftover = data[cut:]

    def _speak_elevenlabs(self, text: str) -> None:
        p = self.cfg["elevenlabs"]
        key = os.environ.get(p["api_key_env"])
        if not key:
            raise RuntimeError(f"Set {p['api_key_env']} to use ElevenLabs")
        rate = 22050
        with requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{p['voice_id']}/stream",
            params={"output_format": f"pcm_{rate}"},
            headers={"xi-api-key": key, "Content-Type": "application/json"},
            json={"text": text, "model_id": p["model"]},
            stream=True,
            timeout=60,
        ) as resp:
            resp.raise_for_status()
            self._play_pcm_stream(resp.iter_content(chunk_size=4096), rate)

    def _speak_piper(self, text: str) -> None:
        p = self.cfg["piper"]
        proc = subprocess.Popen(
            [p["binary"], "--model", p["model_path"], "--output-raw"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        proc.stdin.write(text.encode())
        proc.stdin.close()
        self._play_pcm_stream(iter(lambda: proc.stdout.read(4096), b""), int(p["sample_rate"]))
        proc.wait()
