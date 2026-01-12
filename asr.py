import queue
import threading
import asyncio
import subprocess
from time import time

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import websockets

# ======================================================
# Audio Input
# ======================================================

class AudioInput:
    def __init__(
        self,
        device_id: int,
        input_sr: int = 48000,
        target_sr: int = 16000,
        chunk_seconds: float = 2.5,
        gain: float = 3.5,
        rms_threshold: float = 0.005,
    ):
        self.device_id = device_id
        self.input_sr = input_sr
        self.target_sr = target_sr
        self.chunk_seconds = chunk_seconds
        self.block_size = int(input_sr * chunk_seconds)
        self.gain = gain
        self.rms_threshold = rms_threshold
        self.queue = queue.Queue()

    def _callback(self, indata, frames, time_info, status):
        if status:
            print("Audio status:", status)

        audio = indata[:, 0]
        audio = np.clip(audio * self.gain, -1.0, 1.0)
        self.queue.put(audio.copy())

    def start(self):
        return sd.InputStream(
            device=self.device_id,
            samplerate=self.input_sr,
            channels=1,
            dtype="float32",
            blocksize=self.block_size,
            callback=self._callback,
        )

    def get_chunk(self):
        audio = self.queue.get()
        audio = self._resample(audio)

        rms = np.sqrt(np.mean(audio ** 2))
        if rms < self.rms_threshold:
            return None

        return audio

    def _resample(self, audio):
        if self.input_sr == self.target_sr:
            return audio

        ratio = self.target_sr / self.input_sr
        new_len = int(len(audio) * ratio)

        return np.interp(
            np.linspace(0, len(audio), new_len, endpoint=False),
            np.arange(len(audio)),
            audio
        ).astype(np.float32)

# ======================================================
# Whisper ASR
# ======================================================

class WhisperASR:
    def __init__(self):
        self.model = WhisperModel(
            "small",
            device="cpu",
            compute_type="int8"
        )

    def transcribe(self, audio_chunk):
        segments, _ = self.model.transcribe(
            audio_chunk,
            language="ko",
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300)
        )
        return [seg.text.strip() for seg in segments if seg.text.strip()]

# ======================================================
# Ollama Translator (CLI)
# ======================================================

class OllamaTranslator:
    def __init__(self, model: str):
        self.model = model
        self.cache = {}

    def translate(self, text: str) -> str:
        if text in self.cache:
            return self.cache[text]

        prompt = (
            "你是一個即時字幕翻譯引擎。\n"
            "請將下面的韓文翻譯成自然、口語、繁體中文。\n"
            "只輸出翻譯結果，不要解釋。\n\n"
            f"韓文：{text}\n"
            "繁體中文："
        )

        proc = subprocess.Popen(
            ["ollama", "run", self.model],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="ignore"
        )

        stdout, stderr = proc.communicate(prompt, timeout=120)

        if proc.returncode != 0:
            raise RuntimeError(stderr.strip())

        result = stdout.strip()
        if not result:
            raise RuntimeError("Ollama 沒有輸出任何內容")

        self.cache[text] = result
        return result

# ======================================================
# WebSocket Subtitle Server
# ======================================================

class SubtitleServer:
    def __init__(self, host="localhost", port=8765):
        self.host = host
        self.port = port
        self.clients = set()

    async def _handler(self, websocket):
        self.clients.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            self.clients.remove(websocket)

    async def start(self):
        async with websockets.serve(self._handler, self.host, self.port):
            await asyncio.Future()

    async def broadcast(self, text: str):
        if self.clients:
            await asyncio.gather(
                *[client.send(text) for client in self.clients]
            )

# ======================================================
# ASR Pipeline（主控）
# ======================================================

class ASRPipeline:
    def __init__(self):
        self.audio = AudioInput(device_id=17)
        self.asr = WhisperASR()
        self.translator = OllamaTranslator("qwen2.5:7b")
        self.subtitle_server = SubtitleServer()

        self.ko_buffer = []
        self.last_flush_time = time()
        self.last_ko_text = ""

    def start(self):
        threading.Thread(
            target=lambda: asyncio.run(self.subtitle_server.start()),
            daemon=True
        ).start()

        print("🎧 開始監聽聲音（Ctrl+C 結束）...")

        with self.audio.start():
            while True:
                audio_chunk = self.audio.get_chunk()
                if audio_chunk is None:
                    continue

                texts = self.asr.transcribe(audio_chunk)
                for text in texts:
                    if len(text) < 6 or text == self.last_ko_text:
                        continue
                    self.last_ko_text = text
                    self.ko_buffer.append(text)

                now = time()
                if self.ko_buffer and now - self.last_flush_time >= 1.2:
                    merged = " ".join(self.ko_buffer)
                    self.ko_buffer.clear()
                    self.last_flush_time = now

                    print(f"🇰🇷 {merged}")

                    try:
                        zh = self.translator.translate(merged)
                    except Exception as e:
                        print("⚠️ 翻譯失敗：", e)
                        continue

                    print(f"🇹🇼 {zh}\n")
                    asyncio.run(self.subtitle_server.broadcast(zh))

# ======================================================
# Entry Point
# ======================================================

if __name__ == "__main__":
    ASRPipeline().start()
