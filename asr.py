import os
import queue
from time import time

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from google import genai

import asyncio
import websockets
# ======================================================
# 基本設定
# ======================================================

connected_clients = set()

async def ws_handler(websocket):
    connected_clients.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        connected_clients.remove(websocket)

async def start_ws_server():
    async with websockets.serve(ws_handler, "localhost", 8765):
        await asyncio.Future()  # run forever

async def broadcast_subtitle(text):
    if connected_clients:
        await asyncio.gather(
            *[client.send(text) for client in connected_clients]
        )


load_dotenv()

# Gemini client（新版 SDK）
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# Whisper 模型（即時字幕建議 small）
whisper_model = WhisperModel(
    "small",
    device="cpu",        # 有 GPU 可改 "cuda"
    compute_type="int8"  # CPU 省資源
)

# Audio 設定
SAMPLE_RATE = 16000
CHUNK_SECONDS = 1.5
BLOCK_SIZE = int(SAMPLE_RATE * CHUNK_SECONDS)

audio_queue = queue.Queue()

# ======================================================
# Step 1：短句過濾
# ======================================================

MIN_KO_LENGTH = 6   # 少於 6 字的韓文不翻

# ======================================================
# Step 2：重複句去重
# ======================================================

# last_ko_text = ""

# ======================================================
# Step 3：多段合併（節流）
# ======================================================

# ko_buffer = []
# last_flush_time = time()
FLUSH_INTERVAL = 1.2

# ======================================================
# Step 4：翻譯結果快取
# ======================================================

# translation_cache = {}

# ======================================================
# Audio callback
# ======================================================

def audio_callback(indata, frames, time_info, status):
    if status:
        print("Audio status:", status)
    audio_queue.put(indata.copy())

# ======================================================
# Gemini 翻譯函式
# ======================================================

def translate_ko_to_zh(text: str) -> str:
    prompt = f"""
你是一個即時直播字幕翻譯引擎。
請將下面的韓文翻譯成「自然、口語、繁體中文」。
只輸出翻譯結果，不要解釋。

韓文：
{text}

繁體中文：
"""
    response = client.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=prompt,
    )
    return response.text.strip()

# ======================================================
# 主流程
# ======================================================
def main():
    print("🎧 開始監聽聲音（Ctrl+C 結束）...")

    ko_buffer = []
    last_flush_time = time()
    last_ko_text = ""
    translation_cache = {}

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=BLOCK_SIZE,
        callback=audio_callback,
    ):
        while True:
            audio_chunk = audio_queue.get()
            audio_chunk = audio_chunk.flatten()

            # Whisper ASR
            segments, _ = whisper_model.transcribe(
                audio_chunk,
                language="ko"
            )

            for seg in segments:
                ko_text = seg.text.strip()
                if not ko_text:
                    continue

                # --------------------------------------------------
                # Step 1：短句過濾
                # --------------------------------------------------
                if len(ko_text) < MIN_KO_LENGTH:
                    continue

                # --------------------------------------------------
                # Step 2：重複句去重
                # --------------------------------------------------
                if ko_text == last_ko_text:
                    continue
                last_ko_text = ko_text

                # --------------------------------------------------
                # Step 3：累積到 buffer
                # --------------------------------------------------
                ko_buffer.append(ko_text)

            # --------------------------------------------------
            # Step 3：定時 flush buffer
            # --------------------------------------------------
            now = time()
            if ko_buffer and (now - last_flush_time >= FLUSH_INTERVAL):
                merged_text = " ".join(ko_buffer)
                ko_buffer.clear()
                last_flush_time = now

                if len(merged_text) < MIN_KO_LENGTH:
                    continue

                print(f"🇰🇷 {merged_text}")

                # --------------------------------------------------
                # Step 4：翻譯快取
                # --------------------------------------------------
                if merged_text in translation_cache:
                    zh_text = translation_cache[merged_text]
                else:
                    try:
                        zh_text = translate_ko_to_zh(merged_text)
                        translation_cache[merged_text] = zh_text
                    except Exception as e:
                        print("⚠️ 翻譯失敗：", e)
                        continue

                print(f"🇹🇼 {zh_text}\n")
                asyncio.run(broadcast_subtitle(zh_text))

if __name__ == "__main__":
    import threading

    # 啟動 WebSocket Server（背景）
    threading.Thread(
        target=lambda: asyncio.run(start_ws_server()),
        daemon=True
    ).start()

    # 啟動主流程
    main()