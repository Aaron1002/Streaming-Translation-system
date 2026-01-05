import os
from dotenv import load_dotenv
from google import genai

# 載入 .env
load_dotenv()

# 建立 Gemini client
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

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
        model="models/gemini-2.5-flash",  # ✅ 關鍵在這行
        contents=prompt,
    )

    return response.text.strip()


if __name__ == "__main__":
    print(translate_ko_to_zh("오늘 방송 진짜 재밌어요"))
