import asyncio
import time
import httpx
import os
from datetime import datetime, timedelta

XI_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
VOICE_ID = "Xb7hH8MSUJpSbSDYk0k2"

async def generate_audio(text):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": XI_API_KEY
    }
    data = {
        "text": text,
        "model": "eleven_turbo_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.8
        }
    }

    async with httpx.AsyncClient() as client:
        async with client.stream("POST", url, json=data, headers=headers) as response:
            async for chunk in response.aiter_bytes():
                yield chunk

async def main():
    text = "This is a test of the ElevenLabs text-to-speech API with async streaming. This is a test of the ElevenLabs text-to-speech API with async streaming. This is a test of the ElevenLabs text-to-speech API with async streaming. This is a test of the ElevenLabs text-to-speech API with async streaming. This is a test of the ElevenLabs text-to-speech API with async streaming. This is a test of the ElevenLabs text-to-speech API with async streaming. This is a test of the ElevenLabs text-to-speech API with async streaming. This is a test of the ElevenLabs text-to-speech API with async streaming. This is a test of the ElevenLabs text-to-speech API with async streaming."
    start_time = time.time()
    chunk_count = 0
    total_bytes = 0

    async for chunk in generate_audio(text):
        chunk_count += 1
        total_bytes += len(chunk)
        elapsed = timedelta(seconds=time.time() - start_time)
        timestamp = f"[{elapsed.seconds:02d}:{elapsed.microseconds//10000:02d}.{elapsed.microseconds%10000//100:02d}]"
        print(f"{timestamp} Received chunk {chunk_count}: {len(chunk)} bytes")

    end_time = time.time()
    print(f"\nTotal chunks: {chunk_count}")
    print(f"Total bytes: {total_bytes}")
    print(f"Total time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())
