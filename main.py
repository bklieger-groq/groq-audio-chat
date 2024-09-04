from fastapi import FastAPI, Request, UploadFile, File, WebSocket
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, Response
import os
from groq import Groq
import asyncio
import json
import httpx
import time
from uuid import uuid4

app = FastAPI()

# Initialize Groq client
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ElevenLabs constants
XI_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
VOICE_ID = "Xb7hH8MSUJpSbSDYk0k2"

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Initialize conversation history
default_system_prompt = "You are an AI travel agent tasked with preparing a comprehensive travel itinerary for the human. Your itinerary should include places to see, dining recommendations, and a detailed schedule. Before creating the itinerary, ask clarifying questions to understand the traveler's preferences. Be concise in your responses, except when providing the actual itinerary. Limit your responses to 1-3 sentences unless you're detailing the itinerary."

# Store conversations for each session
conversations = {}

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    contents = await file.read()
    transcription = groq_client.audio.transcriptions.create(
        file=("audio.wav", contents),
        model="distil-whisper-large-v3-en",
        response_format="text",
        language="en"
    )
    return {"transcription": transcription}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid4())
    conversations[session_id] = [
        {"role": "system", "content": default_system_prompt}
    ]
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "user_message":
                conversations[session_id].append({"role": "user", "content": message["content"]})
                ai_response = await get_ai_response(conversations[session_id])
                conversations[session_id].append({"role": "assistant", "content": ai_response})
                await websocket.send_text(json.dumps({"type": "ai_response", "content": ai_response}))
            elif message["type"] == "update_system_prompt":
                # Update the system prompt for this session
                conversations[session_id][0] = {"role": "system", "content": message["content"]}
                await websocket.send_text(json.dumps({"type": "system_prompt_updated"}))
    finally:
        # Clean up the conversation when the WebSocket connection closes
        del conversations[session_id]

async def get_ai_response(conversation):
    max_retries = 3
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            chat_completion = groq_client.chat.completions.create(
                messages=conversation,
                model="llama3-8b-8192",
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Error getting AI response (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"Failed to get AI response after {max_retries} attempts: {e}")
                raise

@app.get("/text-to-speech")
async def text_to_speech(text: str):
    async def generate():
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": XI_API_KEY
        }
        data = {
            "text": text,
            "model":"eleven_turbo_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.8
            }
        }

        async with httpx.AsyncClient() as client:
            async with client.stream("POST", url, json=data, headers=headers) as response:
                async for chunk in response.aiter_bytes():
                    yield chunk

    async def test_gen():
        for i in range(10):
            yield str(i).encode()
            print("i: ", i)
            await asyncio.sleep(1)

    # return StreamingResponse(test_gen(), media_type="audio/mpeg")

    return StreamingResponse(generate(), media_type="audio/mpeg")
