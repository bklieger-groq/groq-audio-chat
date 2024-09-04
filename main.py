from fastapi import FastAPI, Request, UploadFile, File, WebSocket
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
import os
import requests
from groq import Groq
import io
import asyncio
import json

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
conversation = [
    {"role": "system", "content": "You are an AI travel agent who will prepare a full travel itinerary for the human. It should include places to see, places to dine, and a full schedule. Ask clarifying questions before creating it. BE CONCISE UNLESS YOU ARE PROVIDING THE ITINERARY. ONLY 1-3 SENTENCES MAX."}
]

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
    while True:
        data = await websocket.receive_text()
        message = json.loads(data)
        
        if message["type"] == "user_message":
            conversation.append({"role": "user", "content": message["content"]})
            ai_response = await get_ai_response(conversation)
            conversation.append({"role": "assistant", "content": ai_response})
            await websocket.send_text(json.dumps({"type": "ai_response", "content": ai_response}))

async def get_ai_response(conversation):
    chat_completion = groq_client.chat.completions.create(
        messages=conversation,
        model="llama3-8b-8192",
    )
    return chat_completion.choices[0].message.content

@app.get("/text-to-speech")
async def text_to_speech(text: str):
    tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream"
    headers = {
        "Accept": "audio/mpeg",
        "xi-api-key": XI_API_KEY,
        "Content-Type": "application/json"
    }
    data = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.8
        }
    }
    response = requests.post(tts_url, json=data, headers=headers, stream=True)
    if response.ok:
        return StreamingResponse(response.iter_content(chunk_size=4096), media_type="audio/mpeg")
    else:
        return {"error": f"Error generating audio: {response.text}"}
