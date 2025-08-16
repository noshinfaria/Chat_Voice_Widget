import os
import threading
import queue
import time
import json
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from deepgram import DeepgramClient, DeepgramClientOptions, AgentWebSocketEvents, AgentKeepAlive
from deepgram.clients.agent.v1.websocket.options import SettingsOptions

load_dotenv()
app = FastAPI()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
deepgram_client = DeepgramClient(DEEPGRAM_API_KEY, DeepgramClientOptions(options={"keepalive": "true"}))

# Serve static folder
app.mount("/static", StaticFiles(directory="static"), name="static")
@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.websocket("/ws/call/{agent_id}")
async def websocket_call(websocket: WebSocket, agent_id: str):
    await websocket.accept()

    client_audio_queue = queue.Queue()   # User -> Deepgram
    agent_audio_queue = queue.Queue()    # Deepgram -> Client
    transcript_queue = queue.Queue()     # Transcripts -> Client
    processing_complete = threading.Event()

    session_id = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    conversation_audio_file = f"conversation_{session_id}.wav"
    transcript_file = f"transcript_{session_id}.txt"

    def create_wav_header(sample_rate=24000, bits_per_sample=16, channels=1):
        byte_rate = sample_rate * channels * (bits_per_sample // 8)
        block_align = channels * (bits_per_sample // 8)
        header = bytearray(44)
        header[0:4] = b'RIFF'
        header[4:8] = (36).to_bytes(4, 'little')
        header[8:12] = b'WAVE'
        header[12:16] = b'fmt '
        header[16:20] = (16).to_bytes(4, 'little')
        header[20:22] = (1).to_bytes(2, 'little')
        header[22:24] = channels.to_bytes(2, 'little')
        header[24:28] = sample_rate.to_bytes(4, 'little')
        header[28:32] = byte_rate.to_bytes(4, 'little')
        header[32:34] = block_align.to_bytes(2, 'little')
        header[34:36] = bits_per_sample.to_bytes(2, 'little')
        header[36:40] = b'data'
        header[40:44] = (0).to_bytes(4, 'little')
        return header

    # Thread for Deepgram Agent
    def deepgram_agent_thread():
        try:
            connection = deepgram_client.agent.websocket.v("1")
            options = SettingsOptions()
            options.audio.input.encoding = "linear16"
            options.audio.input.sample_rate = 24000
            options.audio.output.encoding = "linear16"
            options.audio.output.sample_rate = 24000
            options.audio.output.container = "wav"
            options.agent.language = "en"
            options.agent.listen.provider.type = "deepgram"
            options.agent.listen.provider.model = "nova-3"
            options.agent.think.provider.type = "open_ai"
            options.agent.think.provider.model = "gpt-4o-mini"
            options.agent.think.prompt = "You are a friendly AI assistant."
            options.agent.speak.provider.type = "deepgram"
            options.agent.speak.provider.model = "aura-2-thalia-en"
            options.agent.greeting = "Hello! How can I help you today?"

            def keep_alive():
                while not processing_complete.is_set():
                    connection.send(str(AgentKeepAlive()))
                    time.sleep(5)
            threading.Thread(target=keep_alive, daemon=True).start()

            def on_audio_data(self, data, **kwargs):
                agent_audio_queue.put(data)

            def on_conversation_text(self, conversation_text, **kwargs):
                transcript_queue.put(conversation_text.__dict__)

            connection.on(AgentWebSocketEvents.AudioData, on_audio_data)
            connection.on(AgentWebSocketEvents.ConversationText, on_conversation_text)

            if not connection.start(options):
                processing_complete.set()
                return

            while not processing_complete.is_set():
                try:
                    chunk = client_audio_queue.get(timeout=1)
                    connection.send(chunk)
                except queue.Empty:
                    continue

            connection.finish()
        except Exception as e:
            print("Deepgram agent thread error:", e)
            processing_complete.set()

    threading.Thread(target=deepgram_agent_thread, daemon=True).start()

    # Open WAV file to write conversation
    with open(conversation_audio_file, "wb") as conv_file:
        conv_file.write(create_wav_header())

        try:
            while True:
                data = await websocket.receive_bytes()
                client_audio_queue.put(data)
                conv_file.write(data)  # Save user audio

                # Send agent audio to client and save
                while not agent_audio_queue.empty():
                    audio_chunk = agent_audio_queue.get_nowait()
                    conv_file.write(audio_chunk)
                    await websocket.send_bytes(audio_chunk)

                # Send transcript to client and save
                while not transcript_queue.empty():
                    transcript = transcript_queue.get_nowait()
                    await websocket.send_json({"type": "transcript", "data": transcript})
                    with open(transcript_file, "a") as f:
                        f.write(json.dumps(transcript) + "\n")

        except WebSocketDisconnect:
            processing_complete.set()
            print(f"Conversation saved: {conversation_audio_file}, Transcript: {transcript_file}")
