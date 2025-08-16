import os
import threading
import queue
import time
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient

from fastapi import FastAPI, HTTPException, WebSocket, Body, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv

from deepgram import DeepgramClient, DeepgramClientOptions, AgentWebSocketEvents, AgentKeepAlive
from deepgram.clients.agent.v1.websocket.options import SettingsOptions

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()
app = FastAPI()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "agentdb")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
deepgram_client = DeepgramClient(DEEPGRAM_API_KEY, DeepgramClientOptions(options={"keepalive": "true"}))

# Serve static folder
app.mount("/static", StaticFiles(directory="static"), name="static")


# -----------------------------------------------------------------------------
# Mongo helpers
# -----------------------------------------------------------------------------
client = AsyncIOMotorClient(MONGO_URI)
db = client[DB_NAME]
agents_col = db["agents"]

class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    @classmethod
    def validate(cls, v):
        if isinstance(v, ObjectId):
            return v
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)
    @classmethod
    def __get_pydantic_json_schema__(cls, field_schema):
        field_schema.update(type="string")

# -----------------------------------------------------------------------------
# Data models
# -----------------------------------------------------------------------------
class ChatConfig(BaseModel):
    llm_model: str = "gpt-4o-mini"
    temperature: float = 0.3
    system_prompt: str = "You are a helpful, concise assistant."
    kb: Optional[str] = None  # knowledge base id/ref (optional)

class VoiceConfig(BaseModel):
    # For Deepgram Live + TTS
    stt_model: str = "nova-2"
    tts_voice: str = "aura-asteria-en"  # pick any supported voice
    enable_tts: bool = True

class WidgetConfig(BaseModel):
    enable_text: bool = True
    enable_voice: bool = True
    suggested_prompts: List[str] = Field(default_factory=lambda: ["Help me get started", "What can you do?"])

class AgentCreate(BaseModel):
    agent_name: str
    welcome_msg: str = "Hi! How can I help?"
    chat: ChatConfig = ChatConfig()
    voice: VoiceConfig = VoiceConfig()
    widget: WidgetConfig = WidgetConfig()
    # Optional owner scoping
    user_id: Optional[str] = None

class AgentDB(AgentCreate):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {ObjectId: str}
        populate_by_name = True

class ChatMessage(BaseModel):
    role: str  # "user" | "assistant" | "system"
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.websocket("/ws/call/{agent_id}")
async def websocket_call(websocket: WebSocket, agent_id: str):
    await websocket.accept()

    # Thread-safe queues
    client_audio_queue = queue.Queue()   # User -> Deepgram
    agent_audio_queue = queue.Queue()    # Deepgram -> Client
    transcript_queue = queue.Queue()     # Transcripts -> Client
    stop_event = threading.Event()

    # Files
    session_id = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    conversation_audio_file = f"conversation_{session_id}.wav"
    transcript_file = f"transcript_{session_id}.txt"

    # WAV header
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

    # ---------------- Deepgram Agent Thread ----------------
    def deepgram_agent():
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

            # Keep-alive thread
            def keep_alive():
                while not stop_event.is_set():
                    try:
                        connection.send(str(AgentKeepAlive()))
                        time.sleep(5)
                    except Exception:
                        break
            threading.Thread(target=keep_alive, daemon=True).start()

            # Callbacks
            def on_audio_data(self, data, **kwargs):
                agent_audio_queue.put(data)

            def on_conversation_text(self, conversation_text, **kwargs):
                transcript_queue.put(conversation_text.__dict__)

            connection.on(AgentWebSocketEvents.AudioData, on_audio_data)
            connection.on(AgentWebSocketEvents.ConversationText, on_conversation_text)

            if not connection.start(options):
                stop_event.set()
                return

            # Send client audio to Deepgram
            while not stop_event.is_set():
                try:
                    chunk = client_audio_queue.get(timeout=1)
                    connection.send(chunk)
                except queue.Empty:
                    continue

            connection.finish()
        except Exception as e:
            print("Deepgram agent error:", e)
            stop_event.set()

    threading.Thread(target=deepgram_agent, daemon=True).start()

    # ---------------- WebSocket Receive ----------------
    with open(conversation_audio_file, "wb") as conv_file:
        conv_file.write(create_wav_header())
        try:
            while not stop_event.is_set():
                # Receive from client
                data = await websocket.receive_bytes()
                client_audio_queue.put(data)
                conv_file.write(data)

                # Send agent audio to client
                while not agent_audio_queue.empty():
                    chunk = agent_audio_queue.get_nowait()
                    conv_file.write(chunk)
                    await websocket.send_bytes(chunk)

                # Send transcript
                while not transcript_queue.empty():
                    transcript = transcript_queue.get_nowait()
                    await websocket.send_json({"type": "transcript", "data": transcript})
                    with open(transcript_file, "a") as f:
                        f.write(json.dumps(transcript) + "\n")

        except WebSocketDisconnect:
            stop_event.set()
            print(f"Conversation saved: {conversation_audio_file}, Transcript: {transcript_file}")
        except Exception as e:
            stop_event.set()
            print("WebSocket error:", e)



# -----------------------------------------------------------------------------
# LangGraph: a tiny, extensible graph for text chat
# -----------------------------------------------------------------------------
def build_text_graph(system_prompt: str, model_name: str, temperature: float):
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model=model_name, temperature=temperature)

    class ChatState(BaseModel):
        history: List[Any]  # LC message objects

    def call_llm(state: ChatState):
        msgs = state.history
        resp = llm.invoke(msgs)
        return ChatState(history=msgs + [resp])

    graph = StateGraph(ChatState)
    graph.add_node("llm", call_llm)
    graph.add_edge(START, "llm")
    graph.add_edge("llm", END)
    app_ = graph.compile()

    def run_chat(user_text: str, prior: List[Any] = None):
        prior = prior or []
        state = ChatState(history=[SystemMessage(content=system_prompt)] + prior + [HumanMessage(content=user_text)])
        out = app_.invoke(state)
        return out.history

    return run_chat

# -----------------------------------------------------------------------------
# Routes: Agents CRUD
# -----------------------------------------------------------------------------
@app.post("/agents", response_model=AgentDB)
async def create_agent(payload: AgentCreate):
    doc = payload.model_dump()
    doc["created_at"] = datetime.utcnow()
    doc["updated_at"] = datetime.utcnow()
    res = await agents_col.insert_one(doc)
    saved = await agents_col.find_one({"_id": res.inserted_id})
    return AgentDB(**saved)

@app.get("/agents/{agent_id}", response_model=AgentDB)
async def get_agent(agent_id: str):
    agent = await agents_col.find_one({"_id": ObjectId(agent_id)})
    if not agent:
        raise HTTPException(404, "Agent not found")
    return AgentDB(**agent)

@app.patch("/agents/{agent_id}", response_model=AgentDB)
async def update_agent(agent_id: str, patch: Dict[str, Any] = Body(...)):
    patch["updated_at"] = datetime.utcnow()
    await agents_col.update_one({"_id": ObjectId(agent_id)}, {"$set": patch})
    agent = await agents_col.find_one({"_id": ObjectId(agent_id)})
    if not agent:
        raise HTTPException(404, "Agent not found after update")
    return AgentDB(**agent)

@app.delete("/agents/{agent_id}")
async def delete_agent(agent_id: str):
    await agents_col.delete_one({"_id": ObjectId(agent_id)})
    return {"ok": True}


# -----------------------------------------------------------------------------
# Route: Widget config (combine text+voice)
# -----------------------------------------------------------------------------
@app.get("/agents/{agent_id}/widget")
async def get_widget(agent_id: str):
    agent = await agents_col.find_one({"_id": ObjectId(agent_id)})
    if not agent:
        raise HTTPException(404, "Agent not found")
    return {
        "agent_name": agent["agent_name"],
        "welcome_msg": agent["welcome_msg"],
        "text": agent["chat"],
        "voice": agent["voice"],
        "widget": agent["widget"],
    }

# -----------------------------------------------------------------------------
# Text Chat endpoint (LangGraph)
# -----------------------------------------------------------------------------
@app.post("/agents/{agent_id}/chat")
async def chat(agent_id: str, req: ChatRequest):
    agent = await agents_col.find_one({"_id": ObjectId(agent_id)})
    if not agent:
        raise HTTPException(404, "Agent not found")

    chat_cfg = ChatConfig(**agent["chat"])
    # Build graph (could be cached per agent in production)
    run_chat = build_text_graph(
        system_prompt=chat_cfg.system_prompt,
        model_name=chat_cfg.llm_model,
        temperature=chat_cfg.temperature,
    )
    # Turn ChatRequest to LC messages (only user/assistant are needed)
    prior = []
    for m in req.messages:
        if m.role == "assistant":
            prior.append(AIMessage(content=m.content))
        elif m.role == "user":
            prior.append(HumanMessage(content=m.content))
        # ignore extra system messages; we inject our own

    history = run_chat(user_text=req.messages[-1].content, prior=prior[:-1] if len(prior) > 0 else [])
    # Return only the last assistant message
    last = next((m for m in reversed(history) if isinstance(m, AIMessage)), None)
    return {"reply": last.content if last else ""}