from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from backend.orchestrator.voice_session import VoiceSession
import json
import os

app = FastAPI(title="VĀYU Voice RAG Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session = VoiceSession(websocket)
    try:
        while True:
            # We expect a mix of binary (audio) and text (events)
            message = await websocket.receive()
            if "bytes" in message:
                await session.handle_audio_chunk(message["bytes"])
            elif "text" in message:
                data = json.loads(message["text"])
                if data.get("event") == "PARTIAL":
                    await session.handle_partial_transcript(data["text"])
                elif data.get("event") == "FINAL":
                    await session.handle_endpoint(data["text"])
    except WebSocketDisconnect:
        print("Client disconnected")

@app.get("/api/benchmark/results")
async def get_benchmark_results():
    results_path = os.path.join(os.path.dirname(__file__), "../benchmark_results.json")
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            return json.load(f)
    return {"message": "Benchmark not run yet."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
