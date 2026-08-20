from dotenv import load_dotenv
import os

# Load .env before importing other modules
load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from backend.orchestrator.voice_session import VoiceSession
import json
import socket as _socket

app = FastAPI(title="VĀYU Voice RAG Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _enable_tcp_nodelay_for_websockets():
    """uvicorn's WebSocket paths (websockets-sansio / legacy) do NOT set
    TCP_NODELAY, so Nagle holds small frames until the peer's delayed ACK
    (~40ms per round-trip — the 'ato:40' seen in ss). Patch the active
    protocol class to disable Nagle on every accepted WebSocket socket.

    Measured effect: WebSocket round-trip ~44ms -> ~2ms (loopback).
    Critical for the 50-100ms voice-RAG latency budget.
    """
    try:
        from uvicorn.protocols.websockets import websockets_sansio_impl

        _proto_cls = websockets_sansio_impl.WebSocketsSansIOProtocol
        _orig = _proto_cls.connection_made

        def connection_made(self, transport):
            _orig(self, transport)
            try:
                sock = transport.get_extra_info("socket")
                if sock is not None:
                    sock.setsockopt(_socket.IPPROTO_TCP, _socket.TCP_NODELAY, 1)
            except Exception:
                pass  # non-TCP transports — ignore

        _proto_cls.connection_made = connection_made
        print("[vayu] TCP_NODELAY enabled (websockets-sansio)")
    except Exception as exc:
        # Fall back to the legacy websockets implementation
        try:
            from uvicorn.protocols.websockets import websockets_impl

            _proto_cls = websockets_impl.WebSocketProtocol
            _orig = _proto_cls.connection_made

            def connection_made(self, transport):
                _orig(self, transport)
                try:
                    sock = transport.get_extra_info("socket")
                    if sock is not None:
                        sock.setsockopt(_socket.IPPROTO_TCP, _socket.TCP_NODELAY, 1)
                except Exception:
                    pass

            _proto_cls.connection_made = connection_made
            print("[vayu] TCP_NODELAY enabled (legacy websockets)")
        except Exception as exc2:
            print(f"[vayu] nodelay patch skipped: {exc} / {exc2}")


_enable_tcp_nodelay_for_websockets()


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
                    await session.handle_endpoint(
                        data.get("text", ""),
                        mime_type=data.get("mime_type", ""),
                    )
    except (WebSocketDisconnect, RuntimeError):
        # RuntimeError: "Cannot call receive once a disconnect message has been received"
        await session.close()
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
    # reload only in dev; set VAYU_ENV=production in deployment
    reload = os.getenv("VAYU_ENV", "dev") != "production"
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=reload)
