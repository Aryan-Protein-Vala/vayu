import asyncio
import json
import os
import re
import os
import time
from dotenv import load_dotenv

# Load .env FIRST so module-level key reads below (groq, sarvam) see them.
# Path: vayu-backend/.env (two levels up from this file).
load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))

from groq import AsyncGroq
from backend.retrieval.engine import get_engine
from backend.guardrails.rules import Guardrails
from backend.stt.sarvam import get_sarvam
from backend.netstatus import groq_available, sarvam_available
from starlette.websockets import WebSocket

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Short timeout + no SDK-internal retries: fast-fail instead of hanging the
# WebSocket handler when the network is down. Our own retry loop handles retry.
groq_client = (
    AsyncGroq(api_key=GROQ_API_KEY, timeout=5.0, max_retries=0)
    if GROQ_API_KEY
    else None
)


class VoiceSession:
    """Orchestrates the full voice pipeline as a structured harness:

    audio bytes -> (barge-in check) -> buffer
    FINAL       -> Sarvam STT (fallback: browser transcript)
               -> guardrails + speculative retrieval (parallel)
               -> grounded generation (Groq, fallback: context answer)
               -> output grounding validation
               -> Sarvam TTS (fallback: browser SpeechSynthesis)
    """

    def __init__(self, websocket: WebSocket):
        self.ws = websocket
        self.is_generating = False
        self.generate_task = None
        self.engine = get_engine()
        self.sarvam = get_sarvam()
        self.transcript_buffer = ""
        self.speculative_context = []
        self.audio_buffer = bytearray()
        self.audio_mime = "audio/webm"

    # ------------------------------------------------------------ audio path
    async def handle_audio_chunk(self, chunk: bytes):
        """Buffer mic audio for Sarvam STT + detect barge-in."""
        # Barge-in: new audio while generating -> interrupt
        if self.is_generating and self.generate_task and not self.generate_task.done():
            self.generate_task.cancel()
            self.is_generating = False
            await self.ws.send_json({"event": "INTERRUPT"})

        # Buffer for STT (cap ~10s @48kHz stereo 16-bit ≈ 1.9 MB, we use ~4MB cap)
        if len(self.audio_buffer) < 4_000_000:
            self.audio_buffer.extend(chunk)

    # -------------------------------------------------------- partial path
    async def handle_partial_transcript(self, text: str):
        """Browser STT partials -> live echo + speculative retrieval."""
        self.transcript_buffer = text
        await self.ws.send_json({"event": "PARTIAL_TRANSCRIPT", "text": text})

        if len(text.split()) > 3:
            asyncio.create_task(self._speculative_retrieve(text))

    async def _speculative_retrieve(self, text: str):
        await self.ws.send_json({"event": "STATE", "state": "RETRIEVING"})
        results = await self.engine.search_async(text, top_k=3)
        self.speculative_context = results

    # ---------------------------------------------------------- final path
    async def handle_endpoint(self, final_text: str, mime_type: str = ""):
        """Speech stopped. Authoritative transcript via Sarvam STT (when audio
        was captured), then run the retrieval + generation harness."""
        await self.ws.send_json({"event": "STATE", "state": "PROCESSING"})

        query = (final_text or "").strip()

        # 1. Sarvam STT on buffered audio (requirement #1: Sarvam STT).
        #    Circuit-breaker: only call Sarvam when reachable; otherwise fall
        #    back to the browser transcript immediately (no multi-second stall).
        if self.audio_buffer:
            if mime_type:
                self.audio_mime = mime_type
            if await sarvam_available():
                stt_text = await self.sarvam.transcribe(bytes(self.audio_buffer), self.audio_mime)
                self.audio_buffer.clear()
                if stt_text:
                    query = stt_text
                    await self.ws.send_json({"event": "STT_RESULT", "text": query, "engine": "sarvam"})
                else:
                    await self.ws.send_json({"event": "STT_RESULT", "text": query, "engine": "browser"})
            else:
                self.audio_buffer.clear()
                await self.ws.send_json({"event": "STT_RESULT", "text": query, "engine": "browser"})

        if not query:
            query = self.transcript_buffer.strip() or "What is the main topic in the retrieved context?"

        # 2. Guardrails + retrieval in parallel
        guardrail_task = asyncio.create_task(
            Guardrails.run_parallel_input_guardrail(query)
        )
        retrieval_task = asyncio.create_task(self.engine.search_async(query, top_k=3))
        guardrail_result, final_context = await asyncio.gather(guardrail_task, retrieval_task)

        if not guardrail_result["safe"]:
            await self.ws.send_json({
                "event": "FINAL_ANSWER",
                "answer": guardrail_result["reason"],
                "sources": [],
                "grounded": False,
                "latency_ms": 0.0,
            })
            await self.ws.send_json({"event": "STATE", "state": "COMPLETE"})
            return

        # 3. Grounded generation (Groq with fallback)
        self.generate_task = asyncio.create_task(
            self._generate_response(query, final_context)
        )

    # ------------------------------------------------------ generation path
    async def _generate_response(self, query: str, context: list):
        """Grounded generation + output grounding validation + Sarvam TTS."""
        self.is_generating = True
        t_start = time.perf_counter()
        await self.ws.send_json({"event": "STATE", "state": "GENERATING"})

        context_str = "\n\n".join(
            [f"[ID:{c['parent_id']}] {c['text']}" for c in context]
        )
        prompt = f"""You are VĀYU, a highly precise AI assistant.
Answer the user's question based strictly on the provided context.
If the user says hello or greets you, greet them back warmly and ask how you can help.
If the answer is not in the context, say so.
When you use information from a passage, cite it exactly like [ID: <parent_id>]
(e.g. [ID: doc_001]). Never invent citations.

Context:
{context_str}

Question:
{query}
"""
        full_answer = ""
        try:
            # Groq generation with a single retry (harness error recovery).
            # Circuit-breaker: skip straight to fallback when unreachable.
            if groq_client and await groq_available():
                for attempt in range(2):
                    try:
                        response = await groq_client.chat.completions.create(
                            messages=[{"role": "user", "content": prompt}],
                            model="openai/gpt-oss-20b",
                            temperature=0.1,
                            stream=True,
                        )
                        async for chunk in response:
                            if chunk.choices[0].delta.content:
                                full_answer += chunk.choices[0].delta.content
                        break
                    except Exception as exc:
                        print(f"[groq] attempt {attempt + 1} failed: {exc}")
                        if attempt == 0:
                            await asyncio.sleep(0.2)
                        else:
                            full_answer = ""
            else:
                full_answer = ""

            if not full_answer.strip():
                # Offline fallback: answer verbatim from retrieved passage,
                # grounded by construction with an explicit citation.
                if context:
                    best = context[0]
                    full_answer = (
                        f"Based on [ID:{best['parent_id']}] the retrieved "
                        f"passage says: {best['text'][:300]}..."
                    )
                else:
                    full_answer = (
                        "I couldn't find relevant information in the retrieved context."
                    )

            # Output grounding validation (hallucination defense)
            t_ground = time.perf_counter_ns()
            retrieved_ids = {c["parent_id"] for c in context}
            grounded = Guardrails.check_grounding(full_answer, retrieved_ids)
            ground_ms = (time.perf_counter_ns() - t_ground) / 1e6

            total_ms = round((time.perf_counter() - t_start) * 1000, 1)

            sources = []
            for c in context:
                preview = c.get("text", "")[:100] + "..."
                sources.append([
                    f"Passage {c['parent_id']}",
                    str(round(c.get("score", 0), 2)),
                    preview,
                ])

            clean_answer = re.sub(r'\[ID:[^\]]+\]\s*', '', full_answer).strip()

            # Send the text answer immediately (low latency UX)...
            await self.ws.send_json({
                "event": "FINAL_ANSWER",
                "answer": clean_answer,
                "sources": sources,
                "grounded": grounded,
                "latency_ms": total_ms,
                "grounding_ms": round(ground_ms, 3),
            })
            await self.ws.send_json({"event": "STATE", "state": "COMPLETE"})

            # ...then pipe the pretty voice (Sarvam TTS) — non-blocking.
            asyncio.create_task(self._send_tts(clean_answer))

        except asyncio.CancelledError:
            print("Generation cancelled due to barge-in.")
        except Exception as e:
            print(f"Generation error: {e}")
            try:
                await self.ws.send_json({
                    "event": "FINAL_ANSWER",
                    "answer": "Sorry, I encountered an error while generating the response.",
                    "sources": [],
                    "grounded": False,
                    "latency_ms": round((time.perf_counter() - t_start) * 1000, 1),
                })
                await self.ws.send_json({"event": "STATE", "state": "COMPLETE"})
            except Exception:
                pass
        finally:
            self.is_generating = False

    async def _send_tts(self, text: str):
        """Sarvam TTS -> wav audio over WebSocket. Frontend falls back to
        browser SpeechSynthesis if this event never arrives."""
        try:
            if await sarvam_available():
                result = await self.sarvam.synthesize(text)
                if result.get("audio"):
                    await self.ws.send_json({
                        "event": "TTS_AUDIO",
                        "audio": result["audio"],
                        "format": result.get("format", "wav"),
                        "engine": "sarvam",
                    })
                    return
            # Unreachable / no audio -> browser voice
            await self.ws.send_json({"event": "TTS_AUDIO", "engine": "browser"})
        except Exception as exc:
            print(f"[sarvam] TTS send failed: {exc}")
            try:
                await self.ws.send_json({"event": "TTS_AUDIO", "engine": "browser"})
            except Exception:
                pass

    async def close(self):
        """Clean up when the client disconnects."""
        if self.generate_task and not self.generate_task.done():
            self.generate_task.cancel()
        self.is_generating = False
