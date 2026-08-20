import asyncio
import json
import os
import time
from groq import AsyncGroq
from backend.retrieval.engine import get_engine
from backend.guardrails.rules import Guardrails
from starlette.websockets import WebSocket

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")

groq_client = AsyncGroq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


class VoiceSession:
    def __init__(self, websocket: WebSocket):
        self.ws = websocket
        self.is_generating = False
        self.generate_task = None
        self.engine = get_engine()
        self.transcript_buffer = ""
        self.speculative_context = []

    async def handle_audio_chunk(self, chunk: bytes):
        """Receives audio bytes (future Sarvam STT pipe). For now: barge-in only."""
        if self.is_generating and self.generate_task and not self.generate_task.done():
            self.generate_task.cancel()
            self.is_generating = False
            await self.ws.send_json({"event": "INTERRUPT"})

    async def handle_partial_transcript(self, text: str):
        """Interim transcripts -> live echo + speculative retrieval."""
        self.transcript_buffer = text
        await self.ws.send_json({"event": "PARTIAL_TRANSCRIPT", "text": text})

        if len(text.split()) > 3:
            asyncio.create_task(self._speculative_retrieve(text))

    async def _speculative_retrieve(self, text: str):
        await self.ws.send_json({"event": "STATE", "state": "RETRIEVING"})
        results = await self.engine.search_async(text, top_k=3)
        self.speculative_context = results

    async def handle_endpoint(self, final_text: str):
        """Final transcript -> guardrails + retrieval in parallel -> generation."""
        await self.ws.send_json({"event": "STATE", "state": "PROCESSING"})

        guardrail_task = asyncio.create_task(
            Guardrails.run_parallel_input_guardrail(final_text)
        )
        retrieval_task = asyncio.create_task(
            self.engine.search_async(final_text, top_k=3)
        )

        guardrail_result, final_context = await asyncio.gather(
            guardrail_task, retrieval_task
        )

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

        self.generate_task = asyncio.create_task(
            self._generate_response(final_text, final_context)
        )

    async def _generate_response(self, query: str, context: list):
        """Grounded generation + output grounding validation + timing."""
        self.is_generating = True
        t_start = time.perf_counter()
        await self.ws.send_json({"event": "STATE", "state": "GENERATING"})

        context_str = "\n\n".join(
            [f"[ID:{c['parent_id']}] {c['text']}" for c in context]
        )
        prompt = f"""You are VĀYU, a highly precise AI assistant.
Answer the user's question based strictly on the provided context.
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
            if groq_client:
                response = await groq_client.chat.completions.create(
                    messages=[{"role": "system", "content": prompt}],
                    model="llama3-8b-8192",
                    temperature=0.1,
                    stream=True,
                )
                async for chunk in response:
                    if chunk.choices[0].delta.content:
                        full_answer += chunk.choices[0].delta.content
            else:
                # Offline fallback: answer is constructed verbatim from the
                # retrieved parent passage with an explicit citation, so it is
                # grounded by construction.
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

            # ---- Output grounding validation (hallucination defense) ----
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

            await self.ws.send_json({
                "event": "FINAL_ANSWER",
                "answer": full_answer,
                "sources": sources,
                "grounded": grounded,
                "latency_ms": total_ms,
                "grounding_ms": round(ground_ms, 3),
            })
            await self.ws.send_json({"event": "STATE", "state": "COMPLETE"})

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

    async def close(self):
        """Clean up when the client disconnects."""
        if self.generate_task and not self.generate_task.done():
            self.generate_task.cancel()
        self.is_generating = False
