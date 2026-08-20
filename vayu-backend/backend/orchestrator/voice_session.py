import asyncio
import json
import httpx
from groq import AsyncGroq
from backend.retrieval.engine import get_engine
from backend.guardrails.rules import Guardrails
from starlette.websockets import WebSocket

from dotenv import load_dotenv
import os

load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

groq_client = AsyncGroq(api_key=GROQ_API_KEY if GROQ_API_KEY else "dummy")

class VoiceSession:
    def __init__(self, websocket: WebSocket):
        self.ws = websocket
        self.is_generating = False
        self.generate_task = None
        self.engine = get_engine()
        self.transcript_buffer = ""
        self.speculative_context = []
        
    async def handle_audio_chunk(self, chunk: bytes):
        """
        Receives audio chunk, pipes to Sarvam.
        In a real Sarvam streaming setup, we'd have a persistent WS to saaras:v3.
        For this prototype, we simulate the STT partials and barge-in.
        """
        # Barge-in Logic: If we receive new audio while generating, INTERRUPT
        if self.is_generating and self.generate_task and not self.generate_task.done():
            self.generate_task.cancel()
            self.is_generating = False
            await self.ws.send_json({"event": "INTERRUPT"})
            
        # Here we'd send audio to Sarvam.
        # We will simulate receiving a partial transcript for speculative retrieval.
        pass

    async def handle_partial_transcript(self, text: str):
        self.transcript_buffer = text
        await self.ws.send_json({"event": "PARTIAL_TRANSCRIPT", "text": text})
        
        # Trigger speculative retrieval concurrently
        if len(text.split()) > 3:
            # We don't await this directly in the main stream path to avoid blocking
            asyncio.create_task(self._speculative_retrieve(text))

    async def _speculative_retrieve(self, text: str):
        await self.ws.send_json({"event": "STATE", "state": "RETRIEVING"})
        results = await self.engine.search_async(text, top_k=3)
        self.speculative_context = results

    async def handle_endpoint(self, final_text: str):
        """Speech has stopped. Final transcript received."""
        await self.ws.send_json({"event": "STATE", "state": "PROCESSING"})
        
        # 1. Parallel Guardrails & Verification
        guardrail_task = asyncio.create_task(Guardrails.run_parallel_input_guardrail(final_text))
        
        # If we didn't speculatively retrieve, or want to verify, we retrieve again
        retrieval_task = asyncio.create_task(self.engine.search_async(final_text, top_k=3))
        
        guardrail_result, final_context = await asyncio.gather(guardrail_task, retrieval_task)
        
        if not guardrail_result["safe"]:
            await self.ws.send_json({
                "event": "FINAL_ANSWER",
                "answer": guardrail_result["reason"],
                "sources": []
            })
            await self.ws.send_json({"event": "STATE", "state": "COMPLETE"})
            return

        # 2. Generation with Groq
        self.generate_task = asyncio.create_task(self._generate_response(final_text, final_context))

    async def _generate_response(self, query: str, context: list):
        self.is_generating = True
        await self.ws.send_json({"event": "STATE", "state": "GENERATING"})
        
        context_str = "\n\n".join([f"[ID: {c['parent_id']}] {c['text']}" for c in context])
        prompt = f"""
You are VĀYU, a highly precise AI assistant. 
Answer the user's question based strictly on the provided context. 
If the answer is not in the context, say so.
Format citations exactly like [ID: 12345].

Context:
{context_str}

Question:
{query}
"""
        try:
            # Requires strict JSON output schema
            response = await groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": prompt}
                ],
                model="llama3-8b-8192",
                temperature=0.1,
                stream=True,
                # For JSON output, Llama3-8b can sometimes use response_format={"type": "json_object"}
            )
            
            full_answer = ""
            async for chunk in response:
                if chunk.choices[0].delta.content:
                    delta = chunk.choices[0].delta.content
                    full_answer += delta
                    # We can stream tokens to the frontend if desired
            
            # Extract citations for frontend
            sources = []
            for c in context:
                sources.append([f"Passage {c['parent_id']}", str(round(c['score'], 2)), c['text'][:100] + "..."])
                
            await self.ws.send_json({
                "event": "FINAL_ANSWER",
                "answer": full_answer,
                "sources": sources
            })
            await self.ws.send_json({"event": "STATE", "state": "COMPLETE"})
            
        except asyncio.CancelledError:
            print("Generation cancelled due to barge-in.")
        finally:
            self.is_generating = False
