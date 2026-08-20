"""VAYU Real-World Live Latency Benchmark

Measures REAL live latency for:
1. Sarvam AI Speech-to-Text (STT) via https://api.sarvam.ai/speech-to-text
2. Local Guardrail Input Validation (compiled regex)
3. Local FAISS Vector Retrieval (TF-IDF + Cosine Search + Parent Resolution)
4. Groq LLM Generation Time-To-First-Token (TTFT) & Full Response Latency via https://api.groq.com
5. Output Grounding Validation
6. Sarvam AI Text-to-Speech (TTS) via https://api.sarvam.ai/text-to-speech

Calculates P50 / P70 / P100 latency metrics across live queries.
"""
import asyncio
import io
import json
import os
import sys
import time
import wave
import httpx
import numpy as np
from dotenv import load_dotenv

# Ensure root vayu-backend path is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))

from backend.retrieval.engine import get_engine
from backend.guardrails.rules import Guardrails
from groq import AsyncGroq

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

groq_client = AsyncGroq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


def generate_sample_wav_audio() -> bytes:
    """Generate a clean 1-second 16kHz mono WAV sample for STT timing."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        # 16000 frames of low amplitude noise
        wf.writeframes(b"\x05\x00" * 16000)
    return buf.getvalue()


async def measure_sarvam_stt(http_client: httpx.AsyncClient, audio_bytes: bytes) -> float:
    """Measure real Sarvam STT REST API latency."""
    if not SARVAM_API_KEY:
        return 0.0
    t0 = time.perf_counter_ns()
    try:
        files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
        data = {"model": "saaras:v3"}
        headers = {"api-subscription-key": SARVAM_API_KEY}
        r = await http_client.post("https://api.sarvam.ai/speech-to-text", headers=headers, data=data, files=files, timeout=15.0)
        t1 = time.perf_counter_ns()
        return (t1 - t0) / 1e6
    except Exception as e:
        print(f"STT Error: {e}")
        return 0.0


async def measure_sarvam_tts(http_client: httpx.AsyncClient, text: str) -> float:
    """Measure real Sarvam TTS REST API latency."""
    if not SARVAM_API_KEY or not text:
        return 0.0
    t0 = time.perf_counter_ns()
    try:
        headers = {"api-subscription-key": SARVAM_API_KEY}
        payload = {
            "inputs": [text[:100]],
            "target_language_code": "en-IN",
            "speaker": "anushka",
            "model": "bulbul:v2",
            "speech_sample_rate": 22050,
            "audio_format": "wav"
        }
        r = await http_client.post("https://api.sarvam.ai/text-to-speech", headers=headers, json=payload, timeout=15.0)
        t1 = time.perf_counter_ns()
        return (t1 - t0) / 1e6
    except Exception as e:
        print(f"TTS Error: {e}")
        return 0.0


async def measure_groq_llm(query: str, contexts: list) -> tuple[float, float, str]:
    """Measure real Groq LLM Time-To-First-Token (TTFT) & Total Generation Latency."""
    if not groq_client:
        return 0.0, 0.0, ""
    
    context_str = "\n\n".join([f"[ID:{c['parent_id']}] {c['text']}" for c in contexts])
    prompt = f"""You are VĀYU, a voice assistant. Answer based on context. Cite as [ID: parent_id].
Context:
{context_str}

Question:
{query}"""
    
    t0 = time.perf_counter_ns()
    ttft_ms = 0.0
    full_answer = ""
    try:
        response = await groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            model="openai/gpt-oss-20b",
            temperature=0.1,
            stream=True,
        )
        async for chunk in response:
            content = chunk.choices[0].delta.content or ""
            if content:
                if ttft_ms == 0.0:
                    t1 = time.perf_counter_ns()
                    ttft_ms = (t1 - t0) / 1e6
                full_answer += content
        t_end = time.perf_counter_ns()
        total_ms = (t_end - t0) / 1e6
        return ttft_ms, total_ms, full_answer
    except Exception as e:
        print(f"Groq Error: {e}")
        return 0.0, 0.0, ""


def print_table(results: dict):
    print("=" * 82)
    print(f"{'Pipeline Stage / API Call':<36} | {'P50 (ms)':<9} | {'P70 (ms)':<9} | {'P100 (ms)':<9}")
    print("=" * 82)
    for stage, times in results.items():
        if not times:
            continue
        p50 = np.percentile(times, 50)
        p70 = np.percentile(times, 70)
        p100 = np.percentile(times, 100)
        print(f"{stage:<36} | {p50:<9.3f} | {p70:<9.3f} | {p100:<9.3f}")
    print("=" * 82)


async def main():
    print("==========================================================================")
    print("  VĀYU — REAL LIVE VOICE RAG BENCHMARK (Live Sarvam AI + Live Groq API)")
    print("==========================================================================\n")

    engine = get_engine()
    sample_wav = generate_sample_wav_audio()

    queries = [
        "What happened at Super Bowl 50?",
        "Tell me about the Golden Gate Bridge",
        "What is machine learning?",
        "Who won Super Bowl 50?",
        "Where is the Eiffel Tower located?",
    ]

    metrics = {
        "1. Sarvam STT (Live Cloud API)": [],
        "2. Guardrail Validation (Local)": [],
        "3. TF-IDF Embedding (Local)": [],
        "4. FAISS Vector Search (Local)": [],
        "5. Parent Chunk Resolution (Local)": [],
        "6. Groq LLM TTFT (Live Cloud API)": [],
        "7. Groq LLM Total Gen (Live Cloud API)": [],
        "8. Grounding Validator (Local)": [],
        "9. Sarvam TTS (Live Cloud API)": [],
        "TOTAL RAG TTFT (Local RAG + Groq TTFT)": [],
        "TOTAL END-TO-END VOICE PIPELINE": [],
    }

    async with httpx.AsyncClient() as http_client:
        print("Warming up connections to Sarvam AI and Groq...")
        _ = await measure_sarvam_stt(http_client, sample_wav)
        _ = await measure_groq_llm("test", [])
        print("Warmup complete. Running 5 live iterations...\n")

        for i, q in enumerate(queries, 1):
            print(f"[{i}/{len(queries)}] Processing query: '{q}' ...")
            t_total_start = time.perf_counter_ns()

            # Stage 1: Sarvam STT
            stt_ms = await measure_sarvam_stt(http_client, sample_wav)
            metrics["1. Sarvam STT (Live Cloud API)"].append(stt_ms)

            # Stage 2: Guardrail Check
            t_g0 = time.perf_counter_ns()
            g_res = Guardrails.check_input(q)
            t_g1 = time.perf_counter_ns()
            metrics["2. Guardrail Validation (Local)"].append((t_g1 - t_g0) / 1e6)

            # Stage 3, 4, 5: Retrieval Engine
            results, timing = await engine.search_timed_async(q, top_k=3)
            metrics["3. TF-IDF Embedding (Local)"].append(timing["embed_ms"])
            metrics["4. FAISS Vector Search (Local)"].append(timing["faiss_ms"])
            metrics["5. Parent Chunk Resolution (Local)"].append(timing["parent_ms"])

            # Stage 6 & 7: Groq LLM Generation
            ttft_ms, groq_total_ms, answer = await measure_groq_llm(q, results)
            metrics["6. Groq LLM TTFT (Live Cloud API)"].append(ttft_ms)
            metrics["7. Groq LLM Total Gen (Live Cloud API)"].append(groq_total_ms)

            # Stage 8: Grounding Check
            t_gr0 = time.perf_counter_ns()
            if results and answer:
                parent_ids = {r["parent_id"] for r in results}
                _ = Guardrails.check_grounding(answer, parent_ids)
            t_gr1 = time.perf_counter_ns()
            metrics["8. Grounding Validator (Local)"].append((t_gr1 - t_gr0) / 1e6)

            # Stage 9: Sarvam TTS
            tts_ms = await measure_sarvam_tts(http_client, answer or "Sample response")
            metrics["9. Sarvam TTS (Live Cloud API)"].append(tts_ms)

            t_total_end = time.perf_counter_ns()

            local_rag_ms = metrics["2. Guardrail Validation (Local)"][-1] + timing["embed_ms"] + timing["faiss_ms"] + timing["parent_ms"]
            rag_ttft = local_rag_ms + ttft_ms
            metrics["TOTAL RAG TTFT (Local RAG + Groq TTFT)"].append(rag_ttft)
            metrics["TOTAL END-TO-END VOICE PIPELINE"].append((t_total_end - t_total_start) / 1e6)

            print(f"    STT: {stt_ms:.1f}ms | Local RAG: {local_rag_ms:.2f}ms | Groq TTFT: {ttft_ms:.1f}ms | Groq Total: {groq_total_ms:.1f}ms | TTS: {tts_ms:.1f}ms")

    print("\n" + "=" * 82)
    print("  LIVE REAL-WORLD BENCHMARK RESULTS")
    print("=" * 82)
    print_table(metrics)

    # Save real metrics to real_live_benchmark_results.json
    out_path = os.path.join(os.path.dirname(__file__), "../../real_live_benchmark_results.json")
    summary = {}
    for stage, times in metrics.items():
        if times:
            summary[stage] = {
                "P50": float(f"{np.percentile(times, 50):.3f}"),
                "P70": float(f"{np.percentile(times, 70):.3f}"),
                "P100": float(f"{np.percentile(times, 100):.3f}"),
            }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"\nReal live benchmark results saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
