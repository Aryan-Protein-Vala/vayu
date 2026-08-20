"""Sarvam AI integration — STT (saaras) + TTS (bulbul).

Task requirement #1 mandates Sarvam (or ElevenLabs) for speech-to-text, and a
nice voice for output comes from Sarvam TTS. This client is a thin, retrying
wrapper around the Sarvam REST API:

  POST /speech-to-text   -> transcript text   (model: saaras:v2)
  POST /text-to-speech   -> base64 audio      (model: bulbul:v2, speakers like meera/arvind)

Design notes:
- Every call is guarded: if no key is set, the endpoint is unreachable, or the
  call fails after retries, it returns None / "" so the orchestrator can fall
  back to browser STT / browser SpeechSynthesis without breaking the demo.
- Retries (3 attempts, exponential backoff) satisfy the "harness" requirement
  of structured error recovery around model calls.
"""
import asyncio
import os
import httpx

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
SARVAM_BASE_URL = os.getenv("SARVAM_BASE_URL", "https://api.sarvam.ai")
STT_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v3")
TTS_MODEL = os.getenv("SARVAM_TTS_MODEL", "bulbul:v2")
TTS_SPEAKER = os.getenv("SARVAM_TTS_SPEAKER", "anushka")
TTS_LANGUAGE = os.getenv("SARVAM_TTS_LANGUAGE", "en-IN")
TTS_SAMPLE_RATE = int(os.getenv("SARVAM_TTS_SAMPLE_RATE", "22050"))
TTS_AUDIO_FORMAT = os.getenv("SARVAM_TTS_AUDIO_FORMAT", "wav")


class SarvamClient:
    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        stt_model: str = "",
        tts_model: str = "",
        tts_speaker: str = "",
        tts_language: str = "",
        max_retries: int = 3,
    ):
        self.api_key = api_key or SARVAM_API_KEY
        self.base_url = base_url or SARVAM_BASE_URL
        self.stt_model = stt_model or STT_MODEL
        self.tts_model = tts_model or TTS_MODEL
        self.tts_speaker = tts_speaker or TTS_SPEAKER
        self.tts_language = tts_language or TTS_LANGUAGE
        self.max_retries = max_retries
        self._http = None

    @property
    def http(self) -> httpx.AsyncClient:
        if self._http is None:
            headers = {"api-subscription-key": self.api_key, "Accept": "application/json"}
            self._http = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=30.0,
            )
        return self._http

    async def close(self):
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    async def _post_with_retry(self, url: str, **kwargs):
        """POST with exponential-backoff retries (harness error recovery)."""
        last_exc = None
        for attempt in range(self.max_retries):
            try:
                resp = await self.http.post(url, **kwargs)
                resp.raise_for_status()
                return resp.json()
            except (httpx.HTTPError, httpx.TimeoutException) as exc:
                last_exc = exc
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(0.3 * (2 ** attempt))
        print(f"[sarvam] request to {url} failed after {self.max_retries} attempts: {last_exc}")
        return None

    # ------------------------------------------------------------------ STT
    async def transcribe(self, audio_bytes: bytes, mime_type: str = "audio/webm") -> str:
        """Sarvam speech-to-text. Returns transcript ('' if unavailable)."""
        if not self.api_key or not audio_bytes:
            return ""
        try:
            ext = "webm"
            if "mp3" in mime_type:
                ext = "mp3"
            elif "wav" in mime_type:
                ext = "wav"
            elif "ogg" in mime_type:
                ext = "ogg"
            payload = await self._post_with_retry(
                "/speech-to-text-translate",
                data={"model": "saaras:v1"},
                files={"file": (f"audio.{ext}", audio_bytes, mime_type or "audio/webm")},
            )
        except Exception as exc:  # never let STT break the pipeline
            print(f"[sarvam] STT exception: {exc}")
            return ""
        if not payload:
            return ""
        transcript = (
            payload.get("transcript")
            or payload.get("text")
            or payload.get("transcription")
            or ""
        )
        return str(transcript).strip()

    # ------------------------------------------------------------------ TTS
    async def synthesize(self, text: str) -> dict:
        """Sarvam text-to-speech. Returns {'audio': <base64>, 'format': 'wav'}
        or {} if unavailable."""
        if not self.api_key or not text:
            return {}
        try:
            payload = await self._post_with_retry(
                "/text-to-speech",
                json={
                    "target_language_code": self.tts_language,
                    "speaker": self.tts_speaker,
                    "pitch": 0,
                    "pace": 1.0,
                    "loudness": 1.0,
                    "speech_sample_rate": TTS_SAMPLE_RATE,
                    "audio_format": TTS_AUDIO_FORMAT,
                    "model": self.tts_model,
                    "inputs": [text],
                },
            )
        except Exception as exc:
            print(f"[sarvam] TTS exception: {exc}")
            return {}
        if not payload:
            return {}
        audio_b64 = payload.get("audios", [""])[0] if payload.get("audios") else ""
        fmt = payload.get("audio_format") or TTS_AUDIO_FORMAT
        return {"audio": audio_b64, "format": fmt} if audio_b64 else {}


_sarvam_client = None


def get_sarvam() -> SarvamClient:
    global _sarvam_client
    if _sarvam_client is None:
        _sarvam_client = SarvamClient()
    return _sarvam_client
