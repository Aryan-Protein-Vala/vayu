"""VAYU API key & connectivity diagnostic.

Run this on the machine where you'll demo (or the deployed server):

    cd vayu-backend
    python scripts/check_api_keys.py

It reports, without printing secrets:
  1. whether .env exists and the three keys are present (masked)
  2. whether api.groq.com and api.sarvam.ai are reachable from THIS machine
  3. whether the Groq key is VALID (real API call to /models)
  4. whether the Sarvam key is VALID (real API call, TTS with a tiny payload)

PASS on every line = the app will run in LIVE mode (Sarvam voices + Groq LLM).
Any FAIL = fix that line, then restart the backend (python -m backend.main).
"""
import os
import sys
import json
import urllib.request
import urllib.error

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ENV_PATH = os.path.join(os.path.dirname(__file__), "../.env")


def mask(key: str) -> str:
    if not key:
        return "(missing)"
    return f"{key[:6]}...{key[-4:]} (len={len(key)})"


def load_env_keys():
    keys = {"SARVAM_API_KEY": "", "GROQ_API_KEY": "", "HF_TOKEN": ""}
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                keys[k.strip()] = v.strip().strip('"').strip("'")
    return keys


def reachable(url: str, timeout: int = 6) -> bool:
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status < 500  # any 4xx/3xx/2xx = reachable
    except urllib.error.HTTPError as e:
        return e.code < 500
    except Exception:
        return False


def groq_key_valid(key: str, timeout: int = 8) -> bool:
    try:
        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {key}"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except urllib.error.HTTPError as e:
        return e.code in (200, 404)  # 404 = auth OK, endpoint path differs
    except Exception:
        return False


def sarvam_key_valid(key: str, timeout: int = 10) -> bool:
    """Real tiny TTS call — the definitive test for the Sarvam key."""
    import base64
    try:
        payload = json.dumps({
            "target_language_code": "en-IN",
            "speaker": "meera",
            "pitch": 0,
            "pace": 1.0,
            "loudness": 1.0,
            "speech_sample_rate": 22050,
            "audio_format": "wav",
            "model": "bulbul:v2",
            "text": "Testing one two three",
        }).encode()
        req = urllib.request.Request(
            "https://api.sarvam.ai/text-to-speech",
            data=payload,
            headers={
                "api-subscription-key": key,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
            return bool(body.get("audio"))
    except urllib.error.HTTPError as e:
        print(f"    (HTTP {e.code}: {e.read()[:120]})")
        return False
    except Exception:
        return False


def main():
    print("=" * 62)
    print("  VAYU — API key & connectivity diagnostic")
    print("=" * 62)

    keys = load_env_keys()
    env_exists = os.path.exists(ENV_PATH)
    print(f"\n[1] .env file: {'FOUND at vayu-backend/.env' if env_exists else '❌ MISSING — create it'}")
    for k in keys:
        print(f"    {k:<18} {mask(keys[k])}")

    print("\n[2] Network reachability (from THIS machine):")
    g = reachable("https://api.groq.com")
    s = reachable("https://api.sarvam.ai")
    print(f"    {'✅' if g else '❌'} api.groq.com   reachable={g}")
    print(f"    {'✅' if s else '❌'} api.sarvam.ai  reachable={s}")
    if not g or not s:
        print("    ⚠️  If unreachable: this machine has no outbound HTTPS to these")
        print("        hosts (VPN? firewall? offline?). The app runs in FALLBACK mode.")
        print("        At HH Goa / any normal internet connection it goes LIVE.")

    print("\n[3] Key validity:")
    if keys["GROQ_API_KEY"]:
        gv = groq_key_valid(keys["GROQ_API_KEY"])
        print(f"    {'✅' if gv else '❌'} GROQ key valid={gv}")
    else:
        print("    ❌ GROQ key missing — add it to .env")

    if keys["SARVAM_API_KEY"]:
        sv = sarvam_key_valid(keys["SARVAM_API_KEY"])
        print(f"    {'✅' if sv else '❌'} SARVAM key valid (real TTS call)={sv}")
        print("        TIP: if key is invalid, get a fresh one at")
        print("        https://dashboard.sarvam.ai/api-keys (they rotate often)")
    else:
        print("    ❌ SARVAM key missing — add it to .env")

    print("\n" + "=" * 62)
    if all([env_exists, keys["GROQ_API_KEY"], keys["SARVAM_API_KEY"], g, s]):
        print("  ✅ Everything present + reachable → app runs in LIVE mode")
        print("     (Sarvam STT/TTS + Groq Llama-3). Restart backend to apply.")
    else:
        print("  ⚠️  Something is missing/unreachable → app runs in FALLBACK mode")
        print("     (browser STT/TTS + grounded context answers).")
    print("=" * 62)


if __name__ == "__main__":
    main()
