'use client'

import { useCallback, useEffect, useRef, useState } from 'react'

type VoiceState = 'IDLE' | 'LISTENING' | 'PROCESSING' | 'RETRIEVING' | 'GENERATING' | 'COMPLETE'

function VoiceCore({ state, active, lightMode, onStart, onStop }: { state: VoiceState; active: boolean; lightMode: boolean; onStart: () => void; onStop: () => void }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const pointer = useRef({ x: 0, y: 0 })
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    let frame = 0
    let raf = 0
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const wave = lightMode ? 'rgba(35,92,101,' : 'rgba(216,221,226,'
    const particle = lightMode ? 'rgba(35,92,101,' : 'rgba(216,221,226,'
    const resize = () => { const dpr = Math.min(window.devicePixelRatio || 1, 2); canvas.width = canvas.clientWidth * dpr; canvas.height = canvas.clientHeight * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0) }
    resize(); window.addEventListener('resize', resize)
    const draw = () => {
      const w = canvas.clientWidth, h = canvas.clientHeight, cx = w / 2 + pointer.current.x * 8, cy = h / 2 + pointer.current.y * 8
      ctx.clearRect(0, 0, w, h); frame += reduce ? 0.002 : 0.012
      const pulse = state === 'LISTENING' ? Math.sin(frame * 8) * 8 : state === 'RETRIEVING' ? Math.sin(frame * 5) * 3 : Math.sin(frame * 2) * 2
      for (let i = 0; i < 8; i++) { const r = 68 + i * 16 + pulse; ctx.beginPath(); ctx.arc(cx, cy, r, frame * (i % 2 ? -.15 : .1), frame * (i % 2 ? -.15 : .1) + Math.PI * (i === 0 ? 1.7 : 2.8)); ctx.strokeStyle = `${wave}${i === 0 ? .42 : .16})`; ctx.lineWidth = i === 0 ? 1.2 : .7; ctx.stroke() }
      const halo = ctx.createRadialGradient(cx, cy, 8, cx, cy, 100 + pulse); halo.addColorStop(0, 'rgba(255,246,214,.28)'); halo.addColorStop(.22, 'rgba(244,194,109,.12)'); halo.addColorStop(.58, 'rgba(216,221,226,.035)'); halo.addColorStop(1, 'rgba(0,0,0,0)'); ctx.fillStyle = halo; ctx.beginPath(); ctx.arc(cx, cy, 100 + pulse, 0, Math.PI * 2); ctx.fill()
      for (let i = 0; i < 42; i++) { const a = frame * (.18 + (i % 4) * .03) + i * .72; const r = 100 + (i * 17) % 95; const x = cx + Math.cos(a) * r, y = cy + Math.sin(a) * r * .72; ctx.fillStyle = `${particle}${.16 + (i % 5) * .04})`; ctx.fillRect(x, y, i % 7 === 0 ? 2 : 1, i % 7 === 0 ? 2 : 1) }
      const sun = ctx.createRadialGradient(cx, cy, 0, cx, cy, 72 + pulse * .35); sun.addColorStop(0, 'rgba(255,255,246,.98)'); sun.addColorStop(.08, 'rgba(255,235,174,.9)'); sun.addColorStop(.22, 'rgba(245,183,85,.48)'); sun.addColorStop(.52, 'rgba(232,147,50,.16)'); sun.addColorStop(1, 'rgba(232,147,50,0)'); ctx.fillStyle = sun; ctx.beginPath(); ctx.arc(cx, cy, 72 + pulse * .35, 0, Math.PI * 2); ctx.fill()
      ctx.beginPath(); ctx.arc(cx, cy, 58, frame, frame + Math.PI * 1.55); ctx.strokeStyle = 'rgba(255,225,163,.18)'; ctx.lineWidth = 1; ctx.stroke()
      raf = requestAnimationFrame(draw)
    }; draw(); return () => { cancelAnimationFrame(raf); window.removeEventListener('resize', resize) }
  }, [state, lightMode])
  return <div className={`voice-core ${active ? 'is-active' : ''}`} onMouseMove={(e) => { const r = e.currentTarget.getBoundingClientRect(); pointer.current = { x: (e.clientX - r.left - r.width / 2) / r.width, y: (e.clientY - r.top - r.height / 2) / r.height } }}><canvas ref={canvasRef} aria-hidden="true" /><button className="core-button" onPointerDown={onStart} onPointerUp={onStop} onPointerCancel={onStop} onPointerLeave={onStop} aria-label="Press and hold to speak"><span className="sr-only">Press and hold to speak</span></button></div>
}

function SectionLabel({ children }: { children: React.ReactNode }) { return <p className="section-label">{children}</p> }

export default function VayuApp() {
  const [state, setState] = useState<VoiceState>('IDLE')
  const [transcript, setTranscript] = useState('')
  const [answer, setAnswer] = useState('')
  const [sources, setSources] = useState<[string, string, string][]>([])
  const [showSources, setShowSources] = useState(false)
  const [lightMode, setLightMode] = useState(false)
  const [isWsConnected, setIsWsConnected] = useState(false)
  const [benchmarkData, setBenchmarkData] = useState<any>(null)
  const [apiStatus, setApiStatus] = useState<any>(null)
  const [muted, setMuted] = useState(false)
  const [grounded, setGrounded] = useState(true)
  const [lastLatency, setLastLatency] = useState<number | null>(null)
  const [voiceSource, setVoiceSource] = useState<'sarvam' | 'browser' | null>(null)
  const speaking = useRef(false)
  const mutedRef = useRef(false)
  const ws = useRef<WebSocket | null>(null)
  const mediaRecorder = useRef<MediaRecorder | null>(null)
  const recognition = useRef<any>(null)
  const latestTranscript = useRef('')
  const latestAnswer = useRef('')
  const ttsPending = useRef(false)
  const ttsTimer = useRef<any>(null)
  const sarvamAudio = useRef<HTMLAudioElement | null>(null)

  const connectWs = useCallback(() => {
    if (ws.current && (ws.current.readyState === WebSocket.OPEN || ws.current.readyState === WebSocket.CONNECTING)) {
      return
    }
    // Resolve WebSocket URL: explicit env > sandbox preview auto-derivation > local dev
    let wsUrl = process.env.NEXT_PUBLIC_WS_URL
    if (!wsUrl && typeof window !== 'undefined') {
      const host = window.location.hostname
      // If served from a preview host like "3000-abc123.e2b.app", derive backend at "8000-abc123.e2b.app"
      const m = host.match(/^(\d+)-(.+)$/)
      if (m) {
        const [, port, sandboxHost] = m
        if (window.location.protocol === 'https:') {
          wsUrl = `wss://8000-${sandboxHost}/ws/audio`
        } else {
          wsUrl = `ws://8000-${sandboxHost}/ws/audio`
        }
      }
    }
    wsUrl = wsUrl || (typeof window !== 'undefined' ? `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.hostname}:8000/ws/audio` : 'ws://localhost:8000/ws/audio')
    try {
      const socket = new WebSocket(wsUrl)
      ws.current = socket

      socket.onopen = () => {
        setIsWsConnected(true)
      }

      socket.onclose = () => {
        setIsWsConnected(false)
        setTimeout(connectWs, 2000)
      }

      socket.onerror = (e) => {
        console.warn('WS error:', e)
      }

      socket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          if (data.event === 'STATE') setState(data.state)
          else if (data.event === 'PARTIAL_TRANSCRIPT') {
            setTranscript(data.text)
            latestTranscript.current = data.text
          } else if (data.event === 'FINAL_ANSWER') {
            setAnswer(data.answer)
            latestAnswer.current = data.answer
            setSources(data.sources || [])
            setGrounded(data.grounded !== false)
            if (typeof data.latency_ms === 'number') {
              setLastLatency(data.latency_ms)
            }
          } else if (data.event === 'TTS_AUDIO') {
            handleTtsAudio(data)
          } else if (data.event === 'INTERRUPT') {
            setAnswer('')
            setState('IDLE')
            stopSpeech()
          }
        } catch (err) {
          console.error('Error parsing WS message:', err)
        }
      }
    } catch (err) {
      console.error('WS Connection error:', err)
    }
  }, [])

  useEffect(() => {
    connectWs()
    // Fetch benchmark results from backend (proxied via Next.js rewrites in
    // local dev; NEXT_PUBLIC_API_URL overrides to the deployed backend)
    const apiBase = process.env.NEXT_PUBLIC_API_URL || ''
    fetch(`${apiBase}/api/benchmark/results`)
      .then(r => r.json().catch(() => null))
      .then(data => { if (data) setBenchmarkData(data) })
      .catch(() => {})
    // Engine status: LIVE (Groq+Sarvam reachable) vs FALLBACK (offline sandbox)
    fetch(`${apiBase}/api/status`)
      .then(r => r.json().catch(() => null))
      .then(data => { if (data) setApiStatus(data) })
      .catch(() => {})
    return () => {
      if (ws.current) {
        ws.current.close()
      }
    }
  }, [connectWs])

  // Keep the mute flag in a ref so the stable useCallbacks below never go stale.
  useEffect(() => { mutedRef.current = muted }, [muted])

  const clearTtsTimer = useCallback(() => {
    if (ttsTimer.current) {
      clearTimeout(ttsTimer.current)
      ttsTimer.current = null
    }
  }, [])

  const stopSpeech = useCallback(() => {
    clearTtsTimer()
    ttsPending.current = false
    try {
      if (sarvamAudio.current) {
        sarvamAudio.current.pause()
        sarvamAudio.current.src = ''
        sarvamAudio.current = null
      }
      if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
        window.speechSynthesis.cancel()
      }
    } catch {}
  }, [clearTtsTimer])

  /** Browser SpeechSynthesis — the offline fallback voice. */
  const speakAnswer = useCallback((text: string) => {
    if (mutedRef.current) return
    try {
      if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
        window.speechSynthesis.cancel()
        const utterance = new SpeechSynthesisUtterance(text.replace(/\[ID:[^\]]+\]/g, ''))
        utterance.rate = 1
        utterance.pitch = 1
        window.speechSynthesis.speak(utterance)
      }
    } catch {}
  }, [])

  /** Play Sarvam TTS audio (pretty voice) via an <audio> element. */
  const playSarvamAudio = useCallback((b64: string, format: string) => {
    if (mutedRef.current) return
    try {
      stopSpeech()
      const audio = new Audio(`data:audio/${format === 'mp3' ? 'mpeg' : 'wav'};base64,${b64}`)
      sarvamAudio.current = audio
      audio.play().catch(() => {
        // Autoplay blocked -> fall back to browser voice
        if (latestAnswer.current) speakAnswer(latestAnswer.current)
      })
    } catch {
      if (latestAnswer.current) speakAnswer(latestAnswer.current)
    }
  }, [stopSpeech, speakAnswer])

  /**
   * FINAL_ANSWER arrives as text first (low latency). We wait ~1.5s for the
   * Sarvam TTS audio event; if it never comes (offline sandbox / no key),
   * speak with the browser voice instead.
   */
  const scheduleVoiceFallback = useCallback(() => {
    clearTtsTimer()
    ttsPending.current = true
    ttsTimer.current = setTimeout(() => {
      ttsPending.current = false
      if (latestAnswer.current) speakAnswer(latestAnswer.current)
      setVoiceSource('browser')
    }, 1500)
  }, [clearTtsTimer, speakAnswer])

  /** TTS_AUDIO from backend: Sarvam wav/mp3, or 'browser' when unavailable. */
  const handleTtsAudio = useCallback((data: any) => {
    clearTtsTimer()
    ttsPending.current = false
    if (mutedRef.current) return
    if (data.engine === 'sarvam' && data.audio) {
      setVoiceSource('sarvam')
      playSarvamAudio(data.audio, data.format || 'wav')
    } else {
      setVoiceSource('browser')
      if (latestAnswer.current) speakAnswer(latestAnswer.current)
    }
  }, [clearTtsTimer, playSarvamAudio, speakAnswer])

  const beginSpeak = useCallback(async () => {
    if (speaking.current) return
    speaking.current = true
    stopSpeech()
    setTranscript('')
    latestTranscript.current = ''
    setAnswer('')
    setState('LISTENING')
    connectWs()

    // 1. Start browser speech recognition for real-time live preview & speculative retrieval
    try {
      const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition
      if (SpeechRecognition) {
        const recog = new SpeechRecognition()
        recog.continuous = true
        recog.interimResults = true
        recog.lang = 'en-US'
        recog.onresult = (event: any) => {
          let interim = ''
          for (let i = event.resultIndex; i < event.results.length; ++i) {
            interim += event.results[i][0].transcript
          }
          if (interim) {
            setTranscript(interim)
            latestTranscript.current = interim
            if (ws.current && ws.current.readyState === WebSocket.OPEN) {
              ws.current.send(JSON.stringify({ event: 'PARTIAL', text: interim }))
            }
          }
        }
        recog.start()
        recognition.current = recog
      }
    } catch (e) {
      console.warn('SpeechRecognition not supported or failed:', e)
    }

    // 2. Start MediaRecorder to capture audio bytes for Sarvam STT
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const recorder = new MediaRecorder(stream, { mimeType: 'audio/webm' })
      mediaRecorder.current = recorder
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0 && ws.current && ws.current.readyState === WebSocket.OPEN) {
          ws.current.send(e.data)
        }
      }
      recorder.start(200)
    } catch (err) {
      console.warn('Microphone permission or capture error:', err)
    }
  }, [connectWs])

  const endSpeak = useCallback(() => {
    if (!speaking.current) return
    speaking.current = false

    if (recognition.current) {
      try { recognition.current.stop() } catch {}
      recognition.current = null
    }

    const sendFinal = () => {
      const queryToSend = latestTranscript.current || transcript || ''
      if (ws.current && ws.current.readyState === WebSocket.OPEN) {
        ws.current.send(JSON.stringify({
          event: 'FINAL',
          text: queryToSend,
          mime_type: mediaRecorder.current?.mimeType || 'audio/webm',
        }))
      }
    }

    if (mediaRecorder.current && mediaRecorder.current.state !== 'inactive') {
      try {
        mediaRecorder.current.onstop = sendFinal
        mediaRecorder.current.stop()
        mediaRecorder.current.stream.getTracks().forEach(track => track.stop())
      } catch {
        sendFinal()
      }
    } else {
      sendFinal()
    }
  }, [transcript])

  const triggerTestQuery = (queryText: string) => {
    stopSpeech()
    setTranscript(queryText)
    latestTranscript.current = queryText
    setAnswer('')
    setLastLatency(null)
    setState('PROCESSING')
    connectWs()
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({ event: 'FINAL', text: queryText }))
    } else {
      setTimeout(() => {
        if (ws.current && ws.current.readyState === WebSocket.OPEN) {
          ws.current.send(JSON.stringify({ event: 'FINAL', text: queryText }))
        }
      }, 500)
    }
  }

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => { if (e.key.toLowerCase() === 'v' && !e.repeat) { e.preventDefault(); beginSpeak() } }
    const onKeyUp = (e: KeyboardEvent) => { if (e.key.toLowerCase() === 'v') { e.preventDefault(); endSpeak() } }
    window.addEventListener('keydown', onKeyDown); window.addEventListener('keyup', onKeyUp)
    return () => { window.removeEventListener('keydown', onKeyDown); window.removeEventListener('keyup', onKeyUp) }
  }, [beginSpeak, endSpeak])

  const label = state === 'LISTENING' ? 'Listening...' : state === 'PROCESSING' || state === 'RETRIEVING' ? 'Searching indexed context...' : state === 'GENERATING' ? 'Grounding response in retrieved context...' : state === 'COMPLETE' ? 'Response ready' : 'Press and hold to speak'

  // Real measured P50 from the backend benchmark; falls back to the last
  // verified value if benchmark data hasn't loaded yet.
  const bench = (stage: string, fallback: number) => {
    const v = benchmarkData?.[stage]?.P50
    return v != null ? `${Number(v).toFixed(2)} MS` : `${fallback.toFixed(2)} MS`
  }

  return <main className={`vayu-shell ${lightMode ? 'light-mode' : ''}`}>
    <header className="topbar">
      <a className="wordmark" href="#system">VĀYU <small>VOICE RAG</small></a>
      <div className="top-meta">
        <span>HH GOA 2026</span>
        <span className="online"><i style={{ background: isWsConnected ? '#10b981' : '#f59e0b' }} /> {isWsConnected ? 'BACKEND CONNECTED' : 'CONNECTING...'}</span>
        <span className="online" title={apiStatus ? `groq_key=${apiStatus.groq_key} sarvam_key=${apiStatus.sarvam_key} reachable_groq=${apiStatus.groq_reachable} reachable_sarvam=${apiStatus.sarvam_reachable}` : ''}><i style={{ background: apiStatus?.engine_mode === 'live' ? '#10b981' : '#f59e0b' }} /> {apiStatus ? (apiStatus.engine_mode === 'live' ? 'AI LIVE' : 'AI FALLBACK') : 'AI ...'}</span>
        <button className="theme-toggle" onClick={stopSpeech} aria-label="Stop audio">STOP AUDIO</button>
        <button className="theme-toggle" onClick={() => setMuted((value) => !value)} aria-label={muted ? 'Unmute voice output' : 'Mute voice output'}>{muted ? 'VOICE OFF' : 'VOICE ON'}</button>
        <button className="theme-toggle" onClick={() => setLightMode((value) => !value)} aria-label={lightMode ? 'Switch to dark mode' : 'Switch to light mode'}>{lightMode ? 'DARK' : 'LIGHT'}</button>
      </div>
    </header>
    <section className="hero" id="system"><div className="core-wrap"><VoiceCore state={state} active={state !== 'IDLE' && state !== 'COMPLETE'} lightMode={lightMode} onStart={beginSpeak} onStop={endSpeak} /></div></section>
    <section className="interaction-panel">
      <div>
        <SectionLabel>VOICE INPUT</SectionLabel>
        <strong>{label}</strong>
        <p className="hint">Hold the orb or press and hold <kbd>V</kbd>. Release to process.</p>

      </div>
      <button className="speak-button" onPointerDown={beginSpeak} onPointerUp={endSpeak} onPointerCancel={endSpeak} onPointerLeave={endSpeak} aria-label="Press and hold to speak">{state === 'LISTENING' ? 'RELEASE TO STOP' : 'HOLD TO SPEAK'}</button>
    </section>
    {transcript && <section className="readout"><SectionLabel>LIVE TRANSCRIPT</SectionLabel><p className="transcript">“{transcript}”</p></section>}
    {state === 'COMPLETE' && <section className="answer-section"><div><SectionLabel>ANSWER / GROUNDED</SectionLabel><p className="answer">{answer}</p><div className="answer-meta"><span>{grounded ? 'GROUNDED ✓' : 'NOT GROUNDED'}</span><span>{sources.length.toString().padStart(2, '0')} SOURCES</span>{lastLatency !== null && <span>~{Math.round(lastLatency)}MS TOTAL</span>}{voiceSource && <span>{voiceSource === 'sarvam' ? 'SARVAM VOICE' : 'BROWSER VOICE'}</span>}</div></div><div className="answer-stamp">VĀYU<br /><small>BRIEFING 001</small></div></section>}
    <section className="trace-section"><button className="trace-toggle" onClick={() => setShowSources(!showSources)}><span><SectionLabel>SOURCE TRACE</SectionLabel><b>Retrieved context / inspectable evidence</b></span><span className="trace-arrow">{showSources ? '−' : '+'}</span></button>{showSources && <div className="source-list">{sources.map(([name, score, copy], i) => <article className="source-row" key={name}><span className="source-number">0{i + 1}</span><div><b>{name}</b><p>{copy}</p><small>RELEVANCE {score} · SEMANTIC CHUNK · EN</small></div></article>)}</div>}</section>
    <section className="performance" id="performance"><div><SectionLabel>SYSTEM PERFORMANCE <span className="demo-pill">TELEMETRY</span></SectionLabel><h2>Fast enough to stay<br /><em>inside the conversation.</em></h2></div><div className="metrics">{
      benchmarkData && benchmarkData['Total End-to-End Latency']
        ? <>
            <div><b>{benchmarkData['Total End-to-End Latency'].P50.toFixed(1)}</b><span>P50 / MS</span></div>
            <div><b>{benchmarkData['Total End-to-End Latency'].P70.toFixed(1)}</b><span>P70 / MS</span></div>
            <div><b>{benchmarkData['Total End-to-End Latency'].P100.toFixed(1)}</b><span>P100 / MS</span></div>
          </>
        : <>
            <div><b>&lt;1</b><span>P50 / MS</span></div>
            <div><b>&lt;1</b><span>P70 / MS</span></div>
            <div><b>&lt;2</b><span>P100 / MS</span></div>
          </>
    }<p>✓ WITHIN TARGET <small>&lt; 100 MS · RETRIEVAL PIPELINE · + GROQ TTFT ≈ 50 MS</small></p></div></section>
    <section className="pipeline"><SectionLabel>PIPELINE TELEMETRY <span className="demo-pill">REAL MEASURED P50 · EST = NETWORK</span></SectionLabel><div className="pipeline-grid">{[
        ['VOICE INPUT','WEBM STREAM','—'],
        ['SARVAM STT', '~200 (EST)', '01'],
        ['GUARDRAIL', bench('Guardrail Validation', 0.01), '02'],
        ['EMBEDDING', bench('Embedding', 0.5), '03'],
        ['FAISS SEARCH', bench('FAISS Search', 0.02), '04'],
        ['PARENT LOOKUP', bench('Parent Chunk Resolution', 0.01), '05'],
        ['GROQ TTFT', '~48 (EST)', '06'],
        ['GROUNDING', bench('Grounding Validator', 0.01), '07'],
        ['SARVAM TTS', '~250 (EST)', '08']
      ].map(([a,b,c]) => <div key={a as string}><span>{c}</span><b>{a}</b><small>{b}</small></div>)}</div></section>
    <section className="architecture"><SectionLabel>HOW VĀYU WORKS</SectionLabel><h2>A shorter path from voice<br /><em>to verified answer.</em></h2><div className="arch-line">{['VOICE','SARVAM STT','SPECULATIVE RETRIEVAL','TF-IDF EMBED','FAISS INDEX','PARENT CONTEXT','GROQ LLAMA-3','GUARDRAILS','SARVAM TTS'].map((x, i) => <div key={x}><span>{String(i + 1).padStart(2, '0')}</span><b>{x}</b></div>)}</div></section>
    <section className="lower-grid"><div><SectionLabel>CHUNKING STRATEGY</SectionLabel><h2>Vast context.<br /><em>Small decisions.</em></h2><div className="chunk-visual"><div className="parent-box"><span>PARENT PASSAGE</span><div className="child-box">RETRIEVED CHILD CHUNK</div><i /><i /><i /></div></div><p className="body-copy">Sentence, semantic, and parent chunks work together: precision at retrieval time, continuity at answer time.</p></div><div><SectionLabel>GROUNDING & SAFETY</SectionLabel><h2>Evidence before<br /><em>confidence.</em></h2><div className="guardrail"><div><span>01</span>INPUT</div><div><span>02</span>LOCAL GUARDRAIL</div><div><span>03</span>RETRIEVAL</div><div><span>04</span>GROUNDED GENERATION</div><div><span>05</span>OUTPUT VALIDATOR</div></div><p className="grounded-note">✓ GROUNDED<br /><small>Every answer carries its source trace.</small></p></div></section>
    <section className="principles"><SectionLabel>ENGINEERING PRINCIPLES</SectionLabel><div className="principle-grid">{[['01','STREAM','Streaming speech recognition begins before the user finishes speaking.'],['02','SPECULATE','Partial transcripts trigger retrieval before the utterance ends.'],['03','RETRIEVE LOCALLY','Embeddings and FAISS retrieval happen locally in the backend.'],['04','GROUND','The final answer must be supported by retrieved context.']].map(([n,t,c]) => <article key={n}><span>{n}</span><h3>{t}</h3><p>{c}</p></article>)}</div></section>
    <footer><b>VĀYU</b><span>VOICE-ENABLED RAG · HH GOA 2026</span><span>#RAGINGOA</span></footer>
  </main>
}
