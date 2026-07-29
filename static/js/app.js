/* Storm-Voice Application Logic & Real-Time Audio Pipeline */

let socket = null;
let audioCtx = null;
let micStream = null;
let scriptProcessor = null;
let dummyGainNode = null;
let isMicActive = false;

// Audio Playback Queue
let audioQueue = [];
let isPlayingAudio = false;
let currentAudioElement = null;

// App State
let currentBotEntry = null;
let currentBotText = "";
let vadRmsHistory = new Array(60).fill(0);
let botState = "READY"; // READY, LISTENING, THINKING, TALKING

// Helper DOM Element Getter
const getEl = (id) => document.getElementById(id);

// Downsample Float32 buffer to 16kHz PCM for Whisper STT
function resampleTo16k(inputBuffer, fromSampleRate) {
  if (fromSampleRate === 16000) return inputBuffer;
  const ratio = fromSampleRate / 16000;
  const newLength = Math.floor(inputBuffer.length / ratio);
  const result = new Float32Array(newLength);
  for (let i = 0; i < newLength; i++) {
    const origIndex = Math.floor(i * ratio);
    result[i] = inputBuffer[origIndex];
  }
  return result;
}

// Initialize on DOM Ready
document.addEventListener("DOMContentLoaded", () => {
  initSystemStatus();
  loadPersonalities();
  loadVoiceProfiles();
  initWebSocket();
  startOrbAnimation();
});

// Telemetry & Status Fetch
async function initSystemStatus() {
  try {
    const res = await fetch("/api/status");
    const status = await res.json();
    
    if (getEl("val-storage")) getEl("val-storage").innerText = status.isolation_status || "D:\\ Drive Locked";
    if (getEl("val-lmstudio")) getEl("val-lmstudio").innerText = status.llm_backend?.online ? "Connected" : "Offline / Local";
    
    const modelName = (status.llm_backend?.available_models && status.llm_backend.available_models.length > 0) 
      ? status.llm_backend.available_models[0] 
      : "Gemma 4 E2B";
    if (getEl("val-llm")) getEl("val-llm").innerText = modelName;
  } catch (err) {
    console.warn("Storm Telemetry Notice:", err);
  }
}

// Load Demo Personalities
async function loadPersonalities() {
  try {
    const res = await fetch("/api/personalities");
    const data = await res.json();
    const container = getEl("personality-list");
    if (!container) return;
    
    container.innerHTML = "";

    if (data.personalities) {
      for (const [key, item] of Object.entries(data.personalities)) {
        const card = document.createElement("div");
        card.className = `persona-card ${key === data.active ? 'selected' : ''}`;
        card.onclick = () => switchPersonality(key);
        card.innerHTML = `
          <div class="persona-title">${item.name}</div>
          <div class="persona-desc">${item.description}</div>
        `;
        container.appendChild(card);
      }
    }
  } catch (err) {
    console.error("Failed to load personalities:", err);
  }
}

async function switchPersonality(key) {
  try {
    await fetch("/api/personalities/select", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ personality: key })
    });
    loadPersonalities();
  } catch (err) {
    console.error("Failed to switch personality:", err);
  }
}

// Load Voice Clone Profiles
async function loadVoiceProfiles() {
  try {
    const res = await fetch("/api/voices");
    const data = await res.json();
    const select = getEl("voice-profile-select");
    if (!select) return;
    
    select.innerHTML = "";
    
    if (data.voices && Array.isArray(data.voices)) {
      data.voices.forEach(v => {
        const opt = document.createElement("option");
        opt.value = v.id;
        opt.innerText = v.name;
        if (v.id === data.active) opt.selected = true;
        select.appendChild(opt);
      });
    }
  } catch (err) {
    console.error("Failed to load voice profiles:", err);
  }
}

async function switchVoiceProfile(voiceId) {
  try {
    await fetch("/api/voices/select", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ voice_id: voiceId })
    });
  } catch (err) {
    console.error("Failed to switch voice profile:", err);
  }
}

// WebSocket Audio Connection
function initWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws/audio`;

  socket = new WebSocket(wsUrl);

  socket.onopen = () => {
    console.log("[Storm WebSocket] Connected to Core Pipeline.");
    setBotState("READY");
  };

  socket.onmessage = (event) => {
    const msg = JSON.parse(event.data);

    if (msg.type === "vad_meter") {
      updateVadMeter(msg.rms);
    } else if (msg.type === "user_transcript") {
      appendUserTranscript(msg.text);
    } else if (msg.type === "bot_thinking") {
      setBotState("THINKING");
    } else if (msg.type === "bot_text_chunk") {
      setBotState("TALKING");
      appendBotChunk(msg.chunk);
    } else if (msg.type === "bot_audio_chunk") {
      enqueueAudio(msg.audio);
    } else if (msg.type === "barge_in_stop") {
      stopCurrentAudio();
      setBotState("LISTENING");
    } else if (msg.type === "bot_finished") {
      setTimeout(() => setBotState("LISTENING"), 1000);
    }
  };

  socket.onclose = () => {
    console.warn("[Storm WebSocket] Connection closed. Reconnecting in 3s...");
    setTimeout(initWebSocket, 3000);
  };
}

// Microphone Control
async function toggleMicrophone() {
  if (isMicActive) {
    stopMicrophone();
  } else {
    await startMicrophone();
  }
}

async function startMicrophone() {
  try {
    // Native hardware sample rate (e.g. 48kHz) to avoid locking Windows WASAPI audio device
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    if (audioCtx.state === "suspended") {
      await audioCtx.resume();
    }
    
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        channelCount: 1
      }
    });
    
    const source = audioCtx.createMediaStreamSource(micStream);
    scriptProcessor = audioCtx.createScriptProcessor(4096, 1, 1);

    // Mute mic speaker feedback using a zero-gain node so mic doesn't play back out of speakers
    dummyGainNode = audioCtx.createGain();
    dummyGainNode.gain.value = 0;

    source.connect(scriptProcessor);
    scriptProcessor.connect(dummyGainNode);
    dummyGainNode.connect(audioCtx.destination);

    scriptProcessor.onaudioprocess = (e) => {
      if (!isMicActive || !socket || socket.readyState !== WebSocket.OPEN) return;

      // Do not send mic frames while Storm-Bot is outputting audio to avoid speaker echo feedback
      if (isPlayingAudio) return;

      const inputBuffer = e.inputBuffer.getChannelData(0);
      const resampled = resampleTo16k(inputBuffer, audioCtx.sampleRate);

      const pcm16 = new Int16Array(resampled.length);
      for (let i = 0; i < resampled.length; i++) {
        let s = Math.max(-1, Math.min(1, resampled[i]));
        pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
      }
      socket.send(pcm16.buffer);
    };

    isMicActive = true;
    const btn = getEl("mic-toggle-btn");
    if (btn) btn.classList.add("active");
    setBotState("LISTENING");
  } catch (err) {
    alert("Microphone Access Required: Please allow browser microphone permission for Storm-Voice.");
  }
}

function stopMicrophone() {
  if (scriptProcessor) scriptProcessor.disconnect();
  if (dummyGainNode) dummyGainNode.disconnect();
  if (micStream) micStream.getTracks().forEach(track => track.stop());
  isMicActive = false;
  const btn = getEl("mic-toggle-btn");
  if (btn) btn.classList.remove("active");
  setBotState("READY");
}

function triggerBargeIn() {
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ action: "interrupt" }));
  }
  stopCurrentAudio();
}

let currentAudioSource = null;

// Global Audio Context Unlock on User Gesture
function unlockAudioContext() {
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  if (audioCtx.state === "suspended") {
    audioCtx.resume().then(() => {
      console.log("[Storm Audio] AudioContext resumed successfully!");
    }).catch(err => console.warn("AudioContext resume notice:", err));
  }
}

document.addEventListener("click", unlockAudioContext, { passive: true });
document.addEventListener("keydown", unlockAudioContext, { passive: true });

// Audio Playback Queue via WebAudio API
function enqueueAudio(base64Data) {
  if (!base64Data) return;
  audioQueue.push(base64Data);
  if (!isPlayingAudio) {
    playNextAudioChunk();
  }
}

async function playNextAudioChunk() {
  if (audioQueue.length === 0) {
    isPlayingAudio = false;
    currentAudioSource = null;
    return;
  }

  isPlayingAudio = true;
  const base64Data = audioQueue.shift();

  try {
    unlockAudioContext();

    const binaryString = window.atob(base64Data);
    const len = binaryString.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }

    const audioBuffer = await audioCtx.decodeAudioData(bytes.buffer);
    const source = audioCtx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(audioCtx.destination);
    
    currentAudioSource = source;

    source.onended = () => {
      playNextAudioChunk();
    };

    source.start(0);
  } catch (err) {
    console.warn("Storm Audio Playback Notice:", err);
    playNextAudioChunk();
  }
}

function stopCurrentAudio() {
  audioQueue = [];
  if (currentAudioSource) {
    try {
      currentAudioSource.stop();
      currentAudioSource.disconnect();
    } catch (e) {}
    currentAudioSource = null;
  }
  isPlayingAudio = false;
}

// UI State Updates
function setBotState(state) {
  botState = state;
  const stateText = getEl("bot-state-text");
  if (stateText) {
    stateText.innerText = state;
    stateText.style.color = state === "TALKING" ? "#00f0ff" : state === "THINKING" ? "#8b5cf6" : "#10b981";
  }
}

function updateVadMeter(rms) {
  vadRmsHistory.push(rms);
  vadRmsHistory.shift();

  const valEl = getEl("vad-meter-val");
  if (valEl) valEl.innerText = `RMS: ${rms.toFixed(4)}`;

  const vadCanvas = getEl("vad-spectrum-canvas");
  if (!vadCanvas) return;
  const vadCtx = vadCanvas.getContext("2d");
  
  const w = vadCanvas.width = vadCanvas.clientWidth;
  const h = vadCanvas.height = vadCanvas.clientHeight;

  vadCtx.clearRect(0, 0, w, h);
  vadCtx.beginPath();
  vadCtx.strokeStyle = "#00f0ff";
  vadCtx.lineWidth = 2;

  const step = w / vadRmsHistory.length;
  for (let i = 0; i < vadRmsHistory.length; i++) {
    const y = h - (vadRmsHistory[i] * h * 8);
    if (i === 0) vadCtx.moveTo(0, y);
    else vadCtx.lineTo(i * step, y);
  }
  vadCtx.stroke();
}

function appendUserTranscript(text) {
  const box = getEl("transcript-box");
  if (!box) return;

  const entry = document.createElement("div");
  entry.className = "transcript-entry user";
  entry.innerHTML = `
    <div class="entry-speaker">YOU <span>${new Date().toLocaleTimeString()}</span></div>
    ${text}
  `;
  box.appendChild(entry);
  box.scrollTop = box.scrollHeight;

  // Prepare next Bot Entry
  currentBotEntry = document.createElement("div");
  currentBotEntry.className = "transcript-entry bot";
  currentBotEntry.innerHTML = `
    <div class="entry-speaker">STORM-BOT <span>${new Date().toLocaleTimeString()}</span></div>
    <span class="bot-text">...</span>
  `;
  box.appendChild(currentBotEntry);
  currentBotText = "";
}

function appendBotChunk(chunk) {
  if (!currentBotEntry) return;
  currentBotText += chunk;
  const span = currentBotEntry.querySelector(".bot-text");
  if (span) span.innerText = currentBotText;
  
  const box = getEl("transcript-box");
  if (box) box.scrollTop = box.scrollHeight;
}

// 3D-Like Storm Core Reactor Canvas Animation
function startOrbAnimation() {
  const orbCanvas = getEl("storm-orb-canvas");
  if (!orbCanvas) return;
  const orbCtx = orbCanvas.getContext("2d");
  let angle = 0;

  function render() {
    const w = orbCanvas.width = orbCanvas.clientWidth;
    const h = orbCanvas.height = orbCanvas.clientHeight;
    const cx = w / 2;
    const cy = h / 2;

    orbCtx.clearRect(0, 0, w, h);

    let pulseScale = 1.0;
    if (botState === "TALKING") pulseScale = 1.15 + Math.sin(angle * 4) * 0.08;
    else if (botState === "THINKING") pulseScale = 1.0 + Math.sin(angle * 6) * 0.04;
    else if (botState === "LISTENING") pulseScale = 1.05 + Math.sin(angle * 2) * 0.03;

    const rad = 75 * pulseScale;
    const grad = orbCtx.createRadialGradient(cx, cy, 10, cx, cy, rad);
    
    if (botState === "TALKING") {
      grad.addColorStop(0, "rgba(0, 240, 255, 0.9)");
      grad.addColorStop(0.5, "rgba(59, 130, 246, 0.6)");
      grad.addColorStop(1, "rgba(139, 92, 246, 0.0)");
    } else if (botState === "THINKING") {
      grad.addColorStop(0, "rgba(139, 92, 246, 0.9)");
      grad.addColorStop(0.5, "rgba(236, 72, 153, 0.5)");
      grad.addColorStop(1, "rgba(0, 0, 0, 0.0)");
    } else {
      grad.addColorStop(0, "rgba(16, 185, 129, 0.8)");
      grad.addColorStop(0.5, "rgba(0, 240, 255, 0.4)");
      grad.addColorStop(1, "rgba(0, 0, 0, 0.0)");
    }

    orbCtx.fillStyle = grad;
    orbCtx.beginPath();
    orbCtx.arc(cx, cy, rad, 0, Math.PI * 2);
    orbCtx.fill();

    orbCtx.save();
    orbCtx.translate(cx, cy);
    orbCtx.rotate(angle);
    
    orbCtx.strokeStyle = "rgba(0, 240, 255, 0.4)";
    orbCtx.lineWidth = 2;
    orbCtx.beginPath();
    orbCtx.ellipse(0, 0, rad * 1.3, rad * 0.6, Math.PI / 4, 0, Math.PI * 2);
    orbCtx.stroke();

    orbCtx.rotate(-angle * 1.8);
    orbCtx.strokeStyle = "rgba(139, 92, 246, 0.4)";
    orbCtx.beginPath();
    orbCtx.ellipse(0, 0, rad * 1.4, rad * 0.5, -Math.PI / 3, 0, Math.PI * 2);
    orbCtx.stroke();

    orbCtx.restore();

    angle += 0.02;
    requestAnimationFrame(render);
  }

  render();
}

// Modal & Export Functions
function openVoiceCloneModal() {
  const modal = getEl("voice-clone-modal");
  if (modal) modal.classList.add("active");
}

function closeVoiceCloneModal() {
  const modal = getEl("voice-clone-modal");
  if (modal) modal.classList.remove("active");
}

async function submitVoiceClone() {
  const nameInput = getEl("clone-name-input");
  const fileInput = getEl("clone-file-input");

  if (!nameInput || !fileInput || !nameInput.value.trim() || fileInput.files.length === 0) {
    alert("Please enter a persona name and select an audio sample file.");
    return;
  }

  const formData = new FormData();
  formData.append("name", nameInput.value.trim());
  formData.append("file", fileInput.files[0]);

  try {
    const res = await fetch("/api/voice-clone", {
      method: "POST",
      body: formData
    });
    const result = await res.json();
    alert(result.message);
    closeVoiceCloneModal();
    loadVoiceProfiles();
  } catch (err) {
    alert("Voice cloning failed: " + err);
  }
}

async function exportLogs(format) {
  try {
    const res = await fetch("/api/export-logs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ format: format })
    });
    const data = await res.json();
    window.open(data.download_url, "_blank");
  } catch (err) {
    alert("Export failed: " + err);
  }
}
