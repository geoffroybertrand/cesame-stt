/* ============================================================
   CESAME STT — Shared JavaScript utilities
   ============================================================ */

const SPEAKER_COLORS = [
    '#3b82f6', // blue
    '#22c55e', // green
    '#f97316', // orange
    '#a855f7', // purple
    '#ec4899', // pink
    '#14b8a6', // teal
    '#eab308', // yellow
    '#ef4444', // red
];

const SPEAKER_BG_COLORS = [
    'rgba(59,130,246,0.08)',
    'rgba(34,197,94,0.08)',
    'rgba(249,115,22,0.08)',
    'rgba(168,85,247,0.08)',
    'rgba(236,72,153,0.08)',
    'rgba(20,184,166,0.08)',
    'rgba(234,179,8,0.08)',
    'rgba(239,68,68,0.08)',
];

function getSpeakerColor(index) {
    return SPEAKER_COLORS[index % SPEAKER_COLORS.length];
}

function getSpeakerBg(index) {
    return SPEAKER_BG_COLORS[index % SPEAKER_BG_COLORS.length];
}

function formatTime(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

function formatDuration(seconds) {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    if (h > 0) return `${h}h ${String(m).padStart(2, '0')}m`;
    return `${m}m ${String(s).padStart(2, '0')}s`;
}

/**
 * Render diarized transcript in Quill Meeting style
 */
function renderQuillTranscript(container, turns, speakerNames, speakerMap) {
    container.innerHTML = '';

    turns.forEach((turn, i) => {
        const speakerId = turn.speaker;
        const displayName = (speakerNames && speakerNames[speakerId]) || speakerId;
        const speakerIdx = speakerMap ? speakerMap[speakerId] : 0;
        const color = getSpeakerColor(speakerIdx);
        const bg = getSpeakerBg(speakerIdx);
        const initials = displayName.substring(0, 2).toUpperCase();

        const turnEl = document.createElement('div');
        turnEl.className = 'turn';
        turnEl.style.borderLeft = `3px solid ${color}`;
        turnEl.style.background = bg;

        turnEl.innerHTML = `
            <div class="turn-header">
                <div class="turn-avatar" style="background:${color}">${initials}</div>
                <div class="turn-speaker" style="color:${color}">${escapeHtml(displayName)}</div>
                <div class="turn-time">${formatTime(turn.start)} - ${formatTime(turn.end)}</div>
            </div>
            <div class="turn-text">${escapeHtml(turn.text)}</div>
        `;

        container.appendChild(turnEl);
    });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Create AudioWorklet for capturing microphone chunks
 */
async function createAudioCapture(chunkDurationS = 4) {
    const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
            sampleRate: 16000,
            channelCount: 1,
            echoCancellation: true,
            noiseSuppression: true,
        }
    });

    const audioContext = new AudioContext({ sampleRate: 16000 });

    const workletCode = `
        class ChunkProcessor extends AudioWorkletProcessor {
            constructor() {
                super();
                this.buffer = [];
                this.bufferSize = 16000 * ${chunkDurationS};
            }
            process(inputs) {
                const input = inputs[0];
                if (input.length > 0) {
                    const channelData = input[0];
                    for (let i = 0; i < channelData.length; i++) {
                        this.buffer.push(channelData[i]);
                    }
                    if (this.buffer.length >= this.bufferSize) {
                        this.port.postMessage(new Float32Array(this.buffer));
                        this.buffer = [];
                    }
                }
                return true;
            }
        }
        registerProcessor('chunk-processor', ChunkProcessor);
    `;
    const blob = new Blob([workletCode], { type: 'application/javascript' });
    const url = URL.createObjectURL(blob);
    await audioContext.audioWorklet.addModule(url);
    URL.revokeObjectURL(url);

    const workletNode = new AudioWorkletNode(audioContext, 'chunk-processor');
    const source = audioContext.createMediaStreamSource(stream);
    source.connect(workletNode);
    // Don't connect to destination to avoid echo
    // workletNode.connect(audioContext.destination);

    return { stream, audioContext, workletNode, source };
}

function destroyAudioCapture(capture) {
    if (!capture) return;
    if (capture.workletNode) capture.workletNode.disconnect();
    if (capture.source) capture.source.disconnect();
    if (capture.audioContext) capture.audioContext.close();
    if (capture.stream) capture.stream.getTracks().forEach(t => t.stop());
}

/**
 * Create a MediaRecorder for WAV-compatible recording (as webm, converted server-side)
 */
function createMediaRecorder(stream) {
    const options = { mimeType: 'audio/webm;codecs=opus' };
    if (!MediaRecorder.isTypeSupported(options.mimeType)) {
        // Fallback
        return new MediaRecorder(stream);
    }
    return new MediaRecorder(stream, options);
}
