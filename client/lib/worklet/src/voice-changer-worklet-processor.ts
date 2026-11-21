export const RequestType = {
    voice: "voice",
    config: "config",
    start: "start",
    stop: "stop",
    trancateBuffer: "trancateBuffer",
} as const;
export type RequestType = (typeof RequestType)[keyof typeof RequestType];

export const ResponseType = {
    inputData: "inputData",
    inputChunk: "inputChunk",
    start_ok: "start_ok",
    stop_ok: "stop_ok",
    pop_detected: "pop_detected",
} as const;
export type ResponseType = (typeof ResponseType)[keyof typeof ResponseType];

export type VoiceChangerWorkletProcessorRequest = {
    requestType: RequestType;
    voice: Float32Array;
    cushion?: number;
    inputChunkNum?: number;
};

export type VoiceChangerWorkletProcessorResponse = {
    responseType: ResponseType;
    recordData?: Float32Array[];
    inputData?: Float32Array;
    inputChunk?: Int16Array;
    type?: string;
    count?: number;
};

const RENDER_QUANTUM_FRAMES = 128;
const RING_BUFFER_SIZE = 96000;
const CROSSFADE_SAMPLES = 512;
const MAX_ADAPTIVE_CUSHION = 8;
const CATCHUP_RATE = 1.04;
const CATCHUP_TRIGGER_MARGIN = 512;
const CATCHUP_STOP_MARGIN = 128;
const HARD_EXCESS_LIMIT = 16384;

class VoiceChangerWorkletProcessor extends AudioWorkletProcessor {
    private isRecording = false;

    // Input capture accumulation (Int16 chunk-level IPC)
    private inputChunkNum = 1;
    private hasExplicitChunkNum = false;
    private inputAccum = new Int16Array(RENDER_QUANTUM_FRAMES);
    private inputOffset = 0;

    // Preallocated circular ring buffer for jitter-free playback
    private ringBuffer = new Float32Array(RING_BUFFER_SIZE);
    private scratchFadeBuf = new Float32Array(CROSSFADE_SAMPLES);
    private readIdx = 0;
    private writeIdx = 0;
    private availableSamples = 0;
    private readFrac = 0;

    // Playback cushion & continuity state
    private cushion = 2;
    private adaptiveCushion = 0;
    private wasPlaying = false;
    private isCatchingUp = false;
    private needsFadeIn = true;
    private lastPlaySample = 0;

    constructor() {
        super();
        this.port.onmessage = this.handleMessage.bind(this);
    }

    private getTargetSamples(): number {
        const effectiveCushion = Math.min(
            this.cushion + this.adaptiveCushion,
            this.cushion + MAX_ADAPTIVE_CUSHION
        );
        return Math.max(RENDER_QUANTUM_FRAMES, effectiveCushion * RENDER_QUANTUM_FRAMES);
    }

    private getConfiguredChunkSamples(): number {
        return this.hasExplicitChunkNum ? this.inputChunkNum * RENDER_QUANTUM_FRAMES : 0;
    }

    private resetPlaybackState(): void {
        this.readIdx = 0;
        this.writeIdx = 0;
        this.availableSamples = 0;
        this.readFrac = 0;
        this.wasPlaying = false;
        this.isCatchingUp = false;
        this.needsFadeIn = true;
        this.lastPlaySample = 0;
        this.adaptiveCushion = 0;
    }

    private pushVoice(voice: Float32Array): void {
        const len = voice.length;
        if (len === 0) {
            return;
        }

        if (len >= RING_BUFFER_SIZE) {
            const start = len - RING_BUFFER_SIZE;
            this.ringBuffer.set(voice.subarray(start), 0);
            this.readIdx = 0;
            this.writeIdx = 0;
            this.availableSamples = RING_BUFFER_SIZE;
            this.readFrac = 0;
            this.isCatchingUp = false;
        } else {
            const overflow = this.availableSamples + len - RING_BUFFER_SIZE;
            if (overflow > 0) {
                this.readIdx = (this.readIdx + overflow) % RING_BUFFER_SIZE;
                this.availableSamples -= overflow;
                this.readFrac = 0;
            }

            const firstPart = Math.min(len, RING_BUFFER_SIZE - this.writeIdx);
            this.ringBuffer.set(voice.subarray(0, firstPart), this.writeIdx);
            if (firstPart < len) {
                this.ringBuffer.set(voice.subarray(firstPart), 0);
            }
            this.writeIdx = (this.writeIdx + len) % RING_BUFFER_SIZE;
            this.availableSamples += len;
        }

        const targetSamples = this.getTargetSamples();
        const configuredChunkSamples = this.getConfiguredChunkSamples();
        const hardCap = Math.min(
            RING_BUFFER_SIZE - RENDER_QUANTUM_FRAMES,
            targetSamples + configuredChunkSamples + HARD_EXCESS_LIMIT
        );
        if (this.wasPlaying && this.availableSamples > hardCap) {
            this.truncateTo(targetSamples + configuredChunkSamples);
        }
    }

    private truncateTo(targetSamples: number): void {
        const keepSamples = Math.max(RENDER_QUANTUM_FRAMES, Math.floor(targetSamples));
        const dropCount = this.availableSamples - keepSamples;
        if (dropCount <= 0) {
            return;
        }

        const fadeLen = Math.min(CROSSFADE_SAMPLES, keepSamples);
        const oldIdx = this.readIdx;
        for (let i = 0; i < fadeLen; i++) {
            this.scratchFadeBuf[i] = this.ringBuffer[(oldIdx + i) % RING_BUFFER_SIZE];
        }

        const newReadIdx = (oldIdx + dropCount) % RING_BUFFER_SIZE;
        for (let i = 0; i < fadeLen; i++) {
            const alpha = i / fadeLen;
            const dstIdx = (newReadIdx + i) % RING_BUFFER_SIZE;
            const oldSample = this.scratchFadeBuf[i];
            const newSample = this.ringBuffer[dstIdx];
            this.ringBuffer[dstIdx] = oldSample * (1.0 - alpha) + newSample * alpha;
        }

        this.readIdx = newReadIdx;
        this.availableSamples = keepSamples;
        this.readFrac = 0;
        this.isCatchingUp = false;
    }

    private applyInputChunkNum(chunkNum: number): void {
        const normalized = Math.max(1, Math.floor(chunkNum));
        this.hasExplicitChunkNum = true;
        if (normalized !== this.inputChunkNum || this.inputAccum.length !== normalized * RENDER_QUANTUM_FRAMES) {
            this.inputChunkNum = normalized;
            this.inputAccum = new Int16Array(normalized * RENDER_QUANTUM_FRAMES);
            this.inputOffset = 0;
        }
    }

    private handleMessage(event: MessageEvent): void {
        const data = event.data as VoiceChangerWorkletProcessorRequest | undefined;
        if (!data) {
            return;
        }

        if (data.requestType === RequestType.voice) {
            this.pushVoice(data.voice);
        } else if (data.requestType === RequestType.config) {
            if (typeof data.cushion === "number" && Number.isFinite(data.cushion)) {
                this.cushion = Math.max(1, Math.floor(data.cushion));
                this.adaptiveCushion = 0;
            }
            if (typeof data.inputChunkNum === "number" && Number.isFinite(data.inputChunkNum) && data.inputChunkNum > 0) {
                this.applyInputChunkNum(data.inputChunkNum);
            }
        } else if (data.requestType === RequestType.start) {
            if (typeof data.inputChunkNum === "number" && Number.isFinite(data.inputChunkNum) && data.inputChunkNum > 0) {
                this.applyInputChunkNum(data.inputChunkNum);
            }
            this.isRecording = true;
            this.inputOffset = 0;
            this.resetPlaybackState();
            const res: VoiceChangerWorkletProcessorResponse = {
                responseType: ResponseType.start_ok,
            };
            this.port.postMessage(res);
        } else if (data.requestType === RequestType.stop) {
            this.isRecording = false;
            this.inputOffset = 0;
            this.resetPlaybackState();
            const res: VoiceChangerWorkletProcessorResponse = {
                responseType: ResponseType.stop_ok,
            };
            this.port.postMessage(res);
        } else if (data.requestType === RequestType.trancateBuffer) {
            this.adaptiveCushion = 0;
            this.truncateTo(this.getTargetSamples());
        }
    }

    private captureInput(inputs: Float32Array[][]): void {
        if (!this.isRecording || inputs.length === 0 || inputs[0].length === 0) {
            return;
        }
        const inputChannel = inputs[0][0];
        if (!inputChannel || inputChannel.length === 0) {
            return;
        }

        let srcIdx = 0;
        while (srcIdx < inputChannel.length) {
            const spaceLeft = this.inputAccum.length - this.inputOffset;
            const copyCount = Math.min(spaceLeft, inputChannel.length - srcIdx);
            const dst = this.inputAccum;
            const base = this.inputOffset;

            for (let i = 0; i < copyCount; i++) {
                const raw = inputChannel[srcIdx + i];
                const clamped = raw < -1.0 ? -1.0 : raw > 1.0 ? 1.0 : raw;
                dst[base + i] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
            }

            this.inputOffset += copyCount;
            srcIdx += copyCount;

            if (this.inputOffset >= this.inputAccum.length) {
                const completedChunk = this.inputAccum;
                this.inputAccum = new Int16Array(this.inputChunkNum * RENDER_QUANTUM_FRAMES);
                this.inputOffset = 0;
                const res: VoiceChangerWorkletProcessorResponse = {
                    responseType: ResponseType.inputChunk,
                    inputChunk: completedChunk,
                };
                this.port.postMessage(res, [completedChunk.buffer]);
            }
        }
    }

    private renderOutput(outputs: Float32Array[][]): void {
        if (outputs.length === 0 || outputs[0].length === 0) {
            return;
        }
        const outputChannels = outputs[0];
        const primaryOut = outputChannels[0];
        const blockFrames = primaryOut.length;
        const targetSamples = this.getTargetSamples();

        if (!this.wasPlaying) {
            if (this.availableSamples < targetSamples || this.availableSamples < blockFrames) {
                primaryOut.fill(0);
                this.lastPlaySample = 0;
                for (let ch = 1; ch < outputChannels.length; ch++) {
                    outputChannels[ch].fill(0);
                }
                return;
            }
            this.wasPlaying = true;
            this.needsFadeIn = true;
        }

        if (this.availableSamples < blockFrames) {
            // Smooth cosine fade-out on underrun to suppress hard edge click
            const prev = this.lastPlaySample;
            if (prev !== 0) {
                for (let i = 0; i < blockFrames; i++) {
                    const fade = 0.5 * (1.0 + Math.cos((Math.PI * (i + 1)) / blockFrames));
                    primaryOut[i] = prev * fade;
                }
            } else {
                primaryOut.fill(0);
            }
            this.lastPlaySample = 0;
            this.wasPlaying = false;
            this.isCatchingUp = false;
            this.readFrac = 0;
            this.needsFadeIn = true;

            if (this.adaptiveCushion < MAX_ADAPTIVE_CUSHION) {
                this.adaptiveCushion++;
            }

            const res: VoiceChangerWorkletProcessorResponse = {
                responseType: ResponseType.pop_detected,
                type: "underflow",
                count: 1,
            };
            this.port.postMessage(res);

            for (let ch = 1; ch < outputChannels.length; ch++) {
                outputChannels[ch].set(primaryOut);
            }
            return;
        }

        const configuredChunkSamples = this.getConfiguredChunkSamples();
        const catchUpTrigger = targetSamples + configuredChunkSamples + CATCHUP_TRIGGER_MARGIN;
        const catchUpStop = targetSamples + configuredChunkSamples + CATCHUP_STOP_MARGIN;

        if (!this.isCatchingUp && this.availableSamples > catchUpTrigger) {
            this.isCatchingUp = true;
        } else if (this.isCatchingUp && this.availableSamples <= catchUpStop) {
            this.isCatchingUp = false;
        }

        const step = this.isCatchingUp ? CATCHUP_RATE : 1.0;
        const fadeIn = this.needsFadeIn;
        const baseIdx = this.readIdx;
        const avail = this.availableSamples;

        if (step === 1.0 && this.readFrac === 0) {
            for (let i = 0; i < blockFrames; i++) {
                let s = this.ringBuffer[(baseIdx + i) % RING_BUFFER_SIZE];
                if (fadeIn) {
                    s *= (i + 1) / blockFrames;
                }
                primaryOut[i] = s;
            }
            this.readIdx = (baseIdx + blockFrames) % RING_BUFFER_SIZE;
            this.availableSamples -= blockFrames;
        } else {
            let pos = this.readFrac;
            const lastValid = avail - 1;
            for (let i = 0; i < blockFrames; i++) {
                const idx0 = Math.floor(pos);
                const frac = pos - idx0;
                const c0 = idx0 < lastValid ? idx0 : lastValid;
                const c1 = idx0 + 1 < lastValid ? idx0 + 1 : lastValid;
                const s0 = this.ringBuffer[(baseIdx + c0) % RING_BUFFER_SIZE];
                const s1 = this.ringBuffer[(baseIdx + c1) % RING_BUFFER_SIZE];
                let s = s0 + (s1 - s0) * frac;
                if (fadeIn) {
                    s *= (i + 1) / blockFrames;
                }
                primaryOut[i] = s;
                pos += step;
            }
            const consumed = Math.min(avail, Math.floor(pos));
            this.readFrac = consumed >= avail ? 0 : pos - Math.floor(pos);
            this.readIdx = (baseIdx + consumed) % RING_BUFFER_SIZE;
            this.availableSamples -= consumed;
        }

        this.needsFadeIn = false;
        this.lastPlaySample = primaryOut[blockFrames - 1];

        for (let ch = 1; ch < outputChannels.length; ch++) {
            outputChannels[ch].set(primaryOut);
        }
    }

    process(
        inputs: Float32Array[][],
        outputs: Float32Array[][],
        _parameters: Record<string, Float32Array>
    ): boolean {
        this.captureInput(inputs);
        this.renderOutput(outputs);
        return true;
    }
}

registerProcessor("voice-changer-worklet-processor", VoiceChangerWorkletProcessor);
