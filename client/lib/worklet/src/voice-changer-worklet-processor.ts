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
    start_ok: "start_ok",
    stop_ok: "stop_ok",
} as const;
export type ResponseType = (typeof ResponseType)[keyof typeof ResponseType];

export type VoiceChangerWorkletProcessorRequest = {
    requestType: RequestType;
    voice: Float32Array;
    cushion?: number;
};

export type VoiceChangerWorkletProcessorResponse = {
    responseType: ResponseType;
    recordData?: Float32Array[];
    inputData?: Float32Array;
};

class VoiceChangerWorkletProcessor extends AudioWorkletProcessor {
    private BLOCK_SIZE = 128;
    private initialized = false;

    private isRecording = false;
    private lastPlaySample = 0;
    private wasPlaying = false;
    private unpushedF32Data: Float32Array = new Float32Array(0);
    private chunkSize = 4;
    private cushion = 2; // User-configured cushion
    private adaptiveCushion = 2; // Current dynamic safety cushion
    private maxCushion = 8; // Maximum cushion limit
    private stableBlocksCount = 0; // Stable blocks counter without underflows

    playBuffer: Float32Array[] = [];
    /**
     * @constructor
     */
    constructor() {
        super();
        // console.log("[AudioWorkletProcessor] created.");
        this.initialized = true;
        this.port.onmessage = this.handleMessage.bind(this);
    }

    trancateBuffer = (start: number, end?: number) => {
        // Apply cross-fade at the truncation boundary to prevent clicks and pitch jumps
        const discardCount = start;
        if (discardCount > 0 && discardCount < this.playBuffer.length) {
            const discardBlock = this.playBuffer[discardCount - 1];
            const keepBlock = this.playBuffer[discardCount];
            if (discardBlock && keepBlock) {
                for (let i = 0; i < this.BLOCK_SIZE; i++) {
                    const ratio = i / this.BLOCK_SIZE;
                    keepBlock[i] = keepBlock[i] * ratio + discardBlock[i] * (1.0 - ratio);
                }
            }
        }
        this.playBuffer = this.playBuffer.slice(start, end);
    };
    handleMessage(event: any) {
        const request = event.data as VoiceChangerWorkletProcessorRequest;
        if (request.requestType === "config") {
            if (request.cushion !== undefined) {
                this.cushion = request.cushion;
                this.adaptiveCushion = Math.max(this.cushion, this.adaptiveCushion);
            }
            return;
        } else if (request.requestType === "start") {
            if (this.isRecording) {
                console.warn("[worklet] recoring is already started");
                return;
            }
            this.isRecording = true;
            const startResponse: VoiceChangerWorkletProcessorResponse = {
                responseType: "start_ok",
            };
            this.port.postMessage(startResponse);
            return;
        } else if (request.requestType === "stop") {
            if (!this.isRecording) {
                console.warn("[worklet] recoring is not started");
                return;
            }
            this.isRecording = false;
            const stopResponse: VoiceChangerWorkletProcessorResponse = {
                responseType: "stop_ok",
            };
            this.port.postMessage(stopResponse);
            return;
        } else if (request.requestType === "trancateBuffer") {
            this.trancateBuffer(0, 0);
            this.unpushedF32Data = new Float32Array(0);
            return;
        }

        const f32Data = request.voice;
        const concatedF32Data = new Float32Array(this.unpushedF32Data.length + f32Data.length);
        concatedF32Data.set(this.unpushedF32Data);
        concatedF32Data.set(f32Data, this.unpushedF32Data.length);

        const chunkSize = Math.floor(concatedF32Data.length / this.BLOCK_SIZE);
        this.chunkSize = chunkSize;
        // Allow a dynamic jitter buffer headroom based on the adaptive cushion
        // to prevent packet arrival jitter from causing constant sample drops.
        const maxBufferBlocks = this.chunkSize * (this.adaptiveCushion + 2) + 16;
        if (this.playBuffer.length > maxBufferBlocks) {
            this.trancateBuffer(this.playBuffer.length - (this.chunkSize * Math.ceil(this.adaptiveCushion))); 
        }

        for (let i = 0; i < chunkSize; i++) {
            const block = concatedF32Data.slice(i * this.BLOCK_SIZE, (i + 1) * this.BLOCK_SIZE);
            this.playBuffer.push(block);
        }
        this.unpushedF32Data = concatedF32Data.slice(chunkSize * this.BLOCK_SIZE);
    }

    pushData = (inputData: Float32Array) => {
        const volumeResponse: VoiceChangerWorkletProcessorResponse = {
            responseType: ResponseType.inputData,
            inputData: inputData,
        };
        this.port.postMessage(volumeResponse, [inputData.buffer]);
    };

    process(_inputs: Float32Array[][], outputs: Float32Array[][], _parameters: Record<string, Float32Array>) {
        if (!this.initialized) {
            // console.warn("[worklet] worklet_process not ready");
            return true;
        }

        if (this.isRecording) {
            if (_inputs.length > 0 && _inputs[0].length > 0) {
                this.pushData(_inputs[0][0]);
            }
        }

        let voice: Float32Array | undefined = undefined;
        // Silence-Aware Catch-up:
        // If our playBuffer length exceeds our target cushion + 4 blocks,
        // and we are actively playing, scan and skip silent blocks from the front.
        const targetBlocks = this.chunkSize * this.adaptiveCushion;
        const catchUpThreshold = targetBlocks + 4;
        if (this.wasPlaying && this.playBuffer.length > catchUpThreshold) {
            const frontBlock = this.playBuffer[0];
            if (frontBlock) {
                let isSilent = true;
                for (let i = 0; i < frontBlock.length; i++) {
                    if (Math.abs(frontBlock[i]) >= 0.02) {
                        isSilent = false;
                        break;
                    }
                }
                if (isSilent) {
                    this.playBuffer.shift();
                }
            }
        }

        const minStartBlocks = Math.max(8, targetBlocks);
        if (!this.wasPlaying) {
            if (this.playBuffer.length >= minStartBlocks) {
                voice = this.playBuffer.shift();
            }
        } else {
            voice = this.playBuffer.shift();
        }
        if (voice) {
            outputs[0][0].set(voice);
            if (outputs[0].length == 2) {
                outputs[0][1].set(voice);
            }

            // Real-time Pop and Clipping Detection
            let clickCount = 0;
            let clipCount = 0;
            let lastSample = this.lastPlaySample;
            for (let i = 0; i < voice.length; i++) {
                const sample = voice[i];
                if (Math.abs(sample) >= 0.999) {
                    clipCount++;
                }
                if (Math.abs(sample - lastSample) > 0.8) {
                    clickCount++;
                }
                lastSample = sample;
            }
            this.lastPlaySample = lastSample;
            this.wasPlaying = true;

            // Decay the adaptive cushion slowly if we are playing stably
            if (this.adaptiveCushion > this.cushion) {
                this.stableBlocksCount++;
                if (this.stableBlocksCount >= 300) {
                    this.adaptiveCushion = Math.max(this.cushion, this.adaptiveCushion - 0.2);
                    this.stableBlocksCount = 0;
                }
            } else {
                this.stableBlocksCount = 0;
            }

            if (clickCount > 0) {
                this.port.postMessage({ responseType: "pop_detected", type: "click", count: clickCount });
            }
            if (clipCount > 0) {
                this.port.postMessage({ responseType: "pop_detected", type: "clipping", count: clipCount });
            }
        } else {
            if (this.wasPlaying) {
                this.wasPlaying = false;
                this.port.postMessage({ responseType: "pop_detected", type: "underflow" });
                
                // Increase safety cushion to absorb future spikes
                if (this.adaptiveCushion < this.maxCushion) {
                    this.adaptiveCushion = Math.min(this.maxCushion, this.adaptiveCushion + 1.0);
                }
                this.stableBlocksCount = 0;
            }
        }

        return true;
    }
}
registerProcessor("voice-changer-worklet-processor", VoiceChangerWorkletProcessor);
