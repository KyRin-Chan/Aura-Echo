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

    playBuffer: Float32Array[] = [];
    /**
     * @constructor
     */
    constructor() {
        super();
        console.log("[AudioWorkletProcessor] created.");
        this.initialized = true;
        this.port.onmessage = this.handleMessage.bind(this);
    }

    trancateBuffer = (start: number, end?: number) => {
        console.log(`[worklet] Play buffer size ${this.playBuffer.length}. Truncating with offset ${start}`);
        this.playBuffer = this.playBuffer.slice(start, end)
    };
    handleMessage(event: any) {
        const request = event.data as VoiceChangerWorkletProcessorRequest;
        if (request.requestType === "config") {
            console.log("[worklet] worklet configured", request);
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
            return;
        }

        const f32Data = request.voice;
        const chunkSize = Math.floor(f32Data.length / this.BLOCK_SIZE);
        // Allow a jitter buffer headroom (chunkSize + 8 blocks, which is ~16ms of delay tolerance at 48kHz)
        // to prevent packet arrival jitter from constantly dropping samples and causing robotic metallic sound.
        const maxBufferBlocks = chunkSize + 8;
        if (this.playBuffer.length > maxBufferBlocks) {
            console.log(`[worklet] Truncate ${this.playBuffer.length} > ${maxBufferBlocks}`);
            this.trancateBuffer(this.playBuffer.length - (chunkSize + 2)); // keep a small safety cushion
        }

        for (let i = 0; i < chunkSize; i++) {
            const block = f32Data.subarray(i * this.BLOCK_SIZE, (i + 1) * this.BLOCK_SIZE);
            this.playBuffer.push(block);
        }
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
            console.warn("[worklet] worklet_process not ready");
            return true;
        }

        if (this.isRecording) {
            if (_inputs.length > 0 && _inputs[0].length > 0) {
                this.pushData(_inputs[0][0]);
            }
        }

        const voice = this.playBuffer.shift();
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
            }
        }

        return true;
    }
}
registerProcessor("voice-changer-worklet-processor", VoiceChangerWorkletProcessor);
