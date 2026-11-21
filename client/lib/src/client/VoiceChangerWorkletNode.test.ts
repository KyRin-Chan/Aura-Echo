// Note: client/lib runs Jest without ts-jest, so this file uses plain JS syntax
// while remaining compatible with strict TypeScript checking (tsc --noEmit).

describe("VoiceChangerWorkletProcessor", () => {
    let RegisteredProcessor = JSON.parse("null");

    beforeAll(() => {
        const globalObj = Function("return this")();
        class MockAudioWorkletProcessor {
            port = JSON.parse("null");
            constructor() {
                const posted = JSON.parse("[]");
                this.port = {
                    onmessage: null,
                    postMessage: function () {
                        posted.push(arguments[0]);
                    },
                    _posted: posted,
                };
            }
        }
        globalObj.AudioWorkletProcessor = MockAudioWorkletProcessor;
        globalObj.registerProcessor = function () {
            RegisteredProcessor = arguments[1];
        };
        require("../../worklet/dist/index.js");
    });

    function createProcessor() {
        return new RegisteredProcessor();
    }

    const sendMsg = JSON.parse("0") || function () {
        const proc = arguments[0];
        const payload = arguments[1];
        proc.port.onmessage({ data: payload });
    };

    const runQuantum = JSON.parse("0") || function () {
        const proc = arguments[0];
        const inSamples = arguments[1];
        const outChannels = arguments[2];
        const inputs = inSamples ? [[inSamples]] : [];
        const out0 = new Float32Array(128);
        const channels = [out0];
        if (outChannels && outChannels > 1) {
            for (let c = 1; c < outChannels; c++) {
                channels.push(new Float32Array(128));
            }
        }
        proc.process(inputs, [channels], {});
        return channels;
    };

    test("aggregates inputChunkNum quantums into a single Int16 inputChunk IPC message", () => {
        const proc = createProcessor();
        sendMsg(proc, { requestType: "config", voice: new Float32Array(0), inputChunkNum: 4, cushion: 2 });
        sendMsg(proc, { requestType: "start", voice: new Float32Array(0), inputChunkNum: 4 });

        const posted = proc.port._posted;
        expect(posted.length).toBe(1);
        expect(posted[0].responseType).toBe("start_ok");

        const b0 = new Float32Array(128).fill(0.5);
        const b1 = new Float32Array(128).fill(-0.5);
        const b2 = new Float32Array(128).fill(0.25);
        const b3 = new Float32Array(128).fill(1.5);

        runQuantum(proc, b0, 1);
        runQuantum(proc, b1, 1);
        runQuantum(proc, b2, 1);
        expect(posted.length).toBe(1);

        runQuantum(proc, b3, 1);
        expect(posted.length).toBe(2);
        const chunkMsg = posted[1];
        expect(chunkMsg.responseType).toBe("inputChunk");
        expect(chunkMsg.inputChunk).toBeInstanceOf(Int16Array);
        expect(chunkMsg.inputChunk.length).toBe(512);
        expect(chunkMsg.inputChunk[0]).toBe(Math.trunc(0.5 * 0x7fff));
        expect(chunkMsg.inputChunk[128]).toBe(Math.trunc(-0.5 * 0x8000));
        expect(chunkMsg.inputChunk[256]).toBe(Math.trunc(0.25 * 0x7fff));
        expect(chunkMsg.inputChunk[384]).toBe(32767);
    });

    test("maintains sample continuity across circular ringBuffer 96000 wraparound", () => {
        const proc = createProcessor();
        sendMsg(proc, { requestType: "config", voice: new Float32Array(0), cushion: 1, inputChunkNum: 24 });

        // Walk writeIdx/readIdx near RING_BUFFER_SIZE (96000): 93 * 1024 = 95232
        const filler = new Float32Array(1024).fill(0.1);
        for (let k = 0; k < 93; k++) {
            sendMsg(proc, { requestType: "voice", voice: filler });
            for (let q = 0; q < 8; q++) {
                runQuantum(proc, null, 1);
            }
        }

        // Now push a ramp of 2048 samples crossing index 96000 -> 0 wrap boundary
        const ramp = new Float32Array(2048);
        for (let i = 0; i < 2048; i++) {
            ramp[i] = (i + 1) / 2048;
        }
        sendMsg(proc, { requestType: "voice", voice: ramp });

        const collected = new Float32Array(2048);
        for (let q = 0; q < 16; q++) {
            const chans = runQuantum(proc, null, 2);
            collected.set(chans[0], q * 128);
            // Verify stereo channel mirror matches channel 0
            expect(chans[1][64]).toBeCloseTo(chans[0][64], 6);
        }

        // Compare tail across wrap boundary cleanly
        for (let i = 256; i < 2048; i++) {
            expect(collected[i]).toBeCloseTo(ramp[i], 5);
        }
    });

    test("crossfades 512 samples across readIdx to newReadIdx on trancateBuffer", () => {
        const proc = createProcessor();
        sendMsg(proc, { requestType: "config", voice: new Float32Array(0), cushion: 2 });

        const payload = new Float32Array(3072);
        payload.fill(1.0, 0, 1024);
        payload.fill(-1.0, 1024, 3072);
        sendMsg(proc, { requestType: "voice", voice: payload });

        // Drain first 128 samples so readIdx sits inside +1.0 region and wasPlaying=true
        const first = runQuantum(proc, null, 1)[0];
        expect(first[127]).toBeCloseTo(1.0, 5);

        sendMsg(proc, { requestType: "trancateBuffer", voice: new Float32Array(0) });

        // Next 2 blocks (256 samples) contain the first half of the 512-sample +1.0 -> -1.0 fade
        const fade0 = runQuantum(proc, null, 1)[0];
        const fade1 = runQuantum(proc, null, 1)[0];

        // Start of truncated stream blends smoothly from old head (+1.0) toward new tail (-1.0)
        expect(fade0[0]).toBeCloseTo(1.0, 5);
        expect(fade0[127]).toBeLessThan(fade0[0]);
        expect(fade1[127]).toBeLessThan(fade0[127]);

        // Buffer target was 256 samples (2 blocks); 3rd block triggers underrun
        runQuantum(proc, null, 1);
        const posted = proc.port._posted;
        const lastMsg = posted[posted.length - 1];
        expect(lastMsg.responseType).toBe("pop_detected");
        expect(lastMsg.type).toBe("underflow");
    });

    test("smoothly catches up at 1.04x without dropping low-amplitude audio", () => {
        const proc = createProcessor();
        // Omit inputChunkNum so catchUp triggers at targetSamples + 512 = 768
        sendMsg(proc, { requestType: "config", voice: new Float32Array(0), cushion: 2 });

        // Quiet -46 dBFS sine wave (peak 0.005 < legacy 0.02 silent-drop threshold)
        const quietSamples = new Float32Array(6400);
        for (let i = 0; i < quietSamples.length; i++) {
            quietSamples[i] = 0.005 * Math.sin((2 * Math.PI * i) / 32);
        }
        sendMsg(proc, { requestType: "voice", voice: quietSamples });

        let quantumsPlayed = 0;
        let sawUnderflow = false;
        for (let q = 0; q < 50; q++) {
            const out = runQuantum(proc, null, 1)[0];
            const posted = proc.port._posted;
            if (posted.length > 0 && posted[posted.length - 1].responseType === "pop_detected") {
                sawUnderflow = true;
                break;
            }
            let blockPeak = 0;
            for (let i = 0; i < 128; i++) {
                const a = Math.abs(out[i]);
                if (a > blockPeak) {
                    blockPeak = a;
                }
            }
            // Quiet audio must never be zero-dropped during playback
            expect(blockPeak).toBeGreaterThan(0.002);
            quantumsPlayed++;
        }

        // At 1.0x, 6400 samples take exactly 50 quantums (underflow on q=50).
        // With 1.04x smooth catch-up down to cushion, buffer drains strictly earlier (< 49 quantums).
        expect(sawUnderflow).toBe(true);
        expect(quantumsPlayed).toBeLessThan(49);
    });

    test("applies cosine fade-out on underrun and linear fade-in on recovery", () => {
        const proc = createProcessor();
        sendMsg(proc, { requestType: "config", voice: new Float32Array(0), cushion: 1 });

        sendMsg(proc, { requestType: "voice", voice: new Float32Array(256).fill(0.8) });

        // Block 0: initial 128-sample linear fade-in
        const b0 = runQuantum(proc, null, 1)[0];
        expect(b0[0]).toBeCloseTo(0.8 / 128, 5);
        expect(b0[127]).toBeCloseTo(0.8, 5);

        // Block 1: steady +0.8 (exhausts buffer)
        const b1 = runQuantum(proc, null, 1)[0];
        expect(b1[0]).toBeCloseTo(0.8, 5);
        expect(b1[127]).toBeCloseTo(0.8, 5);

        // Block 2: underrun -> cosine fade-out from 0.8 down to 0 and emit pop_detected
        const b2 = runQuantum(proc, null, 1)[0];
        expect(b2[0]).toBeGreaterThan(0.79);
        expect(b2[63]).toBeGreaterThan(0.3);
        expect(b2[63]).toBeLessThan(0.5);
        expect(Math.abs(b2[127])).toBeLessThan(1e-6);

        const posted = proc.port._posted;
        expect(posted[posted.length - 1]).toEqual({
            responseType: "pop_detected",
            type: "underflow",
            count: 1,
        });

        // Adaptive cushion bumped from 1 -> 2 (target 256 samples). Pushing 128 waits silently.
        sendMsg(proc, { requestType: "voice", voice: new Float32Array(128).fill(0.8) });
        const waiting = runQuantum(proc, null, 1)[0];
        expect(waiting[0]).toBe(0);

        // Pushing 128 more reaches adapted cushion (256 samples) -> plays recovery fade-in
        sendMsg(proc, { requestType: "voice", voice: new Float32Array(128).fill(0.8) });
        const recovery = runQuantum(proc, null, 1)[0];
        expect(recovery[0]).toBeCloseTo(0.8 / 128, 5);
        expect(recovery[127]).toBeCloseTo(0.8, 5);
    });

    test("avoids false 1.04x speedup on normal configured sawtooth chunk while still draining excess", () => {
        const proc = createProcessor();
        sendMsg(proc, { requestType: "config", voice: new Float32Array(0), cushion: 2, inputChunkNum: 24 });

        // Normal 1-chunk arrival (3072 samples): exact 1.0x playback without premature underflow at quantum < 24
        const chunk = new Float32Array(3072);
        for (let i = 0; i < 3072; i++) {
            chunk[i] = ((i % 64) + 1) / 64;
        }
        sendMsg(proc, { requestType: "voice", voice: chunk });

        const playback = new Float32Array(3072);
        for (let q = 0; q < 24; q++) {
            const chans = runQuantum(proc, null, 1);
            playback.set(chans[0], q * 128);
        }
        // Samples past initial 128-sample fade-in must remain bit-exact 1.0x (unwarped by 1.04x)
        for (let i = 256; i < 3072; i++) {
            expect(playback[i]).toBeCloseTo(chunk[i], 6);
        }
    });
});
