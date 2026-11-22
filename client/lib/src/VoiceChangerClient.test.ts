// Note: client/lib runs Jest without ts-jest, so this file uses plain JS syntax
// while remaining compatible with strict TypeScript checking (tsc --noEmit).

describe("VoiceChangerClient audio gain & settings sync", () => {
    let lib = JSON.parse("null");
    let origAudioWorkletNode = JSON.parse("null");
    let origCreateObjectURL = JSON.parse("null");

    beforeAll(() => {
        const globalObj = Function("return this")();
        origAudioWorkletNode = globalObj.AudioWorkletNode;
        if (globalObj.URL) {
            origCreateObjectURL = globalObj.URL.createObjectURL;
            globalObj.URL.createObjectURL = function () {
                return "blob:mock-worklet";
            };
        }

        class MockAudioWorkletNode {
            port = JSON.parse("null");
            constructor() {
                this.port = {
                    onmessage: null,
                    postMessage: function () {},
                };
            }
            connect() {}
            disconnect() {}
        }

        globalObj.AudioWorkletNode = MockAudioWorkletNode;
        lib = require("../dist/index.js");
    });

    afterAll(() => {
        const globalObj = Function("return this")();
        if (origAudioWorkletNode === undefined) {
            delete globalObj.AudioWorkletNode;
        } else {
            globalObj.AudioWorkletNode = origAudioWorkletNode;
        }
        if (globalObj.URL) {
            if (origCreateObjectURL === undefined) {
                delete globalObj.URL.createObjectURL;
            } else {
                globalObj.URL.createObjectURL = origCreateObjectURL;
            }
        }
    });

    const createMockAudioContext = JSON.parse("0") || function () {
        const gainNodes = JSON.parse("[]");
        const makeGainNode = function () {
            const node = {
                gain: { value: 1 },
                connect: function () {},
                disconnect: function () {},
            };
            gainNodes.push(node);
            return node;
        };
        const makeParam = function () {
            return { value: 0 };
        };
        return {
            sampleRate: 48000,
            state: "running",
            audioWorklet: {
                addModule: async function () {},
            },
            createGain: makeGainNode,
            createMediaStreamDestination: function () {
                return { stream: { id: "mock-dest-stream", getTracks: function () { return []; } } };
            },
            createMediaStreamSource: function () {
                return { connect: function () {}, disconnect: function () {} };
            },
            createOscillator: function () {
                return {
                    frequency: makeParam(),
                    connect: function () {},
                    start: function () {},
                };
            },
            createDynamicsCompressor: function () {
                return {
                    threshold: makeParam(),
                    knee: makeParam(),
                    ratio: makeParam(),
                    attack: makeParam(),
                    release: makeParam(),
                    connect: function () {},
                    disconnect: function () {},
                };
            },
            _gainNodes: gainNodes,
        };
    };

    const createClient = JSON.parse("0") || async function () {
        const ctx = createMockAudioContext();
        const listener = {
            notifySendBufferingTime: function () {},
            notifyPerformanceStats: function () {},
            notifyException: function () {},
        };
        const client = new lib.VoiceChangerClient(ctx, false, listener);
        await client.isInitialized();
        // Set audioInput to "none" to initialize inputGainNode without touching getUserMedia
        await client.updateClientSetting({
            ...lib.DefaultClientSettng.voiceChangerClientSetting,
            audioInput: "none",
        });
        return { client, ctx };
    };

    test("setMonitorGain, setInputGain, and setOutputGain accept 0 (mute) and reject invalid values", async () => {
        const { client } = await createClient();

        expect(client.monitorGainNode.gain.value).toBe(1);
        expect(client.outputGainNode.gain.value).toBe(1);
        expect(client.inputGainNode.gain.value).toBe(1);

        client.setMonitorGain(0);
        expect(client.monitorGainNode.gain.value).toBe(0);
        expect(client.setting.monitorGain).toBe(0);

        client.setOutputGain(0);
        expect(client.outputGainNode.gain.value).toBe(0);
        expect(client.setting.outputGain).toBe(0);

        client.setInputGain(0);
        expect(client.inputGainNode.gain.value).toBe(0);
        expect(client.setting.inputGain).toBe(0);

        // Invalid values (NaN, Infinity, negative) must be ignored
        client.setMonitorGain(0.75);
        client.setMonitorGain(Number.NaN);
        client.setMonitorGain(Number.POSITIVE_INFINITY);
        client.setMonitorGain(-0.2);
        expect(client.monitorGainNode.gain.value).toBe(0.75);
        expect(client.setting.monitorGain).toBe(0.75);

        client.setInputGain(1.4);
        client.setInputGain(-1);
        expect(client.inputGainNode.gain.value).toBe(1.4);
    });

    test("updateClientSetting transitions monitorGain 1.0 -> 0 -> 1.5 without mutating DefaultClientSettng", async () => {
        const { client } = await createClient();
        const base = {
            ...lib.DefaultClientSettng.voiceChangerClientSetting,
            audioInput: "none",
        };

        await client.updateClientSetting({ ...base, monitorGain: 0, inputGain: 1.4, outputGain: 0.8 });
        expect(client.monitorGainNode.gain.value).toBe(0);
        expect(client.inputGainNode.gain.value).toBe(1.4);
        expect(client.outputGainNode.gain.value).toBe(0.8);

        await client.updateClientSetting({ ...base, monitorGain: 1.5, inputGain: 1.4, outputGain: 0.8 });
        expect(client.monitorGainNode.gain.value).toBe(1.5);

        // DefaultClientSettng must remain untouched
        expect(lib.DefaultClientSettng.voiceChangerClientSetting.inputGain).toBe(1.0);
        expect(lib.DefaultClientSettng.voiceChangerClientSetting.outputGain).toBe(1.0);
        expect(lib.DefaultClientSettng.voiceChangerClientSetting.monitorGain).toBe(1.0);
        expect(lib.DefaultClientSettng.voiceChangerClientSetting.audioInput).toBeNull();
    });

    test("mergeLoadedClientSettings syncs server gains (including 0) and preserves early pending patches", () => {
        const merged = lib.mergeLoadedClientSettings(
            {
                serverReadChunkSize: 256,
                serverInputAudioGain: 1.4,
                serverOutputAudioGain: 0.8,
                serverMonitorAudioGain: 0,
            },
            {
                voiceChangerClientSetting: {
                    ...lib.DefaultClientSettng.voiceChangerClientSetting,
                    agcEnabled: true,
                    noiseSuppression: true,
                    inputGain: 0.5,
                    outputGain: 0.6,
                    monitorGain: 0.7,
                },
            },
            { audioInput: "mic-device-1" },
            { protocol: "sio" },
            {}
        );

        expect(merged.workletNodeSetting.inputChunkNum).toBe(256);
        expect(merged.workletNodeSetting.protocol).toBe("sio");
        expect(merged.voiceChangerClientSetting.audioInput).toBe("mic-device-1");
        expect(merged.voiceChangerClientSetting.agcEnabled).toBe(true);
        expect(merged.voiceChangerClientSetting.noiseSuppression).toBe(true);
        expect(merged.voiceChangerClientSetting.inputGain).toBe(1.4);
        expect(merged.voiceChangerClientSetting.outputGain).toBe(0.8);
        expect(merged.voiceChangerClientSetting.monitorGain).toBe(0);

        const serverModeMerged = lib.mergeLoadedClientSettings(
            {
                enableServerAudio: 1,
                echoCancel: true,
                noiseSuppression: false,
                noiseSuppression2: true,
            },
            {
                voiceChangerClientSetting: {
                    ...lib.DefaultClientSettng.voiceChangerClientSetting,
                    echoCancel: false,
                    noiseSuppression: true,
                    noiseSuppression2: false,
                },
            },
            {},
            {},
            {}
        );

        expect(serverModeMerged.voiceChangerClientSetting.echoCancel).toBe(true);
        expect(serverModeMerged.voiceChangerClientSetting.noiseSuppression).toBe(false);
        expect(serverModeMerged.voiceChangerClientSetting.noiseSuppression2).toBe(true);
    });

    test("mergeLoadedClientSettings falls back to stored/default gains when server gains are missing or invalid", () => {
        const fromStored = lib.mergeLoadedClientSettings(
            undefined,
            {
                voiceChangerClientSetting: {
                    ...lib.DefaultClientSettng.voiceChangerClientSetting,
                    inputGain: 1.25,
                    outputGain: 0.9,
                    monitorGain: 0,
                },
            }
        );
        expect(fromStored.voiceChangerClientSetting.inputGain).toBe(1.25);
        expect(fromStored.voiceChangerClientSetting.outputGain).toBe(0.9);
        expect(fromStored.voiceChangerClientSetting.monitorGain).toBe(0);

        const fromDefault = lib.mergeLoadedClientSettings(
            {
                serverInputAudioGain: Number.NaN,
                serverOutputAudioGain: -1,
            },
            null
        );
        expect(fromDefault.voiceChangerClientSetting.inputGain).toBe(1.0);
        expect(fromDefault.voiceChangerClientSetting.outputGain).toBe(1.0);
        expect(fromDefault.voiceChangerClientSetting.monitorGain).toBe(1.0);
    });

    test("collectServerSettingUpdates skips omitted keys so rapid partial patches do not revert earlier toggles", () => {
        const baseServer = {
            ...lib.DefaultServerSetting,
            echoCancel: false,
            noiseSuppression: false,
            noiseSuppression2: false,
        };

        const firstUpdates = lib.collectServerSettingUpdates(baseServer, { echoCancel: true });
        expect(firstUpdates).toEqual([{ key: "echoCancel", value: "true" }]);

        const afterFirst = {
            ...baseServer,
            echoCancel: true,
        };
        const secondUpdates = lib.collectServerSettingUpdates(afterFirst, { noiseSuppression: true });
        expect(secondUpdates).toEqual([{ key: "noiseSuppression", value: "true" }]);

        const afterSecond = {
            ...afterFirst,
            noiseSuppression: true,
        };
        const thirdUpdates = lib.collectServerSettingUpdates(afterSecond, { noiseSuppression2: true });
        expect(thirdUpdates).toEqual([{ key: "noiseSuppression2", value: "true" }]);
    });
});
