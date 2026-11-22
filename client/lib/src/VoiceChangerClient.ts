import { VoiceChangerWorkletNode, VoiceChangerWorkletListener } from "./client/VoiceChangerWorkletNode";
// @ts-ignore
import workerjs from "raw-loader!../worklet/dist/index.js";
import { VoiceFocusDeviceTransformer, VoiceFocusTransformDevice } from "amazon-chime-sdk-js";
import { createDummyMediaStream, validateUrl } from "./util";
import { DefaultClientSettng, MergeModelRequest, ServerSettingKey, VoiceChangerClientSetting, WorkletNodeSetting, WorkletSetting } from "./const";
import { ServerConfigurator } from "./client/ServerConfigurator";

// オーディオデータの流れ
// input node(mic or MediaStream) -> [vf node] -> [vc node] ->
//    sio/rest server ->  [vc node] -> output node

import { BlockingQueue } from "./utils/BlockingQueue";

export class VoiceChangerClient {
    private configurator: ServerConfigurator;
    private ctx: AudioContext;
    private vfEnable = false;
    private vf: VoiceFocusDeviceTransformer | null = null;
    private currentDevice: VoiceFocusTransformDevice | null = null;

    private currentMediaStream: MediaStream | null = null;
    private currentMediaStreamAudioSourceNode: MediaStreamAudioSourceNode | null = null;
    private inputGainNode: GainNode | null = null;
    private outputGainNode: GainNode | null = null;
    private monitorGainNode: GainNode | null = null;
    private vcInNode!: VoiceChangerWorkletNode;
    private vcOutNode!: VoiceChangerWorkletNode;
    private compressorNode: DynamicsCompressorNode | null = null;
    private currentMediaStreamAudioDestinationNode!: MediaStreamAudioDestinationNode;
    private currentMediaStreamAudioDestinationMonitorNode!: MediaStreamAudioDestinationNode;

    private promiseForInitialize: Promise<void>;
    private _isVoiceChanging = false;
    private opGeneration = 0;

    private setting: VoiceChangerClientSetting = { ...DefaultClientSettng.voiceChangerClientSetting };

    private sslCertified: string[] = [];

    private sem = new BlockingQueue<number>();

    constructor(ctx: AudioContext, vfEnable: boolean, voiceChangerWorkletListener: VoiceChangerWorkletListener) {
        this.sem.enqueue(0);
        this.configurator = new ServerConfigurator("");
        this.ctx = ctx;
        this.vfEnable = vfEnable;
        this.promiseForInitialize = new Promise<void>(async (resolve) => {
            const scriptUrl = URL.createObjectURL(new Blob([workerjs], { type: "text/javascript" }));

            // await this.ctx.audioWorklet.addModule(scriptUrl)
            // this.vcInNode = new VoiceChangerWorkletNode(this.ctx, voiceChangerWorkletListener); // vc node

            try {
                this.vcInNode = new VoiceChangerWorkletNode(this.ctx, voiceChangerWorkletListener); // vc node
            } catch (err) {
                await this.ctx.audioWorklet.addModule(scriptUrl);
                this.vcInNode = new VoiceChangerWorkletNode(this.ctx, voiceChangerWorkletListener); // vc node
            }
            this.vcInNode.createSocketIO()

            try {
                this.vcOutNode = new VoiceChangerWorkletNode(this.ctx, voiceChangerWorkletListener); // vc node
            } catch (err) {
                await this.ctx.audioWorklet.addModule(scriptUrl);
                this.vcOutNode = new VoiceChangerWorkletNode(this.ctx, voiceChangerWorkletListener); // vc node
            }
            this.vcInNode.setOutputNode(this.vcOutNode);
            this.currentMediaStreamAudioDestinationNode = this.ctx.createMediaStreamDestination(); // output node
            this.outputGainNode = this.ctx.createGain();
            this.outputGainNode.gain.value = this.setting.outputGain;
            this.outputGainNode.connect(this.currentMediaStreamAudioDestinationNode);

            this.currentMediaStreamAudioDestinationMonitorNode = this.ctx.createMediaStreamDestination(); // output node
            this.monitorGainNode = this.ctx.createGain();
            this.monitorGainNode.gain.value = this.setting.monitorGain;
            this.monitorGainNode.connect(this.currentMediaStreamAudioDestinationMonitorNode);

            this.compressorNode = this.ctx.createDynamicsCompressor();
            this.compressorNode.threshold.value = -12;
            this.compressorNode.knee.value = 30;
            this.compressorNode.ratio.value = 12;
            this.compressorNode.attack.value = 0.003;
            this.compressorNode.release.value = 0.25;

            this.updateOutputConnections();

            if (this.vfEnable) {
                this.vf = await VoiceFocusDeviceTransformer.create({ variant: "c20" });
                const dummyMediaStream = createDummyMediaStream(this.ctx);
                this.currentDevice = (await this.vf.createTransformDevice(dummyMediaStream)) || null;
            }
            resolve();
        });
    }

    private lock = async () => {
        const num = await this.sem.dequeue();
        return num;
    };
    private unlock = (num: number) => {
        this.sem.enqueue(num + 1);
    };

    isInitialized = async () => {
        if (this.promiseForInitialize) {
            await this.promiseForInitialize;
        }
        return true;
    };

    /////////////////////////////////////////////////////
    // オペレーション
    /////////////////////////////////////////////////////
    /// Operations ///
    setup = async () => {
        const lockNum = await this.lock();

        console.log(`Input Setup=> echo: ${this.setting.echoCancel}, noise1: ${this.setting.noiseSuppression}, noise2: ${this.setting.noiseSuppression2}`);
        // condition check
        if (!this.vcInNode) {
            console.warn("vc node is not initialized.");
            throw "vc node is not initialized.";
        }

        // Main Process
        //// shutdown & re-generate mediastream
        if (this.currentMediaStream) {
            this.currentMediaStream.getTracks().forEach((x) => {
                x.stop();
            });
            this.currentMediaStream = null;
        }

        //// Input デバイスがnullの時はmicStreamを止めてリターン
        if (!this.setting.audioInput) {
            console.log(`Input Setup=> client mic is disabled. ${this.setting.audioInput}`);
            this.vcInNode.stop();
            await this.unlock(lockNum);
            return;
        }

        if (typeof this.setting.audioInput == "string") {
            try {
                if (this.setting.audioInput == "none") {
                    this.currentMediaStream = createDummyMediaStream(this.ctx);
                } else {
                    this.currentMediaStream = await navigator.mediaDevices.getUserMedia({
                        audio: {
                            deviceId: this.setting.audioInput,
                            channelCount: 1,
                            sampleRate: this.ctx.sampleRate,
                            sampleSize: 16,
                            autoGainControl: false,
                            echoCancellation: this.setting.echoCancel,
                            noiseSuppression: this.setting.noiseSuppression,
                        },
                        video: false
                    });
                }
            } catch (e) {
                console.warn(e);
                this.vcInNode.stop();
                await this.unlock(lockNum);
                throw e;
            }
            // this.currentMediaStream.getAudioTracks().forEach((x) => {
            //     console.log("MIC Setting(cap)", x.getCapabilities())
            //     console.log("MIC Setting(const)", x.getConstraints())
            //     console.log("MIC Setting(setting)", x.getSettings())
            // })
        } else {
            this.currentMediaStream = this.setting.audioInput;
        }

        // connect nodes.
        this.currentMediaStreamAudioSourceNode = this.ctx.createMediaStreamSource(this.currentMediaStream);
        this.inputGainNode = this.ctx.createGain();
        this.inputGainNode.gain.value = this.setting.inputGain;
        this.currentMediaStreamAudioSourceNode.connect(this.inputGainNode);
        if (this.currentDevice && this.setting.noiseSuppression2) {
            this.currentDevice.chooseNewInnerDevice(this.currentMediaStream);
            const voiceFocusNode = await this.currentDevice.createAudioNode(this.ctx); // vf node
            this.inputGainNode.connect(voiceFocusNode.start); // input node -> vf node
            voiceFocusNode.end.connect(this.vcInNode);
        } else {
            // console.log("input___ media stream", this.currentMediaStream)
            // this.currentMediaStream.getTracks().forEach(x => {
            //     console.log("input___ media stream set", x.getSettings())
            //     console.log("input___ media stream con", x.getConstraints())
            //     console.log("input___ media stream cap", x.getCapabilities())
            // })
            // console.log("input___ media node", this.currentMediaStreamAudioSourceNode)
            // console.log("input___ gain node", this.inputGainNode.channelCount, this.inputGainNode)
            this.inputGainNode.connect(this.vcInNode);
        }
        this.vcInNode.setOutputNode(this.vcOutNode);
        console.log("Input Setup=> success");
        await this.unlock(lockNum);
    };
    get stream(): MediaStream {
        return this.currentMediaStreamAudioDestinationNode.stream;
    }
    get monitorStream(): MediaStream {
        return this.currentMediaStreamAudioDestinationMonitorNode.stream;
    }

    start = async () => {
        const gen = ++this.opGeneration;

        // Before starting capture, sync client chunking with server settings
        try {
            const info = await this.configurator.getSettings();
            if (gen !== this.opGeneration) {
                return;
            }
            const current = this.vcInNode.getSettings();
            if (info && typeof info.serverReadChunkSize === 'number' && current.inputChunkNum !== info.serverReadChunkSize) {
                this.updateWorkletNodeSetting({ ...current, inputChunkNum: info.serverReadChunkSize });
            }
        } catch (e) {
            // Non-fatal: fall back to current settings if server not reachable
            console.warn('[VoiceChangerClient] Failed to sync chunk size before start:', e);
            if (gen !== this.opGeneration) {
                return;
            }
        }

        if (this.ctx.state === "suspended") {
            await this.ctx.resume();
            if (gen !== this.opGeneration) {
                return;
            }
        }
        await this.vcInNode.start();
        if (gen !== this.opGeneration) {
            await Promise.all([this.vcInNode.stop(), this.vcOutNode.stop()]);
            return;
        }
        this._isVoiceChanging = true;
    };
    stop = async () => {
        ++this.opGeneration;
        this._isVoiceChanging = false;
        await Promise.all([this.vcInNode.stop(), this.vcOutNode.stop()]);
    };

    get isVoiceChanging(): boolean {
        return this._isVoiceChanging;
    }

    ////////////////////////
    /// 設定
    //////////////////////////////
    setServerUrl = (serverUrl: string, openTab: boolean = false) => {
        const url = validateUrl(serverUrl);
        const pageUrl = `${location.protocol}//${location.host}`;

        if (url != pageUrl && url.length != 0 && location.protocol == "https:" && this.sslCertified.includes(url) == false) {
            if (openTab) {
                const value = window.confirm("MMVC Server is different from this page's origin. Open tab to open ssl connection. OK? (You can close the opened tab after ssl connection succeed.)");
                if (value) {
                    window.open(url, "_blank");
                    this.sslCertified.push(url);
                } else {
                    alert("Your voice conversion may fail...");
                }
            }
        }
        // Update server URL in worklet and REST client
        const cur = this.vcInNode.getSettings();
        this.vcInNode.updateSetting({ ...cur, serverUrl: url });
        this.configurator = new ServerConfigurator(url);

        // Opportunistically sync inputChunkNum with server immediately
        this.configurator.getSettings().then((info) => {
            try {
                if (info && typeof info.serverReadChunkSize === 'number' && cur.inputChunkNum !== info.serverReadChunkSize) {
                    this.updateWorkletNodeSetting({ ...cur, serverUrl: url, inputChunkNum: info.serverReadChunkSize });
                }
            } catch { /* noop */ }
        }).catch(() => {/* ignore */});
    };

    private updateOutputConnections = () => {
        try {
            this.vcOutNode.disconnect();
            if (this.compressorNode) {
                this.compressorNode.disconnect();
            }
        } catch (e) {
            console.warn("disconnect error", e);
        }

        if (this.setting.agcEnabled && this.compressorNode) {
            this.vcOutNode.connect(this.compressorNode);
            this.compressorNode.connect(this.outputGainNode!);
            this.compressorNode.connect(this.monitorGainNode!);
        } else {
            this.vcOutNode.connect(this.outputGainNode!);
            this.vcOutNode.connect(this.monitorGainNode!);
        }
    };

    updateClientSetting = async (setting: VoiceChangerClientSetting) => {
        const prevSetting = this.setting;
        let reconstructInputRequired = false;
        if (
            prevSetting.audioInput !== setting.audioInput ||
            prevSetting.echoCancel !== setting.echoCancel ||
            prevSetting.noiseSuppression !== setting.noiseSuppression ||
            prevSetting.noiseSuppression2 !== setting.noiseSuppression2 ||
            prevSetting.sampleRate !== setting.sampleRate
        ) {
            reconstructInputRequired = true;
        }

        const agcChanged = prevSetting.agcEnabled !== setting.agcEnabled;
        this.setting = { ...setting };

        const isValidGain = (val: number) => typeof val === "number" && Number.isFinite(val) && val >= 0;

        if (prevSetting.inputGain !== setting.inputGain) {
            if (isValidGain(setting.inputGain)) {
                this.setInputGain(setting.inputGain);
            } else {
                this.setting.inputGain = prevSetting.inputGain;
            }
        }
        if (prevSetting.outputGain !== setting.outputGain) {
            if (isValidGain(setting.outputGain)) {
                this.setOutputGain(setting.outputGain);
            } else {
                this.setting.outputGain = prevSetting.outputGain;
            }
        }
        if (prevSetting.monitorGain !== setting.monitorGain) {
            if (isValidGain(setting.monitorGain)) {
                this.setMonitorGain(setting.monitorGain);
            } else {
                this.setting.monitorGain = prevSetting.monitorGain;
            }
        }

        if (agcChanged) {
            this.updateOutputConnections();
        }

        if (reconstructInputRequired) {
            await this.setup();
        }
    };

    setInputGain = (val: number) => {
        if (typeof val !== "number" || !Number.isFinite(val) || val < 0) {
            return;
        }
        this.setting.inputGain = val;
        if (!this.inputGainNode) {
            return;
        }
        this.inputGainNode.gain.value = val;
    };

    setOutputGain = (val: number) => {
        if (typeof val !== "number" || !Number.isFinite(val) || val < 0) {
            return;
        }
        this.setting.outputGain = val;
        if (!this.outputGainNode) {
            return;
        }
        this.outputGainNode.gain.value = val;
    };

    setMonitorGain = (val: number) => {
        if (typeof val !== "number" || !Number.isFinite(val) || val < 0) {
            return;
        }
        this.setting.monitorGain = val;
        if (!this.monitorGainNode) {
            return;
        }
        this.monitorGainNode.gain.value = val;
    };

    /////////////////////////////////////////////////////
    // コンポーネント設定、操作
    /////////////////////////////////////////////////////
    //##  Server ##//
    getModelType = () => {
        return this.configurator.getModelType();
    };
    getOnnx = async () => {
        return this.configurator.export2onnx();
    };
    mergeModel = async (req: MergeModelRequest) => {
        return this.configurator.mergeModel(req);
    };
    updateModelDefault = async () => {
        return this.configurator.updateModelDefault();
    };
    updateModelInfo = async (slot: number, key: string, val: string) => {
        return this.configurator.updateModelInfo(slot, key, val);
    };
    loadModel = (slot: number, isHalf: boolean, params: string) => {
        return this.configurator.loadModel(slot, isHalf, params);
    };
    deleteModel = async (slot: number) => {
        return this.configurator.deleteModel(slot);
    };

    updateServerSettings = (key: ServerSettingKey, val: string) => {
        return this.configurator.updateSettings(key, val);
    };
    
    uploadFile = (buf: ArrayBuffer, filename: string, onprogress: (progress: number, end: boolean) => void) => {
        return this.configurator.uploadFile(buf, filename, onprogress);
    };
    uploadFile2 = (dir: string, file: File, onprogress: (progress: number, end: boolean) => void) => {
        return this.configurator.uploadFile2(dir, file, onprogress);
    };
    concatUploadedFile = (filename: string, chunkNum: number) => {
        return this.configurator.concatUploadedFile(filename, chunkNum);
    };
    uploadAssets = (params: string) => {
        return this.configurator.uploadAssets(params);
    };

    //-------------- Background Sounds --------------
    loadSound = (params: string) => {
        return this.configurator.loadSound(params);
    };
    updateSoundInfo = (slot: string, key: string, val: string) => {
        return this.configurator.updateSoundInfo(slot, key, val);
    };
    deleteSound = (slot: string) => {
        return this.configurator.deleteSound(slot);
    };

    //-------------- Pretrained Models --------------
    downloadPretrained = (model_key: string) => {
        return this.configurator.downloadPretrained(model_key);
    };
    deletePretrained = (model_key: string) => {
        return this.configurator.deletePretrained(model_key);
    };

    //-------------- TensorRT Runtime --------------
    getTensorRTStatus = () => {
        return this.configurator.getTensorRTStatus();
    };
    downloadTensorRT = () => {
        return this.configurator.downloadTensorRT();
    };
    cancelTensorRTDownload = () => {
        return this.configurator.cancelTensorRTDownload();
    };
    deleteTensorRT = () => {
        return this.configurator.deleteTensorRT();
    };

    //##  Worklet ##//
    configureWorklet = (setting: WorkletSetting) => {
        this.vcInNode.configure(setting);
        this.vcOutNode.configure(setting);
    };
    startOutputRecording = () => {
        this.vcOutNode.startOutputRecording();
    };
    stopOutputRecording = () => {
        return this.vcOutNode.stopOutputRecording();
    };
    trancateBuffer = () => {
        this.vcOutNode.trancateBuffer();
    };
    //##  Worklet Node ##//
    updateWorkletNodeSetting = (setting: WorkletNodeSetting) => {
        this.vcInNode.updateSetting(setting);
        this.vcOutNode.updateSetting(setting);
    };

    /////////////////////////////////////////////////////
    // 情報取得
    /////////////////////////////////////////////////////
    // Information
    getClientSettings = () => {
        return this.vcInNode.getSettings();
    };
    getServerSettings = () => {
        return this.configurator.getSettings();
    };
    getPerformance = () => {
        return this.configurator.getPerformance();
    };

    getSocketId = () => {
        return this.vcInNode.getSocketId();
    };
}
