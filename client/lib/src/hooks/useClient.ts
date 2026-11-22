import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { VoiceChangerClient } from "../VoiceChangerClient";
import { useClientSetting } from "./useClientSetting";
import { IndexedDBStateAndMethod, useIndexedDB } from "./useIndexedDB";
import { ServerSettingState, useServerSetting } from "./useServerSetting";
import { useWorkletNodeSetting } from "./useWorkletNodeSetting";
import { useWorkletSetting } from "./useWorkletSetting";
import { ClientSetting, DefaultClientSettng, ServerInfo, VoiceChangerClientSetting, WorkletNodeSetting, WorkletSetting } from "../const";

export type VoiceChangerClientSettingUpdater =
    | VoiceChangerClientSetting
    | ((prev: VoiceChangerClientSetting) => VoiceChangerClientSetting);

const isValidGain = (val: unknown): val is number =>
    typeof val === "number" && Number.isFinite(val) && val >= 0;

export const mergeLoadedClientSettings = (
    server: Partial<ServerInfo> | null | undefined,
    stored: Partial<ClientSetting> | null | undefined,
    pendingVcPatch?: Partial<VoiceChangerClientSetting>,
    pendingWorkletNodePatch?: Partial<WorkletNodeSetting>,
    pendingWorkletPatch?: Partial<WorkletSetting>
): ClientSetting => {
    const serverChunk =
        typeof server?.serverReadChunkSize === "number"
            ? server.serverReadChunkSize
            : DefaultClientSettng.workletNodeSetting.inputChunkNum;

    const loadedVc: VoiceChangerClientSetting = {
        ...DefaultClientSettng.voiceChangerClientSetting,
        ...(stored?.voiceChangerClientSetting || {}),
    };

    const inGain = server?.serverInputAudioGain;
    if (isValidGain(inGain)) {
        loadedVc.inputGain = inGain;
    }
    const outGain = server?.serverOutputAudioGain;
    if (isValidGain(outGain)) {
        loadedVc.outputGain = outGain;
    }
    const monGain = server?.serverMonitorAudioGain;
    if (isValidGain(monGain)) {
        loadedVc.monitorGain = monGain;
    }
    if (
        typeof server?.echoCancel === "boolean" &&
        (server.enableServerAudio === 1 || stored?.voiceChangerClientSetting?.echoCancel === undefined)
    ) {
        loadedVc.echoCancel = server.echoCancel;
    }
    if (
        typeof server?.noiseSuppression === "boolean" &&
        (server.enableServerAudio === 1 || stored?.voiceChangerClientSetting?.noiseSuppression === undefined)
    ) {
        loadedVc.noiseSuppression = server.noiseSuppression;
    }
    if (
        typeof server?.noiseSuppression2 === "boolean" &&
        (server.enableServerAudio === 1 || stored?.voiceChangerClientSetting?.noiseSuppression2 === undefined)
    ) {
        loadedVc.noiseSuppression2 = server.noiseSuppression2;
    }

    return {
        workletSetting: {
            ...DefaultClientSettng.workletSetting,
            ...(stored?.workletSetting || {}),
            ...(pendingWorkletPatch || {}),
        },
        workletNodeSetting: {
            ...DefaultClientSettng.workletNodeSetting,
            ...(stored?.workletNodeSetting || {}),
            inputChunkNum: serverChunk,
            ...(pendingWorkletNodePatch || {}),
        },
        voiceChangerClientSetting: {
            ...loadedVc,
            ...(pendingVcPatch || {}),
        },
    };
};

const collectChangedPatch = <T extends object>(prev: T, next: T, existing: Partial<T>): Partial<T> => {
    const patch: Record<string, unknown> = { ...existing };
    const prevRec = prev as Record<string, unknown>;
    const nextRec = next as Record<string, unknown>;
    for (const key of Object.keys(nextRec)) {
        if (nextRec[key] !== prevRec[key]) {
            patch[key] = nextRec[key];
        }
    }
    return patch as Partial<T>;
};

export type UseClientProps = {
    audioContext: AudioContext | null;
};

export type ClientState = {
    initialized: boolean;
    setting: ClientSetting;
    // 各種設定I/Fへの参照
    setVoiceChangerClientSetting: (_voiceChangerClientSetting: VoiceChangerClientSettingUpdater) => void;
    setServerUrl: (url: string) => void;
    start: () => Promise<void>;
    stop: () => Promise<void>;
    reloadClientSetting: () => Promise<void>;

    setWorkletNodeSetting: (_workletNodeSetting: WorkletNodeSetting) => void;
    startOutputRecording: () => void;
    stopOutputRecording: () => Promise<Float32Array>;
    trancateBuffer: () => Promise<void>;

    setWorkletSetting: (_workletSetting: WorkletSetting) => void;
    // workletSetting: WorkletSetting
    // workletSetting: WorkletSettingState
    // clientSetting: ClientSettingState
    // workletNodeSetting: WorkletNodeSettingState
    serverSetting: ServerSettingState;
    indexedDBState: IndexedDBStateAndMethod;

    // モニタリングデータ
    bufferingTime: number;
    performance: PerformanceStats;
    // setClientType: (val: ClientType) => void

    // 情報取得
    getInfo: () => Promise<void>;
    // 設定クリア
    clearSetting: () => Promise<void>;
    // AudioOutputElement  設定
    setAudioOutputElementId: (elemId: string) => void;
    setAudioMonitorElementId: (elemId: string) => void;

    errorMessage: string;
    resetErrorMessage: () => void;
    popLogs: { type: string; timestamp: string; count: number }[];
    clearPopLogs: () => void;
};

export type PerformanceStats = {
    vol: number;
    responseTime: number;
    preprocessTime: number;
    mainprocessTime: number;
    postprocessTime: number;
};

export const useClient = (props: UseClientProps): ClientState => {
    const [initialized, setInitialized] = useState<boolean>(false);
    const [setting, setSetting] = useState<ClientSetting>(() => ({
        workletSetting: { ...DefaultClientSettng.workletSetting },
        workletNodeSetting: { ...DefaultClientSettng.workletNodeSetting },
        voiceChangerClientSetting: { ...DefaultClientSettng.voiceChangerClientSetting },
    }));
    const settingRef = useRef<ClientSetting>(setting);
    const settingsLoadedRef = useRef<boolean>(false);
    const pendingVcPatchRef = useRef<Partial<VoiceChangerClientSetting>>({});
    const pendingWorkletNodePatchRef = useRef<Partial<WorkletNodeSetting>>({});
    const pendingWorkletPatchRef = useRef<Partial<WorkletSetting>>({});
    // (1-1) クライアント
    const voiceChangerClientRef = useRef<VoiceChangerClient | null>(null);
    const [voiceChangerClient, setVoiceChangerClient] = useState<VoiceChangerClient | null>(voiceChangerClientRef.current);
    //// クライアント初期化待ち用フラグ
    const initializedResolveRef = useRef<(value: void | PromiseLike<void>) => void>();
    const initializedPromise = useMemo(() => {
        return new Promise<void>((resolve) => {
            initializedResolveRef.current = resolve;
        });
    }, []);

    // (1-2) 各種設定I/F
    const voiceChangerClientSetting = useClientSetting({ voiceChangerClient, voiceChangerClientSetting: setting.voiceChangerClientSetting });
    const workletNodeSetting = useWorkletNodeSetting({ voiceChangerClient: voiceChangerClient, workletNodeSetting: setting.workletNodeSetting });
    useWorkletSetting({ voiceChangerClient, workletSetting: setting.workletSetting });
    const serverSetting = useServerSetting({ voiceChangerClient });
    const indexedDBState = useIndexedDB({ clientType: null });

    // (1-3) モニタリングデータ
    const [bufferingTime, setBufferingTime] = useState<number>(0);
    const [performance, setPerformance] = useState<PerformanceStats>({
        vol: 0,
        responseTime: 0,
        preprocessTime: 0,
        mainprocessTime: 0,
        postprocessTime: 0,
    });
    const [errorMessage, setErrorMessage] = useState<string>("");

    const resetErrorMessage = () => {
        setErrorMessage("");
    };

    // 設定データ管理
    const { setItem, getItem, removeItem } = useIndexedDB({ clientType: null });
    const persistClientSetting = useCallback(
        (nextSetting: ClientSetting) => {
            const storeData: ClientSetting = {
                ...nextSetting,
                voiceChangerClientSetting: { ...nextSetting.voiceChangerClientSetting },
            };
            if (typeof storeData.voiceChangerClientSetting.audioInput !== "string") {
                storeData.voiceChangerClientSetting.audioInput = "none";
            }
            setItem("clientSetting", storeData);
        },
        [setItem]
    );

    // 設定データ初期化
    useEffect(() => {
        if (!voiceChangerClient) {
            return;
        }
        let cancelled = false;
        settingsLoadedRef.current = false;
        const loadSettings = async () => {
            let server: Partial<ServerInfo> | undefined;
            let _setting: Partial<ClientSetting> | null = null;
            try {
                const [srv, stored] = await Promise.all([
                    serverSetting.reloadServerInfo(),
                    getItem("clientSetting") as Promise<Partial<ClientSetting> | null>,
                ]);
                server = srv as Partial<ServerInfo> | undefined;
                _setting = stored;
            } catch (e) {
                console.warn("[useClient] Failed to load initial settings:", e);
            }
            if (cancelled) {
                return;
            }

            const pendingVc = pendingVcPatchRef.current;
            const pendingWorkletNode = pendingWorkletNodePatchRef.current;
            const pendingWorklet = pendingWorkletPatchRef.current;
            const hadPending =
                Object.keys(pendingVc).length > 0 ||
                Object.keys(pendingWorkletNode).length > 0 ||
                Object.keys(pendingWorklet).length > 0;

            const mergedSetting = mergeLoadedClientSettings(
                server,
                _setting,
                pendingVc,
                pendingWorkletNode,
                pendingWorklet
            );

            settingsLoadedRef.current = true;
            pendingVcPatchRef.current = {};
            pendingWorkletNodePatchRef.current = {};
            pendingWorkletPatchRef.current = {};
            settingRef.current = mergedSetting;
            if (hadPending) {
                persistClientSetting(mergedSetting);
            }
            setSetting(mergedSetting);
        };
        loadSettings();
        return () => {
            cancelled = true;
        };
    }, [voiceChangerClient]);

    const [popLogs, setPopLogs] = useState<{ type: string; timestamp: string; count: number }[]>([]);
    const clearPopLogs = () => setPopLogs([]);

    // (2-1) クライアント初期化処理
    useEffect(() => {
        const initialized = async () => {
            if (!props.audioContext) {
                return;
            }
            const voiceChangerClient = new VoiceChangerClient(props.audioContext, true, {
                notifySendBufferingTime: (val: number) => {
                    setBufferingTime(val);
                },
                notifyPerformanceStats: (responseTime: number, vol: number, perf: number[]) => {
                    const [preprocessTime, mainprocessTime, postprocessTime] = perf;
                    setPerformance({
                        vol,
                        responseTime,
                        preprocessTime: Math.ceil(preprocessTime * 1000),
                        mainprocessTime: Math.ceil(mainprocessTime * 1000),
                        postprocessTime: Math.ceil(postprocessTime * 1000),
                    });
                },
                notifyException: (_: string, mes: string) => {
                    // TODO: Refactor
                    // const serverError = `Error code: ${code}\n\n${mes}`
                    // console.error(serverError);
                    setErrorMessage(mes);
                },
                notifyPop: (type: string, timestamp: string, count: number) => {
                    setPopLogs(prev => {
                        const newLogs = [...prev, { type, timestamp, count }];
                        if (newLogs.length > 50) {
                            newLogs.shift();
                        }
                        return newLogs;
                    });
                }
            });

            await voiceChangerClient.isInitialized();
            voiceChangerClientRef.current = voiceChangerClient;
            setVoiceChangerClient(voiceChangerClientRef.current);
            console.log("[useClient] client initialized");

            // const audio = document.getElementById(props.audioOutputElementId) as HTMLAudioElement
            // audio.srcObject = voiceChangerClientRef.current.stream
            // audio.play()
            initializedResolveRef.current!();
            setInitialized(true);
        };
        initialized();
    }, [props.audioContext]);

    const setAudioOutputElementId = (elemId: string) => {
        if (!voiceChangerClientRef.current) {
            console.warn("[voiceChangerClient] is not ready for set audio output.");
            return;
        }
        const audio = document.getElementById(elemId) as HTMLAudioElement;
        if (audio.paused) {
            audio.srcObject = voiceChangerClientRef.current.stream;
            audio.play().catch(() => {});
        }
    };

    const setAudioMonitorElementId = (elemId: string) => {
        if (!voiceChangerClientRef.current) {
            console.warn("[voiceChangerClient] is not ready for set audio output.");
            return;
        }
        const audio = document.getElementById(elemId) as HTMLAudioElement;
        if (audio.paused) {
            audio.srcObject = voiceChangerClientRef.current.monitorStream;
            audio.play().catch(() => {});
        }
    };

    // (2-2) 情報リロード
    const getInfo = useMemo(() => {
        return async () => {
            await initializedPromise;
            await voiceChangerClientSetting.reloadClientSetting(); // 実質的な処理の意味はない
            // await serverSetting.reloadServerInfo();
        };
    }, [voiceChangerClientSetting.reloadClientSetting]);

    const clearSetting = async () => {
        await removeItem("clientSetting");
    };

    // 設定変更
    const setVoiceChangerClientSetting = useCallback(
        (_voiceChangerClientSetting: VoiceChangerClientSettingUpdater) => {
            const prevVc = settingRef.current.voiceChangerClientSetting;
            const nextVc =
                typeof _voiceChangerClientSetting === "function"
                    ? _voiceChangerClientSetting(prevVc)
                    : _voiceChangerClientSetting;

            if (!settingsLoadedRef.current) {
                pendingVcPatchRef.current = collectChangedPatch(
                    prevVc,
                    nextVc,
                    pendingVcPatchRef.current
                );
            }

            const nextSetting: ClientSetting = {
                ...settingRef.current,
                voiceChangerClientSetting: nextVc,
            };
            console.log("setting.voiceChangerClientSetting", nextSetting.voiceChangerClientSetting);
            settingRef.current = nextSetting;
            if (settingsLoadedRef.current) {
                persistClientSetting(nextSetting);
            }
            setSetting(nextSetting);
        },
        [persistClientSetting]
    );

    const setWorkletNodeSetting = useCallback(
        (_workletNodeSetting: WorkletNodeSetting) => {
            const prevWn = settingRef.current.workletNodeSetting;
            if (!settingsLoadedRef.current) {
                pendingWorkletNodePatchRef.current = collectChangedPatch(
                    prevWn,
                    _workletNodeSetting,
                    pendingWorkletNodePatchRef.current
                );
            }

            const nextSetting: ClientSetting = {
                ...settingRef.current,
                workletNodeSetting: _workletNodeSetting,
            };
            console.log("setting.workletNodeSetting", nextSetting.workletNodeSetting);
            settingRef.current = nextSetting;
            if (settingsLoadedRef.current) {
                persistClientSetting(nextSetting);
            }
            setSetting(nextSetting);
        },
        [persistClientSetting]
    );

    const setWorkletSetting = useCallback(
        (_workletSetting: WorkletSetting) => {
            const prevWs = settingRef.current.workletSetting;
            if (!settingsLoadedRef.current) {
                pendingWorkletPatchRef.current = collectChangedPatch(
                    prevWs,
                    _workletSetting,
                    pendingWorkletPatchRef.current
                );
            }

            const nextSetting: ClientSetting = {
                ...settingRef.current,
                workletSetting: _workletSetting,
            };
            console.log("setting.workletSetting", nextSetting.workletSetting);
            settingRef.current = nextSetting;
            if (settingsLoadedRef.current) {
                persistClientSetting(nextSetting);
            }
            setSetting(nextSetting);
        },
        [persistClientSetting]
    );

    const stop = useMemo(() => {
        return async () => {
            await voiceChangerClientSetting.stop();
            setPerformance({
                vol: 0,
                responseTime: 0,
                preprocessTime: 0,
                mainprocessTime: 0,
                postprocessTime: 0,
            });
        };
    }, [voiceChangerClientSetting.stop]);

    return {
        initialized,
        setting,
        // 各種設定I/Fへの参照
        setVoiceChangerClientSetting,
        setServerUrl: voiceChangerClientSetting.setServerUrl,
        start: voiceChangerClientSetting.start,
        stop,
        reloadClientSetting: voiceChangerClientSetting.reloadClientSetting,

        setWorkletNodeSetting,
        startOutputRecording: workletNodeSetting.startOutputRecording,
        stopOutputRecording: workletNodeSetting.stopOutputRecording,
        trancateBuffer: workletNodeSetting.trancateBuffer,

        setWorkletSetting,
        // workletSetting: workletSettingIF.setting,
        serverSetting,
        indexedDBState,

        // モニタリングデータ
        bufferingTime,
        performance,

        // 情報取得
        getInfo,

        // 設定クリア
        clearSetting,

        // AudioOutputElement  設定
        setAudioOutputElementId,
        setAudioMonitorElementId,

        errorMessage,
        resetErrorMessage,
        popLogs,
        clearPopLogs,
    };
};
