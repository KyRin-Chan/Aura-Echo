import { useEffect, useMemo, useRef, useState } from "react";
import { VoiceChangerClient } from "../VoiceChangerClient";
import { useClientSetting } from "./useClientSetting";
import { IndexedDBStateAndMethod, useIndexedDB } from "./useIndexedDB";
import { ServerSettingState, useServerSetting } from "./useServerSetting";
import { useWorkletNodeSetting } from "./useWorkletNodeSetting";
import { useWorkletSetting } from "./useWorkletSetting";
import { ClientSetting, DefaultClientSettng, VoiceChangerClientSetting, WorkletNodeSetting, WorkletSetting } from "../const";

export type UseClientProps = {
    audioContext: AudioContext | null;
};

export type ClientState = {
    initialized: boolean;
    setting: ClientSetting;
    // 各種設定I/Fへの参照
    setVoiceChangerClientSetting: (_voiceChangerClientSetting: VoiceChangerClientSetting) => void;
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
    const [setting, setSetting] = useState<ClientSetting>(DefaultClientSettng);
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
    // 設定データの更新と保存
    const _setSetting = (_setting: ClientSetting) => {
        const storeData = { ..._setting };
        storeData.voiceChangerClientSetting = { ...storeData.voiceChangerClientSetting };
        if (typeof storeData.voiceChangerClientSetting.audioInput != "string") {
            storeData.voiceChangerClientSetting.audioInput = "none";
        }
        setItem("clientSetting", storeData);

        setSetting(_setting);
    };
    // 設定データ初期化
    useEffect(() => {
        if (!voiceChangerClient) {
            return;
        }
        const loadSettings = async () => {
            const server = await serverSetting.reloadServerInfo();
            const _setting = (await getItem("clientSetting")) as Partial<ClientSetting> | null;
            const serverChunk =
                typeof server?.serverReadChunkSize === "number"
                    ? server.serverReadChunkSize
                    : DefaultClientSettng.workletNodeSetting.inputChunkNum;
            const mergedSetting: ClientSetting = {
                workletSetting: {
                    ...DefaultClientSettng.workletSetting,
                    ...(_setting?.workletSetting || {}),
                },
                workletNodeSetting: {
                    ...DefaultClientSettng.workletNodeSetting,
                    ...(_setting?.workletNodeSetting || {}),
                    inputChunkNum: serverChunk,
                },
                voiceChangerClientSetting: {
                    ...DefaultClientSettng.voiceChangerClientSetting,
                    ...(_setting?.voiceChangerClientSetting || {}),
                },
            };
            setSetting(mergedSetting);
        };
        loadSettings();
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
    const setVoiceChangerClientSetting = (_voiceChangerClientSetting: VoiceChangerClientSetting) => {
        const nextSetting: ClientSetting = {
            ...setting,
            voiceChangerClientSetting: _voiceChangerClientSetting,
        };
        console.log("setting.voiceChangerClientSetting", nextSetting.voiceChangerClientSetting);
        // workletSettingIF.setSetting(_workletSetting)
        _setSetting(nextSetting);
    };

    const setWorkletNodeSetting = (_workletNodeSetting: WorkletNodeSetting) => {
        const nextSetting: ClientSetting = {
            ...setting,
            workletNodeSetting: _workletNodeSetting,
        };
        console.log("setting.workletNodeSetting", nextSetting.workletNodeSetting);
        // workletSettingIF.setSetting(_workletSetting)
        _setSetting(nextSetting);
    };

    const setWorkletSetting = (_workletSetting: WorkletSetting) => {
        const nextSetting: ClientSetting = {
            ...setting,
            workletSetting: _workletSetting,
        };
        console.log("setting.workletSetting", nextSetting.workletSetting);
        // workletSettingIF.setSetting(_workletSetting)
        _setSetting(nextSetting);
    };

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
