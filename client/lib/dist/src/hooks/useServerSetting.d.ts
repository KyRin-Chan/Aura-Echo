import { VoiceChangerServerSetting, ServerInfo, OnnxExporterInfo, MergeModelRequest, VoiceChangerType, TensorRTStatus } from "../const";
import { VoiceChangerClient } from "../VoiceChangerClient";
export declare const ModelAssetName: {
    readonly iconFile: "iconFile";
};
export type ModelAssetName = (typeof ModelAssetName)[keyof typeof ModelAssetName];
export declare const ModelFileKind: {
    readonly rvcModel: "rvcModel";
    readonly rvcIndex: "rvcIndex";
};
export type ModelFileKind = (typeof ModelFileKind)[keyof typeof ModelFileKind];
export type ModelFile = {
    file: File;
    kind: ModelFileKind;
    dir: string;
};
export type ModelUploadSetting = {
    voiceChangerType: VoiceChangerType;
    slot: number;
    files: ModelFile[];
    embedder: string;
    params: any;
};
export type ModelFileForServer = Omit<ModelFile, "file"> & {
    name: string;
    kind: ModelFileKind;
};
export type ModelUploadSettingForServer = Omit<ModelUploadSetting, "files"> & {
    files: ModelFileForServer[];
};
export type BackgroundSoundsFile = {
    file: File;
    dir: string;
};
export type BackgroundSoundsUploadSetting = {
    file: BackgroundSoundsFile;
    params: any;
};
export type BackgroundSoundsFileForServer = Omit<BackgroundSoundsFile, "file"> & {
    name: string;
};
export type BackgroundSoundsUploadSettingForServer = Omit<BackgroundSoundsUploadSetting, "file"> & {
    file: BackgroundSoundsFileForServer;
};
export type ServerSettingUpdateEntry = {
    key: keyof VoiceChangerServerSetting;
    value: string;
};
export declare const collectServerSettingUpdates: (current: ServerInfo, patch: Partial<ServerInfo>) => ServerSettingUpdateEntry[];
export type UseServerSettingProps = {
    voiceChangerClient: VoiceChangerClient | null;
};
export type ServerSettingState = {
    serverSetting: ServerInfo;
    updateServerSettings: (setting: Partial<ServerInfo>) => Promise<void>;
    reloadServerInfo: () => Promise<any>;
    uploadBackgroundSound: (setting: BackgroundSoundsUploadSetting) => Promise<void>;
    uploadModel: (setting: ModelUploadSetting) => Promise<ServerInfo>;
    deleteModel: (slot: number) => Promise<void>;
    uploadProgress: number;
    isUploading: boolean;
    getOnnx: () => Promise<OnnxExporterInfo>;
    mergeModel: (request: MergeModelRequest) => Promise<ServerInfo>;
    updateModelDefault: () => Promise<ServerInfo>;
    updateModelInfo: (slot: number, key: string, val: string) => Promise<ServerInfo>;
    updateSoundInfo: (slot: string, key: string, val: string) => Promise<ServerInfo>;
    deleteSound: (soundId: string) => Promise<ServerInfo>;
    uploadAssets: (slot: number, name: ModelAssetName, file: File) => Promise<void>;
    downloadPretrained: (model_key: string) => Promise<ServerInfo>;
    deletePretrained: (model_key: string) => Promise<ServerInfo>;
    getTensorRTStatus: () => Promise<TensorRTStatus>;
    downloadTensorRT: () => Promise<TensorRTStatus>;
    cancelTensorRTDownload: () => Promise<TensorRTStatus>;
    deleteTensorRT: () => Promise<TensorRTStatus>;
};
export declare const useServerSetting: (props: UseServerSettingProps) => ServerSettingState;
