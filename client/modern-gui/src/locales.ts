export type LangType = 'en' | 'zh' | 'ja';

// Simple language detector
const getBrowserLang = (): LangType => {
    const lang = localStorage.getItem('app_language') || navigator.language || 'en';
    if (lang.startsWith('zh')) return 'zh';
    if (lang.startsWith('ja')) return 'ja';
    return 'en';
};

export let currentLang: LangType = getBrowserLang();

export const setLang = (lang: LangType) => {
    localStorage.setItem('app_language', lang);
    currentLang = lang;
};

export const locales = {
    en: {
        // UI Prompts
        modelUploadSuccess: "Model uploaded successfully!",
        modelUploadFailed: "Failed to upload model. The model file was not properly saved.",
        modelUploadSelectFile: "Please select a model file.",
        modelUploadEnterName: "Please enter a model name.",
        modelUploadNoSlot: "No empty model slot available. Please clear a slot or manage existing ones.",
        modelUploadedMsg: "Model uploaded successfully!",
        
        // Voice Analyzer
        voiceAnalyzerTitle: "Voice Lab & Analyzer",
        voiceAnalyzerDesc: "Upload a target reference sample (e.g. the voice model's original speaker) and your own voice input sample. The analyzer will automatically calculate recommendations for your Pitch (transpose) and Formant Shift.",
        voiceAnalyzerStandardTarget: "Standard Target Reference",
        voiceAnalyzerYourInput: "Your Input Sample",
        voiceAnalyzerComparing: "Comparing & Analyzing...",
        voiceAnalyzerStartAnalysis: "Start Comparison Analysis",
        voiceAnalyzerDiagnostics: "Analysis Diagnostics",
        voiceAnalyzerRecommendedParams: "Recommended Adaptation Parameters",
        voiceAnalyzerPitchComparison: "Pitch (F0) Comparison",
        voiceAnalyzerRecommendedPitch: "Recommended Pitch Shift:",
        voiceAnalyzerApplyPitch: "Apply Pitch",
        voiceAnalyzerRecommendedFormant: "Recommended Formant Shift:",
        voiceAnalyzerApplyFormant: "Apply Formant",
        voiceAnalyzerApplyAll: "Apply All Recommendations",
        voiceAnalyzerAppliedPitchSuccess: "Applied recommended Pitch: ",
        voiceAnalyzerAppliedBothSuccess: "Applied both parameters! Pitch: ",
        
        // Generic UI & Errors
        errorTitle: "Error",
        confirmTitle: "Confirm",
        loadingSwappingModel: "Swapping to model: ",
        
        // Audio Effects & Merge
        mergeSuccess: "Models uploaded successfully!",
        mergeNoAction: "No action selected. Please select at least one action.",
        mergeError: "Error merging models: ",
        
        // Model Settings
        pitchLabel: "Pitch:",
        f0DetectorLabel: "Pitch Extraction Algorithm",
        noPitchExtractors: "No downloaded pitch extractors available",

        // New Voice Analyzer & General Translations
        selectModelFirst: "Select a voice model first.",
        selectAudioInputFirst: "Select an audio input device.",
        selectAudioOutputFirst: "Select an audio output device.",
        voiceAnalyzerUploadBoth: "Please upload both Reference and Input samples.",
        voiceAnalyzerUploadWarning: "Warning",
        voiceAnalyzerFailedAnalysis: "Failed to analyze audio: ",
        voiceAnalyzerAppliedFormantSuccess: "Applied recommended Formant Shift: ",
        voiceAnalyzerFormantLabel: ", Formant: ",
        voiceAnalyzerPreviewAudio: "Preview Audio:",
        voiceAnalyzerStandardTargetLabel: "Standard Target:",
        voiceAnalyzerYourInputLabel: "Your Input:",
        voiceAnalyzerFormantResonanceLabel: "Formant Resonance",
        voiceAnalyzerSemitones: "semitones",

        mergeLabTitle: "Merge Lab",
        advancedSettingsTitle: "Advanced Settings",
        serverInfoTitle: "Server Info",
        clientInfoTitle: "Client Info",
        btnStop: "Stop",
        btnStart: "Start",
        passthroughOn: "Passthrough ON",
        passthroughOff: "Passthrough OFF",

        modelSelectorTitle: "Model Selector",
        availableModelsLabel: "Available Models",
        uploadNewModelTitle: "Upload New Model"
    },
    zh: {
        // UI Prompts
        modelUploadSuccess: "模型上传成功！",
        modelUploadFailed: "模型上传失败。模型文件未能成功保存。",
        modelUploadSelectFile: "请选择一个模型文件。",
        modelUploadEnterName: "请输入模型名称。",
        modelUploadNoSlot: "没有空闲的模型槽位。请清理一个槽位或管理现有槽位。",
        modelUploadedMsg: "模型上传成功！",
        
        // Voice Analyzer
        voiceAnalyzerTitle: "语音分析实验室 (Voice Lab)",
        voiceAnalyzerDesc: "上传目标参考音频（例如：声线模型原唱/原作者的说话声）以及你自己的输入音频。分析器将自动计算并推荐你的音高偏移（协变）与基频偏移（共振峰）。",
        voiceAnalyzerStandardTarget: "标准参考样本 (Target)",
        voiceAnalyzerYourInput: "输入样本音频 (Input)",
        voiceAnalyzerComparing: "正在对比分析...",
        voiceAnalyzerStartAnalysis: "开始对比分析",
        voiceAnalyzerDiagnostics: "声学对比分析结果",
        voiceAnalyzerRecommendedParams: "推荐的适配参数",
        voiceAnalyzerPitchComparison: "音高 (F0) 对比",
        voiceAnalyzerRecommendedPitch: "推荐音高偏移：",
        voiceAnalyzerApplyPitch: "应用音高",
        voiceAnalyzerRecommendedFormant: "推荐共振峰偏移：",
        voiceAnalyzerApplyFormant: "应用共振峰",
        voiceAnalyzerApplyAll: "应用所有推荐",
        voiceAnalyzerAppliedPitchSuccess: "已应用推荐音高：",
        voiceAnalyzerAppliedBothSuccess: "已成功应用两个参数！音高：",
        
        // Generic UI & Errors
        errorTitle: "错误",
        confirmTitle: "确认",
        loadingSwappingModel: "正在切换模型：",
        
        // Audio Effects & Merge
        mergeSuccess: "模型融合并上传成功！",
        mergeNoAction: "未选择操作。请至少选择一个操作。",
        mergeError: "融合模型出错：",
        
        // Model Settings
        pitchLabel: "音高变调 (Pitch)：",
        f0DetectorLabel: "音高提取算法",
        noPitchExtractors: "无已下载的音高提取算法可用",

        // New Voice Analyzer & General Translations
        selectModelFirst: "请先选择一个声线模型。",
        selectAudioInputFirst: "请选择音频输入设备。",
        selectAudioOutputFirst: "请选择音频输出设备。",
        voiceAnalyzerUploadBoth: "请同时上传参考样本与输入样本。",
        voiceAnalyzerUploadWarning: "警告",
        voiceAnalyzerFailedAnalysis: "音频分析失败：",
        voiceAnalyzerAppliedFormantSuccess: "已应用推荐共振峰偏移：",
        voiceAnalyzerFormantLabel: ", 共振峰: ",
        voiceAnalyzerPreviewAudio: "预览音频:",
        voiceAnalyzerStandardTargetLabel: "标准参考:",
        voiceAnalyzerYourInputLabel: "你的输入:",
        voiceAnalyzerFormantResonanceLabel: "共振峰共鸣",
        voiceAnalyzerSemitones: "半音",

        mergeLabTitle: "融合实验室",
        advancedSettingsTitle: "高级设置",
        serverInfoTitle: "服务端信息",
        clientInfoTitle: "客户端信息",
        btnStop: "停止",
        btnStart: "启动",
        passthroughOn: "直通已开启",
        passthroughOff: "直通已关闭",

        modelSelectorTitle: "模型选择",
        availableModelsLabel: "可用模型",
        uploadNewModelTitle: "上传新模型"
    },
    ja: {
        // UI Prompts
        modelUploadSuccess: "モデルが正常にアップロードされました！",
        modelUploadFailed: "モデルのアップロードに失敗しました。モデルファイルが正常に保存されませんでした。",
        modelUploadSelectFile: "モデルファイルを選択してください。",
        modelUploadEnterName: "モデル名を入力してください。",
        modelUploadNoSlot: "空いているモデルスロットがありません。スロットをクリアするか、既存のスロットを管理してください。",
        modelUploadedMsg: "モデルのアップロードに成功しました！",
        
        // Voice Analyzer
        voiceAnalyzerTitle: "音声分析ラボ (Voice Lab)",
        voiceAnalyzerDesc: "ターゲットの参照サンプル（例：ボイスモデルのオリジナル話者）と自身のボイス入力サンプルをアップロードします。アナライザーはピッチ（キー）およびフォルマントシフトの推奨値を自動的に計算します。",
        voiceAnalyzerStandardTarget: "標準参照サンプル (Target)",
        voiceAnalyzerYourInput: "入力オーディオサンプル (Input)",
        voiceAnalyzerComparing: "比較分析中...",
        voiceAnalyzerStartAnalysis: "比較分析を開始",
        voiceAnalyzerDiagnostics: "音響比較分析結果",
        voiceAnalyzerRecommendedParams: "推奨のアダプテーションパラメータ",
        voiceAnalyzerPitchComparison: "ピッチ（F0）の比較",
        voiceAnalyzerRecommendedPitch: "推奨ピッチシフト：",
        voiceAnalyzerApplyPitch: "ピッチを適用",
        voiceAnalyzerRecommendedFormant: "推奨フォルマントシフト：",
        voiceAnalyzerApplyFormant: "フォルマントを適用",
        voiceAnalyzerApplyAll: "すべての推奨値を適用",
        voiceAnalyzerAppliedPitchSuccess: "推奨ピッチを適用しました：",
        voiceAnalyzerAppliedBothSuccess: "両方のパラメータを適用しました！ピッチ：",
        
        // Generic UI & Errors
        errorTitle: "エラー",
        confirmTitle: "確認",
        loadingSwappingModel: "モデル切り替え中: ",
        
        // Audio Effects & Merge
        mergeSuccess: "モデルのマージとアップロードが成功しました！",
        mergeNoAction: "アクションが選択されていません。少なくとも1つのアクションを選択してください。",
        mergeError: "モデルのマージ中にエラーが発生しました: ",
        
        // Model Settings
        pitchLabel: "ピッチ:",
        f0DetectorLabel: "ピッチ抽出アルゴリズム",
        noPitchExtractors: "ダウンロード済みのピッチ抽出アルゴリズムがありません",

        // New Voice Analyzer & General Translations
        selectModelFirst: "最初にボイスモデルを選択してください。",
        selectAudioInputFirst: "オーディオ入力デバイスを選択してください。",
        selectAudioOutputFirst: "オーディオ出力デバイスを選択してください。",
        voiceAnalyzerUploadBoth: "参照サンプルと入力サンプルの両方をアップロードしてください。",
        voiceAnalyzerUploadWarning: "警告",
        voiceAnalyzerFailedAnalysis: "オーディオ分析に失敗しました: ",
        voiceAnalyzerAppliedFormantSuccess: "推奨フォルマントシフトを適用しました：",
        voiceAnalyzerFormantLabel: ", フォルマント: ",
        voiceAnalyzerPreviewAudio: "プレビューオーディオ:",
        voiceAnalyzerStandardTargetLabel: "標準参照:",
        voiceAnalyzerYourInputLabel: "入力オーディオ:",
        voiceAnalyzerFormantResonanceLabel: "フォルマント共鳴",
        voiceAnalyzerSemitones: "半音",

        mergeLabTitle: "マージラボ",
        advancedSettingsTitle: "詳細設定",
        serverInfoTitle: "サーバー情報",
        clientInfoTitle: "クライアント情報",
        btnStop: "停止",
        btnStart: "起動",
        passthroughOn: "パススルー ON",
        passthroughOff: "パススルー OFF",

        modelSelectorTitle: "モデル選択",
        availableModelsLabel: "利用可能なモデル",
        uploadNewModelTitle: "新規モデルのアップロード"
    }
};

export const t = (key: keyof typeof locales['en']): string => {
    return locales[currentLang][key] || locales['en'][key] || String(key);
};
