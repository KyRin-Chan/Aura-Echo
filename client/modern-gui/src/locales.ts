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
        voiceAnalyzerDesc: "Upload a target reference sample (e.g. the voice model's original speaker) and your own voice input sample. The analyzer will automatically calculate recommendations for your Pitch (transpose) and Formant Shift.",
        voiceAnalyzerPitchComparison: "Pitch (F0) Comparison",
        voiceAnalyzerRecommendedPitch: "Recommended Pitch Shift:",
        voiceAnalyzerApplyPitch: "Apply Pitch",
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
        noPitchExtractors: "No downloaded pitch extractors available"
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
        voiceAnalyzerDesc: "上传目标参考音频（例如：声线模型原唱/原作者的说话声）以及你自己的输入音频。分析器将自动计算并推荐你的音高偏移（协变）与基频偏移（共振峰）。",
        voiceAnalyzerPitchComparison: "音高 (F0) 对比",
        voiceAnalyzerRecommendedPitch: "推荐音高偏移：",
        voiceAnalyzerApplyPitch: "应用音高",
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
        noPitchExtractors: "无已下载的音高提取算法可用"
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
        voiceAnalyzerDesc: "ターゲットの参照サンプル（例：ボイスモデルのオリジナル話者）と自身のボイス入力サンプルをアップロードします。アナライザーはピッチ（キー）およびフォルマントシフトの推奨値を自动的に计算します。",
        voiceAnalyzerPitchComparison: "ピッチ（F0）の比較",
        voiceAnalyzerRecommendedPitch: "推奨ピッチシフト：",
        voiceAnalyzerApplyPitch: "ピッチを適用",
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
        noPitchExtractors: "ダウンロード済みのピッチ抽出アルゴリズムがありません"
    }
};

export const t = (key: keyof typeof locales['en']): string => {
    return locales[currentLang][key] || locales['en'][key] || String(key);
};
