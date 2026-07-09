export const CSS_CLASSES = {
    // Form Controls
    select: "w-full p-2.5 border border-slate-200 dark:border-gray-700 rounded-2xl focus:outline-none focus:ring-2 focus:ring-[var(--macaron-mint)] bg-slate-50 dark:bg-gray-800 text-slate-800 dark:text-gray-100 text-sm transition-all duration-200 shadow-sm",
    input: "w-full p-2.5 border border-slate-200 dark:border-gray-700 rounded-2xl focus:outline-none focus:ring-2 focus:ring-[var(--macaron-mint)] bg-slate-50 dark:bg-gray-800 text-slate-800 dark:text-gray-100 text-sm transition-all duration-200 shadow-sm",
    fileInput: `w-full p-2.5 border border-slate-200 dark:border-gray-700 rounded-2xl focus:outline-none focus:ring-2 focus:ring-[var(--macaron-mint)] bg-slate-50 dark:bg-gray-800 text-slate-800 dark:text-gray-100 text-sm file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-[var(--macaron-blue)] file:text-slate-800 hover:file:opacity-90 dark:file:bg-gray-700 dark:file:text-gray-200 transition-all`,
    range: "w-full h-2 bg-slate-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer accent-[var(--macaron-mint)] dark:accent-[var(--macaron-mint)] transition-colors duration-150",
    rangeDisabled: "w-full h-2 bg-slate-100 dark:bg-gray-800 rounded-lg appearance-none cursor-not-allowed accent-slate-300 dark:accent-gray-600 opacity-50 transition-colors duration-150",
    checkbox: "mr-2 accent-[var(--macaron-mint)]",
    radioButton: "mr-2 accent-[var(--macaron-mint)]",

    // Typography
    label: "block text-sm font-medium text-slate-500 dark:text-gray-400 mt-1 mb-1 tracking-wide",
    heading: "text-lg font-semibold text-slate-700 dark:text-gray-200 tracking-tight",
    sliderValue: "text-xs font-semibold text-slate-500 dark:text-gray-400 text-right",
    checkboxLabel: "flex items-center text-sm text-slate-700 dark:text-gray-300 cursor-pointer",
    radioLabel: "inline-flex items-center mr-4 text-sm text-slate-700 dark:text-gray-300 cursor-pointer",

    // Buttons
    iconButton: "p-2.5 text-slate-500 hover:text-slate-700 hover:bg-slate-100 dark:text-gray-400 dark:hover:text-gray-200 dark:hover:bg-gray-700 rounded-full transition-all duration-150 hover:scale-105 active:scale-95",
    primaryButton: "px-5 py-2.5 bg-[var(--macaron-mint)] hover:bg-[var(--macaron-mint-hover)] text-slate-800 font-medium rounded-full shadow-sm transition-all duration-150 hover:scale-103 active:scale-97 hover:shadow-md",
    modalPrimaryButton: "px-5 py-2.5 rounded-full text-sm font-semibold focus:outline-none focus:ring-2 focus:ring-offset-2 bg-[var(--macaron-mint)] hover:bg-[var(--macaron-mint-hover)] text-slate-800 shadow-sm transition-all duration-150 hover:scale-103 active:scale-97",
    modalSecondaryButton: "px-5 py-2.5 rounded-full text-sm font-semibold focus:outline-none focus:ring-2 focus:ring-offset-2 bg-slate-100 hover:bg-slate-200 text-slate-700 dark:bg-gray-700 dark:hover:bg-gray-600 dark:text-gray-200 transition-all duration-150 hover:scale-103 active:scale-97",

    // Layout
    card: "p-5 border border-slate-100 dark:border-gray-800 rounded-3xl shadow-sm bg-white dark:bg-gray-900 transition-all duration-300 hover:shadow-md",
    cardHeader: "flex justify-between items-center mb-4 pb-2.5 border-b border-slate-100 dark:border-gray-800",

    // States
    error: "p-3 mb-4 text-sm text-red-700 bg-[var(--macaron-coral)]/20 dark:bg-[var(--macaron-coral)]/10 dark:text-red-400 rounded-2xl border border-[var(--macaron-coral)]/30",
    success: "p-3 mb-4 text-sm text-green-700 bg-[var(--macaron-mint)]/20 dark:bg-[var(--macaron-mint)]/10 dark:text-green-400 rounded-2xl border border-[var(--macaron-mint)]/30",
    warning: "p-3 mb-4 text-sm text-yellow-700 bg-[var(--macaron-yellow)]/20 dark:bg-[var(--macaron-yellow)]/10 dark:text-yellow-400 rounded-2xl border border-[var(--macaron-yellow)]/30",
    loading: "text-[var(--macaron-mint)] animate-spin"
};

export const AUDIO_KEYS = {
    AUDIO_ELEMENT_FOR_PLAY_RESULT: "audio-result",
    AUDIO_ELEMENT_FOR_PLAY_MONITOR: "audio-monitor",
    AUDIO_ELEMENT_FOR_TEST_ORIGINAL: "audio-test-original",
    AUDIO_ELEMENT_FOR_TEST_CONVERTED: "audio-test-converted",
    AUDIO_ELEMENT_FOR_TEST_CONVERTED_ECHOBACK: "audio-test-converted-echoback",
    AUDIO_ELEMENT_FOR_SAMPLING_INPUT: "body-wav-container-wav-input",
    AUDIO_ELEMENT_FOR_SAMPLING_OUTPUT: "body-wav-container-wav-output",
};

export const INDEXEDDB_KEYS = {
    INDEXEDDB_KEY_AUDIO_INPUT: "INDEXEDDB_KEY_AUDIO_INPUT",
    INDEXEDDB_KEY_AUDIO_OUTPUT: "INDEXEDDB_KEY_AUDIO_OUTPUT",
    INDEXEDDB_KEY_AUDIO_MONITOR: "INDEXEDDB_KEY_AUDIO_MONITOR",
    INDEXEDDB_KEY_DEFAULT_MODEL_TYPE: "INDEXEDDB_KEY_DEFALT_MODEL_TYPE",
    INDEXEDDB_KEY_NOISE1: "INDEXEDDB_KEY_NOISE1",
    INDEXEDDB_KEY_NOISE2: "INDEXEDDB_KEY_NOISE2",
    INDEXEDDB_KEY_ECHO: "INDEXEDDB_KEY_ECHO",
};

export const isDesktopApp = () => {
    if (navigator.userAgent.indexOf('Electron') >= 0) {
        return true;
    } else {
        return false;
    }
};