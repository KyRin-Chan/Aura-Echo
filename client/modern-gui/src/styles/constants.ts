export const CSS_CLASSES = {
    // Form Controls
    select: "w-full p-2.5 border border-[var(--border-primary)] rounded-2xl focus:outline-none focus:ring-2 focus:ring-[var(--macaron-mint)] bg-[var(--bg-tertiary)] text-[var(--text-primary)] text-sm transition-all duration-200 shadow-sm",
    input: "w-full p-2.5 border border-[var(--border-primary)] rounded-2xl focus:outline-none focus:ring-2 focus:ring-[var(--macaron-mint)] bg-[var(--bg-tertiary)] text-[var(--text-primary)] text-sm transition-all duration-200 shadow-sm",
    fileInput: `w-full p-2.5 border border-[var(--border-primary)] rounded-2xl focus:outline-none focus:ring-2 focus:ring-[var(--macaron-mint)] bg-[var(--bg-tertiary)] text-[var(--text-primary)] text-sm file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-[var(--macaron-blue)] file:text-[var(--text-primary)] hover:file:opacity-90 transition-all`,
    range: "w-full h-2 bg-[var(--border-primary)] rounded-lg appearance-none cursor-pointer accent-[var(--macaron-mint)] transition-colors duration-150",
    rangeDisabled: "w-full h-2 bg-[var(--bg-tertiary)] rounded-lg appearance-none cursor-not-allowed accent-[var(--border-secondary)] opacity-50 transition-colors duration-150",
    checkbox: "mr-2 accent-[var(--macaron-mint)]",
    radioButton: "mr-2 accent-[var(--macaron-mint)]",

    // Typography
    label: "block text-sm font-medium text-[var(--text-secondary)] mt-1 mb-1 tracking-wide",
    heading: "text-lg font-semibold text-[var(--text-primary)] tracking-tight",
    sliderValue: "text-xs font-semibold text-[var(--text-secondary)] text-right",
    checkboxLabel: "flex items-center text-sm text-[var(--text-primary)] cursor-pointer",
    radioLabel: "inline-flex items-center mr-4 text-sm text-[var(--text-primary)] cursor-pointer",

    // Buttons
    iconButton: "p-2.5 text-[var(--text-tertiary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-tertiary)] rounded-full transition-all duration-150 hover:scale-105 active:scale-95",
    primaryButton: "px-5 py-2.5 bg-[var(--macaron-mint)] hover:bg-[var(--macaron-mint-hover)] text-[var(--text-primary)] font-medium rounded-full shadow-sm transition-all duration-150 hover:scale-103 active:scale-97 hover:shadow-md",
    modalPrimaryButton: "px-5 py-2.5 rounded-full text-sm font-semibold focus:outline-none focus:ring-2 focus:ring-offset-2 bg-[var(--macaron-mint)] hover:bg-[var(--macaron-mint-hover)] text-[#3a3530] shadow-sm transition-all duration-150 hover:scale-103 active:scale-97",
    modalSecondaryButton: "px-5 py-2.5 rounded-full text-sm font-semibold focus:outline-none focus:ring-2 focus:ring-offset-2 bg-[var(--bg-tertiary)] hover:bg-[var(--border-primary)] text-[var(--text-primary)] border border-[var(--border-primary)] transition-all duration-150 hover:scale-103 active:scale-97",

    // Layout
    card: "p-5 border border-[var(--border-primary)] rounded-3xl shadow-sm bg-[var(--bg-secondary)] transition-all duration-300 hover:shadow-md",
    cardHeader: "flex justify-between items-center mb-4 pb-2.5 border-b border-[var(--border-primary)]",

    // States
    error: "p-3 mb-4 text-sm text-red-700 dark:text-red-400 bg-[var(--macaron-coral)]/20 rounded-2xl border border-[var(--macaron-coral)]/30",
    success: "p-3 mb-4 text-sm text-green-700 dark:text-green-400 bg-[var(--macaron-mint)]/20 rounded-2xl border border-[var(--macaron-mint)]/30",
    warning: "p-3 mb-4 text-sm text-yellow-700 dark:text-yellow-400 bg-[var(--macaron-yellow)]/20 rounded-2xl border border-[var(--macaron-yellow)]/30",
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