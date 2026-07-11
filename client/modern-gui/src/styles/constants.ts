export const CSS_CLASSES = {
    // Form Controls
    select: "w-full px-3 py-2 bg-surface-container border border-outline rounded-xs focus:outline-none focus:border-primary text-on-surface text-sm transition-all duration-200",
    input: "w-full px-3 py-2 bg-surface-container border border-outline rounded-xs focus:outline-none focus:border-primary text-on-surface text-sm transition-all duration-200",
    fileInput: "w-full px-3 py-2 bg-surface-container border border-outline rounded-xs focus:outline-none focus:border-primary text-on-surface text-sm file:mr-4 file:py-1.5 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-primary-container file:text-on-primary-container hover:file:opacity-90 transition-all",
    range: "w-full h-1 bg-surface-container-highest rounded-full appearance-none cursor-pointer accent-primary transition-colors duration-150",
    rangeDisabled: "w-full h-1 bg-surface-container rounded-full appearance-none cursor-not-allowed accent-outline opacity-50 transition-colors duration-150",
    checkbox: "mr-2 accent-primary cursor-pointer",
    radioButton: "mr-2 accent-primary cursor-pointer",

    // Typography
    label: "block text-sm font-medium text-on-surface-variant mb-1 tracking-wide",
    heading: "text-lg font-semibold text-on-surface tracking-tight",
    sliderValue: "text-xs font-semibold text-on-surface-variant text-right",
    checkboxLabel: "flex items-center text-sm text-on-surface cursor-pointer",
    radioLabel: "inline-flex items-center mr-4 text-sm text-on-surface cursor-pointer",

    // Buttons
    iconButton: "p-2.5 text-on-surface-variant hover:text-on-surface hover:bg-surface-variant/20 rounded-full transition-all duration-150 active:scale-95",
    primaryButton: "px-6 py-2 bg-primary hover:bg-primary/90 text-on-primary font-semibold rounded-full shadow-elevation-1 hover:shadow-elevation-2 active:shadow-elevation-1 transition-all duration-150",
    modalPrimaryButton: "px-6 py-2.5 bg-primary hover:bg-primary/90 text-on-primary font-semibold rounded-full shadow-elevation-1 hover:shadow-elevation-2 active:shadow-elevation-1 transition-all duration-150",
    modalSecondaryButton: "px-6 py-2.5 bg-surface-container-high text-primary border border-outline font-semibold rounded-full transition-all duration-150 hover:bg-primary/8 active:scale-97",

    // Layout
    card: "p-4 bg-surface-container-low rounded-md shadow-elevation-1 transition-all duration-300 hover:shadow-elevation-2",
    cardHeader: "flex justify-between items-center mb-4 pb-2.5 border-b border-outline-variant",

    // States
    error: "p-3 mb-4 text-sm text-error bg-error-container/20 rounded-md border border-error/30",
    success: "p-3 mb-4 text-sm text-primary bg-primary-container/20 rounded-md border border-primary/30",
    warning: "p-3 mb-4 text-sm text-tertiary bg-tertiary-container/20 rounded-md border border-tertiary/30",
    loading: "text-primary animate-spin"
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