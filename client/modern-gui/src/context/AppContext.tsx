import { ClientState, PerformanceStats } from "@dannadori/voice-changer-client-js";
import React, { useContext, useEffect, useMemo, useRef, useState } from "react";
import { ReactNode } from "react";
import { useVCClient } from "../scripts/useVCClient";
import { useAppRoot } from "./AppRootProvider";

type Props = {
    children: ReactNode;
};

export type AppContextValue = ClientState & {
    audioContext: AudioContext;
    initializedRef: React.MutableRefObject<boolean>;
};

const PERFORMANCE_THROTTLE_MS = 100;

const DEFAULT_PERFORMANCE: PerformanceStats = {
    vol: 0,
    responseTime: 0,
    preprocessTime: 0,
    mainprocessTime: 0,
    postprocessTime: 0,
};

// Create context
const AppStateContext = React.createContext<AppContextValue | null>(null);
const PerformanceContext = React.createContext<PerformanceStats>(DEFAULT_PERFORMANCE);

// Create hook
export const useAppState = (): AppContextValue => {
    const state = useContext(AppStateContext);
    if (!state) {
        throw new Error("useAppState must be used within AppContextProvider");
    }
    return state;
};

export const usePerformance = (): PerformanceStats => {
    return useContext(PerformanceContext);
};

// Create provider
export const AppContextProvider = ({ children }: Props) => {
    // Get audio context and setup client
    const appRoot = useAppRoot();
    const clientState = useVCClient({ audioContext: appRoot.audioContextState.audioContext });
    const initializedRef = useRef<boolean>(false);

    const cs = clientState.clientState;
    const rawPerformance = cs.performance;
    const [throttledPerformance, setThrottledPerformance] = useState<PerformanceStats>(DEFAULT_PERFORMANCE);
    const lastPerfUpdateRef = useRef<number>(0);
    const trailingPerfTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
    const latestPerformanceRef = useRef<PerformanceStats>(rawPerformance);
    latestPerformanceRef.current = rawPerformance;

    // Set initialized flag of the client
    useEffect(() => {
        if (cs.initialized) {
            initializedRef.current = true;
            cs.getInfo();
        }
    }, [cs.initialized]);

    // Throttle high-frequency audio performance callbacks before exposing to UI consumers
    useEffect(() => {
        const isReset =
            rawPerformance.vol === 0 &&
            rawPerformance.responseTime === 0 &&
            rawPerformance.preprocessTime === 0 &&
            rawPerformance.mainprocessTime === 0 &&
            rawPerformance.postprocessTime === 0;

        if (isReset) {
            if (trailingPerfTimerRef.current) {
                clearTimeout(trailingPerfTimerRef.current);
                trailingPerfTimerRef.current = null;
            }
            lastPerfUpdateRef.current = Date.now();
            setThrottledPerformance(DEFAULT_PERFORMANCE);
            return;
        }

        const now = Date.now();
        const elapsed = now - lastPerfUpdateRef.current;

        if (elapsed >= PERFORMANCE_THROTTLE_MS) {
            if (trailingPerfTimerRef.current) {
                clearTimeout(trailingPerfTimerRef.current);
                trailingPerfTimerRef.current = null;
            }
            lastPerfUpdateRef.current = now;
            setThrottledPerformance(rawPerformance);
        } else if (!trailingPerfTimerRef.current) {
            trailingPerfTimerRef.current = setTimeout(() => {
                trailingPerfTimerRef.current = null;
                lastPerfUpdateRef.current = Date.now();
                setThrottledPerformance(latestPerformanceRef.current);
            }, PERFORMANCE_THROTTLE_MS - elapsed);
        }
    }, [rawPerformance]);

    useEffect(() => {
        return () => {
            if (trailingPerfTimerRef.current) {
                clearTimeout(trailingPerfTimerRef.current);
            }
        };
    }, []);

    // Provide app state (excluding live performance to avoid re-rendering unrelated consumers)
    const providerValue = useMemo<AppContextValue>(
        () => ({
            audioContext: appRoot.audioContextState.audioContext!,
            ...cs,
            performance: DEFAULT_PERFORMANCE,
            initializedRef,
        }),
        [
            appRoot.audioContextState.audioContext,
            cs.initialized,
            cs.setting,
            cs.setVoiceChangerClientSetting,
            cs.setServerUrl,
            cs.start,
            cs.stop,
            cs.reloadClientSetting,
            cs.setWorkletNodeSetting,
            cs.startOutputRecording,
            cs.stopOutputRecording,
            cs.trancateBuffer,
            cs.setWorkletSetting,
            cs.serverSetting,
            cs.indexedDBState,
            cs.bufferingTime,
            cs.getInfo,
            cs.clearSetting,
            cs.setAudioOutputElementId,
            cs.setAudioMonitorElementId,
            cs.errorMessage,
            cs.resetErrorMessage,
            cs.popLogs,
            cs.clearPopLogs,
        ]
    );

    return (
        <AppStateContext.Provider value={providerValue}>
            <PerformanceContext.Provider value={throttledPerformance}>
                {children}
            </PerformanceContext.Provider>
        </AppStateContext.Provider>
    );
};