import { useEffect, useState } from "react"

export type AudioConfigState = {
    audioContext: AudioContext | null
}
export const useAudioConfig = (): AudioConfigState => {

    // ---------------- State ----------------
    const [audioContext, setAudioContext] = useState<AudioContext | null>(null)

    // ---------------- Hooks ----------------
    // Create AudioContext immediately on mount so signaling channels initialize,
    // and resume it on first user gesture if the browser starts it suspended.
    useEffect(() => {
        const url = new URL(window.location.href);
        const sampleRateParam = url.searchParams.get('sample_rate');
        const sampleRate = sampleRateParam ? Number(sampleRateParam) : 48000;

        let ctx: AudioContext | null = null;
        try {
            ctx = new AudioContext({ sampleRate });
            setAudioContext(ctx);
        } catch (err) {
            console.warn('Initial AudioContext creation deferred until user gesture:', err);
        }

        const unlockAudioContext = () => {
            if (!ctx) {
                try {
                    ctx = new AudioContext({ sampleRate });
                    setAudioContext(ctx);
                } catch (err) {
                    console.error('Failed to create AudioContext on user gesture:', err);
                    return;
                }
            }
            if (ctx.state === 'suspended') {
                ctx.resume()
                    .then(() => {
                        if (ctx && ctx.state === 'running') {
                            removeListeners();
                        }
                    })
                    .catch(() => {});
            } else if (ctx.state === 'running') {
                removeListeners();
            }
        };

        const removeListeners = () => {
            document.removeEventListener('touchstart', unlockAudioContext);
            document.removeEventListener('mousedown', unlockAudioContext);
            document.removeEventListener('keydown', unlockAudioContext);
        };

        document.addEventListener('touchstart', unlockAudioContext, false);
        document.addEventListener('mousedown', unlockAudioContext, false);
        document.addEventListener('keydown', unlockAudioContext, false);

        return () => {
            removeListeners();
        };
    }, []);

    return {
        audioContext
    }
}