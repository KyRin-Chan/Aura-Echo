import { useMemo } from 'react';

// Placeholder for pictures (model icons)
export const useInitialPlaceholder = (name: string, options?: { size?: number; backgroundColor?: string; textColor?: string; shape?: string; fontSize?: number; fontFamily?: string }): string => {
    // ---------------- Hooks ---------------- 
    return useMemo(() => {
        // Default settings
        const {
            size = 100,
            backgroundColor = null,
            textColor = '#ffffff',
            shape = 'rounded',
            fontSize = null,
            fontFamily = 'Comfortaa, Outfit, "Noto Sans SC", "Microsoft YaHei", sans-serif'
        } = options || {};

        // Use initial letter for placeholder
        const initial = name.charAt(0).toUpperCase();

        // Auto-generate background color if not specified
        let bgColor = backgroundColor;
        if (!bgColor) {
            const hash = name.split('').reduce((a, b) => {
                a = ((a << 5) - a) + b.charCodeAt(0);
                return a & a;
            }, 0);
            const colors = [
                '#6750A4', // MD3 Purple
                '#625B71', // MD3 Secondary
                '#7D5260', // MD3 Tertiary
                '#386A20', // MD3 Green
                '#00639B', // MD3 Blue
                '#9C423E', // MD3 Red
                '#8F4E00', // MD3 Orange
                '#7F5600'  // MD3 Amber
            ];
            bgColor = colors[Math.abs(hash) % colors.length];
        }

        // Calculate font size if not specified
        const calculatedFontSize = fontSize || size * 0.4;

        // Return SVG as data URL
        const svgString = `
            <svg width="${size}" height="${size}" xmlns="http://www.w3.org/2000/svg">
                <defs>
                    <clipPath id="clip">
                        <rect width="${size}" height="${size}" rx="${shape === 'circle' ? size / 2 : (shape === 'rounded' ? size * 0.1 : 0)}" ry="${shape === 'circle' ? size / 2 : (shape === 'rounded' ? size * 0.1 : 0)}"/>
                    </clipPath>
                </defs>
                <rect width="100%" height="100%" fill="${bgColor}" clip-path="url(#clip)"/>
                <text x="50%" y="50%" font-family="${fontFamily}" font-size="${calculatedFontSize}" 
                      fill="${textColor}" text-anchor="middle" dy=".3em" clip-path="url(#clip)">${initial}</text>
            </svg>
        `;
        return `data:image/svg+xml;base64,${btoa(unescape(encodeURIComponent(svgString.trim())))}`;
    }, [name, options]);
};