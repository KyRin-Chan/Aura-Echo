// Material Design 3 Color Engine
// Generates MD3 tonal palettes and applies CSS variables dynamically

export interface MD3Tones {
  0: string;
  10: string;
  20: string;
  30: string;
  40: string;
  50: string;
  60: string;
  70: string;
  80: string;
  90: string;
  95: string;
  98: string;
  99: string;
  100: string;
}

export interface MD3Palettes {
  primary: MD3Tones;
  secondary: MD3Tones;
  tertiary: MD3Tones;
  neutral: MD3Tones;
  neutralVariant: MD3Tones;
  error: MD3Tones;
}

// Convert Hex to HSL
function hexToHsl(hex: string): { h: number; s: number; l: number } {
  hex = hex.replace(/^#/, '');
  if (hex.length === 3) {
    hex = hex.split('').map(char => char + char).join('');
  }
  const r = parseInt(hex.substring(0, 2), 16) / 255;
  const g = parseInt(hex.substring(2, 4), 16) / 255;
  const b = parseInt(hex.substring(4, 6), 16) / 255;

  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  let h = 0;
  let s = 0;
  const l = (max + min) / 2;

  if (max !== min) {
    const d = max - min;
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
    switch (max) {
      case r:
        h = (g - b) / d + (g < b ? 6 : 0);
        break;
      case g:
        h = (b - r) / d + 2;
        break;
      case b:
        h = (r - g) / d + 4;
        break;
    }
    h /= 6;
  }

  return {
    h: Math.round(h * 360),
    s: Math.round(s * 100),
    l: Math.round(l * 100)
  };
}

// Convert HSL to Hex
function hslToHex(h: number, s: number, l: number): string {
  s /= 100;
  l /= 100;
  const k = (n: number) => (n + h / 30) % 12;
  const a = s * Math.min(l, 1 - l);
  const f = (n: number) => {
    const u = Math.max(-1, Math.min(k(n) - 3, 9 - k(n), 1));
    return Math.round(255 * (l - a * u));
  };
  const toHex = (x: number) => {
    const hex = x.toString(16);
    return hex.length === 1 ? '0' + hex : hex;
  };
  return `#${toHex(f(0))}${toHex(f(8))}${toHex(f(4))}`;
}

// Generate MD3 Tonal Palette for a given H, S and an adjustment factor for S
function generateTonalPalette(h: number, s: number): MD3Tones {
  const tonesList = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98, 99, 100] as const;
  const tones = {} as MD3Tones;

  tonesList.forEach(t => {
    const lightness = t;
    tones[t as keyof MD3Tones] = hslToHex(h, s, lightness);
  });

  return tones;
}

// Main generation function
export function generateMD3Palettes(seedHex: string): MD3Palettes {
  const { h, s } = hexToHsl(seedHex);

  // Error palette is standard red in MD3 (H=0 or H=360, S=85)
  const errorPalette = generateTonalPalette(0, 85);

  return {
    primary: generateTonalPalette(h, s),
    // Secondary is muted (approx 1/3 saturation)
    secondary: generateTonalPalette(h, Math.max(10, Math.round(s * 0.33))),
    // Tertiary is shifted in hue (+60 deg) and slightly muted
    tertiary: generateTonalPalette((h + 60) % 360, Math.max(15, Math.round(s * 0.5))),
    // Neutral is extremely desaturated (approx 4-8% saturation)
    neutral: generateTonalPalette(h, Math.min(8, Math.max(4, Math.round(s * 0.08)))),
    // Neutral Variant is slightly more saturated grey (approx 12-16% saturation)
    neutralVariant: generateTonalPalette(h, Math.min(16, Math.max(8, Math.round(s * 0.16)))),
    error: errorPalette
  };
}

// Apply MD3 variables to the document root element
export function applyMD3Theme(palettes: MD3Palettes, mode: 'light' | 'dark') {
  const root = document.documentElement;
  const { primary, secondary, tertiary, neutral, neutralVariant, error } = palettes;

  const vars: Record<string, string> = {};

  if (mode === 'light') {
    // Generate light beige neutral and neutralVariant (Hue: 38, Saturation: 10% / 16%)
    const lightNeutral = generateTonalPalette(38, 10);
    const lightNeutralVariant = generateTonalPalette(38, 16);

    // MD3 Light theme colors
    vars['--md-sys-color-primary'] = primary[40];
    vars['--md-sys-color-on-primary'] = primary[100];
    vars['--md-sys-color-primary-container'] = primary[90];
    vars['--md-sys-color-on-primary-container'] = primary[10];

    vars['--md-sys-color-secondary'] = secondary[40];
    vars['--md-sys-color-on-secondary'] = secondary[100];
    vars['--md-sys-color-secondary-container'] = secondary[90];
    vars['--md-sys-color-on-secondary-container'] = secondary[10];

    vars['--md-sys-color-tertiary'] = tertiary[40];
    vars['--md-sys-color-on-tertiary'] = tertiary[100];
    vars['--md-sys-color-tertiary-container'] = tertiary[90];
    vars['--md-sys-color-on-tertiary-container'] = tertiary[10];

    vars['--md-sys-color-error'] = error[40];
    vars['--md-sys-color-on-error'] = error[100];
    vars['--md-sys-color-error-container'] = error[90];
    vars['--md-sys-color-on-error-container'] = error[10];

    vars['--md-sys-color-surface'] = lightNeutral[98];
    vars['--md-sys-color-on-surface'] = lightNeutral[10];
    vars['--md-sys-color-surface-variant'] = lightNeutralVariant[90];
    vars['--md-sys-color-on-surface-variant'] = lightNeutralVariant[30];

    vars['--md-sys-color-outline'] = lightNeutralVariant[50];
    vars['--md-sys-color-outline-variant'] = lightNeutralVariant[80];

    vars['--md-sys-color-surface-container-lowest'] = lightNeutral[100];
    vars['--md-sys-color-surface-container-low'] = lightNeutral[96];
    vars['--md-sys-color-surface-container'] = lightNeutral[94];
    vars['--md-sys-color-surface-container-high'] = lightNeutral[92];
    vars['--md-sys-color-surface-container-highest'] = lightNeutral[90];

    vars['--md-sys-color-inverse-surface'] = lightNeutral[20];
    vars['--md-sys-color-inverse-on-surface'] = lightNeutral[95];
    vars['--md-sys-color-scrim'] = '#000000';
  } else {
    // Generate unified dark neutral and neutralVariant (Hue: 224, Saturation: 6% / 10%)
    const darkNeutral = generateTonalPalette(224, 6);
    const darkNeutralVariant = generateTonalPalette(224, 10);

    // MD3 Dark theme colors
    vars['--md-sys-color-primary'] = primary[80];
    vars['--md-sys-color-on-primary'] = primary[20];
    vars['--md-sys-color-primary-container'] = primary[30];
    vars['--md-sys-color-on-primary-container'] = primary[90];

    vars['--md-sys-color-secondary'] = secondary[80];
    vars['--md-sys-color-on-secondary'] = secondary[20];
    vars['--md-sys-color-secondary-container'] = secondary[30];
    vars['--md-sys-color-on-secondary-container'] = secondary[90];

    vars['--md-sys-color-tertiary'] = tertiary[80];
    vars['--md-sys-color-on-tertiary'] = tertiary[20];
    vars['--md-sys-color-tertiary-container'] = tertiary[30];
    vars['--md-sys-color-on-tertiary-container'] = tertiary[90];

    vars['--md-sys-color-error'] = error[80];
    vars['--md-sys-color-on-error'] = error[20];
    vars['--md-sys-color-error-container'] = error[30];
    vars['--md-sys-color-on-error-container'] = error[90];

    vars['--md-sys-color-surface'] = darkNeutral[6];
    vars['--md-sys-color-on-surface'] = darkNeutral[90];
    vars['--md-sys-color-surface-variant'] = darkNeutralVariant[30];
    vars['--md-sys-color-on-surface-variant'] = darkNeutralVariant[80];

    vars['--md-sys-color-outline'] = darkNeutralVariant[60];
    vars['--md-sys-color-outline-variant'] = darkNeutralVariant[30];

    vars['--md-sys-color-surface-container-lowest'] = darkNeutral[4];
    vars['--md-sys-color-surface-container-low'] = darkNeutral[10];
    vars['--md-sys-color-surface-container'] = darkNeutral[12];
    vars['--md-sys-color-surface-container-high'] = darkNeutral[17];
    vars['--md-sys-color-surface-container-highest'] = darkNeutral[22];

    vars['--md-sys-color-inverse-surface'] = darkNeutral[90];
    vars['--md-sys-color-inverse-on-surface'] = darkNeutral[20];
    vars['--md-sys-color-scrim'] = '#000000';
  }

  // Set CSS variables on document element
  Object.entries(vars).forEach(([key, val]) => {
    root.style.setProperty(key, val);
  });
}

// Preset color options for configuring themes
export const PRESET_SEED_COLORS = [
  { name: 'Mint Green', hex: '#3ae1a5' },
  { name: 'Aura Violet', hex: '#b792e8' },
  { name: 'Ocean Blue', hex: '#00b4d8' },
  { name: 'Crimson Pink', hex: '#ff6083' },
  { name: 'Citron Gold', hex: '#ffd066' }
];
