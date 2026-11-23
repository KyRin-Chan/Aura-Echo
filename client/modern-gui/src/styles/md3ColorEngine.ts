// Material Design 3 Color Engine
// Generates MD3 tonal palettes and applies CSS variables dynamically

export interface MD3Tones {
  0: string;
  4: string;
  6: string;
  10: string;
  12: string;
  15: string;
  17: string;
  20: string;
  22: string;
  30: string;
  40: string;
  50: string;
  60: string;
  70: string;
  80: string;
  90: string;
  92: string;
  94: string;
  95: string;
  96: string;
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

// Convert Hex to space-separated RGB channels ("R G B") for Tailwind alpha modifiers
function hexToRgbChannels(hex: string): string {
  let clean = hex.replace(/^#/, '');
  if (clean.length === 3) {
    clean = clean.split('').map(char => char + char).join('');
  }
  const r = parseInt(clean.substring(0, 2), 16) || 0;
  const g = parseInt(clean.substring(2, 4), 16) || 0;
  const b = parseInt(clean.substring(4, 6), 16) || 0;
  return `${r} ${g} ${b}`;
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

// Generate MD3 Tonal Palette for a given H, S and optional mid-tone chroma boost
function generateTonalPalette(h: number, s: number, boostMidTones = false): MD3Tones {
  const tonesList = [0, 4, 6, 10, 12, 15, 17, 20, 22, 30, 40, 50, 60, 70, 80, 90, 92, 94, 95, 96, 98, 99, 100] as const;
  const tones = {} as Record<number, string>;

  tonesList.forEach(t => {
    let toneSat = s;
    if (boostMidTones) {
      if (t >= 30 && t <= 70) {
        if (s >= 5 && s < 48) {
          toneSat = Math.max(s, Math.min(48, Math.round(Math.max(s * 1.35, 24))));
        } else if (s > 65) {
          // Keep ultra-light pastel seeds (e.g. #f9bfc6) from turning overly vivid in mid tones
          toneSat = Math.round(58 + (s - 65) * 0.25);
        }
      } else if (t === 80 && s >= 5 && s < 48) {
        toneSat = Math.max(s, Math.min(42, Math.round(Math.max(s * 1.25, 22))));
      }
    }
    tones[t] = hslToHex(h, toneSat, t);
  });

  return tones as unknown as MD3Tones;
}

// Main generation function
export function generateMD3Palettes(seedHex: string): MD3Palettes {
  const { h, s } = hexToHsl(seedHex);

  // Error palette follows MD3 warm red (H=4, S=72)
  const errorPalette = generateTonalPalette(4, 72);

  // Tertiary hue rotation: avoid muddy yellow-olive zones (45°-80°)
  let tertiaryHue = (h + 60) % 360;
  if (tertiaryHue >= 45 && tertiaryHue <= 80) {
    tertiaryHue = (h - 45 + 360) % 360;
  }

  return {
    primary: generateTonalPalette(h, s, true),
    // Secondary is muted (approx 1/3 saturation, kept expressive enough for chips/badges)
    secondary: generateTonalPalette(h, Math.max(14, Math.min(36, Math.round(s * 0.38)))),
    // Tertiary is shifted in hue and balanced in saturation
    tertiary: generateTonalPalette(tertiaryHue, Math.max(24, Math.min(55, Math.round(s * 0.65)))),
    // Neutral is derived from seed hue with subtle chroma
    neutral: generateTonalPalette(h, Math.min(10, Math.max(4, Math.round(s * 0.12)))),
    // Neutral Variant is slightly more saturated for outlines and variant containers
    neutralVariant: generateTonalPalette(h, Math.min(18, Math.max(8, Math.round(s * 0.22)))),
    error: errorPalette
  };
}

// Apply MD3 variables to the document root element
export function applyMD3Theme(palettes: MD3Palettes, mode: 'light' | 'dark', seedHex?: string) {
  const root = document.documentElement;
  const { primary, secondary, tertiary, neutral, neutralVariant, error } = palettes;
  const { h, s } = hexToHsl(seedHex || primary[50]);

  const vars: Record<string, string> = {};

  if (mode === 'light') {
    const lightNeutral = neutral;
    const lightNeutralVariant = neutralVariant;

    // MD3 Light theme colors
    if (seedHex === '#f9bfc6') {
      // Sakura Snow: keep soft sakura pink containers while ensuring WCAG AA contrast for primary text/icons
      vars['--md-sys-color-primary'] = hslToHex(h, Math.min(62, s), 46);
      vars['--md-sys-color-on-primary'] = '#ffffff';
      vars['--md-sys-color-primary-container'] = primary[90];
      vars['--md-sys-color-on-primary-container'] = primary[15];
    } else {
      vars['--md-sys-color-primary'] = primary[40];
      vars['--md-sys-color-on-primary'] = primary[100];
      vars['--md-sys-color-primary-container'] = primary[90];
      vars['--md-sys-color-on-primary-container'] = primary[10];
    }

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

    vars['--md-sys-color-success'] = hslToHex(152, 62, 32);
    vars['--md-sys-color-on-success'] = '#ffffff';
    vars['--md-sys-color-success-container'] = hslToHex(152, 48, 90);
    vars['--md-sys-color-on-success-container'] = hslToHex(152, 60, 12);

    vars['--md-sys-color-warning'] = hslToHex(35, 85, 35);
    vars['--md-sys-color-on-warning'] = '#ffffff';
    vars['--md-sys-color-warning-container'] = hslToHex(38, 85, 90);
    vars['--md-sys-color-on-warning-container'] = hslToHex(35, 85, 14);

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
    // Derive dark neutral and neutralVariant from the seed hue at controlled low saturation
    const darkNeutral = generateTonalPalette(h, Math.min(10, Math.max(5, Math.round(s * 0.14))));
    const darkNeutralVariant = generateTonalPalette(h, Math.min(18, Math.max(9, Math.round(s * 0.24))));

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

    vars['--md-sys-color-success'] = hslToHex(152, 62, 64);
    vars['--md-sys-color-on-success'] = hslToHex(152, 60, 14);
    vars['--md-sys-color-success-container'] = hslToHex(152, 45, 24);
    vars['--md-sys-color-on-success-container'] = hslToHex(152, 60, 88);

    vars['--md-sys-color-warning'] = hslToHex(42, 92, 68);
    vars['--md-sys-color-on-warning'] = hslToHex(35, 80, 16);
    vars['--md-sys-color-warning-container'] = hslToHex(38, 70, 26);
    vars['--md-sys-color-on-warning-container'] = hslToHex(42, 90, 90);

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

  // Set CSS variables and RGB channels on document element
  Object.entries(vars).forEach(([key, val]) => {
    root.style.setProperty(key, val);
    root.style.setProperty(`${key}-rgb`, hexToRgbChannels(val));
  });
}

// Preset color options for configuring themes
export interface PresetColor {
  name: string;
  hex: string;
  langKey: 'colorSageGreen' | 'colorLavenderWisteria' | 'colorSlateBlue' | 'colorDustyRose' | 'colorSoftSand' | 'colorSakuraSnow';
}

export const PRESET_SEED_COLORS: PresetColor[] = [
  { name: 'Sage Green', hex: '#8fa89c', langKey: 'colorSageGreen' },
  { name: 'Lavender Wisteria', hex: '#b5a6c9', langKey: 'colorLavenderWisteria' },
  { name: 'Slate Blue', hex: '#8da9c4', langKey: 'colorSlateBlue' },
  { name: 'Dusty Rose', hex: '#d9a0a0', langKey: 'colorDustyRose' },
  { name: 'Soft Sand', hex: '#d6b785', langKey: 'colorSoftSand' },
  { name: 'Sakura Snow', hex: '#f9bfc6', langKey: 'colorSakuraSnow' }
];
