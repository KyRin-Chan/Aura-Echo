const colors = require('tailwindcss/colors');

// Delete deprecated colors to suppress Tailwind warnings during build
delete colors['lightBlue'];
delete colors['warmGray'];
delete colors['trueGray'];
delete colors['coolGray'];
delete colors['blueGray'];

function withOpacity(variableName) {
  return ({ opacityValue }) => {
    if (opacityValue !== undefined) {
      return `rgb(var(${variableName}-rgb) / ${opacityValue})`;
    }
    return `var(${variableName})`;
  };
}

module.exports = {
  darkMode: 'class',
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    fontFamily: {
      sans: ['Comfortaa', 'Outfit', '"Noto Sans SC"', '"Microsoft YaHei"', 'sans-serif'],
    },
    colors: {
      ...colors,
      transparent: 'transparent',
      current: 'currentColor',
      // MD3 Semantic Colors mapped to CSS variables with RGB alpha support
      primary: withOpacity('--md-sys-color-primary'),
      'on-primary': withOpacity('--md-sys-color-on-primary'),
      'primary-container': withOpacity('--md-sys-color-primary-container'),
      'on-primary-container': withOpacity('--md-sys-color-on-primary-container'),
      secondary: withOpacity('--md-sys-color-secondary'),
      'on-secondary': withOpacity('--md-sys-color-on-secondary'),
      'secondary-container': withOpacity('--md-sys-color-secondary-container'),
      'on-secondary-container': withOpacity('--md-sys-color-on-secondary-container'),
      tertiary: withOpacity('--md-sys-color-tertiary'),
      'on-tertiary': withOpacity('--md-sys-color-on-tertiary'),
      'tertiary-container': withOpacity('--md-sys-color-tertiary-container'),
      'on-tertiary-container': withOpacity('--md-sys-color-on-tertiary-container'),
      error: withOpacity('--md-sys-color-error'),
      'on-error': withOpacity('--md-sys-color-on-error'),
      'error-container': withOpacity('--md-sys-color-error-container'),
      'on-error-container': withOpacity('--md-sys-color-on-error-container'),
      success: withOpacity('--md-sys-color-success'),
      'on-success': withOpacity('--md-sys-color-on-success'),
      'success-container': withOpacity('--md-sys-color-success-container'),
      'on-success-container': withOpacity('--md-sys-color-on-success-container'),
      warning: withOpacity('--md-sys-color-warning'),
      'on-warning': withOpacity('--md-sys-color-on-warning'),
      'warning-container': withOpacity('--md-sys-color-warning-container'),
      'on-warning-container': withOpacity('--md-sys-color-on-warning-container'),
      surface: withOpacity('--md-sys-color-surface'),
      'on-surface': withOpacity('--md-sys-color-on-surface'),
      'surface-variant': withOpacity('--md-sys-color-surface-variant'),
      'on-surface-variant': withOpacity('--md-sys-color-on-surface-variant'),
      outline: withOpacity('--md-sys-color-outline'),
      'outline-variant': withOpacity('--md-sys-color-outline-variant'),
      'surface-container-lowest': withOpacity('--md-sys-color-surface-container-lowest'),
      'surface-container-low': withOpacity('--md-sys-color-surface-container-low'),
      'surface-container': withOpacity('--md-sys-color-surface-container'),
      'surface-container-high': withOpacity('--md-sys-color-surface-container-high'),
      'surface-container-highest': withOpacity('--md-sys-color-surface-container-highest'),
      'inverse-surface': withOpacity('--md-sys-color-inverse-surface'),
      'inverse-on-surface': withOpacity('--md-sys-color-inverse-on-surface'),
      scrim: withOpacity('--md-sys-color-scrim'),
    },
    extend: {
      opacity: {
        '4': '0.04',
        '8': '0.08',
        '12': '0.12',
        '15': '0.15',
      },
      scale: {
        '97': '0.97',
        '98': '0.98',
        '102': '1.02',
        '103': '1.03',
      },
      borderRadius: {
        'xs': 'var(--md-sys-shape-corner-extra-small)',
        'sm': 'var(--md-sys-shape-corner-small)',
        'md': 'var(--md-sys-shape-corner-medium)',
        'lg': 'var(--md-sys-shape-corner-large)',
        'xl': 'var(--md-sys-shape-corner-extra-large)',
        'full': 'var(--md-sys-shape-corner-full)',
      },
      boxShadow: {
        'elevation-1': 'var(--md-sys-elevation-level1)',
        'elevation-2': 'var(--md-sys-elevation-level2)',
        'elevation-3': 'var(--md-sys-elevation-level3)',
        'elevation-4': 'var(--md-sys-elevation-level4)',
        'elevation-5': 'var(--md-sys-elevation-level5)',
      },
      keyframes: {
        modalFadeInScaleUp: {
          '0%': { opacity: '0', transform: 'scale(0.95) translateY(10px)' },
          '100%': { opacity: '1', transform: 'scale(1) translateY(0px)' },
        }
      },
      animation: {
        modalFadeInScaleUp: 'modalFadeInScaleUp 0.3s cubic-bezier(0.2, 0, 0, 1) forwards',
      }
    },
  },
  plugins: [],
}