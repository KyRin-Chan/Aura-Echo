const colors = require('tailwindcss/colors');

// Delete deprecated colors to suppress Tailwind warnings during build
delete colors['lightBlue'];
delete colors['warmGray'];
delete colors['trueGray'];
delete colors['coolGray'];
delete colors['blueGray'];

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
      // MD3 Semantic Colors mapped to CSS variables
      primary: 'var(--md-sys-color-primary)',
      'on-primary': 'var(--md-sys-color-on-primary)',
      'primary-container': 'var(--md-sys-color-primary-container)',
      'on-primary-container': 'var(--md-sys-color-on-primary-container)',
      secondary: 'var(--md-sys-color-secondary)',
      'on-secondary': 'var(--md-sys-color-on-secondary)',
      'secondary-container': 'var(--md-sys-color-secondary-container)',
      'on-secondary-container': 'var(--md-sys-color-on-secondary-container)',
      tertiary: 'var(--md-sys-color-tertiary)',
      'on-tertiary': 'var(--md-sys-color-on-tertiary)',
      'tertiary-container': 'var(--md-sys-color-tertiary-container)',
      'on-tertiary-container': 'var(--md-sys-color-on-tertiary-container)',
      error: 'var(--md-sys-color-error)',
      'on-error': 'var(--md-sys-color-on-error)',
      'error-container': 'var(--md-sys-color-error-container)',
      'on-error-container': 'var(--md-sys-color-on-error-container)',
      surface: 'var(--md-sys-color-surface)',
      'on-surface': 'var(--md-sys-color-on-surface)',
      'surface-variant': 'var(--md-sys-color-surface-variant)',
      'on-surface-variant': 'var(--md-sys-color-on-surface-variant)',
      outline: 'var(--md-sys-color-outline)',
      'outline-variant': 'var(--md-sys-color-outline-variant)',
      'surface-container-lowest': 'var(--md-sys-color-surface-container-lowest)',
      'surface-container-low': 'var(--md-sys-color-surface-container-low)',
      'surface-container': 'var(--md-sys-color-surface-container)',
      'surface-container-high': 'var(--md-sys-color-surface-container-high)',
      'surface-container-highest': 'var(--md-sys-color-surface-container-highest)',
      'inverse-surface': 'var(--md-sys-color-inverse-surface)',
      'inverse-on-surface': 'var(--md-sys-color-inverse-on-surface)',
      scrim: 'var(--md-sys-color-scrim)',
    },
    extend: {
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