/** @type {import('tailwindcss').Config} */
const colors = require('tailwindcss/colors');
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
      // Cream-white base for light theme
      cream: {
        50: '#fdfbf7',
        100: '#faf6ee',
        200: '#f5efe3',
        300: '#ede5d4',
        400: '#e0d5be',
        500: '#d4c8ab',
      },
      // Navy-blue base for dark theme
      navy: {
        50: '#e8ecf4',
        100: '#c5cde0',
        200: '#8a9bc0',
        300: '#5a6f9a',
        400: '#3a4d73',
        500: '#2a3a5c',
        600: '#1c2541',
        700: '#151d35',
        800: '#0f1629',
        900: '#0b1329',
        950: '#070d1e',
      },
      // Macaron accent palette
      macaron: {
        mint: '#a8e6cf',
        'mint-hover': '#8fd9b8',
        pink: '#ffd3b6',
        'pink-hover': '#ffbfa0',
        yellow: '#ffe49c',
        coral: '#ffaaa6',
        lavender: '#d4a5c9',
        'lavender-hover': '#c490b8',
        blue: '#aec6cf',
        'blue-hover': '#98b5bf',
      },
    },
    extend: {
      keyframes: {
        modalFadeInScaleUp: {
          '0%': { opacity: '0', transform: 'scale(0.95) translateY(10px)' },
          '100%': { opacity: '1', transform: 'scale(1) translateY(0px)' },
        }
      },
      animation: {
        modalFadeInScaleUp: 'modalFadeInScaleUp 0.3s ease-out forwards',
      }
    },
  },
  plugins: [],
}