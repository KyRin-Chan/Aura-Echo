/** @type {import('tailwindcss').Config} */
const colors = require('tailwindcss/colors');
module.exports = {
  darkMode: 'class',
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    colors: {
      ...colors,
      white: '#F5F5F5',
      slate: {
        ...colors.slate,
        50: '#F0F2F5',
        100: '#E9EDF0',
      },
      blue: {
        50: '#FAFDFB',
        100: '#E2F0D9',
        200: '#C9DAF8',
        300: '#AEC6CF',
        400: '#92B1BD',
        500: '#A8E6CF', // Macaron Mint
        600: '#92DBA7', // Macaron Mint Hover
        700: '#D291BC', // Macaron Purple
        800: '#C17CA9',
        900: '#5C3A51',
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