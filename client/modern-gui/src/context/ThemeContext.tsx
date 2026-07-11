import React, { createContext, useState, useEffect, ReactNode, useContext } from 'react';
import { generateMD3Palettes, applyMD3Theme } from '../styles/md3ColorEngine';

interface ThemeContextType {
  theme: string;
  toggleTheme: () => void;
  seedColor: string;
  setSeedColor: (color: string) => void;
}

interface ThemeProviderProps {
  children: ReactNode;
}

// Create context
export const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

// Create hook
export const useThemeContext = (): ThemeContextType => {
  const state = useContext(ThemeContext);
  if (!state) {
    throw new Error("useThemeContext must be used within ThemeProvider");
  }
  return state;
};

// Create provider
export const ThemeProvider: React.FC<ThemeProviderProps> = ({ children }) => {
  const [theme, setTheme] = useState<string>(() => {
    const storedTheme = localStorage.getItem('theme');
    if (storedTheme) {
      return storedTheme;
    }
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
      return 'dark';
    }
    return 'dark'; // Default to dark theme instead of light
  });

  const [seedColor, setSeedColorState] = useState<string>(() => {
    return localStorage.getItem('seedColor') || '#3ae1a5'; // Default to mint green
  });

  useEffect(() => {
    localStorage.setItem('theme', theme);
    if (theme === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [theme]);

  // Apply MD3 palettes whenever theme or seedColor changes
  useEffect(() => {
    const palettes = generateMD3Palettes(seedColor);
    applyMD3Theme(palettes, theme as 'light' | 'dark');
  }, [theme, seedColor]);

  const toggleTheme = () => {
    setTheme(prevTheme => (prevTheme === 'light' ? 'dark' : 'light'));
  };

  const setSeedColor = (color: string) => {
    setSeedColorState(color);
    localStorage.setItem('seedColor', color);
  };

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme, seedColor, setSeedColor }}>
      {children}
    </ThemeContext.Provider>
  );
}; 