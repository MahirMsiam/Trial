'use client';

import { ThemeProvider as NextThemesProvider, useTheme as useNextTheme } from 'next-themes';
import { ReactNode } from 'react';

type Theme = 'light' | 'dark';

export interface ThemeContextValue {
  theme: Theme;
  setTheme: (theme: Theme) => void;
}

/**
 * ThemeProvider component wrapping next-themes ThemeProvider
 * Manages theme state ('light' | 'dark') persisted to localStorage
 * and applies class to <html> element via attribute="class"
 */
export function ThemeProvider({ children }: { children: ReactNode }) {
  return (
    <NextThemesProvider attribute="class" defaultTheme="light" enableSystem>
      {children}
    </NextThemesProvider>
  );
}

/**
 * useTheme hook - access theme and setTheme from next-themes
 * Returns { theme, setTheme } where theme is 'light' | 'dark' (never undefined)
 */
export function useTheme(): ThemeContextValue {
  const { theme, setTheme } = useNextTheme();
  
  // Normalize undefined/system theme to 'light'
  const normalizedTheme = (theme === 'dark' ? 'dark' : 'light') as Theme;
  
  return {
    theme: normalizedTheme,
    setTheme: (newTheme: Theme) => {
      setTheme(newTheme);
    },
  };
}

