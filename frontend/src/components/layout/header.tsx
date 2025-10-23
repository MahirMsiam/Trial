'use client';

import { useTheme } from '@/components/theme-provider';
import { Button } from '@/components/ui/button';
import apiClient from '@/lib/api-client';
import { Moon, Scale, Sun } from 'lucide-react';
import Link from 'next/link';
import { useEffect, useState } from 'react';

interface HeaderProps {
  currentPage: 'search' | 'chat' | 'about';
}

export default function Header({ currentPage }: HeaderProps) {
  const [isHealthy, setIsHealthy] = useState<boolean | null>(null);
  const { theme, setTheme } = useTheme();

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const health = await apiClient.getHealth();
        const statusLower = health.status?.toLowerCase?.();
        setIsHealthy(
          statusLower === 'healthy' || 
          statusLower === 'ok' || 
          health.database_connected === true
        );
      } catch {
        setIsHealthy(false);
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 60000); // Check every minute
    return () => clearInterval(interval);
  }, []);

  const toggleTheme = () => {
    setTheme(theme === 'dark' ? 'light' : 'dark');
  };

  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-16 items-center">
        <Link href="/" className="flex items-center space-x-2">
          <Scale className="h-6 w-6" />
          <span className="font-bold">BD Supreme Court</span>
          {isHealthy !== null && (
            <span 
              className={`ml-2 h-2 w-2 rounded-full ${
                isHealthy ? 'bg-green-500' : 'bg-red-500'
              }`}
              title={isHealthy ? 'API Healthy' : 'API Down'}
            />
          )}
        </Link>

        <nav className="ml-auto flex items-center space-x-2">
          <Link href="/">
            <Button variant={currentPage === 'search' ? 'default' : 'ghost'}>
              Search
            </Button>
          </Link>
          <Link href="/chat">
            <Button variant={currentPage === 'chat' ? 'default' : 'ghost'}>
              Chat
            </Button>
          </Link>
          <Button variant="ghost" size="icon" onClick={toggleTheme}>
            {theme === 'dark' ? (
              <Sun className="h-5 w-5" />
            ) : (
              <Moon className="h-5 w-5" />
            )}
            <span className="sr-only">Toggle theme</span>
          </Button>
        </nav>
      </div>
    </header>
  );
}
