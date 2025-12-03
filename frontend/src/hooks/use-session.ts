import apiClient from '@/lib/api-client';
import { getSessionId, removeSessionId, setSessionId as saveSessionId } from '@/lib/utils';
import { useCallback, useEffect, useRef, useState } from 'react';

export function useSession() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [isSessionExpiring, setIsSessionExpiring] = useState(false);
  const refreshTimerRef = useRef<NodeJS.Timeout | null>(null);

  // Get session timeout from environment (in seconds, default 30 minutes)
  const sessionTimeout = parseInt(
    process.env.NEXT_PUBLIC_SESSION_TIMEOUT || '1800',
    10
  );
  
  // Warning threshold: show warning 5 minutes before expiry
  const warningThreshold = 300;

  // Initialize session on mount
  useEffect(() => {
    const initializeSession = async () => {
      // Check for existing session in localStorage (stored as plain string)
      if (typeof window !== 'undefined') {
        const existingSessionId = getSessionId();
        
        if (existingSessionId) {
          try {
            // Verify session still exists on backend
            await apiClient.getSession(existingSessionId);
            setSessionId(existingSessionId);
            // Start refresh timer for this session
            startSessionRefreshTimer(existingSessionId);
          } catch (err) {
            // Session expired or invalid, create new one
            console.log('Existing session invalid, creating new session');
            await createNewSession();
          }
        } else {
          // No existing session, create new one
          await createNewSession();
        }
      }
    };

    initializeSession();

    return () => {
      // Cleanup timer on unmount
      if (refreshTimerRef.current) {
        clearTimeout(refreshTimerRef.current);
      }
    };
  }, []);

  const createNewSession = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await apiClient.createSession();
      const newSessionId = response.session_id;
      
      setSessionId(newSessionId);
      // Store as plain string using dedicated helper
      if (typeof window !== 'undefined') {
        saveSessionId(newSessionId);
      }
      
      // Start refresh timer for new session
      startSessionRefreshTimer(newSessionId);
      
      console.log('New session created:', newSessionId);
    } catch (err) {
      const error = err instanceof Error ? err : new Error('Failed to create session');
      setError(error);
      console.error('Error creating session:', error);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const startSessionRefreshTimer = useCallback((sid: string) => {
    // Clear existing timer
    if (refreshTimerRef.current) {
      clearTimeout(refreshTimerRef.current);
    }

    // Set timer to refresh before expiry, with warning threshold
    const refreshTime = (sessionTimeout - warningThreshold) * 1000; // Refresh 5 min before expiry
    
    refreshTimerRef.current = setTimeout(() => {
      setIsSessionExpiring(true);
      console.warn('Session will expire in 5 minutes');
      
      // Auto-refresh session
      refreshSession();
    }, refreshTime);
  }, [sessionTimeout]);

  const clearSession = useCallback(async () => {
    if (!sessionId) return;

    try {
      await apiClient.deleteSession(sessionId);
      setSessionId(null);
      removeSessionId();
      setIsSessionExpiring(false);
      
      // Clear timer
      if (refreshTimerRef.current) {
        clearTimeout(refreshTimerRef.current);
      }
      
      // Create a new session immediately
      await createNewSession();
    } catch (err) {
      const error = err instanceof Error ? err : new Error('Failed to clear session');
      setError(error);
      console.error('Error clearing session:', error);
    }
  }, [sessionId, createNewSession]);

  const refreshSession = useCallback(async () => {
    if (!sessionId) return;

    try {
      await apiClient.getSession(sessionId);
      setIsSessionExpiring(false);
      
      // Restart the refresh timer
      startSessionRefreshTimer(sessionId);
      
      console.log('Session refreshed:', sessionId);
    } catch (err) {
      const error = err instanceof Error ? err : new Error('Failed to refresh session');
      setError(error);
      console.error('Error refreshing session:', error);
      
      // If refresh fails, create new session
      await createNewSession();
    }
  }, [sessionId, createNewSession, startSessionRefreshTimer]);

  return {
    sessionId,
    isLoading,
    error,
    isSessionExpiring,
    createNewSession,
    clearSession,
    refreshSession,
  };
}
