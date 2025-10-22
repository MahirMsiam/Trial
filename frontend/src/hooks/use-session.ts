import apiClient from '@/lib/api-client';
import { STORAGE_KEYS, removeFromStorage } from '@/lib/utils';
import { useCallback, useEffect, useState } from 'react';

export function useSession() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  // Initialize session on mount
  useEffect(() => {
    const initializeSession = async () => {
      // Check for existing session in localStorage (stored as plain string)
      if (typeof window !== 'undefined') {
        const existingSessionId = localStorage.getItem(STORAGE_KEYS.SESSION_ID);
        
        if (existingSessionId) {
          try {
            // Verify session still exists on backend
            await apiClient.getSession(existingSessionId);
            setSessionId(existingSessionId);
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
  }, []);

  const createNewSession = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await apiClient.createSession();
      const newSessionId = response.session_id;
      
      setSessionId(newSessionId);
      // Store as plain string, not JSON
      if (typeof window !== 'undefined') {
        localStorage.setItem(STORAGE_KEYS.SESSION_ID, newSessionId);
      }
      
      console.log('New session created:', newSessionId);
    } catch (err) {
      const error = err instanceof Error ? err : new Error('Failed to create session');
      setError(error);
      console.error('Error creating session:', error);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const clearSession = useCallback(async () => {
    if (!sessionId) return;

    try {
      await apiClient.deleteSession(sessionId);
      setSessionId(null);
      removeFromStorage(STORAGE_KEYS.SESSION_ID);
      
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
      console.log('Session refreshed:', sessionId);
    } catch (err) {
      const error = err instanceof Error ? err : new Error('Failed to refresh session');
      setError(error);
      console.error('Error refreshing session:', error);
      
      // If refresh fails, create new session
      await createNewSession();
    }
  }, [sessionId, createNewSession]);

  return {
    sessionId,
    isLoading,
    error,
    createNewSession,
    clearSession,
    refreshSession,
  };
}
