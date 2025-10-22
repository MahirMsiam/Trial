import { type ClassValue, clsx } from 'clsx';
import { format, formatDistanceToNow } from 'date-fns';
import { twMerge } from 'tailwind-merge';

// Class Name Utilities
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// Date Formatting
export function formatDate(date?: string | Date | null): string {
  if (!date) return 'N/A';
  
  try {
    const d = typeof date === 'string' ? new Date(date) : date;
    if (isNaN(d.getTime())) return 'N/A';
    return format(d, 'MMM dd, yyyy');
  } catch {
    return 'N/A';
  }
}

export function formatDateTime(date?: string | Date | null): string {
  if (!date) return 'N/A';
  
  try {
    const d = typeof date === 'string' ? new Date(date) : date;
    if (isNaN(d.getTime())) return 'N/A';
    return format(d, 'MMM dd, yyyy hh:mm a');
  } catch {
    return 'N/A';
  }
}

export function formatRelativeTime(date?: string | Date | null): string {
  if (!date) return 'N/A';
  
  try {
    const d = typeof date === 'string' ? new Date(date) : date;
    if (isNaN(d.getTime())) return 'N/A';
    return formatDistanceToNow(d, { addSuffix: true });
  } catch {
    return 'N/A';
  }
}

// Text Utilities
export function truncate(text: string, maxLength: number): string {
  if (text.length <= maxLength) return text;
  return text.slice(0, maxLength) + '...';
}

function escapeRegExp(str: string): string {
  return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

export function highlightText(text: string, query: string): string {
  if (!query) return text;
  
  const escapedQuery = escapeRegExp(query);
  const regex = new RegExp(`(${escapedQuery})`, 'gi');
  return text.replace(regex, '<mark class="bg-yellow-200 dark:bg-yellow-800">$1</mark>');
}

export function extractExcerpt(text: string, query: string, contextLength: number = 150): string {
  if (!query || !text) return truncate(text, contextLength * 2);
  
  const lowerText = text.toLowerCase();
  const lowerQuery = query.toLowerCase();
  const index = lowerText.indexOf(lowerQuery);
  
  if (index === -1) return truncate(text, contextLength * 2);
  
  const start = Math.max(0, index - contextLength);
  const end = Math.min(text.length, index + query.length + contextLength);
  
  let excerpt = text.slice(start, end);
  
  if (start > 0) excerpt = '...' + excerpt;
  if (end < text.length) excerpt = excerpt + '...';
  
  return excerpt;
}

// Case Utilities
export function formatCaseId(caseNumber?: string, caseYear?: string, caseType?: string): string {
  if (!caseNumber && !caseYear && !caseType) return 'N/A';
  
  const parts = [];
  if (caseType) parts.push(caseType);
  if (caseNumber) parts.push(caseNumber);
  if (caseYear) parts.push(caseYear);
  
  return parts.join(' ');
}

export function getCaseTypeColor(caseType?: string | null): string {
  if (!caseType) return 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200';
  
  const type = caseType.toLowerCase();
  
  if (type.includes('criminal')) return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200';
  if (type.includes('civil')) return 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200';
  if (type.includes('writ')) return 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200';
  if (type.includes('revision')) return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200';
  if (type.includes('appeal')) return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200';
  
  return 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200';
}

export function getOutcomeColor(outcome?: string | null): string {
  if (!outcome) return 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200';
  
  const o = outcome.toLowerCase();
  
  if (o.includes('allow') || o.includes('granted') || o.includes('absolute')) {
    return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200';
  }
  if (o.includes('dismiss') || o.includes('reject')) {
    return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200';
  }
  if (o.includes('discharge')) {
    return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200';
  }
  if (o.includes('disposed')) {
    return 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200';
  }
  
  return 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200';
}

// Validation
export function isValidSessionId(sessionId: string): boolean {
  return sessionId.length > 0 && sessionId.length <= 100;
}

export function isValidCaseId(caseId: number): boolean {
  return Number.isInteger(caseId) && caseId > 0;
}

// Storage Utilities
export const STORAGE_KEYS = {
  SESSION_ID: 'session_id',
  SEARCH_HISTORY: 'search_history',
  FILTERS: 'filters',
  THEME: 'theme',
} as const;

export function getFromStorage<T>(key: string, defaultValue: T): T {
  if (typeof window === 'undefined') return defaultValue;
  
  try {
    const item = localStorage.getItem(key);
    return item ? JSON.parse(item) : defaultValue;
  } catch (error) {
    console.error(`Error reading from localStorage:`, error);
    return defaultValue;
  }
}

export function setInStorage(key: string, value: any): void {
  if (typeof window === 'undefined') return;
  
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch (error) {
    console.error(`Error writing to localStorage:`, error);
  }
}

export function removeFromStorage(key: string): void {
  if (typeof window === 'undefined') return;
  
  try {
    localStorage.removeItem(key);
  } catch (error) {
    console.error(`Error removing from localStorage:`, error);
  }
}
