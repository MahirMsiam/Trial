import apiClient from '@/lib/api-client';
import { STORAGE_KEYS, getFromStorage, setInStorage } from '@/lib/utils';
import type {
    CaseResult,
    ChunkResponse,
    CrimeSearchResponse,
    KeywordFilters,
    SearchResultResponse,
} from '@/types/api';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useCallback, useEffect, useRef, useState } from 'react';

export type SearchMode = 'keyword' | 'semantic' | 'hybrid' | 'crime';

type SearchResults = SearchResultResponse | ChunkResponse[] | CaseResult[] | CrimeSearchResponse | null;

export function useSearch() {
  const queryClient = useQueryClient();
  const [searchMode, setSearchMode] = useState<SearchMode>('keyword');
  const [query, setQuery] = useState('');
  const [filters, setFilters] = useState<KeywordFilters>({});
  const [shouldSearch, setShouldSearch] = useState(false);
  const debounceTimerRef = useRef<NodeJS.Timeout | null>(null);

  // Keyword search query
  const keywordSearchQuery = useQuery({
    queryKey: ['search', 'keyword', query, filters],
    queryFn: () => apiClient.searchKeyword({ query, filters }),
    enabled: shouldSearch && searchMode === 'keyword',
    staleTime: 5 * 60 * 1000, // 5 minutes
  });

  // Semantic search query
  const semanticSearchQuery = useQuery({
    queryKey: ['search', 'semantic', query, filters],
    queryFn: () => apiClient.searchSemantic({ query, filters, top_k: 10 }),
    enabled: shouldSearch && searchMode === 'semantic',
    staleTime: 5 * 60 * 1000,
  });

  // Hybrid search query
  const hybridSearchQuery = useQuery({
    queryKey: ['search', 'hybrid', query, filters],
    queryFn: () => apiClient.searchHybrid({ query, filters, top_k: 10 }),
    enabled: shouldSearch && searchMode === 'hybrid',
    staleTime: 5 * 60 * 1000,
  });

  // Crime search query
  const crimeSearchQuery = useQuery({
    queryKey: ['search', 'crime', query],
    queryFn: () => apiClient.searchCrime({ query, limit: 20 }),
    enabled: shouldSearch && searchMode === 'crime',
    staleTime: 5 * 60 * 1000,
  });

  // Get current results based on mode
  const getCurrentResults = (): SearchResults => {
    switch (searchMode) {
      case 'keyword':
        return keywordSearchQuery.data || null;
      case 'semantic':
        return semanticSearchQuery.data || null;
      case 'hybrid':
        return hybridSearchQuery.data || null;
      case 'crime':
        return crimeSearchQuery.data || null;
      default:
        return null;
    }
  };

  const results = getCurrentResults();

  // Helper to convert unknown error to Error
  const toError = (err: unknown): Error => {
    if (err instanceof Error) return err;
    if (typeof err === 'string') return new Error(err);
    return new Error(JSON.stringify(err));
  };

  // Computed values
  const isLoading = 
    keywordSearchQuery.isLoading ||
    semanticSearchQuery.isLoading ||
    hybridSearchQuery.isLoading ||
    crimeSearchQuery.isLoading;

  const error: Error | null =
    (keywordSearchQuery.error && toError(keywordSearchQuery.error)) ||
    (semanticSearchQuery.error && toError(semanticSearchQuery.error)) ||
    (hybridSearchQuery.error && toError(hybridSearchQuery.error)) ||
    (crimeSearchQuery.error && toError(crimeSearchQuery.error)) ||
    null;

  const hasResults = results !== null;

  const getResultCount = (): number => {
    if (!results) return 0;
    
    if (Array.isArray(results)) {
      return results.length;
    }
    
    if ('results' in results) {
      return results.count;
    }
    
    if ('cases' in results) {
      return results.count;
    }
    
    return 0;
  };

  const resultCount = getResultCount();

  // Execute search with debounce - accepts parameters to avoid stale state
  const executeSearch = useCallback((params?: { 
    query?: string; 
    mode?: SearchMode; 
    filters?: KeywordFilters; 
    immediate?: boolean 
  }) => {
    const searchQuery = params?.query ?? query;
    const searchModeParam = params?.mode ?? searchMode;
    const searchFilters = params?.filters ?? filters;
    const immediate = params?.immediate ?? false;
    
    if (!searchQuery.trim()) return;
    
    // Update state if parameters provided
    if (params?.query !== undefined) setQuery(params.query);
    if (params?.mode !== undefined) setSearchMode(params.mode);
    if (params?.filters !== undefined) setFilters(params.filters);
    
    // Clear existing timer
    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
    }
    
    const performSearch = () => {
      setShouldSearch(true);
      
      // Save to search history
      const history = getFromStorage<string[]>(STORAGE_KEYS.SEARCH_HISTORY, []);
      const newHistory = [searchQuery, ...history.filter(q => q !== searchQuery)].slice(0, 10);
      setInStorage(STORAGE_KEYS.SEARCH_HISTORY, newHistory);
    };
    
    if (immediate) {
      performSearch();
    } else {
      // Debounce by 500ms
      debounceTimerRef.current = setTimeout(performSearch, 500);
    }
  }, [query, searchMode, filters]);

  // Clear results
  const clearResults = useCallback(() => {
    setShouldSearch(false);
    queryClient.removeQueries({ queryKey: ['search', 'keyword', query, filters] });
    queryClient.removeQueries({ queryKey: ['search', 'semantic', query, filters] });
    queryClient.removeQueries({ queryKey: ['search', 'hybrid', query, filters] });
    queryClient.removeQueries({ queryKey: ['search', 'crime', query] });
  }, [queryClient, query, filters]);

  // Clear filters
  const clearFilters = useCallback(() => {
    setFilters({});
    setInStorage(STORAGE_KEYS.FILTERS, {});
  }, []);

  // Update filters in storage when changed
  useEffect(() => {
    if (Object.keys(filters).length > 0) {
      setInStorage(STORAGE_KEYS.FILTERS, filters);
    }
  }, [filters]);

  // Load saved filters on mount
  useEffect(() => {
    const savedFilters = getFromStorage<KeywordFilters>(STORAGE_KEYS.FILTERS, {});
    if (Object.keys(savedFilters).length > 0) {
      setFilters(savedFilters);
    }
  }, []);

  // Cleanup debounce timer on unmount
  useEffect(() => {
    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }
    };
  }, []);

  return {
    searchMode,
    query,
    filters,
    results,
    isLoading,
    error,
    hasResults,
    resultCount,
    setSearchMode,
    setQuery,
    setFilters,
    executeSearch,
    clearResults,
    clearFilters,
  };
}
