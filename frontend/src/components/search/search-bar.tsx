'use client';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { SearchMode } from '@/hooks/use-search';
import { STORAGE_KEYS, getFromStorage } from '@/lib/utils';
import { Search, X } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';

interface SearchBarProps {
  onSearch: (query: string, mode: SearchMode) => void;
  initialQuery?: string;
  initialMode?: SearchMode;
  placeholder?: string;
}

export default function SearchBar({
  onSearch,
  initialQuery = '',
  initialMode = 'keyword',
  placeholder,
}: SearchBarProps) {
  const [query, setQuery] = useState(initialQuery);
  const [mode, setMode] = useState<SearchMode>(initialMode);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const suggestionsRef = useRef<HTMLDivElement>(null);

  // Check if crime search is enabled
  const crimeSearchEnabled = process.env.NEXT_PUBLIC_ENABLE_CRIME_SEARCH !== 'false';

  // Load search history for suggestions
  useEffect(() => {
    const history = getFromStorage<string[]>(STORAGE_KEYS.SEARCH_HISTORY, []);
    setSuggestions(history.slice(0, 5)); // Show last 5 queries
  }, []);

  // Close suggestions when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (suggestionsRef.current && !suggestionsRef.current.contains(e.target as Node)) {
        setShowSuggestions(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleSearch = () => {
    // Allow search with empty query if filters will be applied via FiltersPanel
    // The hook's guard (query || filters) will decide if search proceeds
    onSearch(query, mode);
    setShowSuggestions(false);
  };

  const handleSuggestionClick = (suggestion: string) => {
    setQuery(suggestion);
    setShowSuggestions(false);
    onSearch(suggestion, mode);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  const placeholderText = placeholder || {
    keyword: 'Search by case number, parties, laws...',
    semantic: 'Ask a question in natural language...',
    hybrid: 'Search using keywords or natural language...',
    crime: 'Search by crime category (murder, theft, etc.)...',
  }[mode];

  return (
    <div className="space-y-4">
      <Tabs value={mode} onValueChange={(v) => setMode(v as SearchMode)}>
        <TabsList className={`grid w-full ${crimeSearchEnabled ? 'grid-cols-4' : 'grid-cols-3'}`}>
          <TabsTrigger value="keyword">Keyword</TabsTrigger>
          <TabsTrigger value="semantic">Semantic</TabsTrigger>
          <TabsTrigger value="hybrid">Hybrid</TabsTrigger>
          {crimeSearchEnabled && <TabsTrigger value="crime">Crime</TabsTrigger>}
        </TabsList>
      </Tabs>

      <div className="flex gap-2 relative">
        <div className="relative flex-1" ref={suggestionsRef}>
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            onFocus={() => suggestions.length > 0 && setShowSuggestions(true)}
            placeholder={placeholderText}
            className="pl-10 pr-10"
          />
          {query && (
            <button
              onClick={() => setQuery('')}
              className="absolute right-3 top-1/2 -translate-y-1/2"
            >
              <X className="h-4 w-4 text-muted-foreground" />
            </button>
          )}

          {/* Suggestions Dropdown */}
          {showSuggestions && suggestions.length > 0 && (
            <div className="absolute top-full left-0 right-0 mt-1 border rounded-md bg-background shadow-md z-50">
              <div className="p-2 space-y-1">
                {suggestions.map((suggestion, idx) => (
                  <button
                    key={idx}
                    onClick={() => handleSuggestionClick(suggestion)}
                    className="w-full text-left px-3 py-2 rounded-md hover:bg-accent text-sm text-muted-foreground hover:text-foreground transition-colors"
                  >
                    <Search className="h-3 w-3 inline mr-2" />
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
        <Button onClick={handleSearch}>
          Search
        </Button>
      </div>
    </div>
  );
}
