'use client';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { SearchMode } from '@/hooks/use-search';
import { Search, X } from 'lucide-react';
import { useState } from 'react';

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

  // Check if crime search is enabled
  const crimeSearchEnabled = process.env.NEXT_PUBLIC_ENABLE_CRIME_SEARCH !== 'false';

  const handleSearch = () => {
    if (query.trim()) {
      onSearch(query, mode);
    }
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

      <div className="flex gap-2">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
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
        </div>
        <Button onClick={handleSearch} disabled={!query.trim()}>
          Search
        </Button>
      </div>
    </div>
  );
}
