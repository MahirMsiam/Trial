'use client';

import CaseDetailsModal from '@/components/case/case-details-modal';
import MainLayout from '@/components/layout/main-layout';
import CrimeCategories from '@/components/search/crime-categories';
import FiltersPanel from '@/components/search/filters-panel';
import SearchBar from '@/components/search/search-bar';
import SearchResults from '@/components/search/search-results';
import { Button } from '@/components/ui/button';
import { useSearch } from '@/hooks/use-search';
import { useSession } from '@/hooks/use-session';
import { CrimeCategory } from '@/types/crime';
import { Filter } from 'lucide-react';
import { useState } from 'react';

export default function HomePage() {
  const { sessionId } = useSession();
  const search = useSearch();
  const [selectedCaseId, setSelectedCaseId] = useState<number | null>(null);
  const [isFiltersOpen, setIsFiltersOpen] = useState(false);
  const [selectedCrime, setSelectedCrime] = useState<CrimeCategory | null>(null);

  // Check if crime search is enabled
  const crimeSearchEnabled = process.env.NEXT_PUBLIC_ENABLE_CRIME_SEARCH !== 'false';

  return (
    <MainLayout currentPage="search">
      <div className="max-w-7xl mx-auto space-y-8">
        {/* Hero Section */}
        <div className="text-center space-y-4 py-8">
          <h1 className="text-4xl font-bold tracking-tight">
            Bangladesh Supreme Court Legal Research
          </h1>
          <p className="text-lg text-muted-foreground">
            Search and analyze 8000+ legal judgments using AI
          </p>
        </div>

        {/* Search Section */}
        <div className="flex gap-2 items-start">
          <div className="flex-1">
            <SearchBar
              onSearch={(query, mode) => {
                // Prevent crime mode if disabled
                const searchMode = (!crimeSearchEnabled && mode === 'crime') ? 'keyword' : mode;
                search.executeSearch({ query, mode: searchMode, immediate: true });
              }}
              initialQuery={search.query}
              initialMode={search.searchMode}
            />
          </div>
          <Button
            variant="outline"
            size="lg"
            onClick={() => setIsFiltersOpen(true)}
            className="mt-0"
          >
            <Filter className="h-4 w-4 mr-2" />
            Filters
          </Button>
        </div>

        {/* Crime Categories - show when no results and feature is enabled */}
        {crimeSearchEnabled && !search.hasResults && !search.isLoading && (
          <CrimeCategories
            selectedCategory={selectedCrime}
            onCategorySelect={(category) => {
              setSelectedCrime(category);
              if (category) {
                search.executeSearch({ 
                  query: category, 
                  mode: 'crime', 
                  immediate: true 
                });
              }
            }}
          />
        )}

        {/* Results Section */}
        {search.hasResults && (
          <SearchResults
            results={search.results}
            searchMode={search.searchMode}
            isLoading={search.isLoading}
            error={search.error}
            onCaseClick={(caseId: number) => setSelectedCaseId(caseId)}
          />
        )}

        {/* Filters Panel */}
        <FiltersPanel
          filters={search.filters}
          onFiltersChange={(newFilters) => {
            search.setFilters(newFilters);
            search.executeSearch({ filters: newFilters, immediate: true });
          }}
          onClear={() => {
            search.clearFilters();
            search.executeSearch({ filters: {}, immediate: true });
          }}
          isOpen={isFiltersOpen}
          onClose={() => setIsFiltersOpen(false)}
          searchMode={search.searchMode}
        />

        {/* Case Details Modal */}
        {selectedCaseId && (
          <CaseDetailsModal
            caseId={selectedCaseId}
            isOpen={!!selectedCaseId}
            onClose={() => setSelectedCaseId(null)}
          />
        )}
      </div>
    </MainLayout>
  );
}
