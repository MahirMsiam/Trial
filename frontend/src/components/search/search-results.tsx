'use client';

import CaseComparisonModal from '@/components/case/case-comparison-modal';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { SearchMode } from '@/hooks/use-search';
import type {
    CaseResult,
    ChunkResponse,
    CrimeSearchResponse,
    SearchResultResponse,
} from '@/types/api';
import { GitCompare, Loader2 } from 'lucide-react';
import { useState } from 'react';
import CaseCard from './case-card';

interface SearchResultsProps {
  results: SearchResultResponse | ChunkResponse[] | CaseResult[] | CrimeSearchResponse | null;
  searchMode: SearchMode;
  isLoading: boolean;
  error: Error | null;
  onCaseClick: (caseId: number) => void;
}

export default function SearchResults({
  results,
  searchMode,
  isLoading,
  error,
  onCaseClick,
}: SearchResultsProps) {
  const [selectedCaseIds, setSelectedCaseIds] = useState<number[]>([]);
  const [showComparison, setShowComparison] = useState(false);
  const [isCompareMode, setIsCompareMode] = useState(false);

  const toggleCaseSelection = (caseId: number) => {
    setSelectedCaseIds(prev =>
      prev.includes(caseId)
        ? prev.filter(id => id !== caseId)
        : [...prev, caseId]
    );
  };

  const handleCompare = () => {
    if (selectedCaseIds.length >= 2) {
      setShowComparison(true);
    }
  };

  const handleCloseComparison = () => {
    setShowComparison(false);
    setSelectedCaseIds([]);
    setIsCompareMode(false);
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }

  if (error) {
    return (
      <Alert variant="destructive">
        <AlertDescription>
          Error loading results: {error.message || 'Unknown error'}
        </AlertDescription>
      </Alert>
    );
  }

  if (!results) {
    return null;
  }

  // Render based on search mode
  if (searchMode === 'keyword' && 'results' in results) {
    return (
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-semibold">
            Found {results.count} {results.count === 1 ? 'result' : 'results'}
          </h2>
          <div className="flex gap-2">
            {results.results.length > 1 && (
              <Button
                variant={isCompareMode ? 'default' : 'outline'}
                size="sm"
                onClick={() => {
                  setIsCompareMode(!isCompareMode);
                  if (isCompareMode) {
                    setSelectedCaseIds([]);
                  }
                }}
              >
                <GitCompare className="h-4 w-4 mr-2" />
                {isCompareMode ? 'Cancel Compare' : 'Compare Cases'}
              </Button>
            )}
            {isCompareMode && selectedCaseIds.length >= 2 && (
              <Button size="sm" onClick={handleCompare}>
                Compare {selectedCaseIds.length} Cases
              </Button>
            )}
          </div>
        </div>
        <div className="grid gap-4">
          {results.results.map((caseData) => (
            <CaseCard
              key={caseData.id}
              caseData={caseData}
              onClick={() => onCaseClick(caseData.id)}
              isSelected={selectedCaseIds.includes(caseData.id)}
              onSelect={(_selected: boolean) => toggleCaseSelection(caseData.id)}
              showCheckbox={isCompareMode}
            />
          ))}
        </div>
        
        {/* Comparison Modal */}
        <CaseComparisonModal
          isOpen={showComparison}
          onClose={handleCloseComparison}
          caseIds={selectedCaseIds}
        />
      </div>
    );
  }

  if (searchMode === 'semantic' && Array.isArray(results)) {
    // Type guard for ChunkResponse
    const chunks = results.filter((r): r is ChunkResponse => 'chunk_text' in r);
    
    return (
      <div className="space-y-4">
        <h2 className="text-xl font-semibold">
          Found {chunks.length} relevant {chunks.length === 1 ? 'chunk' : 'chunks'}
        </h2>
        <div className="grid gap-4">
          {chunks.map((chunk, idx) => (
            <div
              key={idx}
              className="p-4 border rounded-lg hover:bg-accent cursor-pointer"
              onClick={() => onCaseClick(chunk.case_id)}
            >
              <p className="text-sm mb-2">{chunk.chunk_text}</p>
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <span>Case ID: {chunk.case_id}</span>
                {chunk.similarity !== undefined && (
                  <span>• Similarity: {(chunk.similarity * 100).toFixed(1)}%</span>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (searchMode === 'hybrid' && Array.isArray(results)) {
    // Type guard for CaseResult
    const caseResults = results.filter((r): r is CaseResult => 'hybrid_score' in r);
    
    return (
      <div className="space-y-4">
        <h2 className="text-xl font-semibold">
          Found {caseResults.length} relevant {caseResults.length === 1 ? 'case' : 'cases'}
        </h2>
        <div className="grid gap-4">
          {caseResults.map((caseResult, idx) => (
            <div
              key={idx}
              className="border rounded-lg p-4 hover:bg-accent cursor-pointer"
              onClick={() => onCaseClick(caseResult.case_id)}
            >
              <div className="font-medium mb-2">
                {caseResult.full_case_id || `Case #${caseResult.case_id}`}
              </div>
              <div className="text-sm text-muted-foreground mb-2">
                Hybrid Score: {caseResult.hybrid_score.toFixed(3)} • {caseResult.chunk_count} chunks
              </div>
              {caseResult.chunks.slice(0, 2).map((chunk: any, cidx: number) => (
                <p key={cidx} className="text-sm mb-1">
                  {chunk.chunk_text.substring(0, 200)}...
                </p>
              ))}
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (searchMode === 'crime' && 'response' in results) {
    return (
      <div className="space-y-4">
        <div className="p-4 bg-primary/10 rounded-lg">
          <h2 className="text-xl font-semibold mb-2">{results.crime_type}</h2>
          <p className="text-sm">{results.summary}</p>
        </div>
        <h3 className="text-lg font-medium">
          Found {results.count} relevant {results.count === 1 ? 'case' : 'cases'}
        </h3>
        <div className="grid gap-4">
          {results.cases.map((chunk, idx) => (
            <div
              key={idx}
              className="p-4 border rounded-lg hover:bg-accent cursor-pointer"
              onClick={() => onCaseClick(chunk.case_id)}
            >
              <p className="text-sm mb-2">{chunk.chunk_text}</p>
              <div className="text-xs text-muted-foreground">
                Case ID: {chunk.case_id}
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return <div>No results to display</div>;
}
