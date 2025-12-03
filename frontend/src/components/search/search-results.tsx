'use client';

import CaseComparisonModal from '@/components/case/case-comparison-modal';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { SearchMode } from '@/hooks/use-search';
import { extractExcerpt, formatDate, getCaseTypeColor } from '@/lib/utils';
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
    
    // Get query from sessionStorage if available for highlighting
    const query = typeof window !== 'undefined' ? sessionStorage.getItem('lastSearchQuery') || '' : '';
    
    return (
      <div className="space-y-4">
        <h2 className="text-xl font-semibold">
          Found {chunks.length} relevant {chunks.length === 1 ? 'chunk' : 'chunks'}
        </h2>
        <div className="grid gap-4">
          {chunks.map((chunk, idx) => (
            <div
              key={idx}
              className="p-4 border rounded-lg hover:bg-accent cursor-pointer transition-colors"
              onClick={() => onCaseClick(chunk.case_id)}
            >
              {/* Case Metadata Header */}
              <div className="flex items-center justify-between mb-3">
                <div className="space-y-1">
                  <p className="font-medium text-sm">
                    {chunk.full_case_id || `Case #${chunk.case_id}`}
                  </p>
                  {chunk.case_type && (
                    <Badge variant="outline" className={getCaseTypeColor(chunk.case_type)}>
                      {chunk.case_type}
                    </Badge>
                  )}
                </div>
                {chunk.similarity !== undefined && (
                  <div className="text-right">
                    <p className="text-sm font-medium">
                      Similarity: <span className="text-primary">{(chunk.similarity * 100).toFixed(1)}%</span>
                    </p>
                  </div>
                )}
              </div>

              {/* Parties Information */}
              {(chunk.petitioner || chunk.respondent) && (
                <div className="text-xs text-muted-foreground mb-3 space-y-1">
                  {chunk.petitioner && <div><strong>Petitioner:</strong> {chunk.petitioner}</div>}
                  {chunk.respondent && <div><strong>Respondent:</strong> {chunk.respondent}</div>}
                </div>
              )}

              {/* Highlighted Excerpt */}
              <p className="text-sm mb-3 leading-relaxed">
                {extractExcerpt(chunk.chunk_text, query, 150)}
              </p>

              {/* Footer with Date and Case ID */}
              <div className="flex items-center justify-between text-xs text-muted-foreground pt-2 border-t">
                {chunk.judgment_date && (
                  <span>Judgment: {formatDate(chunk.judgment_date)}</span>
                )}
                <span>Case ID: {chunk.case_id}</span>
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
    
    // Get query from sessionStorage if available for highlighting
    const query = typeof window !== 'undefined' ? sessionStorage.getItem('lastSearchQuery') || '' : '';
    
    return (
      <div className="space-y-4">
        <h2 className="text-xl font-semibold">
          Found {caseResults.length} relevant {caseResults.length === 1 ? 'case' : 'cases'}
        </h2>
        <div className="grid gap-4">
          {caseResults.map((caseResult, idx) => (
            <div
              key={idx}
              className="border rounded-lg p-4 hover:bg-accent cursor-pointer transition-colors"
              onClick={() => onCaseClick(caseResult.case_id)}
            >
              {/* Case Header with Score */}
              <div className="flex items-center justify-between mb-3">
                <div className="space-y-1">
                  <p className="font-medium">
                    {caseResult.full_case_id || `Case #${caseResult.case_id}`}
                  </p>
                  {caseResult.case_type && (
                    <Badge variant="outline" className={getCaseTypeColor(caseResult.case_type)}>
                      {caseResult.case_type}
                    </Badge>
                  )}
                </div>
                <div className="text-right">
                  <p className="text-sm font-medium">
                    Score: <span className="text-primary">{caseResult.hybrid_score.toFixed(3)}</span>
                  </p>
                  <p className="text-xs text-muted-foreground">{caseResult.chunk_count} chunks</p>
                </div>
              </div>

              {/* Parties Information */}
              {(caseResult.petitioner || caseResult.respondent) && (
                <div className="text-xs text-muted-foreground mb-3 space-y-1">
                  {caseResult.petitioner && <div><strong>Petitioner:</strong> {caseResult.petitioner}</div>}
                  {caseResult.respondent && <div><strong>Respondent:</strong> {caseResult.respondent}</div>}
                </div>
              )}

              {/* Top Chunk Excerpts */}
              <div className="space-y-2 mb-3">
                {caseResult.chunks.slice(0, 2).map((chunk: ChunkResponse, cidx: number) => (
                  <div key={cidx} className="bg-muted/50 p-2 rounded text-sm">
                    <p className="leading-relaxed">
                      {extractExcerpt(chunk.chunk_text, query, 100)}
                    </p>
                  </div>
                ))}
              </div>

              {/* Footer with Date */}
              {caseResult.judgment_date && (
                <p className="text-xs text-muted-foreground pt-2 border-t">
                  Judgment: {formatDate(caseResult.judgment_date)}
                </p>
              )}
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (searchMode === 'crime' && 'response' in results) {
    // Get query from sessionStorage if available for highlighting
    const query = typeof window !== 'undefined' ? sessionStorage.getItem('lastSearchQuery') || '' : '';
    
    return (
      <div className="space-y-4">
        <div className="p-4 bg-primary/10 rounded-lg border border-primary/20">
          <h2 className="text-xl font-semibold mb-2">{results.crime_type}</h2>
          <p className="text-sm text-muted-foreground">{results.summary}</p>
        </div>
        <h3 className="text-lg font-medium">
          Found {results.count} relevant {results.count === 1 ? 'case' : 'cases'}
        </h3>
        <div className="grid gap-4">
          {results.cases.map((chunk, idx) => (
            <div
              key={idx}
              className="p-4 border rounded-lg hover:bg-accent cursor-pointer transition-colors"
              onClick={() => onCaseClick(chunk.case_id)}
            >
              {/* Case Metadata */}
              <div className="flex items-center justify-between mb-3">
                <div className="space-y-1">
                  <p className="font-medium text-sm">
                    {chunk.full_case_id || `Case #${chunk.case_id}`}
                  </p>
                  {chunk.case_type && (
                    <Badge variant="outline" className={getCaseTypeColor(chunk.case_type)}>
                      {chunk.case_type}
                    </Badge>
                  )}
                </div>
              </div>

              {/* Parties */}
              {(chunk.petitioner || chunk.respondent) && (
                <div className="text-xs text-muted-foreground mb-3 space-y-1">
                  {chunk.petitioner && <div><strong>Petitioner:</strong> {chunk.petitioner}</div>}
                  {chunk.respondent && <div><strong>Respondent:</strong> {chunk.respondent}</div>}
                </div>
              )}

              {/* Highlighted Excerpt */}
              <p className="text-sm mb-3 leading-relaxed">
                {extractExcerpt(chunk.chunk_text, query, 150)}
              </p>

              {/* Footer */}
              <div className="text-xs text-muted-foreground pt-2 border-t">
                {chunk.judgment_date && (
                  <span>Judgment: {formatDate(chunk.judgment_date)} • </span>
                )}
                <span>Case ID: {chunk.case_id}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return <div>No results to display</div>;
}
