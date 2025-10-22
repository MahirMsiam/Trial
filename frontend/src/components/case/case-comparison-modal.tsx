'use client';

import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import apiClient from '@/lib/api-client';
import type { CaseComparisonResponse } from '@/types/api';
import { useQuery } from '@tanstack/react-query';
import { Loader2 } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

interface CaseComparisonModalProps {
  isOpen: boolean;
  onClose: () => void;
  caseIds: number[];
}

export default function CaseComparisonModal({
  isOpen,
  onClose,
  caseIds,
}: CaseComparisonModalProps) {
  const { data, isLoading, error } = useQuery<CaseComparisonResponse>({
    queryKey: ['case-comparison', caseIds],
    queryFn: () => apiClient.compareCases({ case_ids: caseIds }),
    enabled: isOpen && caseIds.length >= 2,
  });

  return (
    <Dialog open={isOpen} onOpenChange={(open) => { if (!open) onClose(); }}>
      <DialogContent className="max-w-5xl max-h-[85vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Case Comparison ({caseIds.length} cases)</DialogTitle>
        </DialogHeader>

        {isLoading && (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="h-8 w-8 animate-spin" />
          </div>
        )}

        {error && (
          <div className="text-destructive p-4 border border-destructive rounded-md">
            Error loading comparison: {error instanceof Error ? error.message : 'Unknown error'}
          </div>
        )}

        {data && (
          <div className="space-y-6">
            {/* Comparison Analysis */}
            <div className="prose prose-sm max-w-none">
              <h3 className="text-lg font-semibold mb-3">Comparative Analysis</h3>
              <ReactMarkdown>{data.comparison}</ReactMarkdown>
            </div>

            {/* Cases Being Compared */}
            {data.cases && data.cases.length > 0 && (
              <div className="space-y-4">
                <h3 className="text-lg font-semibold">Cases</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {data.cases.map((caseData) => (
                    <div
                      key={caseData.id}
                      className="border rounded-lg p-4 space-y-2 bg-muted/30"
                    >
                      <div className="font-semibold text-sm">
                        {caseData.full_case_id || caseData.file_name}
                      </div>
                      
                      {caseData.petitioner_name && (
                        <div className="text-xs">
                          <span className="text-muted-foreground">Petitioner:</span>{' '}
                          {caseData.petitioner_name}
                        </div>
                      )}
                      
                      {caseData.respondent_name && (
                        <div className="text-xs">
                          <span className="text-muted-foreground">Respondent:</span>{' '}
                          {caseData.respondent_name}
                        </div>
                      )}
                      
                      {caseData.judgment_date && (
                        <div className="text-xs">
                          <span className="text-muted-foreground">Date:</span>{' '}
                          {caseData.judgment_date}
                        </div>
                      )}
                      
                      {caseData.case_type && (
                        <div className="text-xs">
                          <span className="text-muted-foreground">Type:</span>{' '}
                          {caseData.case_type}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        <div className="flex justify-end pt-4 border-t">
          <Button onClick={onClose}>Close</Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
