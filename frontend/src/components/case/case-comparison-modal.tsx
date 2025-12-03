'use client';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import apiClient from '@/lib/api-client';
import { formatDate, getCaseTypeColor } from '@/lib/utils';
import type { CaseComparisonResponse } from '@/types/api';
import { useQuery } from '@tanstack/react-query';
import { Loader2 } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import rehypeSanitize from 'rehype-sanitize';
import remarkGfm from 'remark-gfm';

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
  // Stable query key: serialize case IDs as JSON to prevent cache churn
  const caseIdsKey = JSON.stringify(caseIds.sort((a, b) => a - b));

  const { data, isLoading, error } = useQuery<CaseComparisonResponse>({
    queryKey: ['case-comparison', caseIdsKey],
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
            <Loader2 className="h-8 w-8 animate-spin text-primary" />
          </div>
        )}

        {error && (
          <div className="rounded-lg bg-destructive/10 p-4 text-destructive border border-destructive/20">
            Error loading comparison: {error instanceof Error ? error.message : 'Unknown error'}
          </div>
        )}

        {data && (
          <div className="space-y-6">
            {/* Comparison Analysis */}
            {data.comparison && (
              <div>
                <h3 className="text-lg font-semibold mb-3">Comparative Analysis</h3>
                <div className="prose prose-sm max-w-none bg-muted p-4 rounded-lg dark:prose-invert">
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    rehypePlugins={[rehypeSanitize]}
                  >
                    {data.comparison}
                  </ReactMarkdown>
                </div>
              </div>
            )}

            {/* Cases Being Compared */}
            {data.cases && data.cases.length > 0 && (
              <div className="space-y-4">
                <h3 className="text-lg font-semibold">Cases Compared</h3>
                <div className="space-y-4">
                  {data.cases.map((caseData) => (
                    <div
                      key={caseData.id}
                      className="border rounded-lg p-4 space-y-3 bg-muted/50"
                    >
                      <div>
                        <div className="text-base font-semibold">
                          {caseData.full_case_id || `Case #${caseData.id}`}
                        </div>
                        {caseData.case_type && (
                          <Badge className={getCaseTypeColor(caseData.case_type)}>
                            {caseData.case_type}
                          </Badge>
                        )}
                      </div>

                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <span className="font-medium">Petitioner:</span>
                          <p className="text-muted-foreground">
                            {caseData.petitioner_name || 'N/A'}
                          </p>
                        </div>
                        <div>
                          <span className="font-medium">Respondent:</span>
                          <p className="text-muted-foreground">
                            {caseData.respondent_name || 'N/A'}
                          </p>
                        </div>
                      </div>

                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <span className="font-medium">Judgment Date:</span>
                          <p className="text-muted-foreground">
                            {caseData.judgment_date
                              ? formatDate(caseData.judgment_date)
                              : 'N/A'}
                          </p>
                        </div>
                        <div>
                          <span className="font-medium">Court:</span>
                          <p className="text-muted-foreground">
                            {caseData.court_name || 'N/A'}
                          </p>
                        </div>
                      </div>
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
