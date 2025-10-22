'use client';

import { Badge } from '@/components/ui/badge';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import apiClient from '@/lib/api-client';
import { formatDate, getCaseTypeColor, getOutcomeColor } from '@/lib/utils';
import { useQuery } from '@tanstack/react-query';
import { Loader2 } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

interface CaseDetailsModalProps {
  caseId: number;
  isOpen: boolean;
  onClose: () => void;
}

export default function CaseDetailsModal({ caseId, isOpen, onClose }: CaseDetailsModalProps) {
  const { data, isLoading, error } = useQuery({
    queryKey: ['case', caseId],
    queryFn: () => apiClient.summarizeCase(caseId),
    enabled: isOpen && !!caseId,
  });

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-4xl max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Case Details</DialogTitle>
        </DialogHeader>

        {isLoading && (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="h-8 w-8 animate-spin" />
          </div>
        )}

        {error && (
          <div className="text-destructive">
            Error loading case details: {(error as Error).message}
          </div>
        )}

        {data && (
          <div className="space-y-6">
            <div>
              <h2 className="text-2xl font-bold mb-2">
                {data.case_data.full_case_id || `Case #${data.case_data.id}`}
              </h2>
              {data.case_data.case_type && (
                <Badge className={getCaseTypeColor(data.case_data.case_type)}>
                  {data.case_data.case_type}
                </Badge>
              )}
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <h3 className="font-semibold mb-1">Petitioner</h3>
                <p>{data.case_data.petitioner_name || 'N/A'}</p>
              </div>
              <div>
                <h3 className="font-semibold mb-1">Respondent</h3>
                <p>{data.case_data.respondent_name || 'N/A'}</p>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <h3 className="font-semibold mb-1">Judgment Date</h3>
                <p>{data.case_data.judgment_date ? formatDate(data.case_data.judgment_date) : 'N/A'}</p>
              </div>
              <div>
                <h3 className="font-semibold mb-1">Court</h3>
                <p>{data.case_data.court_name || 'N/A'}</p>
              </div>
            </div>

            {data.case_data.judgment_outcome && (
              <div>
                <h3 className="font-semibold mb-1">Outcome</h3>
                <Badge className={getOutcomeColor(data.case_data.judgment_outcome)}>
                  {data.case_data.judgment_outcome}
                </Badge>
              </div>
            )}

            {data.summary && (
              <div>
                <h3 className="font-semibold mb-2">AI Summary</h3>
                <div className="prose prose-sm max-w-none bg-muted p-4 rounded-lg dark:prose-invert">
                  <ReactMarkdown>{data.summary}</ReactMarkdown>
                </div>
              </div>
            )}

            {data.case_data.advocates && Object.keys(data.case_data.advocates).length > 0 && (
              <div>
                <h3 className="font-semibold mb-2">Advocates</h3>
                <div className="space-y-2">
                  {Object.entries(data.case_data.advocates).map(([party, advocates]) => (
                    <div key={party}>
                      <h4 className="text-sm font-medium">{party}:</h4>
                      <ul className="list-disc list-inside text-sm text-muted-foreground">
                        {advocates.map((adv, idx) => (
                          <li key={idx}>{adv}</li>
                        ))}
                      </ul>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {data.case_data.laws && data.case_data.laws.length > 0 && (
              <div>
                <h3 className="font-semibold mb-2">Laws Cited</h3>
                <div className="flex flex-wrap gap-2">
                  {data.case_data.laws.map((law, idx) => (
                    <Badge key={idx} variant="outline">
                      {law}
                    </Badge>
                  ))}
                </div>
              </div>
            )}

            {data.case_data.judges && data.case_data.judges.length > 0 && (
              <div>
                <h3 className="font-semibold mb-2">Judges</h3>
                <p className="text-sm">{data.case_data.judges.join(', ')}</p>
              </div>
            )}
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
