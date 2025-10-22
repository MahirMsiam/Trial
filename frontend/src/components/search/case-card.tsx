'use client';

import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { formatDate, getCaseTypeColor, getOutcomeColor } from '@/lib/utils';
import { JudgmentResponse } from '@/types/api';
import { Calendar, Scale, Users } from 'lucide-react';

interface CaseCardProps {
  caseData: JudgmentResponse;
  onClick: () => void;
}

export default function CaseCard({ caseData, onClick }: CaseCardProps) {
  return (
    <Card className="cursor-pointer hover:shadow-md transition-shadow" onClick={onClick}>
      <CardHeader>
        <div className="flex items-start justify-between gap-4">
          <div className="space-y-1">
            <div className="font-semibold text-lg">
              {caseData.full_case_id || `Case #${caseData.id}`}
            </div>
            {caseData.case_type && (
              <Badge className={getCaseTypeColor(caseData.case_type)}>
                {caseData.case_type}
              </Badge>
            )}
          </div>
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Calendar className="h-4 w-4" />
            {caseData.judgment_date ? formatDate(caseData.judgment_date) : 'N/A'}
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-3">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <Scale className="h-4 w-4 text-muted-foreground" />
            <span className="font-medium">
              {caseData.petitioner_name || 'Unknown'} vs {caseData.respondent_name || 'Unknown'}
            </span>
          </div>
        </div>

        {caseData.judgment_outcome && (
          <Badge className={getOutcomeColor(caseData.judgment_outcome)}>
            {caseData.judgment_outcome}
          </Badge>
        )}

        {caseData.judgment_summary && (
          <p className="text-sm text-muted-foreground line-clamp-3">
            {caseData.judgment_summary}
          </p>
        )}

        {caseData.court_name && (
          <div className="text-sm text-muted-foreground">
            Court: {caseData.court_name}
          </div>
        )}

        {caseData.advocates && Object.keys(caseData.advocates).length > 0 && (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Users className="h-4 w-4" />
            <span>
              {Object.values(caseData.advocates).flat().slice(0, 2).join(', ')}
              {Object.values(caseData.advocates).flat().length > 2 && '...'}
            </span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
