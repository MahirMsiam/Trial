'use client';

import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import apiClient from '@/lib/api-client';
import { KeywordFilters } from '@/types/api';
import { useEffect, useState } from 'react';

interface FiltersPanelProps {
  filters: KeywordFilters;
  onFiltersChange: (filters: KeywordFilters) => void;
  onClear: () => void;
  isOpen: boolean;
  onClose: () => void;
}

export default function FiltersPanel({
  filters,
  onFiltersChange,
  onClear,
  isOpen,
  onClose,
}: FiltersPanelProps) {
  const [localFilters, setLocalFilters] = useState<KeywordFilters>(filters);
  const [caseTypes, setCaseTypes] = useState<string[]>([]);

  useEffect(() => {
    const fetchCaseTypes = async () => {
      try {
        const stats = await apiClient.getStats();
        if (stats.case_type_breakdown) {
          // Handle both object shape {case_type, count} and array shape [case_type, count]
          const types = stats.case_type_breakdown
            .map((item) => {
              if (typeof item === 'object' && item !== null && 'case_type' in item) {
                return item.case_type;
              } else if (Array.isArray(item) && item.length > 0) {
                return item[0];
              }
              return '';
            })
            .filter((type) => type && type.trim() !== '');
          setCaseTypes(types);
        }
      } catch (error) {
        console.error('Failed to fetch case types:', error);
        // Fallback to default case types
        setCaseTypes([
          'Criminal Appeal',
          'Civil Appeal',
          'Writ Petition',
          'Civil Revision',
          'Criminal Revision',
        ]);
      }
    };

    fetchCaseTypes();
  }, []);

  const handleApply = () => {
    onFiltersChange(localFilters);
    onClose();
  };

  const handleClear = () => {
    setLocalFilters({});
    onClear();
  };

  return (
    <Dialog open={isOpen} onOpenChange={(open) => { if (!open) onClose(); }}>
      <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Advanced Filters</DialogTitle>
        </DialogHeader>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-2">
            <label className="text-sm font-medium">Case Type</label>
            <Select
              value={localFilters.case_type || ''}
              onValueChange={(v) => setLocalFilters({ ...localFilters, case_type: v })}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select case type" />
              </SelectTrigger>
              <SelectContent>
                {caseTypes.map((type) => (
                  <SelectItem key={type} value={type}>
                    {type}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Case Number</label>
            <Input
              value={localFilters.case_number || ''}
              onChange={(e) => setLocalFilters({ ...localFilters, case_number: e.target.value })}
              placeholder="e.g., 123"
            />
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Year</label>
            <Input
              value={localFilters.year || ''}
              onChange={(e) => setLocalFilters({ ...localFilters, year: e.target.value })}
              placeholder="e.g., 2023"
            />
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Petitioner</label>
            <Input
              value={localFilters.petitioner || ''}
              onChange={(e) => setLocalFilters({ ...localFilters, petitioner: e.target.value })}
              placeholder="Petitioner name"
            />
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Respondent</label>
            <Input
              value={localFilters.respondent || ''}
              onChange={(e) => setLocalFilters({ ...localFilters, respondent: e.target.value })}
              placeholder="Respondent name"
            />
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Advocate</label>
            <Input
              value={localFilters.advocate || ''}
              onChange={(e) => setLocalFilters({ ...localFilters, advocate: e.target.value })}
              placeholder="Advocate name"
            />
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Law/Section</label>
            <Input
              value={localFilters.section || ''}
              onChange={(e) => setLocalFilters({ ...localFilters, section: e.target.value })}
              placeholder="e.g., Article 102, Section 302"
            />
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Rule Outcome</label>
            <Select
              value={localFilters.rule_outcome || ''}
              onValueChange={(v) => setLocalFilters({ ...localFilters, rule_outcome: v })}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select outcome" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="Discharged">Discharged</SelectItem>
                <SelectItem value="Made Absolute">Made Absolute</SelectItem>
                <SelectItem value="Disposed of">Disposed of</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Court</label>
            <Select
              value={localFilters.court_name || ''}
              onValueChange={(v) => setLocalFilters({ ...localFilters, court_name: v })}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select court" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="Appellate Division">Appellate Division</SelectItem>
                <SelectItem value="High Court Division">High Court Division</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Date From</label>
            <Input
              type="date"
              value={localFilters.date_from || ''}
              onChange={(e) => setLocalFilters({ ...localFilters, date_from: e.target.value })}
              placeholder="Start date"
            />
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Date To</label>
            <Input
              type="date"
              value={localFilters.date_to || ''}
              onChange={(e) => setLocalFilters({ ...localFilters, date_to: e.target.value })}
              placeholder="End date"
            />
          </div>
        </div>

        <div className="flex justify-between pt-4">
          <Button variant="outline" onClick={handleClear}>
            Clear All
          </Button>
          <div className="flex gap-2">
            <Button variant="outline" onClick={onClose}>
              Cancel
            </Button>
            <Button onClick={handleApply}>Apply Filters</Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
