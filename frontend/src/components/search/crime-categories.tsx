'use client';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import apiClient from '@/lib/api-client';
import { CRIME_CATEGORY_LABELS, type CrimeCategory } from '@/types/crime';
import { useQuery } from '@tanstack/react-query';
import * as LucideIcons from 'lucide-react';
import { useMemo } from 'react';

interface CrimeCategoriesProps {
  onCategorySelect: (category: CrimeCategory | null) => void;
  selectedCategory?: CrimeCategory | null;
}

export default function CrimeCategories({ onCategorySelect, selectedCategory }: CrimeCategoriesProps) {
  // Fetch crime categories from backend, with fallback to CRIME_CATEGORY_LABELS keys
  const { data: categoriesData } = useQuery({
    queryKey: ['crime-categories'],
    queryFn: async () => {
      try {
        return await apiClient.getCrimeCategories();
      } catch (error) {
        // Silently fall back to local list (endpoint may not exist or be temporarily unavailable)
        // Fallback to local list
        return Object.keys(CRIME_CATEGORY_LABELS);
      }
    },
    staleTime: 1000 * 60 * 60, // Cache for 1 hour
  });

  // Memoize fallback to avoid recreating on each render
  const crimeCategories = useMemo(
    () => (categoriesData || Object.keys(CRIME_CATEGORY_LABELS)) as CrimeCategory[],
    [categoriesData]
  );

  const getIcon = (iconName: string) => {
    const Icon = (LucideIcons as any)[iconName] || LucideIcons.AlertCircle;
    return <Icon className="h-4 w-4" />;
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold">Select Crime Category</h3>
        {selectedCategory && (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => onCategorySelect(null)}
          >
            Clear Selection
          </Button>
        )}
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3">
        {crimeCategories.map((category) => {
          const label = CRIME_CATEGORY_LABELS[category];
          if (!label) return null; // Skip categories not in CRIME_CATEGORY_LABELS
          
          const { label: labelText, icon, color } = label;
          const isSelected = selectedCategory === category;

          return (
            <button
              key={category}
              onClick={() => onCategorySelect(category)}
              className={`
                flex flex-col items-center justify-center gap-2 p-4 
                rounded-lg border-2 transition-all hover:shadow-md
                ${isSelected 
                  ? 'border-primary bg-primary/10 shadow-md scale-105' 
                  : 'border-transparent bg-card hover:border-primary/50'
                }
              `}
            >
              <div className={`${color} p-2 rounded-full text-white`}>
                {getIcon(icon)}
              </div>
              <span className="text-xs font-medium text-center">{labelText}</span>
            </button>
          );
        })}
      </div>

      {selectedCategory && (
        <div className="flex items-center gap-2 p-3 bg-muted rounded-lg">
          <span className="text-sm font-medium">Selected:</span>
          {CRIME_CATEGORY_LABELS[selectedCategory] && (
            <Badge className={CRIME_CATEGORY_LABELS[selectedCategory].color}>
              {CRIME_CATEGORY_LABELS[selectedCategory].label}
            </Badge>
          )}
        </div>
      )}
    </div>
  );
}
