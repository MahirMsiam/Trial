'use client';

import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { AlertCircle, RotateCcw } from 'lucide-react';

interface SessionExpiryAlertProps {
  isExpiring: boolean;
  onRefresh: () => Promise<void>;
  isRefreshing?: boolean;
}

export default function SessionExpiryAlert({
  isExpiring,
  onRefresh,
  isRefreshing = false,
}: SessionExpiryAlertProps) {
  if (!isExpiring) return null;

  return (
    <Alert className="border-yellow-500 bg-yellow-50 dark:bg-yellow-950">
      <AlertCircle className="h-4 w-4 text-yellow-600 dark:text-yellow-400" />
      <AlertTitle className="text-yellow-800 dark:text-yellow-200">
        Session Expiring Soon
      </AlertTitle>
      <AlertDescription className="text-yellow-700 dark:text-yellow-300 flex items-center justify-between">
        <span>Your session will expire in 5 minutes. Click refresh to continue.</span>
        <Button
          size="sm"
          variant="outline"
          onClick={onRefresh}
          disabled={isRefreshing}
          className="ml-4"
        >
          <RotateCcw className={`h-3 w-3 mr-1 ${isRefreshing ? 'animate-spin' : ''}`} />
          {isRefreshing ? 'Refreshing...' : 'Refresh'}
        </Button>
      </AlertDescription>
    </Alert>
  );
}
