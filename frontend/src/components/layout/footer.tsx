'use client';

import apiClient from '@/lib/api-client';
import { useEffect, useState } from 'react';

export default function Footer() {
  const [stats, setStats] = useState<{ version?: string; totalCases?: number } | null>(null);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const [versionData, statsData] = await Promise.all([
          apiClient.getVersion().catch(() => ({ version: 'N/A' })),
          apiClient.getStats().catch(() => ({ total_judgments: 0 })),
        ]);
        setStats({
          version: versionData.version,
          totalCases: statsData.total_judgments,
        });
      } catch {
        // Silently fail
      }
    };

    fetchStats();
  }, []);

  return (
    <footer className="border-t">
      <div className="container flex flex-col items-center justify-between gap-4 py-6 md:flex-row">
        <p className="text-sm text-muted-foreground">
          © 2025 Bangladesh Supreme Court Legal Research System
          {stats?.version && ` • v${stats.version}`}
        </p>
        <p className="text-sm text-muted-foreground">
          Powered by AI RAG Technology
          {stats?.totalCases && ` • ${stats.totalCases.toLocaleString()} Cases`}
        </p>
      </div>
    </footer>
  );
}
