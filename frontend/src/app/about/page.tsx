'use client';

import MainLayout from '@/components/layout/main-layout';
import apiClient from '@/lib/api-client';
import { AlertCircle, CheckCircle2, Database, Zap } from 'lucide-react';
import { useEffect, useState } from 'react';

interface SystemInfo {
  status: string;
  database_connected: boolean;
  total_judgments: number;
  case_types: number;
  total_advocates: number;
  total_laws_cited: number;
  faiss_index_loaded: boolean;
  llm_provider: string;
  health_check_time: string;
}

export default function AboutPage() {
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchSystemInfo = async () => {
      try {
        setIsLoading(true);
        const health = await apiClient.getHealth();
        const stats = await apiClient.getStats();

        setSystemInfo({
          status: health.status || 'Unknown',
          database_connected: health.database_connected ?? false,
          total_judgments: stats.total_judgments ?? 0,
          case_types: stats.case_types ?? 0,
          total_advocates: stats.total_advocates ?? 0,
          total_laws_cited: stats.total_laws_cited ?? 0,
          faiss_index_loaded: health.faiss_index_loaded ?? false,
          llm_provider: health.llm_provider || 'Unknown',
          health_check_time: new Date().toLocaleString(),
        });
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch system information');
        console.error('Error fetching system info:', err);
      } finally {
        setIsLoading(false);
      }
    };

    fetchSystemInfo();
  }, []);

  return (
    <MainLayout currentPage="about">
      <main>
        <div className="container py-12">
          <div className="space-y-8">
            {/* Header Section */}
            <div className="space-y-4">
              <h1 className="text-4xl font-bold">About Bangladesh Supreme Court Legal Research</h1>
              <p className="text-lg text-muted-foreground">
                An AI-powered legal research system providing intelligent access to 8000+ Supreme Court judgments
              </p>
            </div>

            {/* System Status */}
            {isLoading ? (
              <div className="rounded-lg border p-8 text-center">
                <p className="text-muted-foreground">Loading system information...</p>
              </div>
            ) : error ? (
              <div className="rounded-lg border border-destructive/50 bg-destructive/10 p-8">
                <div className="flex items-start gap-4">
                  <AlertCircle className="h-6 w-6 text-destructive flex-shrink-0 mt-1" />
                  <div>
                    <h3 className="font-semibold text-destructive mb-2">Unable to Connect</h3>
                    <p className="text-sm text-destructive/80">{error}</p>
                  </div>
                </div>
              </div>
            ) : systemInfo ? (
              <div className="grid gap-6">
                {/* Status Card */}
                <div className="rounded-lg border bg-card p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h2 className="text-xl font-semibold flex items-center gap-2">
                      {systemInfo.database_connected ? (
                        <CheckCircle2 className="h-5 w-5 text-green-500" />
                      ) : (
                        <AlertCircle className="h-5 w-5 text-red-500" />
                      )}
                      System Status
                    </h2>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <p className="text-sm text-muted-foreground mb-1">Status</p>
                      <p className="text-lg font-medium capitalize">{systemInfo.status}</p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground mb-1">LLM Provider</p>
                      <p className="text-lg font-medium">{systemInfo.llm_provider}</p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground mb-1">Database</p>
                      <p className="text-lg font-medium">
                        {systemInfo.database_connected ? 'Connected ✓' : 'Disconnected ✗'}
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground mb-1">Index Loaded</p>
                      <p className="text-lg font-medium">
                        {systemInfo.faiss_index_loaded ? 'Yes ✓' : 'No ✗'}
                      </p>
                    </div>
                  </div>
                </div>

                {/* Statistics Cards */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="rounded-lg border bg-card p-6">
                    <div className="flex items-center gap-4">
                      <div className="rounded-lg bg-blue-500/10 p-3">
                        <Database className="h-6 w-6 text-blue-600 dark:text-blue-400" />
                      </div>
                      <div>
                        <p className="text-sm text-muted-foreground">Total Judgments</p>
                        <p className="text-2xl font-bold">{systemInfo.total_judgments.toLocaleString()}</p>
                      </div>
                    </div>
                  </div>

                  <div className="rounded-lg border bg-card p-6">
                    <div className="flex items-center gap-4">
                      <div className="rounded-lg bg-purple-500/10 p-3">
                        <Zap className="h-6 w-6 text-purple-600 dark:text-purple-400" />
                      </div>
                      <div>
                        <p className="text-sm text-muted-foreground">Case Types</p>
                        <p className="text-2xl font-bold">{systemInfo.case_types.toLocaleString()}</p>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Technical Details */}
                <div className="rounded-lg border bg-card p-6">
                  <h3 className="text-lg font-semibold mb-4">Technical Details</h3>
                  <div className="space-y-3 text-sm">
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">LLM Provider</span>
                      <span className="font-medium">{systemInfo.llm_provider}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Total Advocates</span>
                      <span className="font-medium">{systemInfo.total_advocates.toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Total Laws Cited</span>
                      <span className="font-medium">{systemInfo.total_laws_cited.toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Search Capabilities</span>
                      <span className="font-medium">Keyword, Semantic, Hybrid, Crime</span>
                    </div>
                  </div>
                </div>

                {/* About Section */}
                <div className="rounded-lg border bg-card p-6">
                  <h3 className="text-lg font-semibold mb-4">About This System</h3>
                  <div className="space-y-4 text-sm text-muted-foreground">
                    <p>
                      This legal research platform combines advanced AI with a comprehensive database of Bangladesh Supreme Court
                      judgments. It enables legal professionals, researchers, and citizens to:
                    </p>
                    <ul className="list-disc list-inside space-y-2">
                      <li>Search across 8000+ judgments using natural language queries</li>
                      <li>Find relevant cases using semantic similarity and hybrid search</li>
                      <li>Filter results by case type, date, and crime category</li>
                      <li>Compare multiple cases side-by-side</li>
                      <li>Chat with an AI assistant about legal matters</li>
                      <li>Access comprehensive case metadata and summaries</li>
                    </ul>
                    <p className="mt-4">
                      The system uses state-of-the-art embeddings and retrieval techniques to provide accurate, contextual results
                      from complex legal documents. Data is indexed with FAISS for fast semantic search and processed with modern
                      LLM providers for intelligent analysis.
                    </p>
                  </div>
                </div>
              </div>
            ) : null}
          </div>
        </div>
      </main>
    </MainLayout>
  );
}
