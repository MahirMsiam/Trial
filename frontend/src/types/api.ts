// Request Types
export interface KeywordFilters {
  case_type?: string;
  case_number?: string;
  petitioner?: string;
  respondent?: string;
  advocate?: string;
  section?: string;
  rule_outcome?: string;
  year?: string;
  court_name?: string;
  date_from?: string;
  date_to?: string;
}

// Extended filters for chat (includes fields not in keyword search)
export interface ChatFilters extends KeywordFilters {
  court_name?: string;
}

export interface KeywordSearchRequest {
  query?: string;
  filters?: KeywordFilters;
}

export interface SemanticSearchRequest {
  query: string;
  top_k?: number;
  filters?: Record<string, any>;
}

export interface HybridSearchRequest {
  query: string;
  top_k?: number;
  filters?: Record<string, any>;
}

export interface ChatRequest {
  query: string;
  session_id?: string;
  filters?: Record<string, any>;
  stream?: boolean;
  max_chunks_per_case?: number;
}

export interface CrimeSearchRequest {
  query: string;
  limit?: number;
}

export interface CompareCasesRequest {
  case_ids: number[];
}

export interface SessionCreateRequest {
  metadata?: Record<string, any>;
}

// Response Types
export interface JudgmentResponse {
  id: number;
  file_name: string;
  case_number?: string;
  case_year?: string;
  case_type?: string;
  full_case_id?: string;
  petitioner_name?: string;
  respondent_name?: string;
  judgment_date?: string;
  judgment_outcome?: string;
  judgment_summary?: string;
  court_name?: string;
  rule_type?: string;
  rule_outcome?: string;
  language?: string;
  page_count?: number;
  advocates?: Record<string, string[]>;
  laws?: string[];
  judges?: string[];
}

export interface SearchResultResponse {
  results: JudgmentResponse[];
  count: number;
  query?: string;
}

export interface ChunkResponse {
  chunk_text: string;
  similarity?: number;
  case_id: number;
  case_number?: string;
  case_type?: string;
  judgment_date?: string;
  petitioner?: string;
  respondent?: string;
  full_case_id?: string;
  source?: string;
}

export interface CaseResult {
  case_id: number;
  case_number?: string;
  case_type?: string;
  judgment_date?: string;
  petitioner?: string;
  respondent?: string;
  full_case_id?: string;
  hybrid_score: number;
  chunk_count: number;
  chunks: ChunkResponse[];
}

export interface ChatResponse {
  response: string;
  sources: ChunkResponse[];
  query_type: string;
  session_id: string;
}

export interface CrimeSearchResponse {
  response: string;
  crime_type: string;
  count: number;
  cases: ChunkResponse[];
  summary: string;
}

export interface CaseComparisonResponse {
  comparison: string;
  cases: JudgmentResponse[];
}

export interface CaseSummaryResponse {
  summary: string;
  case_data: JudgmentResponse;
}

export interface SessionResponse {
  session_id: string;
  created_at: string;
  last_active: string;
  message_count: number;
}

export interface SessionTurn {
  query?: string;
  response?: string;
  timestamp?: string | number;
  sources?: ChunkResponse[];
  query_type?: string;
}

export interface HealthResponse {
  status: string;
  database_connected: boolean;
  faiss_index_loaded: boolean;
  llm_provider: string;
}

export interface StatsResponse {
  total_judgments: number;
  case_types: number;
  total_advocates: number;
  total_laws_cited: number; // Primary field from API
  total_laws?: number; // Optional fallback field
  case_type_breakdown: Array<{
    case_type: string;
    count: number;
  }> | Array<[string, number]>; // Support both object and array tuple shapes
}

export interface ErrorResponse {
  error: string;
  detail?: string;
  status_code: number;
}

// Streaming Types
export type StreamChunk =
  | { type: 'sources'; sources: any[] }
  | { type: 'token'; token: string }
  | { type: 'complete'; response: string; query_type: string; session_id: string }
  | { type: 'error'; error: string };
