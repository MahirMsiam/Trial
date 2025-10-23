import { getSessionId } from '@/lib/utils';
import type {
    CaseComparisonResponse,
    CaseResult,
    CaseSummaryResponse,
    ChatRequest,
    ChatResponse,
    ChunkResponse,
    CompareCasesRequest,
    CrimeSearchRequest,
    CrimeSearchResponse,
    ErrorResponse,
    HealthResponse,
    HybridSearchRequest,
    KeywordSearchRequest,
    SearchResultResponse,
    SemanticSearchRequest,
    SessionCreateRequest,
    SessionResponse,
    SessionTurn,
    StatsResponse,
    StreamChunk,
} from '@/types/api';
import axios, { AxiosError, AxiosInstance } from 'axios';

class APIClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: 30000,
    });

    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        // Add session_id only for chat endpoints, avoid modifying session management endpoints
        if (typeof window !== 'undefined' && config.url && config.data) {
          const method = config.method?.toUpperCase();
          const url = config.url;
          
          // Only inject session_id for chat endpoints
          const isChatEndpoint = url.includes('/api/chat') && !url.includes('/api/chat/stream');
          const isChatStreamEndpoint = url.includes('/api/chat/stream');
          
          // Skip injection for session creation (POST /api/session) and deletion (DELETE /api/session/:id)
          const isSessionCreate = method === 'POST' && url === '/api/session';
          const isSessionDelete = method === 'DELETE' && url.match(/^\/api\/session\/[^/]+$/);
          
          if ((isChatEndpoint || isChatStreamEndpoint) && !isSessionCreate && !isSessionDelete) {
            const sessionId = getSessionId();
            if (sessionId) {
              config.data.session_id = config.data.session_id || sessionId;
            }
          }
        }
        
        // Log API calls in development
        if (process.env.NODE_ENV === 'development') {
          console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`, config.data);
        }
        
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => {
        if (process.env.NODE_ENV === 'development') {
          console.log(`API Response: ${response.config.url}`, response.data);
        }
        return response;
      },
      (error: AxiosError) => {
        const errorResponse: ErrorResponse = {
          error: error.message,
          detail: error.response?.data ? JSON.stringify(error.response.data) : undefined,
          status_code: error.response?.status || 500,
        };
        
        console.error('API Error:', errorResponse);
        return Promise.reject(errorResponse);
      }
    );
  }

  // Search Methods
  async searchKeyword(request: KeywordSearchRequest): Promise<SearchResultResponse> {
    try {
      // Sanitize filters to only include fields supported by the backend
      let sanitizedRequest = { ...request };
      if (request.filters) {
        const { court_name, date_from, date_to, ...supportedFilters } = request.filters;
        sanitizedRequest = {
          ...request,
          filters: supportedFilters,
        };
        
        // Log removed fields in development
        if (process.env.NODE_ENV === 'development') {
          const removedFields = [];
          if (court_name) removedFields.push('court_name');
          if (date_from) removedFields.push('date_from');
          if (date_to) removedFields.push('date_to');
          if (removedFields.length > 0) {
            console.warn('Removed unsupported keyword search filters:', removedFields);
          }
        }
      }
      
      const response = await this.client.post<SearchResultResponse>('/api/search/keyword', sanitizedRequest);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async searchSemantic(request: SemanticSearchRequest): Promise<ChunkResponse[]> {
    try {
      const response = await this.client.post<ChunkResponse[]>('/api/search/semantic', request);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async searchHybrid(request: HybridSearchRequest): Promise<CaseResult[]> {
    try {
      const response = await this.client.post<CaseResult[]>('/api/search/hybrid', request);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async searchCrime(request: CrimeSearchRequest): Promise<CrimeSearchResponse> {
    try {
      const response = await this.client.post<CrimeSearchResponse>('/api/search/crime', request);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  // Chat Methods
  async chat(request: ChatRequest): Promise<ChatResponse> {
    try {
      const response = await this.client.post<ChatResponse>('/api/chat', {
        ...request,
        stream: false,
      });
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async *chatStream(request: ChatRequest, signal?: AbortSignal): AsyncGenerator<StreamChunk> {
    const url = `${this.client.defaults.baseURL}/api/chat/stream`;
    
    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream',
        },
        body: JSON.stringify({
          ...request,
          stream: true,
        }),
        signal, // Pass AbortSignal to fetch
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('Response body is not readable');
      }

      const decoder = new TextDecoder();
      let buffer = '';
      let accumulatedData = ''; // Accumulate multi-line data

      while (true) {
        const { done, value } = await reader.read();
        
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        // Split on both \r\n and \n for better CRLF compatibility
        const lines = buffer.split(/\r?\n/);
        buffer = lines.pop() || '';

        for (const line of lines) {
          // Skip heartbeat comments
          if (line.startsWith(':')) {
            continue;
          }
          
          // Handle event: lines (optional, for better compatibility)
          if (line.startsWith('event:')) {
            // Could handle different event types here if needed
            continue;
          }
          
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            accumulatedData += data;
            // Don't parse yet - continue accumulating
            continue;
          }
          
          // Empty line signals end of event - parse accumulated data
          if (!line.trim() && accumulatedData) {
            if (accumulatedData === '[DONE]') {
              return;
            }

            try {
              const parsed = JSON.parse(accumulatedData);
              yield parsed as StreamChunk;
            } catch (e) {
              // If JSON parsing fails but data is non-empty, treat as plain text token
              if (accumulatedData.trim()) {
                console.warn('SSE data is not valid JSON, treating as plain text token:', accumulatedData);
                yield {
                  type: 'token',
                  token: accumulatedData,
                } as StreamChunk;
              } else {
                console.error('Failed to parse SSE data:', e, 'Data:', accumulatedData);
              }
            }
            
            // Reset accumulator
            accumulatedData = '';
          }
        }
      }
      
      // Handle any remaining accumulated data
      if (accumulatedData) {
        if (accumulatedData !== '[DONE]') {
          try {
            const parsed = JSON.parse(accumulatedData);
            yield parsed as StreamChunk;
          } catch (e) {
            // If JSON parsing fails but data is non-empty, treat as plain text token
            if (accumulatedData.trim()) {
              console.warn('Final SSE data is not valid JSON, treating as plain text token:', accumulatedData);
              yield {
                type: 'token',
                token: accumulatedData,
              } as StreamChunk;
            } else {
              console.error('Failed to parse final SSE data:', e, 'Data:', accumulatedData);
            }
          }
        }
      }
    } catch (error) {
      console.error('Stream error:', error);
      yield {
        type: 'error',
        error: error instanceof Error ? error.message : 'Unknown streaming error',
      };
    }
  }

  // Case Methods
  async summarizeCase(caseId: number): Promise<CaseSummaryResponse> {
    try {
      const response = await this.client.post<CaseSummaryResponse>(`/api/case/${caseId}/summary`);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async compareCases(request: CompareCasesRequest): Promise<CaseComparisonResponse> {
    try {
      const response = await this.client.post<CaseComparisonResponse>('/api/cases/compare', request);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  // Session Methods
  async createSession(request?: SessionCreateRequest): Promise<SessionResponse> {
    try {
      const response = await this.client.post<SessionResponse>('/api/session', request || {});
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async getSession(sessionId: string): Promise<SessionResponse> {
    try {
      const response = await this.client.get<SessionResponse>(`/api/session/${sessionId}`);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async getSessionHistory(sessionId: string, maxTurns?: number): Promise<SessionTurn[]> {
    try {
      const params = maxTurns ? { max_turns: maxTurns } : {};
      const response = await this.client.get<SessionTurn[]>(`/api/session/${sessionId}/history`, { params });
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async deleteSession(sessionId: string): Promise<void> {
    try {
      await this.client.delete(`/api/session/${sessionId}`);
    } catch (error) {
      throw this.handleError(error);
    }
  }

  // Utility Methods
  async getHealth(): Promise<HealthResponse> {
    try {
      const response = await this.client.get<HealthResponse>('/api/health');
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async getStats(): Promise<StatsResponse> {
    try {
      const response = await this.client.get<StatsResponse>('/api/stats');
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async getVersion(): Promise<any> {
    try {
      const response = await this.client.get('/api/version');
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  // Error Handling
  private handleError(error: any): ErrorResponse {
    if (error && typeof error === 'object' && 'error' in error) {
      return error as ErrorResponse;
    }

    return {
      error: error instanceof Error ? error.message : 'An unknown error occurred',
      status_code: 500,
    };
  }
}

const apiClient = new APIClient();
export default apiClient;
