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
        // Add session_id only for chat-related endpoints
        if (typeof window !== 'undefined' && config.url) {
          const isChatEndpoint = config.url.includes('/chat') || config.url.includes('/session');
          if (isChatEndpoint) {
            const sessionId = localStorage.getItem('session_id');
            if (sessionId && config.data) {
              // Remove quotes if the value is JSON stringified
              const cleanSessionId = sessionId.replace(/^"(.*)"$/, '$1');
              config.data.session_id = config.data.session_id || cleanSessionId;
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
      const response = await this.client.post<SearchResultResponse>('/api/search/keyword', request);
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

      while (true) {
        const { done, value } = await reader.read();
        
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            
            if (data === '[DONE]') {
              return;
            }

            try {
              const parsed = JSON.parse(data);
              yield parsed as StreamChunk;
            } catch (e) {
              console.error('Failed to parse SSE data:', e);
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

  async getSessionHistory(sessionId: string, maxTurns?: number): Promise<any[]> {
    try {
      const params = maxTurns ? { max_turns: maxTurns } : {};
      const response = await this.client.get<any[]>(`/api/session/${sessionId}/history`, { params });
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
