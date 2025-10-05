import type { DashboardData, AnalysisRun, Topic } from '@/lib/types'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api'

class ApiClient {
  public baseURL: string
  private authToken: string | null = null

  constructor(baseURL: string) {
    this.baseURL = baseURL
    // Try to get auth token from localStorage on client side
    if (typeof window !== 'undefined') {
      this.authToken = localStorage.getItem('dev_auth_token')
    }
  }

  setAuthToken(token: string | null) {
    this.authToken = token
    if (typeof window !== 'undefined') {
      if (token) {
        localStorage.setItem('dev_auth_token', token)
      } else {
        localStorage.removeItem('dev_auth_token')
      }
    }
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseURL}${endpoint}`
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...(options.headers as Record<string, string> || {}),
    }

    if (this.authToken) {
      headers['Authorization'] = `Bearer ${this.authToken}`
    }

    const response = await fetch(url, {
      ...options,
      headers,
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`)
    }

    return response.json()
  }

  // Auth endpoints
  async login(password: string) {
    const response = await this.request<{authenticated: boolean, role: string, message: string, token: string}>(
      '/auth/login',
      {
        method: 'POST',
        body: JSON.stringify({ password }),
      }
    )
    
    if (response.token) {
      this.setAuthToken(response.token)
    }
    
    return response
  }

  async logout() {
    await this.request('/auth/logout', { method: 'POST' })
    this.setAuthToken(null)
  }

  // Dashboard endpoints
  async getDashboard(): Promise<DashboardData> {
    return this.request('/dashboard')
  }

  async getDashboardMetrics(): Promise<DashboardData> {
    return this.request('/dashboard/metrics')
  }

  async getQuestions(params: {
    limit?: number
    offset?: number
    country?: string
    state?: string
    topic?: string
    date_from?: string
    date_to?: string
  } = {}) {
    const searchParams = new URLSearchParams()
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null) {
        searchParams.append(key, value.toString())
      }
    })
    
    const queryString = searchParams.toString()
    return this.request(`/dashboard/questions${queryString ? `?${queryString}` : ''}`)
  }

  async exportQuestions(params: {
    format?: 'csv' | 'json'
    country?: string
    state?: string
    topic?: string
    date_from?: string
    date_to?: string
  } = {}) {
    const searchParams = new URLSearchParams()
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null) {
        searchParams.append(key, value.toString())
      }
    })
    
    const queryString = searchParams.toString()
    return this.request(`/dashboard/export${queryString ? `?${queryString}` : ''}`)
  }

  async getChartData(chartType: string, params: {
    time_range?: string
    country?: string
    topic?: string
  } = {}) {
    const searchParams = new URLSearchParams({ chart_type: chartType })
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null) {
        searchParams.append(key, value.toString())
      }
    })
    
    return this.request(`/dashboard/charts/data?${searchParams.toString()}`)
  }

  // Upload endpoints (developer only)
  async uploadFile(file: File, onProgress?: (progress: number) => void): Promise<{
    upload_id: string;
    filename: string;
    size: number;
    status: string;
  }> {
    const formData = new FormData()
    formData.append('file', file)
    
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest()
      
      xhr.upload.addEventListener('progress', (e) => {
        if (e.lengthComputable && onProgress) {
          const progress = (e.loaded / e.total) * 100
          onProgress(progress)
        }
      })
      
      xhr.addEventListener('load', () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            resolve(JSON.parse(xhr.responseText))
          } catch {
            reject(new Error('Invalid response format'))
          }
        } else {
          reject(new Error(`Upload failed: ${xhr.statusText}`))
        }
      })
      
      xhr.addEventListener('error', () => {
        reject(new Error('Upload failed'))
      })
      
      xhr.open('POST', `${this.baseURL}/upload/upload`)
      if (this.authToken) {
        xhr.setRequestHeader('Authorization', `Bearer ${this.authToken}`)
      }
      xhr.send(formData)
    })
  }

  async processFile(uploadId: string): Promise<{
    processing_id: string;
    upload_id: string;
    status: string;
    message: string;
  }> {
    return this.request(`/upload/process/${uploadId}`, {
      method: 'POST',
    })
  }

  async getProcessingStatus(processingId: string) {
    return this.request(`/upload/status/${processingId}`)
  }

  async getUploadHistory() {
    return this.request('/upload/history')
  }

  async uploadQuestions(file: File) {
    const formData = new FormData()
    formData.append('file', file)
    
    const response = await fetch(`${this.baseURL}/upload/questions`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.authToken}`,
      },
      body: formData,
    })
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      throw new Error(errorData.detail || `Upload failed: ${response.statusText}`)
    }
    
    return response.json()
  }

  async getUploadedFiles() {
    return this.request('/upload/uploads')
  }

  // Analysis endpoints (developer only)
  async startAnalysis(params: {
    mode?: 'sample' | 'all'
    sample_size?: number
  } = {}): Promise<{ run_id: string }> {
    return this.request('/analysis/start', {
      method: 'POST',
      body: JSON.stringify(params),
    })
  }

  async getAnalysisRuns(): Promise<AnalysisRun[]> {
    const response = await this.request<{runs: AnalysisRun[]}>('/analysis/runs')
    return response.runs
  }

  async getTopics(runId: string): Promise<Topic[]> {
    return this.request(`/analysis/runs/${runId}/topics`)
  }

  async exportResults(runId: string) {
    const response = await fetch(`${this.baseURL}/analysis/runs/${runId}/export`, {
      headers: {
        'Authorization': `Bearer ${this.authToken}`,
      },
    })
    
    if (!response.ok) {
      throw new Error(`Export failed: ${response.statusText}`)
    }
    
    return response.blob()
  }

  async runAnalysis(params: {
    mode?: 'sample' | 'all'
    sample_size?: number
  } = {}) {
    return this.request('/analysis/run', {
      method: 'POST',
      body: JSON.stringify(params),
    })
  }

  async getAnalysisStatus(jobId: string) {
    return this.request(`/analysis/status/${jobId}`)
  }

  async getAnalysisHistory(limit: number = 20) {
    return this.request(`/analysis/history?limit=${limit}`)
  }

  async clearAnalysisData(confirm: boolean = false) {
    return this.request(`/analysis/clear?confirm=${confirm}`, {
      method: 'DELETE',
    })
  }

  // Server-Sent Events for analysis progress
  createAnalysisProgressStream(jobId: string): EventSource {
    const url = `${this.baseURL}/analysis/${jobId}/progress`
    const eventSource = new EventSource(url)
    
    // Add auth header (Note: EventSource doesn't support custom headers,
    // so we'll need to handle auth differently for SSE)
    return eventSource
  }
}

export const apiClient = new ApiClient(API_BASE_URL)
export default apiClient