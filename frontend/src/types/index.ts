export interface Question {
  id: string
  text: string
  date?: string
  country?: string
  state?: string
  userLanguage?: string
  similarityScore?: number
  matchedTopic?: string
  isNewTopic: boolean
  topicId?: string
  topic?: Topic
  createdAt: string
  updatedAt: string
}

export interface Topic {
  id: string
  name: string
  subtopic?: string
  representativeQuestion?: string
  isDiscovered: boolean
  discoveredAt?: string
  approvalStatus: 'pending' | 'approved' | 'rejected'
  createdAt: string
  updatedAt: string
}

export interface AnalysisRun {
  id: string
  status: 'queued' | 'running' | 'completed' | 'failed'
  progress: number
  message?: string
  mode: 'sample' | 'all'
  sampleSize?: number
  totalQuestions?: number
  similarQuestions?: number
  newTopicsDiscovered?: number
  similarQuestionsFile?: string
  newTopicsFile?: string
  completeReviewFile?: string
  error?: string
  startedAt: string
  completedAt?: string
  failedAt?: string
}

export interface DashboardMetrics {
  recent_insights: {
    peak_question_time: string
    most_active_region: string
    top_topic: {
      name: string
      percentage: number
    }
    question_rate_trend: {
      unique_percentage: number
      previous_percentage: number
      trend: 'increasing' | 'decreasing' | 'stable'
    }
  }
  totals: {
    total_questions: number
    total_topics: number
    total_countries: number
    last_analysis?: string
  }
}

export interface ChartData {
  chart_type: string
  time_range: string
  data: any[]
  message?: string
}

export interface FilterParams {
  limit?: number
  offset?: number
  country?: string
  state?: string
  topic?: string
  date_from?: string
  date_to?: string
  time_range?: string
}

export interface UploadValidation {
  status: 'valid' | 'error'
  warnings: string[]
  errors: string[]
  stats: {
    total_rows: number
    valid_questions: number
    columns: string[]
    sample_questions: string[]
  }
}

export interface UploadResponse {
  message: string
  filename: string
  rows_count: number
  validation: UploadValidation
  file_path: string
}

export interface AuthResponse {
  authenticated: boolean
  role: string
  message: string
  token: string
}