// Dashboard data types
export interface DashboardData {
  question_count: number;
  topic_count: number;
  last_analysis: string | null;
  last_updated: string | null;
  coverage_percentage: number;
  has_questions: boolean;
}

// Analysis run types
export interface AnalysisRun {
  id: string;
  status: 'running' | 'completed' | 'failed';
  created_at: string;
  completed_at?: string;
  error_message?: string;
  total_questions: number;
  topics_discovered: number;
  parameters: {
    mode: 'sample' | 'all';
    sample_size?: number;
  };
}

// Topic types
export interface Topic {
  id: string;
  name: string;
  description: string;
  question_count: number;
  confidence_score: number;
  keywords: string[];
  representative_questions: string[];
  run_id: string;
  created_at: string;
}

// Question types
export interface Question {
  id: string;
  text: string;
  country?: string;
  state?: string;
  topic_id?: string;
  confidence_score?: number;
  created_at: string;
  uploaded_at: string;
}

// Upload types
export interface UploadResult {
  success: boolean;
  message: string;
  questions_processed: number;
  questions_added: number;
  file_name: string;
}

// Auth types
export interface AuthResponse {
  authenticated: boolean;
  role: string;
  message: string;
  token: string;
}

// API Error types
export interface ApiError {
  detail: string;
  code?: string;
}

// Progress tracking types
export interface ProgressUpdate {
  stage: string;
  progress: number;
  message: string;
  timestamp: string;
}

// Chart data types
export interface ChartData {
  labels: string[];
  datasets: {
    label: string;
    data: number[];
    backgroundColor?: string[];
    borderColor?: string[];
  }[];
}

// Filter types
export interface DashboardFilters {
  country?: string;
  state?: string;
  topic?: string;
  date_from?: string;
  date_to?: string;
}