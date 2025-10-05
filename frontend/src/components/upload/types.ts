export type UploadStatus = "idle" | "uploading" | "uploaded" | "processing" | "completed" | "error";

export interface ProcessingProgress {
  stage: string;
  progress: number;
  message: string;
}

export interface UploadDialogProps {
  open: boolean;
  onClose: () => void;
  onSuccess: (processingResult: any) => void;
}

export interface ProcessingResult {
  status: string;
  statistics?: {
    valid_questions_extracted: number;
    total_rows_processed: number;
    [key: string]: any;
  };
  database?: {
    write_successful: boolean;
    write_result?: {
      rows_written: number;
      duplicates_skipped: number;
      data_errors_skipped: number;
      insertion_errors: number;
      total_processed: number;
    };
    write_error?: {
      message: string;
    };
  };
  google_sheets?: {
    write_successful: boolean;
    write_error?: {
      message: string;
    };
  };
}

export interface UploadState {
  file: File | null;
  status: UploadStatus;
  error: string;
  uploadProgress: number;
  processingProgress: ProcessingProgress;
  uploadId: string;
  processingId: string;
  eventSource: EventSource | null;
  processingResult: ProcessingResult | null;
  processingTimeout: NodeJS.Timeout | null;
  isDragOver: boolean;
}