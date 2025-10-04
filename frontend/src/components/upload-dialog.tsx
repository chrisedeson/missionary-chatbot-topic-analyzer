"use client";

import { useState, useRef, useEffect } from "react";
import { Upload, File, X, Loader2, CheckCircle, AlertCircle, Database } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { apiClient } from "@/lib/api";

interface UploadDialogProps {
  open: boolean;
  onClose: () => void;
  onSuccess: (processingResult: any) => void;
}

type UploadStatus = "idle" | "uploading" | "uploaded" | "processing" | "completed" | "error";

interface ProcessingProgress {
  stage: string;
  progress: number;
  message: string;
}

export function UploadDialog({ open, onClose, onSuccess }: UploadDialogProps) {
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<UploadStatus>("idle");
  const [error, setError] = useState("");
  const [uploadProgress, setUploadProgress] = useState(0);
  const [processingProgress, setProcessingProgress] = useState<ProcessingProgress>({
    stage: "validation",
    progress: 0,
    message: "Preparing..."
  });
  const [uploadId, setUploadId] = useState<string>("");
  const [processingId, setProcessingId] = useState<string>("");
  const [eventSource, setEventSource] = useState<EventSource | null>(null);
  const [processingResult, setProcessingResult] = useState<any>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Clean up event source on unmount or dialog close
  useEffect(() => {
    return () => {
      if (eventSource) {
        eventSource.close();
        setEventSource(null);
      }
    };
  }, [eventSource]);

  useEffect(() => {
    if (!open) {
      if (eventSource) {
        eventSource.close();
        setEventSource(null);
      }
    }
  }, [open, eventSource]);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setStatus("idle");
      setError("");
      setUploadId("");
      setProcessingId("");
    }
  };

  const handleDrop = (event: React.DragEvent) => {
    event.preventDefault();
    const droppedFile = event.dataTransfer.files[0];
    if (droppedFile) {
      setFile(droppedFile);
      setStatus("idle");
      setError("");
      setUploadId("");
      setProcessingId("");
    }
  };

  const handleDragOver = (event: React.DragEvent) => {
    event.preventDefault();
  };

  const handleUpload = async () => {
    if (!file) return;

    setStatus("uploading");
    setUploadProgress(0);
    setError("");

    try {
      // Step 1: Upload file
      const uploadResponse = await apiClient.uploadFile(file, (progressPercent) => {
        setUploadProgress(progressPercent);
      });
      
      setUploadId(uploadResponse.upload_id);
      setStatus("uploaded");
      
      // Automatically start processing
      await handleStartProcessing(uploadResponse.upload_id);
      
    } catch (err) {
      setStatus("error");
      setError(err instanceof Error ? err.message : "Upload failed");
    }
  };

  const handleStartProcessing = async (uploadIdToProcess?: string) => {
    const targetUploadId = uploadIdToProcess || uploadId;
    if (!targetUploadId) return;

    setStatus("processing");
    setProcessingProgress({ stage: "validation", progress: 0, message: "Starting processing..." });

    try {
      // Step 2: Start processing
      const processResponse = await apiClient.processFile(targetUploadId);
      setProcessingId(processResponse.processing_id);

      // Step 3: Listen for real-time progress (add small delay)
      setTimeout(() => {
        const authToken = localStorage.getItem('dev_auth_token');
        const sseUrl = `${process.env.NEXT_PUBLIC_API_URL}/upload/progress/${processResponse.processing_id}`;
        
        console.log("Connecting to SSE:", sseUrl);
        
        const source = new EventSource(sseUrl);
        setEventSource(source);

        source.onopen = () => {
          console.log("SSE connection opened");
        };

        source.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            console.log("SSE message received:", data);
            
            if (data.type === "connected") {
              console.log("Connected to processing stream");
              return;
            }

            if (data.type === "error") {
              setStatus("error");
              setError(data.message);
              source.close();
              return;
            }

            if (data.type === "heartbeat") {
              // Ignore heartbeat messages
              return;
            }

            // Update progress
            setProcessingProgress({
              stage: data.stage || "processing",
              progress: data.progress || 0,
              message: data.message || "Processing..."
            });

            // Check if completed
            if (data.stage === "completion" && data.progress === 100) {
              setStatus("completed");
              source.close();
              
              // Get final results
              setTimeout(async () => {
                try {
                  const result = await apiClient.getProcessingStatus(processResponse.processing_id);
                  setProcessingResult(result);
                  
                  setTimeout(() => {
                    onSuccess(result);
                    handleReset();
                  }, 2000);
                } catch (error) {
                  console.error("Failed to get processing results:", error);
                }
              }, 1000);
            }
          } catch (parseError) {
            console.error("Failed to parse progress data:", parseError);
          }
        };

        source.onerror = (error) => {
          console.error("SSE connection error:", error);
          console.log("SSE readyState:", source.readyState);
          console.log("SSE url:", source.url);
          
          // Don't immediately fail - SSE connections can be flaky
          // Only fail if we've been trying for a while
          setTimeout(() => {
            if (source.readyState === EventSource.CLOSED) {
              setStatus("error");
              setError("Connection lost during processing. Please check your network connection.");
              source.close();
            }
          }, 5000); // Wait 5 seconds before giving up
        };
      }, 500); // Wait 500ms before connecting to SSE

    } catch (err) {
      setStatus("error");
      setError(err instanceof Error ? err.message : "Processing failed");
    }
  };

  const handleReset = () => {
    if (eventSource) {
      eventSource.close();
      setEventSource(null);
    }
    setFile(null);
    setStatus("idle");
    setError("");
    setUploadProgress(0);
    setProcessingProgress({ stage: "validation", progress: 0, message: "Preparing..." });
    setUploadId("");
    setProcessingId("");
    setProcessingResult(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const isValidFile = (file: File) => {
    return file.name.endsWith(".csv");
  };

  const getStageLabel = (stage: string) => {
    const labels: Record<string, string> = {
      validation: "Validating",
      cleaning: "Cleaning Data", 
      sheets_update: "Updating Sheets",
      completion: "Completing",
      error: "Error"
    };
    return labels[stage] || "Processing";
  };

  return (
    <Sheet open={open} onOpenChange={onClose}>
      <SheetContent className="w-full sm:max-w-lg">
        <SheetHeader>
          <SheetTitle className="flex items-center gap-2">
            <Upload className="w-5 h-5" />
            Upload Question Data
          </SheetTitle>
          <SheetDescription>
            Upload a CSV file containing student questions for analysis
          </SheetDescription>
        </SheetHeader>

        <div className="mt-6 space-y-6">
          {/* File Drop Zone */}
          <div
            className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
              status === "uploading" || status === "processing"
                ? "border-muted-foreground/25 bg-muted/25"
                : "border-muted-foreground/25 hover:border-muted-foreground/50 cursor-pointer"
            }`}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onClick={() => !status.match(/uploading|processing/) && fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv"
              onChange={handleFileSelect}
              className="hidden"
              disabled={status === "uploading" || status === "processing"}
            />

            {status === "completed" ? (
              <div className="text-green-600">
                <CheckCircle className="w-12 h-12 mx-auto mb-4" />
                <p className="font-medium">Processing Complete!</p>
                <p className="text-sm text-muted-foreground">
                  {processingResult?.statistics?.valid_questions_extracted || 0} questions extracted
                </p>
              </div>
            ) : status === "error" ? (
              <div className="text-red-600">
                <AlertCircle className="w-12 h-12 mx-auto mb-4" />
                <p className="font-medium">Processing Failed</p>
                <p className="text-sm text-muted-foreground">{error}</p>
              </div>
            ) : file ? (
              <div>
                <File className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
                <p className="font-medium">{file.name}</p>
                <p className="text-sm text-muted-foreground">
                  {(file.size / 1024 / 1024).toFixed(2)} MB
                </p>
                {!isValidFile(file) && (
                  <p className="text-sm text-red-600 mt-2">
                    Only CSV files are supported.
                  </p>
                )}
              </div>
            ) : (
              <div>
                <Upload className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
                <p className="font-medium">Choose a CSV file or drag it here</p>
                <p className="text-sm text-muted-foreground">
                  Questions will be extracted and cleaned automatically
                </p>
              </div>
            )}
          </div>

          {/* Upload Progress Bar */}
          {status === "uploading" && (
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Uploading...</span>
                <span>{Math.round(uploadProgress)}%</span>
              </div>
              <div className="w-full bg-muted rounded-full h-2">
                <div
                  className="bg-primary h-2 rounded-full transition-all duration-300"
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
            </div>
          )}

          {/* Processing Progress */}
          {status === "processing" && (
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <Database className="w-4 h-4" />
                <span className="text-sm font-medium">{getStageLabel(processingProgress.stage)}</span>
              </div>
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>{processingProgress.message}</span>
                  <span>{Math.round(processingProgress.progress)}%</span>
                </div>
                <div className="w-full bg-muted rounded-full h-2">
                  <div
                    className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${processingProgress.progress}%` }}
                  />
                </div>
              </div>
            </div>
          )}

          {/* File Format Guide */}
          <div className="space-y-3">
            <h4 className="font-medium text-sm">Expected CSV Format:</h4>
            <div className="text-xs text-muted-foreground space-y-1">
              <p>• CSV file with questions in any column</p>
              <p>• Supports kwargs JSON format (from Langfuse exports)</p>
              <p>• Automatic data cleaning and extraction</p>
              <p>• Maximum file size: 50 MB</p>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-3">
            <Button
              onClick={handleUpload}
              disabled={!file || !isValidFile(file) || status === "uploading" || status === "processing"}
              className="flex-1"
            >
              {status === "uploading" ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Uploading...
                </>
              ) : status === "processing" ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Processing...
                </>
              ) : (
                <>
                  <Upload className="w-4 h-4 mr-2" />
                  Upload & Process
                </>
              )}
            </Button>

            {file && status !== "uploading" && status !== "processing" && (
              <Button variant="outline" onClick={handleReset}>
                <X className="w-4 h-4 mr-2" />
                Clear
              </Button>
            )}
          </div>
        </div>
      </SheetContent>
    </Sheet>
  );
}