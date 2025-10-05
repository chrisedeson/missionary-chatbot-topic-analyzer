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
  const [processingTimeout, setProcessingTimeout] = useState<NodeJS.Timeout | null>(null);
  
  // Additional missing state variables
  const [isDragOver, setIsDragOver] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [fileName, setFileName] = useState("");
  const [fileSize, setFileSize] = useState("");
  const [isUploaded, setIsUploaded] = useState(false);
  const [processingResults, setProcessingResults] = useState<any>(null);
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Clean up event source on unmount or dialog close
  useEffect(() => {
    return () => {
      if (eventSource) {
        eventSource.close();
        setEventSource(null);
      }
      if (processingTimeout) {
        clearTimeout(processingTimeout);
        setProcessingTimeout(null);
      }
    };
  }, [eventSource, processingTimeout]);

  useEffect(() => {
    if (!open) {
      if (eventSource) {
        eventSource.close();
        setEventSource(null);
      }
    }
  }, [open, eventSource]);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (files && files[0]) {
      const file = files[0]
      setFile(file)  // Use the main file state
      setSelectedFile(file)
      setFileName(file.name)
      
      // Properly calculate file size
      const sizeInMB = file.size / (1024 * 1024)
      setFileSize(sizeInMB.toFixed(2))
      
      setError('')
      setIsUploaded(false)
      setProcessingId('')
      setProcessingResults(null)
    }
  }

    const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDragOver(false)
    
    const files = e.dataTransfer.files
    if (files && files[0]) {
      const file = files[0]
      setFile(file)  // Use the main file state
      setSelectedFile(file)
      setFileName(file.name)
      
      // Properly calculate file size for drag and drop
      const sizeInMB = file.size / (1024 * 1024)
      setFileSize(sizeInMB.toFixed(2))
      
      setError('')
      setIsUploaded(false)
      setProcessingId('')
      setProcessingResults(null)
    }
  }

  const handleDragOver = (event: React.DragEvent) => {
    event.preventDefault();
  };

  const handleDragEnter = (event: React.DragEvent) => {
    event.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (event: React.DragEvent) => {
    event.preventDefault();
    setIsDragOver(false);
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
        const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api';
        const sseUrl = `${apiUrl}/upload/progress/${processResponse.processing_id}`;
        
        console.log("Connecting to SSE:", sseUrl);
        console.log("API URL:", apiUrl);
        
        const source = new EventSource(sseUrl);
        setEventSource(source);

        // Set a timeout to prevent infinite loading (30 seconds)
        const timeout = setTimeout(() => {
          if (status === "processing") {
            setStatus("completed");
            source.close();
            setProcessingProgress({
              stage: "completion",
              progress: 100,
              message: "Processing completed (timed out waiting for updates)"
            });
            onSuccess({ status: "completed", message: "Processing completed" });
          }
        }, 30000);
        setProcessingTimeout(timeout);

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

            // Handle Google Sheets errors (non-fatal)
            if (data.stage === "sheets_error") {
              // Continue processing but note the error
              console.warn("Google Sheets error:", data.message);
              // Set a timeout to complete processing if Google Sheets fails
              setTimeout(() => {
                if (status === "processing") {
                  setStatus("completed");
                  source.close();
                  setProcessingProgress({
                    stage: "completion",
                    progress: 100,
                    message: "Processing completed (Google Sheets unavailable)"
                  });
                  
                  // Get final results
                  setTimeout(async () => {
                    try {
                      const result = await apiClient.getProcessingStatus(processResponse.processing_id);
                      setProcessingResult(result);
                      onSuccess(result);
                    } catch (error) {
                      console.error("Failed to get processing results:", error);
                      // Still complete the process even if we can't get results
                      onSuccess({ status: "completed", google_sheets: { status: "error", message: "Google Sheets unavailable" } });
                    }
                  }, 1000);
                }
              }, 3000); // Wait 3 seconds then complete
            }

            // Check if completed
            if (data.stage === "completion" && data.progress === 100) {
              setStatus("completed");
              source.close();
              if (processingTimeout) {
                clearTimeout(processingTimeout);
                setProcessingTimeout(null);
              }
              
              // Get final results
              setTimeout(async () => {
                try {
                  const result = await apiClient.getProcessingStatus(processResponse.processing_id);
                  setProcessingResult(result);
                  
                  setTimeout(() => {
                    onSuccess(result);
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
      }, 100); // Wait 100ms before connecting to SSE

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
    if (processingTimeout) {
      clearTimeout(processingTimeout);
      setProcessingTimeout(null);
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
      <SheetContent className="w-full sm:max-w-lg flex flex-col">
        <SheetHeader>
          <SheetTitle className="flex items-center gap-2">
            <Upload className="w-5 h-5" />
            Upload Question Data
          </SheetTitle>
          <SheetDescription>
            Upload a CSV file containing student questions for analysis
          </SheetDescription>
        </SheetHeader>

        {/* Scrollable content container with slim dark-mode scrollbar */}
        <div className="flex-1 overflow-y-auto pr-2 mt-6 space-y-6 scrollbar-thin scrollbar-track-gray-100 dark:scrollbar-track-gray-800 scrollbar-thumb-gray-300 dark:scrollbar-thumb-gray-600">
          {/* File Drop Zone */}
          <div
            className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
              isDragOver
                ? "border-primary bg-primary/5"
                : status === "uploading" || status === "processing"
                ? "border-muted-foreground/25 bg-muted/25"
                : "border-muted-foreground/25 hover:border-muted-foreground/50 cursor-pointer"
            }`}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragEnter={handleDragEnter}
            onDragLeave={handleDragLeave}
            onClick={() => !status.match(/uploading|processing/) && fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv"
              onChange={handleFileChange}
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
                
                {/* Database Status */}
                {processingResult?.database && (
                  <div className="mt-3 p-3 bg-blue-50 dark:bg-blue-950/20 rounded-lg">
                    <div className="flex items-center gap-2 text-blue-700 dark:text-blue-300 mb-2">
                      <Database className="w-4 h-4" />
                      <span className="font-medium text-sm">Database Status</span>
                    </div>
                    {processingResult.database.write_successful ? (
                      <div className="text-xs">
                        {processingResult.database.write_result?.rows_written > 0 ? (
                          <div className="text-blue-600 dark:text-blue-400">
                            <p className="font-medium mb-1">✅ Database Write Successful</p>
                            <div className="space-y-1">
                              <p>� {processingResult.database.write_result?.rows_written || 0} new questions saved</p>
                              {processingResult.database.write_result?.duplicates_skipped > 0 && (
                                <p>🔄 {processingResult.database.write_result.duplicates_skipped} duplicates skipped</p>
                              )}
                              {processingResult.database.write_result?.data_errors_skipped > 0 && (
                                <p>⚠️ {processingResult.database.write_result.data_errors_skipped} rows with data errors skipped</p>
                              )}
                              {processingResult.database.write_result?.insertion_errors > 0 && (
                                <p>❌ {processingResult.database.write_result.insertion_errors} database insertion errors</p>
                              )}
                              <p>📊 {processingResult.database.write_result?.total_processed || 0} total questions processed</p>
                            </div>
                            {(processingResult.database.write_result?.duplicates_skipped > 0 || 
                              processingResult.database.write_result?.data_errors_skipped > 0 || 
                              processingResult.database.write_result?.insertion_errors > 0) && (
                              <div className="mt-2 p-2 bg-blue-50 dark:bg-blue-950/30 border border-blue-200 dark:border-blue-800 rounded">
                                <p className="text-blue-700 dark:text-blue-300 text-xs">
                                  💡 Rows skipped due to: duplicates (prevent redundant data), data errors (missing/invalid fields), or database constraints.
                                  Only valid, unique questions are stored for analysis.
                                </p>
                              </div>
                            )}
                          </div>
                        ) : (
                          <div className="text-yellow-600 dark:text-yellow-400">
                            <p className="font-medium mb-1">🔄 All Questions Were Duplicates</p>
                            <div className="space-y-1">
                              <p>🔄 {processingResult.database.write_result?.duplicates_skipped || 0} duplicates skipped</p>
                              <p>📊 {processingResult.database.write_result?.total_processed || 0} total questions processed</p>
                            </div>
                            <div className="mt-2 p-2 bg-yellow-50 dark:bg-yellow-950/30 border border-yellow-200 dark:border-yellow-800 rounded">
                              <p className="text-yellow-700 dark:text-yellow-300 text-xs">
                                💡 No new questions were added because all questions from this file 
                                already exist in the database.
                              </p>
                            </div>
                          </div>
                        )}
                      </div>
                    ) : (
                      <div className="text-xs text-red-600 dark:text-red-400">
                        <p className="mb-2">❌ Database write failed:</p>
                        <div className="bg-red-50 dark:bg-red-950/30 p-2 rounded border">
                          <p className="text-xs">{processingResult.database.write_error?.message}</p>
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* Google Sheets Status */}
                {processingResult?.google_sheets && (
                  <div className="mt-3 p-3 bg-green-50 dark:bg-green-950/20 rounded-lg">
                    <div className="flex items-center gap-2 text-green-700 dark:text-green-300 mb-2">
                      <Database className="w-4 h-4" />
                      <span className="font-medium text-sm">Google Sheets Status</span>
                    </div>
                    {processingResult.google_sheets.write_successful ? (
                      <div className="text-xs text-green-600 dark:text-green-400">
                        <p className="font-medium">✅ Google Sheets sync completed</p>
                      </div>
                    ) : (
                      <div className="text-xs text-red-600 dark:text-red-400">
                        <p className="font-medium">❌ Google Sheets sync failed</p>
                        <div className="bg-red-50 dark:bg-red-950/30 p-2 rounded border mt-1">
                          <p className="text-xs">{processingResult.google_sheets.write_error?.message}</p>
                        </div>
                      </div>
                    )}
                  </div>
                )}

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
            {status === "completed" ? (
              // Success state: Only show Clear button
              <Button variant="outline" onClick={handleReset} className="flex-1">
                <X className="w-4 h-4 mr-2" />
                Close & Clear
              </Button>
            ) : status === "error" ? (
              // Error state: Show both retry and clear buttons
              <>
                <Button
                  onClick={handleUpload}
                  disabled={!file || !isValidFile(file)}
                  className="flex-1"
                >
                  <Upload className="w-4 h-4 mr-2" />
                  Retry Upload
                </Button>
                <Button variant="outline" onClick={handleReset}>
                  <X className="w-4 h-4 mr-2" />
                  Clear
                </Button>
              </>
            ) : (
              // Default state: Show upload and optional clear buttons
              <>
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
              </>
            )}
          </div>
        </div>
      </SheetContent>
    </Sheet>
  );
}