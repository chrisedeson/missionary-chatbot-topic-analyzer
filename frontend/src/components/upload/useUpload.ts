import { useState, useRef, useEffect } from "react";
import { apiClient } from "@/lib/api";
import { UploadStatus, ProcessingProgress, UploadState, ProcessingResult } from "./types";

export function useUpload(onSuccess: (result: ProcessingResult) => void) {
  const [state, setState] = useState<UploadState>({
    file: null,
    status: "idle",
    error: "",
    uploadProgress: 0,
    processingProgress: {
      stage: "validation",
      progress: 0,
      message: "Preparing..."
    },
    uploadId: "",
    processingId: "",
    eventSource: null,
    processingResult: null,
    processingTimeout: null,
    isDragOver: false
  });

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Clean up event source on unmount
  useEffect(() => {
    return () => {
      if (state.eventSource) {
        state.eventSource.close();
      }
      if (state.processingTimeout) {
        clearTimeout(state.processingTimeout);
      }
    };
  }, [state.eventSource, state.processingTimeout]);

  const updateState = (updates: Partial<UploadState>) => {
    setState(prev => ({ ...prev, ...updates }));
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files[0]) {
      const file = files[0];
      updateState({
        file,
        error: "",
        status: "idle",
        processingId: "",
        processingResult: null
      });
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    updateState({ isDragOver: false });
    
    const files = e.dataTransfer.files;
    if (files && files[0]) {
      const file = files[0];
      updateState({
        file,
        error: "",
        status: "idle",
        processingId: "",
        processingResult: null
      });
    }
  };

  const handleDragOver = (event: React.DragEvent) => {
    event.preventDefault();
  };

  const handleDragEnter = (event: React.DragEvent) => {
    event.preventDefault();
    updateState({ isDragOver: true });
  };

  const handleDragLeave = (event: React.DragEvent) => {
    event.preventDefault();
    updateState({ isDragOver: false });
  };

  const handleUpload = async () => {
    if (!state.file) return;

    updateState({
      status: "uploading",
      uploadProgress: 0,
      error: ""
    });

    try {
      // Step 1: Upload file
      const uploadResponse = await apiClient.uploadFile(state.file, (progressPercent) => {
        updateState({ uploadProgress: progressPercent });
      });
      
      updateState({
        uploadId: uploadResponse.upload_id,
        status: "uploaded"
      });
      
      // Automatically start processing
      await handleStartProcessing(uploadResponse.upload_id);
      
    } catch (err) {
      updateState({
        status: "error",
        error: err instanceof Error ? err.message : "Upload failed"
      });
    }
  };

  const handleStartProcessing = async (uploadIdToProcess?: string) => {
    const targetUploadId = uploadIdToProcess || state.uploadId;
    if (!targetUploadId) return;

    updateState({
      status: "processing",
      processingProgress: { stage: "validation", progress: 0, message: "Starting processing..." }
    });

    try {
      // Step 2: Start processing
      const processResponse = await apiClient.processFile(targetUploadId);
      updateState({ processingId: processResponse.processing_id });

      // Step 3: Listen for real-time progress
      setTimeout(() => {
        const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api';
        const sseUrl = `${apiUrl}/upload/progress/${processResponse.processing_id}`;
        
        const source = new EventSource(sseUrl);
        updateState({ eventSource: source });

        // Set a timeout to prevent infinite loading (30 seconds)
        const timeout = setTimeout(() => {
          setState(currentState => {
            if (currentState.status === "processing") {
              source.close();
              onSuccess({ status: "completed" } as ProcessingResult);
              return {
                ...currentState,
                status: "completed" as UploadStatus,
                processingProgress: {
                  stage: "completion",
                  progress: 100,
                  message: "Processing completed (timed out waiting for updates)"
                },
                eventSource: null,
                processingTimeout: null
              };
            }
            return currentState;
          });
        }, 30000);
        updateState({ processingTimeout: timeout });

        source.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            
            if (data.type === "connected" || data.type === "heartbeat") {
              return;
            }

            if (data.type === "error") {
              // Clear timeout and close source immediately on error
              if (timeout) {
                clearTimeout(timeout);
              }
              source.close();
              
              updateState({
                status: "error",
                error: data.message,
                eventSource: null,
                processingTimeout: null
              });
              return;
            }

            // Update progress
            updateState({
              processingProgress: {
                stage: data.stage || "processing",
                progress: data.progress || 0,
                message: data.message || "Processing..."
              }
            });

            // Check if completed
            if (data.stage === "completion" && data.progress === 100) {
              // Clear timeout and close source
              if (timeout) {
                clearTimeout(timeout);
              }
              source.close();
              
              updateState({ 
                status: "completed",
                eventSource: null,
                processingTimeout: null
              });
              
              // Get final results
              setTimeout(async () => {
                try {
                  const result = await apiClient.getProcessingStatus(processResponse.processing_id) as ProcessingResult;
                  updateState({ processingResult: result });
                  
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
          setTimeout(() => {
            if (source.readyState === EventSource.CLOSED) {
              updateState({
                status: "error",
                error: "Connection lost during processing. Please check your network connection.",
                eventSource: null
              });
              source.close();
            }
          }, 5000);
        };
      }, 100);

    } catch (err) {
      updateState({
        status: "error",
        error: err instanceof Error ? err.message : "Processing failed"
      });
    }
  };

  const handleReset = () => {
    if (state.eventSource) {
      state.eventSource.close();
    }
    if (state.processingTimeout) {
      clearTimeout(state.processingTimeout);
    }
    
    setState({
      file: null,
      status: "idle",
      error: "",
      uploadProgress: 0,
      processingProgress: { stage: "validation", progress: 0, message: "Preparing..." },
      uploadId: "",
      processingId: "",
      eventSource: null,
      processingResult: null,
      processingTimeout: null,
      isDragOver: false
    });
    
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const isValidFile = (file: File) => {
    return file.name.endsWith(".csv");
  };

  return {
    state,
    fileInputRef,
    handleFileChange,
    handleDrop,
    handleDragOver,
    handleDragEnter,
    handleDragLeave,
    handleUpload,
    handleReset,
    isValidFile
  };
}