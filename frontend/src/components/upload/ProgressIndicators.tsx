import React from "react";
import { Database } from "lucide-react";
import { UploadStatus, ProcessingProgress } from "./types";

interface ProgressIndicatorsProps {
  status: UploadStatus;
  uploadProgress: number;
  processingProgress: ProcessingProgress;
}

export function ProgressIndicators({
  status,
  uploadProgress,
  processingProgress
}: ProgressIndicatorsProps) {
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

  if (status === "uploading") {
    return (
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
    );
  }

  if (status === "processing") {
    return (
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
    );
  }

  return null;
}