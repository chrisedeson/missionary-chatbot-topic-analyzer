import React from "react";
import { Upload, File, CheckCircle, AlertCircle } from "lucide-react";
import { UploadStatus, ProcessingResult } from "./types";

interface FileDropZoneProps {
  file: File | null;
  status: UploadStatus;
  error: string;
  isDragOver: boolean;
  processingResult: ProcessingResult | null;
  isValidFile: (file: File) => boolean;
  onDrop: (e: React.DragEvent<HTMLDivElement>) => void;
  onDragOver: (e: React.DragEvent) => void;
  onDragEnter: (e: React.DragEvent) => void;
  onDragLeave: (e: React.DragEvent) => void;
  onClick: () => void;
  fileInputRef: React.RefObject<HTMLInputElement | null>;
  onFileChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
}

export function FileDropZone({
  file,
  status,
  error,
  isDragOver,
  processingResult,
  isValidFile,
  onDrop,
  onDragOver,
  onDragEnter,
  onDragLeave,
  onClick,
  fileInputRef,
  onFileChange
}: FileDropZoneProps) {
  return (
    <div
      className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
        isDragOver
          ? "border-primary bg-primary/5"
          : status === "uploading" || status === "processing"
          ? "border-muted-foreground/25 bg-muted/25"
          : "border-muted-foreground/25 hover:border-muted-foreground/50 cursor-pointer"
      }`}
      onDrop={onDrop}
      onDragOver={onDragOver}
      onDragEnter={onDragEnter}
      onDragLeave={onDragLeave}
      onClick={() => !status.match(/uploading|processing/) && onClick()}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept=".csv"
        onChange={onFileChange}
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
          <div className="text-sm text-muted-foreground text-left max-w-md">
            {error.includes("missing required column headers") ? (
              <div className="space-y-2">
                <p className="font-medium text-red-600 text-center">❌ Missing Required Columns</p>
                <div className="bg-red-50 dark:bg-red-950/30 p-3 rounded border border-red-200 dark:border-red-800">
                  <pre className="whitespace-pre-wrap text-xs font-mono">
                    {error}
                  </pre>
                </div>
                <div className="text-xs text-center">
                  <p>💡 <strong>Expected CSV format:</strong></p>
                  <p>date, country, language, state, question</p>
                </div>
              </div>
            ) : (
              <p className="text-center">{error}</p>
            )}
          </div>
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
  );
}