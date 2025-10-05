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
  );
}