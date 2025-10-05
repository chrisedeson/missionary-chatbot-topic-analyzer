"use client";

import { useEffect } from "react";
import { Upload } from "lucide-react";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";

import { useUpload } from "./useUpload";
import { FileDropZone } from "./FileDropZone";
import { ProgressIndicators } from "./ProgressIndicators";
import { StatusDisplay } from "./StatusDisplay";
import { ActionButtons } from "./ActionButtons";
import { UploadDialogProps } from "./types";

export function UploadDialog({ open, onClose, onSuccess }: UploadDialogProps) {
  const {
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
  } = useUpload(onSuccess);

  // Clean up when dialog closes
  useEffect(() => {
    if (!open) {
      if (state.eventSource) {
        state.eventSource.close();
      }
    }
  }, [open, state.eventSource]);

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
          <FileDropZone
            file={state.file}
            status={state.status}
            error={state.error}
            isDragOver={state.isDragOver}
            processingResult={state.processingResult}
            isValidFile={isValidFile}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragEnter={handleDragEnter}
            onDragLeave={handleDragLeave}
            onClick={() => fileInputRef.current?.click()}
            fileInputRef={fileInputRef}
            onFileChange={handleFileChange}
          />

          {/* Progress Indicators */}
          <ProgressIndicators
            status={state.status}
            uploadProgress={state.uploadProgress}
            processingProgress={state.processingProgress}
          />

          {/* Status Display */}
          {state.status === "completed" && (
            <StatusDisplay processingResult={state.processingResult} />
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
          <ActionButtons
            status={state.status}
            file={state.file}
            isValidFile={isValidFile}
            onUpload={handleUpload}
            onReset={handleReset}
          />
        </div>
      </SheetContent>
    </Sheet>
  );
}