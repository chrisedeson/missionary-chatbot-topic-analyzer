"use client";

import { useState, useRef } from "react";
import { Upload, File, X, Loader2, CheckCircle, AlertCircle } from "lucide-react";

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
  onSuccess: () => void;
}

type UploadStatus = "idle" | "uploading" | "success" | "error";

export function UploadDialog({ open, onClose, onSuccess }: UploadDialogProps) {
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<UploadStatus>("idle");
  const [error, setError] = useState("");
  const [progress, setProgress] = useState(0);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setStatus("idle");
      setError("");
    }
  };

  const handleDrop = (event: React.DragEvent) => {
    event.preventDefault();
    const droppedFile = event.dataTransfer.files[0];
    if (droppedFile) {
      setFile(droppedFile);
      setStatus("idle");
      setError("");
    }
  };

  const handleDragOver = (event: React.DragEvent) => {
    event.preventDefault();
  };

  const handleUpload = async () => {
    if (!file) return;

    setStatus("uploading");
    setProgress(0);
    setError("");

    try {
      await apiClient.uploadFile(file, (progressPercent) => {
        setProgress(progressPercent);
      });
      
      setStatus("success");
      setTimeout(() => {
        onSuccess();
        handleReset();
      }, 1500);
    } catch (err) {
      setStatus("error");
      setError(err instanceof Error ? err.message : "Upload failed");
    }
  };

  const handleReset = () => {
    setFile(null);
    setStatus("idle");
    setError("");
    setProgress(0);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const isValidFile = (file: File) => {
    const validTypes = [
      "text/csv",
      "application/vnd.ms-excel",
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
      "application/json",
    ];
    return validTypes.includes(file.type) || file.name.endsWith(".csv");
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
            Upload a CSV, Excel, or JSON file containing student questions
          </SheetDescription>
        </SheetHeader>

        <div className="mt-6 space-y-6">
          {/* File Drop Zone */}
          <div
            className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
              status === "uploading"
                ? "border-muted-foreground/25 bg-muted/25"
                : "border-muted-foreground/25 hover:border-muted-foreground/50 cursor-pointer"
            }`}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onClick={() => !status.match(/uploading/) && fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv,.xlsx,.xls,.json"
              onChange={handleFileSelect}
              className="hidden"
              disabled={status === "uploading"}
            />

            {status === "success" ? (
              <div className="text-green-600">
                <CheckCircle className="w-12 h-12 mx-auto mb-4" />
                <p className="font-medium">Upload Successful!</p>
                <p className="text-sm text-muted-foreground">Data has been processed</p>
              </div>
            ) : status === "error" ? (
              <div className="text-red-600">
                <AlertCircle className="w-12 h-12 mx-auto mb-4" />
                <p className="font-medium">Upload Failed</p>
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
                    File type not supported. Please use CSV, Excel, or JSON.
                  </p>
                )}
              </div>
            ) : (
              <div>
                <Upload className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
                <p className="font-medium">Choose a file or drag it here</p>
                <p className="text-sm text-muted-foreground">
                  Supports CSV, Excel (.xlsx, .xls), and JSON files
                </p>
              </div>
            )}
          </div>

          {/* Progress Bar */}
          {status === "uploading" && (
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Uploading...</span>
                <span>{Math.round(progress)}%</span>
              </div>
              <div className="w-full bg-muted rounded-full h-2">
                <div
                  className="bg-primary h-2 rounded-full transition-all duration-300"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
          )}

          {/* File Format Guide */}
          <div className="space-y-3">
            <h4 className="font-medium text-sm">Expected File Format:</h4>
            <div className="text-xs text-muted-foreground space-y-1">
              <p>• CSV/Excel: Must contain a column with student questions</p>
              <p>• JSON: Array of objects with question text</p>
              <p>• Column names like: "question", "text", "content", "message"</p>
              <p>• Maximum file size: 50 MB</p>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-3">
            <Button
              onClick={handleUpload}
              disabled={!file || !isValidFile(file) || status === "uploading"}
              className="flex-1"
            >
              {status === "uploading" ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Uploading...
                </>
              ) : (
                <>
                  <Upload className="w-4 h-4 mr-2" />
                  Upload File
                </>
              )}
            </Button>

            {file && status !== "uploading" && (
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