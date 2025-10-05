import React from "react";
import { Upload, X, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { UploadStatus } from "./types";

interface ActionButtonsProps {
  status: UploadStatus;
  file: File | null;
  isValidFile: (file: File) => boolean;
  onUpload: () => void;
  onReset: () => void;
}

export function ActionButtons({
  status,
  file,
  isValidFile,
  onUpload,
  onReset
}: ActionButtonsProps) {
  if (status === "completed") {
    return (
      <div className="flex gap-3">
        <Button variant="outline" onClick={onReset} className="flex-1">
          <X className="w-4 h-4 mr-2" />
          Close & Clear
        </Button>
      </div>
    );
  }

  if (status === "error") {
    return (
      <div className="flex gap-3">
        <Button
          onClick={onUpload}
          disabled={!file || !isValidFile(file)}
          className="flex-1"
        >
          <Upload className="w-4 h-4 mr-2" />
          Retry Upload
        </Button>
        <Button variant="outline" onClick={onReset}>
          <X className="w-4 h-4 mr-2" />
          Clear
        </Button>
      </div>
    );
  }

  return (
    <div className="flex gap-3">
      <Button
        onClick={onUpload}
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
        <Button variant="outline" onClick={onReset}>
          <X className="w-4 h-4 mr-2" />
          Clear
        </Button>
      )}
    </div>
  );
}