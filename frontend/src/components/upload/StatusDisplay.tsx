import React from "react";
import { Database } from "lucide-react";
import { ProcessingResult } from "./types";

interface StatusDisplayProps {
  processingResult: ProcessingResult | null;
}

export function StatusDisplay({ processingResult }: StatusDisplayProps) {
  if (!processingResult) return null;

  return (
    <>
      {/* Database Status */}
      {processingResult?.database && (
        <div className="mt-3 p-3 bg-blue-50 dark:bg-blue-950/20 rounded-lg">
          <div className="flex items-center gap-2 text-blue-700 dark:text-blue-300 mb-2">
            <Database className="w-4 h-4" />
            <span className="font-medium text-sm">Database Status</span>
          </div>
          {processingResult.database.write_successful ? (
            <div className="text-xs">
              {(processingResult.database.write_result?.rows_written ?? 0) > 0 ? (
                <div className="text-blue-600 dark:text-blue-400">
                  <p className="font-medium mb-1">✅ Database Write Successful</p>
                  <div className="space-y-1">
                    <p>💾 {processingResult.database.write_result?.rows_written || 0} new questions saved</p>
                    {(processingResult.database.write_result?.duplicates_skipped ?? 0) > 0 && (
                      <p>🔄 {processingResult.database.write_result?.duplicates_skipped} duplicates skipped</p>
                    )}
                    {(processingResult.database.write_result?.data_errors_skipped ?? 0) > 0 && (
                      <p>⚠️ {processingResult.database.write_result?.data_errors_skipped} rows with data errors skipped</p>
                    )}
                    {(processingResult.database.write_result?.insertion_errors ?? 0) > 0 && (
                      <p>❌ {processingResult.database.write_result?.insertion_errors} database insertion errors</p>
                    )}
                    <p>📊 {processingResult.database.write_result?.total_processed || 0} total questions processed</p>
                  </div>
                  {((processingResult.database.write_result?.duplicates_skipped ?? 0) > 0 || 
                    (processingResult.database.write_result?.data_errors_skipped ?? 0) > 0 || 
                    (processingResult.database.write_result?.insertion_errors ?? 0) > 0) && (
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
    </>
  );
}