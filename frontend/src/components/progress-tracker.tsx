"use client";

import { useState, useEffect } from "react";
import { Loader2, CheckCircle, AlertCircle, Clock, X } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { apiClient } from "@/lib/api";

interface ProgressTrackerProps {
  runId: string;
  onClose?: () => void;
}

interface ProgressUpdate {
  stage: string;
  progress: number;
  message: string;
  timestamp: string;
}

export function ProgressTracker({ runId, onClose }: ProgressTrackerProps) {
  const [progress, setProgress] = useState<ProgressUpdate[]>([]);
  const [currentStage, setCurrentStage] = useState<string>("");
  const [overallProgress, setOverallProgress] = useState<number>(0);
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [isVisible, setIsVisible] = useState<boolean>(true);

  const handleClose = () => {
    setIsVisible(false);
    onClose?.();
  };

  if (!isVisible) return null;

  useEffect(() => {
    if (!runId) return;

    let eventSource: EventSource | null = null;

    const connectToProgress = () => {
      try {
        eventSource = new EventSource(`${apiClient.baseURL}/analysis/${runId}/progress`);
        
        eventSource.onopen = () => {
          setIsConnected(true);
        };

        eventSource.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            
            if (data.type === "progress") {
              const newUpdate: ProgressUpdate = {
                stage: data.stage,
                progress: data.progress,
                message: data.message,
                timestamp: new Date().toISOString(),
              };
              
              // Only add if different from last message (deduplicate consecutive identical messages)
              setProgress(prev => {
                const lastUpdate = prev[prev.length - 1];
                if (lastUpdate && lastUpdate.message === newUpdate.message && lastUpdate.stage === newUpdate.stage) {
                  return prev; // Skip duplicate
                }
                return [...prev, newUpdate];
              });
              
              setCurrentStage(data.stage);
              setOverallProgress(data.progress);
            } else if (data.type === "complete") {
              setOverallProgress(100);
              setCurrentStage("complete");
              eventSource?.close();
            } else if (data.type === "error") {
              setCurrentStage("error");
              setOverallProgress(0);
              // Don't close - let user manually close to see error
            }
          } catch (error) {
            console.error("Error parsing progress data:", error);
          }
        };

        eventSource.onerror = () => {
          setIsConnected(false);
          eventSource?.close();
          
          // Retry connection after 5 seconds
          setTimeout(() => {
            if (currentStage !== "complete" && currentStage !== "error") {
              connectToProgress();
            }
          }, 5000);
        };
      } catch (error) {
        console.error("Error connecting to progress stream:", error);
      }
    };

    connectToProgress();

    return () => {
      eventSource?.close();
    };
  }, [runId, currentStage]);

  const getStageIcon = (stage: string) => {
    switch (stage) {
      case "complete":
        return <CheckCircle className="w-4 h-4 text-green-600" />;
      case "error":
        return <AlertCircle className="w-4 h-4 text-red-600" />;
      default:
        return <Loader2 className="w-4 h-4 animate-spin text-blue-600" />;
    }
  };

  const formatStage = (stage: string) => {
    if (!stage) return "Processing";
    return stage
      .split("_")
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(" ");
  };

  return (
    <Card className="relative">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          {getStageIcon(currentStage)}
          Analysis Progress
          {!isConnected && currentStage !== "complete" && currentStage !== "error" && (
            <span className="text-xs text-muted-foreground">(Reconnecting...)</span>
          )}
        </CardTitle>
        <Button
          variant="ghost"
          size="icon"
          className="absolute top-4 right-4 h-6 w-6"
          onClick={handleClose}
        >
          <X className="h-4 w-4" />
        </Button>
      </CardHeader>
      
      <CardContent className="space-y-4">
        {/* Overall Progress Bar */}
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span>Overall Progress</span>
            <span>{Math.round(overallProgress)}%</span>
          </div>
          <div className="w-full bg-muted rounded-full h-2">
            <div
              className={`h-2 rounded-full transition-all duration-500 ${
                currentStage === "error"
                  ? "bg-red-500"
                  : currentStage === "complete"
                  ? "bg-green-500"
                  : "bg-blue-500"
              }`}
              style={{ width: `${overallProgress}%` }}
            />
          </div>
        </div>

        {/* Current Stage */}
        {currentStage && currentStage !== "complete" && currentStage !== "error" && (
          <div className="flex items-center gap-2 text-sm">
            <Loader2 className="w-4 h-4 animate-spin" />
            <span>Current: {formatStage(currentStage)}</span>
          </div>
        )}

        {/* Status Message */}
        {currentStage === "complete" && (
          <div className="flex items-center gap-2 text-sm text-green-600">
            <CheckCircle className="w-4 h-4" />
            <span>Analysis completed successfully!</span>
          </div>
        )}

        {currentStage === "error" && (
          <div className="flex items-center gap-2 text-sm text-red-600">
            <AlertCircle className="w-4 h-4" />
            <span>Analysis failed. Please try again.</span>
          </div>
        )}

        {/* Progress Log */}
        {progress.length > 0 && (
          <div className="space-y-2">
            <h4 className="text-sm font-medium">Progress Log</h4>
            <div className="max-h-32 overflow-y-auto space-y-1">
              {progress.slice(-5).map((update, index) => (
                <div key={index} className="text-xs text-muted-foreground flex items-center gap-2">
                  <Clock className="w-3 h-3" />
                  <span className="flex-1">
                    {formatStage(update.stage)}: {update.message}
                  </span>
                  <span className="text-xs">
                    {new Date(update.timestamp).toLocaleTimeString()}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Stage Breakdown */}
        <div className="grid grid-cols-2 gap-2 text-xs">
          {[
            "generating_embeddings",
            "classifying_questions",
            "discovering_topics", 
            "saving_results",
            "complete"
          ].map((stage) => {
            const stageOrder = [
              "generating_embeddings",
              "classifying_questions", 
              "discovering_topics",
              "saving_results",
              "complete"
            ];
            const currentIndex = stageOrder.indexOf(currentStage);
            const stageIndex = stageOrder.indexOf(stage);
            
            const isActive = currentStage === stage;
            const isCompleted = currentIndex > stageIndex || currentStage === "complete";
            
            return (
              <div
                key={stage}
                className={`flex items-center gap-2 p-2 rounded ${
                  isActive
                    ? "bg-blue-100 border border-blue-300 text-blue-900"
                    : isCompleted
                    ? "bg-green-100 border border-green-400 text-green-900"
                    : "bg-muted border text-gray-200"
                }`}
              >
                {isCompleted ? (
                  <CheckCircle className="w-3 h-3 text-green-700" />
                ) : isActive ? (
                  <Loader2 className="w-3 h-3 animate-spin text-blue-700" />
                ) : (
                  <Clock className="w-3 h-3 text-gray-400" />
                )}
                <span className="text-xs font-medium">{formatStage(stage)}</span>
              </div>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}