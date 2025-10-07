"use client";

import { useState, useEffect } from "react";
import { Loader2, CheckCircle, AlertCircle, Clock } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { apiClient } from "@/lib/api";

interface ProgressTrackerProps {
  runId: string;
}

interface ProgressUpdate {
  stage: string;
  progress: number;
  message: string;
  timestamp: string;
}

export function ProgressTracker({ runId }: ProgressTrackerProps) {
  const [progress, setProgress] = useState<ProgressUpdate[]>([]);
  const [currentStage, setCurrentStage] = useState<string>("");
  const [overallProgress, setOverallProgress] = useState<number>(0);
  const [isConnected, setIsConnected] = useState<boolean>(false);

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
              
              setProgress(prev => [...prev, newUpdate]);
              setCurrentStage(data.stage);
              setOverallProgress(data.progress);
            } else if (data.type === "complete") {
              setOverallProgress(100);
              setCurrentStage("complete");
              eventSource?.close();
            } else if (data.type === "error") {
              setCurrentStage("error");
              eventSource?.close();
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
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          {getStageIcon(currentStage)}
          Analysis Progress
          {!isConnected && currentStage !== "complete" && currentStage !== "error" && (
            <span className="text-xs text-muted-foreground">(Reconnecting...)</span>
          )}
        </CardTitle>
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
            "embedding_generation",
            "clustering",
            "topic_extraction", 
            "classification",
            "finalization"
          ].map((stage) => {
            const isActive = currentStage === stage;
            const isCompleted = progress.some(p => p.stage === stage && p.progress === 100);
            
            return (
              <div
                key={stage}
                className={`flex items-center gap-2 p-2 rounded ${
                  isActive
                    ? "bg-blue-50 border border-blue-200"
                    : isCompleted
                    ? "bg-green-50 border border-green-200"
                    : "bg-muted"
                }`}
              >
                {isCompleted ? (
                  <CheckCircle className="w-3 h-3 text-green-600" />
                ) : isActive ? (
                  <Loader2 className="w-3 h-3 animate-spin text-blue-600" />
                ) : (
                  <Clock className="w-3 h-3 text-muted-foreground" />
                )}
                <span className="text-xs">{formatStage(stage)}</span>
              </div>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}