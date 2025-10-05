"use client";

import { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { Upload, Play, Download, Loader2, FileText, BarChart3 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardContent, CardTitle, CardDescription } from "@/components/ui/card";
import { StatsGrid } from "@/components/stats-grid";
import { AnalysisResults } from "@/components/analysis-results";
import { UploadDialog } from "@/components/upload-dialog";
import { ProgressTracker } from "@/components/progress-tracker";
import { apiClient } from "@/lib/api";
import type { DashboardData, AnalysisRun } from "@/lib/types";

interface DashboardProps {
  isDeveloper: boolean;
}

export function Dashboard({ isDeveloper }: DashboardProps) {
  const [showUpload, setShowUpload] = useState(false);
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [selectedRun, setSelectedRun] = useState<AnalysisRun | null>(null);
  const [lastProcessingResult, setLastProcessingResult] = useState<any>(null);

  // Fetch dashboard data
  const { data: dashboardData, refetch: refetchDashboard, isLoading } = useQuery({
    queryKey: ["dashboard"],
    queryFn: () => apiClient.getDashboardMetrics(),
    refetchInterval: 30000, // Refresh every 30 seconds
  });

  // Fetch analysis runs
  const { data: analysisRuns, refetch: refetchRuns } = useQuery({
    queryKey: ["analysis-runs"],
    queryFn: () => apiClient.getAnalysisRuns(),
    refetchInterval: 10000, // Refresh every 10 seconds
  });

  const handleUploadSuccess = (processingResult: any) => {
    setLastProcessingResult(processingResult);
    refetchDashboard();
  };

  // Check if we have processed questions available for analysis
  const hasProcessedQuestions = dashboardData?.has_questions || 
    (lastProcessingResult?.status === "completed" && 
     lastProcessingResult?.statistics?.valid_questions_extracted > 0);

  const handleStartAnalysis = async () => {
    if (!hasProcessedQuestions) return;
    
    try {
      const response = await apiClient.startAnalysis();
      setActiveRunId(response.run_id);
      refetchRuns();
    } catch (error) {
      console.error("Failed to start analysis:", error);
    }
  };

  const handleExportResults = async () => {
    if (!selectedRun?.id) return;
    
    try {
      const blob = await apiClient.exportResults(selectedRun.id);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `analysis-results-${selectedRun.id}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error("Failed to export results:", error);
    }
  };

  const runningRun = analysisRuns?.find(run => run.status === "running");
  const completedRuns = analysisRuns?.filter(run => run.status === "completed") || [];
  const latestRun = completedRuns[0]; // Most recent completed run

  useEffect(() => {
    if (latestRun && !selectedRun) {
      setSelectedRun(latestRun);
    }
  }, [latestRun, selectedRun]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="w-8 h-8 animate-spin" />
        <span className="ml-2">Loading dashboard...</span>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header Actions */}
      <div className="flex flex-col sm:flex-row gap-4 justify-between">
        <div>
          <h2 className="text-2xl font-bold">Dashboard</h2>
          <p className="text-muted-foreground">
            Analyze student questions to discover topics and insights
          </p>
        </div>
        
        <div className="flex gap-2">
          {isDeveloper && (
            <>
              <Button
                variant="outline"
                onClick={() => setShowUpload(true)}
                disabled={!!runningRun}
              >
                <Upload className="w-4 h-4 mr-2" />
                Upload Data
              </Button>
              
              <Button
                onClick={handleStartAnalysis}
                disabled={!hasProcessedQuestions || !!runningRun}
              >
                {runningRun ? (
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                ) : (
                  <Play className="w-4 h-4 mr-2" />
                )}
                {runningRun ? "Analyzing..." : "Start Analysis"}
              </Button>
            </>
          )}
          
          {selectedRun && (
            <Button
              variant="outline"
              onClick={handleExportResults}
            >
              <Download className="w-4 h-4 mr-2" />
              Export Results
            </Button>
          )}
        </div>
      </div>

      {/* Progress Tracking */}
      {runningRun && (
        <ProgressTracker runId={runningRun.id} />
      )}

      {/* Stats Grid */}
      {dashboardData && (
        <StatsGrid data={dashboardData} />
      )}

      {/* Main Content */}
      <div className="grid gap-6 lg:grid-cols-3">
        {/* Data Status */}
        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileText className="w-5 h-5" />
              Data Status
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">Questions Loaded</span>
              <span className="font-medium">
                {dashboardData?.question_count?.toLocaleString() || 0}
              </span>
            </div>
            
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">Analysis Runs</span>
              <span className="font-medium">{analysisRuns?.length || 0}</span>
            </div>
            
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">Last Updated</span>
              <span className="text-sm">
                {dashboardData?.last_updated 
                  ? new Date(dashboardData.last_updated).toLocaleDateString()
                  : "Never"
                }
              </span>
            </div>

            {!isDeveloper && !hasProcessedQuestions && (
              <div className="p-3 bg-muted/50 rounded-md">
                <p className="text-sm text-muted-foreground">
                  No data available. Contact a developer to upload question data.
                </p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Analysis Results */}
        <div className="lg:col-span-2">
          {selectedRun ? (
            <AnalysisResults 
              run={selectedRun} 
              onRunSelect={setSelectedRun}
              availableRuns={completedRuns}
            />
          ) : (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <BarChart3 className="w-5 h-5" />
                  Analysis Results
                </CardTitle>
                <CardDescription>
                  Run an analysis to see topic discovery results
                </CardDescription>
              </CardHeader>
              <CardContent className="flex items-center justify-center h-64 text-muted-foreground">
                {!hasProcessedQuestions ? (
                  <div className="text-center">
                    <FileText className="w-12 h-12 mx-auto mb-4 opacity-50" />
                    <p>Upload and process question data to get started</p>
                    {lastProcessingResult && (
                      <p className="text-sm mt-2 text-blue-600">
                        Last processing: {lastProcessingResult.statistics?.valid_questions_extracted || 0} questions extracted
                      </p>
                    )}
                  </div>
                ) : (
                  <div className="text-center">
                    <BarChart3 className="w-12 h-12 mx-auto mb-4 opacity-50" />
                    <p>No analysis results yet</p>
                    {isDeveloper && (
                      <p className="text-sm mt-2">Click "Start Analysis" to begin</p>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>
          )}
        </div>
      </div>

      {/* Upload Dialog */}
      {showUpload && (
        <UploadDialog
          open={showUpload}
          onClose={() => setShowUpload(false)}
          onSuccess={handleUploadSuccess}
        />
      )}
    </div>
  );
}