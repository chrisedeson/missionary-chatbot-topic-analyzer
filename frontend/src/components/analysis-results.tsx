"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  BarChart3,
  ChevronDown,
  Loader2,
  Eye,
  Hash,
  Users,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { TopicCard } from "@/components/topic-card";
import { apiClient } from "@/lib/api";
import type { AnalysisRun, Topic } from "@/lib/types";

interface AnalysisResultsProps {
  run: AnalysisRun;
  onRunSelect: (run: AnalysisRun) => void;
  availableRuns: AnalysisRun[];
}

export function AnalysisResults({
  run,
  onRunSelect,
  availableRuns,
}: AnalysisResultsProps) {
  const [showRunSelector, setShowRunSelector] = useState(false);
  const [selectedTopic, setSelectedTopic] = useState<Topic | null>(null);

  const {
    data: topics,
    isLoading,
    error,
  } = useQuery({
    queryKey: ["topics", run.id],
    queryFn: () => apiClient.getTopics(run.id),
    enabled: !!run.id,
  });

  if (error) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center h-64">
          <div className="text-center text-muted-foreground">
            <p>Failed to load analysis results</p>
            <p className="text-sm mt-1">Please try again later</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="w-5 h-5" />
              Analysis Results
            </CardTitle>
            <CardDescription>
              {run.created_at
                ? `Completed ${new Date(run.created_at).toLocaleString()}`
                : "Analysis results"}
            </CardDescription>
          </div>

          {availableRuns.length > 1 && (
            <Sheet open={showRunSelector} onOpenChange={setShowRunSelector}>
              <SheetTrigger asChild>
                <Button variant="outline" size="sm">
                  <ChevronDown className="w-4 h-4 mr-2" />
                  Switch Run
                </Button>
              </SheetTrigger>
              <SheetContent>
                <SheetHeader>
                  <SheetTitle>Analysis Runs</SheetTitle>
                  <SheetDescription>
                    Select a different analysis run to view
                  </SheetDescription>
                </SheetHeader>
                <div className="mt-6 space-y-2">
                  {availableRuns.map((availableRun) => (
                    <Button
                      key={availableRun.id}
                      variant={availableRun.id === run.id ? "default" : "ghost"}
                      className="w-full justify-start"
                      onClick={() => {
                        onRunSelect(availableRun);
                        setShowRunSelector(false);
                      }}
                    >
                      <div className="text-left">
                        <div className="font-medium">
                          {availableRun.created_at
                            ? new Date(availableRun.created_at).toLocaleDateString()
                            : "Unknown Date"}
                        </div>
                        <div className="text-sm text-muted-foreground">
                          {availableRun.created_at
                            ? new Date(availableRun.created_at).toLocaleTimeString()
                            : ""}
                        </div>
                      </div>
                    </Button>
                  ))}
                </div>
              </SheetContent>
            </Sheet>
          )}
        </div>
      </CardHeader>

      <CardContent>
        {isLoading ? (
          <div className="flex items-center justify-center h-64">
            <Loader2 className="w-8 h-8 animate-spin" />
            <span className="ml-2">Loading topics...</span>
          </div>
        ) : topics && topics.length > 0 ? (
          <div className="space-y-4">
            <div className="flex items-center gap-4 text-sm text-muted-foreground">
              <div className="flex items-center gap-1">
                <Hash className="w-4 h-4" />
                <span>{topics.length} topics discovered</span>
              </div>
              <div className="flex items-center gap-1">
                <Users className="w-4 h-4" />
                <span>
                  {topics.reduce((sum, topic) => sum + topic.question_count, 0)}{" "}
                  questions categorized
                </span>
              </div>
            </div>

            <div className="grid gap-4 md:grid-cols-2">
              {topics.map((topic) => (
                <TopicCard
                  key={topic.id}
                  topic={topic}
                  onClick={() => setSelectedTopic(topic)}
                />
              ))}
            </div>
          </div>
        ) : (
          <div className="flex items-center justify-center h-64 text-muted-foreground">
            <div className="text-center">
              <BarChart3 className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <p>No topics discovered in this analysis</p>
              <p className="text-sm mt-1">
                Try adjusting analysis parameters or uploading more data
              </p>
            </div>
          </div>
        )}
      </CardContent>

      {/* Topic Detail Sheet */}
      {selectedTopic && (
        <Sheet
          open={!!selectedTopic}
          onOpenChange={() => setSelectedTopic(null)}
        >
          <SheetContent className="w-full sm:max-w-lg">
            <SheetHeader>
              <SheetTitle className="flex items-center gap-2">
                <Eye className="w-5 h-5" />
                Topic Details
              </SheetTitle>
              <SheetDescription>
                {selectedTopic.name}
              </SheetDescription>
            </SheetHeader>

            <div className="mt-6 space-y-6">
              <div>
                <h4 className="font-medium mb-2">Description</h4>
                <p className="text-sm text-muted-foreground">
                  {selectedTopic.description}
                </p>
              </div>

              <div>
                <h4 className="font-medium mb-2">Statistics</h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Questions:</span>
                    <span className="font-medium">
                      {selectedTopic.question_count}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Confidence:</span>
                    <span className="font-medium">
                      {Math.round(selectedTopic.confidence_score * 100)}%
                    </span>
                  </div>
                </div>
              </div>

              {selectedTopic.keywords && selectedTopic.keywords.length > 0 && (
                <div>
                  <h4 className="font-medium mb-2">Keywords</h4>
                  <div className="flex flex-wrap gap-2">
                    {selectedTopic.keywords.map((keyword, index) => (
                      <span
                        key={index}
                        className="px-2 py-1 bg-muted rounded-md text-xs"
                      >
                        {keyword}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {selectedTopic.representative_questions &&
                selectedTopic.representative_questions.length > 0 && (
                  <div>
                    <h4 className="font-medium mb-2">Sample Questions</h4>
                    <div className="space-y-2">
                      {selectedTopic.representative_questions
                        .slice(0, 5)
                        .map((question, index) => (
                          <div
                            key={index}
                            className="p-3 bg-muted/50 rounded-md text-sm"
                          >
                            {question}
                          </div>
                        ))}
                    </div>
                  </div>
                )}
            </div>
          </SheetContent>
        </Sheet>
      )}
    </Card>
  );
}