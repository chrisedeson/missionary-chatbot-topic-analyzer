"use client";

// Topic card component

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import type { Topic } from "@/lib/types";

interface TopicCardProps {
  topic: Topic;
  onClick: () => void;
}

export function TopicCard({ topic, onClick }: TopicCardProps) {
  return (
    <Card className="cursor-pointer hover:shadow-md transition-shadow" onClick={onClick}>
      <CardHeader className="pb-3">
        <CardTitle className="text-base font-medium leading-tight">
          {topic.name}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <p className="text-sm text-muted-foreground line-clamp-2">
          {topic.description}
        </p>
        
        <div className="flex items-center justify-between text-xs text-muted-foreground">
          <div className="flex items-center gap-1">
            <Users className="w-3 h-3" />
            <span>{topic.question_count} questions</span>
          </div>
          
          <div className="flex items-center gap-1">
            <TrendingUp className="w-3 h-3" />
            <span>{Math.round(topic.confidence_score * 100)}% conf.</span>
          </div>
        </div>
        
        {topic.keywords && topic.keywords.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {topic.keywords.slice(0, 3).map((keyword, index) => (
              <span
                key={index}
                className="px-2 py-0.5 bg-muted rounded text-xs"
              >
                {keyword}
              </span>
            ))}
            {topic.keywords.length > 3 && (
              <span className="px-2 py-0.5 text-xs text-muted-foreground">
                +{topic.keywords.length - 3} more
              </span>
            )}
          </div>
        )}
        
        <Button variant="ghost" size="sm" className="w-full mt-2">
          View Details
        </Button>
      </CardContent>
    </Card>
  );
}