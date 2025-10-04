"use client";

import { FileText, Target, Clock, TrendingUp } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { DashboardData } from "@/lib/types";

interface StatsGridProps {
  data: DashboardData;
}

export function StatsGrid({ data }: StatsGridProps) {
  const stats = [
    {
      title: "Total Questions",
      value: data.question_count?.toLocaleString() || "0",
      icon: FileText,
      description: "Student questions analyzed",
    },
    {
      title: "Topics Discovered",
      value: data.topic_count?.toString() || "0",
      icon: Target,
      description: "Unique topics identified",
    },
    {
      title: "Latest Analysis",
      value: data.last_analysis
        ? new Date(data.last_analysis).toLocaleDateString()
        : "Never",
      icon: Clock,
      description: "Most recent analysis run",
    },
    {
      title: "Coverage",
      value: data.coverage_percentage
        ? `${Math.round(data.coverage_percentage)}%`
        : "0%",
      icon: TrendingUp,
      description: "Questions successfully categorized",
    },
  ];

  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
      {stats.map((stat) => {
        const Icon = stat.icon;
        return (
          <Card key={stat.title}>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">
                {stat.title}
              </CardTitle>
              <Icon className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stat.value}</div>
              <p className="text-xs text-muted-foreground">
                {stat.description}
              </p>
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}