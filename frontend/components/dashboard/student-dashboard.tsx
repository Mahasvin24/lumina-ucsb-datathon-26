"use client";

import { useMemo } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { SectionCard } from "@/components/dashboard/section-card";
import { StatCard } from "@/components/dashboard/stat-card";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { StudentDetail, StudentTagPerformance } from "@/lib/dashboard-api";
import { useProcessedData } from "@/lib/processed-data-context";

type StudentDashboardProps = {
  data: StudentDetail;
};

function clampProbability(value: number): number {
  if (Number.isNaN(value)) return 0;
  if (value < 0) return 0;
  if (value > 1) return 1;
  return value;
}

function barColor(pct: number): string {
  if (pct < 40) return "#f43f5e";
  if (pct < 60) return "#f59e0b";
  if (pct < 80) return "#0ea5e9";
  return "#10b981";
}

function badgeColor(pct: number): { bg: string; text: string; border: string } {
  if (pct < 40) return { bg: "bg-rose-50", text: "text-rose-700", border: "border-rose-300" };
  if (pct < 60) return { bg: "bg-amber-50", text: "text-amber-700", border: "border-amber-300" };
  if (pct < 80) return { bg: "bg-sky-50", text: "text-sky-700", border: "border-sky-300" };
  return { bg: "bg-emerald-50", text: "text-emerald-700", border: "border-emerald-300" };
}

function TagBadge({ tag }: { tag: StudentTagPerformance }) {
  const c = badgeColor(tag.correctPct);
  return (
    <Badge className={`${c.border} ${c.bg} ${c.text} text-xs font-normal`}>
      {tag.tag} — {tag.correctPct.toFixed(1)}%
    </Badge>
  );
}

function ProbabilityBadge({ probability }: { probability: number }) {
  const p = clampProbability(probability);
  const curved = Math.pow(p, 2.2);
  const hue = Math.round(curved * 120);
  const style = {
    backgroundColor: `hsl(${hue} 92% 72%)`,
    color: `hsl(${hue} 12% 16%)`,
  };
  return (
    <Badge className="inline-block border-transparent px-2 py-1 tabular-nums" style={style}>
      {(p * 100).toFixed(0)}%
    </Badge>
  );
}

export function StudentDashboard({ data }: StudentDashboardProps) {
  const { processedResult, testedQuestionIds } = useProcessedData();

  const testedSet = useMemo(() => new Set(testedQuestionIds), [testedQuestionIds]);

  const studentPredictions = useMemo(() => {
    if (!processedResult) return null;
    const entry = processedResult.students.find(
      (s) => s.student_id === data.studentId,
    );
    if (!entry) return null;
    const byQuestion = new Map<number, number>();
    for (const qs of entry.question_summaries) {
      byQuestion.set(qs.question_id, clampProbability(qs.avg_probability));
    }
    return byQuestion;
  }, [processedResult, data.studentId]);

  const hasPredictions = studentPredictions !== null && studentPredictions.size > 0;

  const predictedScore = useMemo(() => {
    if (!studentPredictions || testedQuestionIds.length === 0) return null;
    let sum = 0;
    let count = 0;
    for (const qid of testedQuestionIds) {
      const p = studentPredictions.get(qid);
      if (p !== undefined) {
        sum += p;
        count++;
      }
    }
    if (count === 0) return null;
    return (sum / count) * 100;
  }, [studentPredictions, testedQuestionIds]);

  const predictedTagPerformance = useMemo(() => {
    if (!studentPredictions) return null;
    const tagProbSum = new Map<string, number>();
    const tagProbCount = new Map<string, number>();
    for (const q of data.questions) {
      const p = studentPredictions.get(q.questionId);
      if (p === undefined) continue;
      for (const tag of q.skillTags) {
        tagProbSum.set(tag, (tagProbSum.get(tag) ?? 0) + p);
        tagProbCount.set(tag, (tagProbCount.get(tag) ?? 0) + 1);
      }
    }
    if (tagProbCount.size === 0) return null;
    const tags: StudentTagPerformance[] = [];
    for (const [tag, count] of tagProbCount) {
      const avg = (tagProbSum.get(tag) ?? 0) / count;
      tags.push({
        tag,
        correctCount: Number((tagProbSum.get(tag) ?? 0).toFixed(2)),
        totalCount: count,
        correctPct: Number((avg * 100).toFixed(2)),
      });
    }
    tags.sort((a, b) => a.correctPct - b.correctPct);
    return tags;
  }, [studentPredictions, data.questions]);

  const predictedWeakest = useMemo(
    () => predictedTagPerformance?.slice(0, 3) ?? [],
    [predictedTagPerformance],
  );
  const predictedStrongest = useMemo(
    () =>
      predictedTagPerformance
        ? [...predictedTagPerformance].reverse().slice(0, 3)
        : [],
    [predictedTagPerformance],
  );

  const chartData = useMemo(() => {
    if (!hasPredictions || !predictedTagPerformance) {
      return data.tagPerformance.map((t) => ({
        tag: t.tag,
        accuracy: t.correctPct,
        correct: t.correctCount,
        total: t.totalCount,
      }));
    }
    const predictedMap = new Map(
      predictedTagPerformance.map((t) => [t.tag, t]),
    );
    const allTags = new Set([
      ...data.tagPerformance.map((t) => t.tag),
      ...predictedTagPerformance.map((t) => t.tag),
    ]);
    return [...allTags]
      .map((tag) => {
        const raw = data.tagPerformance.find((t) => t.tag === tag);
        const pred = predictedMap.get(tag);
        return {
          tag,
          accuracy: raw?.correctPct ?? 0,
          correct: raw?.correctCount ?? 0,
          total: raw?.totalCount ?? 0,
          predicted: pred?.correctPct ?? 0,
        };
      })
      .sort((a, b) => a.accuracy - b.accuracy);
  }, [data.tagPerformance, hasPredictions, predictedTagPerformance]);

  const chartHeight = Math.max(260, chartData.length * 36);

  const predictedScoreColor =
    predictedScore !== null
      ? predictedScore >= 80
        ? "text-emerald-700"
        : predictedScore >= 60
          ? "text-sky-700"
          : predictedScore >= 40
            ? "text-amber-600"
            : "text-rose-700"
      : undefined;

  const questionsFedIn = hasPredictions ? testedQuestionIds.length : 0;

  return (
    <>
      <section className="grid gap-3 grid-cols-2 max-w-md">
        {predictedScore !== null && (
          <StatCard label="Predicted Score" value={`${predictedScore.toFixed(2)}%`} valueClassName={predictedScoreColor} />
        )}
        {questionsFedIn > 0 && (
          <StatCard label="Questions fed in" value={String(questionsFedIn)} />
        )}
      </section>

      <div className="grid gap-6 lg:grid-cols-2">
        <SectionCard
          title={hasPredictions ? "Predicted Weakest Tags" : "Weakest Tags"}
          subtitle={<span className="text-sm text-zinc-500">{hasPredictions ? "Lowest predicted accuracy" : "Lowest accuracy concepts"}</span>}
        >
          {(hasPredictions ? predictedWeakest : data.weakestTags).length > 0 ? (
            <div className="space-y-2">
              {(hasPredictions ? predictedWeakest : data.weakestTags).map((tag) => (
                <Card key={tag.tag} className="bg-zinc-50">
                  <CardContent className="flex items-center justify-between gap-3 p-3">
                    <div>
                      <span className="text-sm font-medium text-zinc-800">{tag.tag}</span>
                      <span className="ml-2 text-xs text-zinc-500">
                        {hasPredictions
                          ? `${tag.correctPct.toFixed(1)}% predicted`
                          : `${tag.correctCount}/${tag.totalCount} correct`}
                      </span>
                    </div>
                    <TagBadge tag={tag} />
                  </CardContent>
                </Card>
              ))}
            </div>
          ) : (
            <p className="py-4 text-center text-sm text-zinc-400">No tag data available.</p>
          )}
        </SectionCard>

        <SectionCard
          title={hasPredictions ? "Predicted Strongest Tags" : "Strongest Tags"}
          subtitle={<span className="text-sm text-zinc-500">{hasPredictions ? "Highest predicted accuracy" : "Highest accuracy concepts"}</span>}
        >
          {(hasPredictions ? predictedStrongest : data.strongestTags).length > 0 ? (
            <div className="space-y-2">
              {(hasPredictions ? predictedStrongest : data.strongestTags).map((tag) => (
                <Card key={tag.tag} className="bg-zinc-50">
                  <CardContent className="flex items-center justify-between gap-3 p-3">
                    <div>
                      <span className="text-sm font-medium text-zinc-800">{tag.tag}</span>
                      <span className="ml-2 text-xs text-zinc-500">
                        {hasPredictions
                          ? `${tag.correctPct.toFixed(1)}% predicted`
                          : `${tag.correctCount}/${tag.totalCount} correct`}
                      </span>
                    </div>
                    <TagBadge tag={tag} />
                  </CardContent>
                </Card>
              ))}
            </div>
          ) : (
            <p className="py-4 text-center text-sm text-zinc-400">No tag data available.</p>
          )}
        </SectionCard>
      </div>

      <SectionCard
        title="Tag Performance"
        subtitle={
          <span className="text-sm text-zinc-500">
            {hasPredictions ? "Historical vs. predicted accuracy" : "Accuracy across all concept tags"}
          </span>
        }
      >
        {chartData.length > 0 ? (
          <ResponsiveContainer width="100%" height={chartHeight}>
            <BarChart data={chartData} layout="vertical" margin={{ top: 4, right: 24, bottom: 4, left: 8 }}>
              <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#e4e4e7" />
              <XAxis type="number" domain={[0, 100]} tickFormatter={(v: number) => `${v}%`} tick={{ fontSize: 11, fill: "#71717a" }} />
              <YAxis type="category" dataKey="tag" width={180} tick={{ fontSize: 11, fill: "#3f3f46" }} />
              <Tooltip
                contentStyle={{ fontSize: 12, borderRadius: 6, border: "1px solid #e4e4e7", boxShadow: "0 1px 3px rgba(0,0,0,0.08)" }}
              />
              <Bar dataKey="accuracy" name="Historical" radius={[0, 4, 4, 0]} barSize={hasPredictions ? 14 : 20}>
                {chartData.map((entry) => (
                  <Cell key={entry.tag} fill={barColor(entry.accuracy)} />
                ))}
              </Bar>
              {hasPredictions && (
                <Bar dataKey="predicted" name="Predicted" radius={[0, 4, 4, 0]} barSize={14} fillOpacity={0.7}>
                  {chartData.map((entry) => (
                    <Cell key={entry.tag} fill={barColor(entry.predicted ?? 0)} />
                  ))}
                </Bar>
              )}
              {hasPredictions && <Legend />}
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <p className="py-4 text-center text-sm text-zinc-400">No tag data available.</p>
        )}
      </SectionCard>

      <SectionCard title="Question Breakdown" subtitle={<span className="text-sm text-zinc-500">Performance on each test question</span>}>
        <Table className="min-w-full text-left">
          <TableHeader>
            <TableRow>
              <TableHead className="w-24">Question</TableHead>
              <TableHead className="w-24">Status</TableHead>
              {hasPredictions && <TableHead className="w-28">Predicted</TableHead>}
              <TableHead>Skill Tags</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {data.questions.map((q) => {
              const prediction = studentPredictions?.get(q.questionId);
              const isTested = testedSet.has(q.questionId);
              return (
                <TableRow
                  key={q.questionId}
                  className={q.state === "wrong" ? "bg-red-50 hover:bg-red-100" : ""}
                >
                  <TableCell className="font-medium">Q{q.questionId}</TableCell>
                  <TableCell>
                    {q.state === "correct" && <Badge variant="success">Correct</Badge>}
                    {q.state === "wrong" && <Badge variant="destructive">Wrong</Badge>}
                    {q.state === "unanswered" && <Badge variant="secondary">Unanswered</Badge>}
                  </TableCell>
                  {hasPredictions && (
                    <TableCell>
                      {isTested && prediction !== undefined ? (
                        <ProbabilityBadge probability={prediction} />
                      ) : (
                        <span className="text-zinc-400">—</span>
                      )}
                    </TableCell>
                  )}
                  <TableCell>
                    {q.skillTags.length > 0 ? (
                      <div className="flex flex-wrap gap-1">
                        {q.skillTags.map((tag) => (
                          <Badge key={tag} variant="outline" className="text-xs font-normal">
                            {tag}
                          </Badge>
                        ))}
                      </div>
                    ) : (
                      <span className="text-zinc-400">—</span>
                    )}
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </SectionCard>
    </>
  );
}
