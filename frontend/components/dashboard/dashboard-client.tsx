"use client";

import { useMemo, useState } from "react";

import { MatrixView } from "@/components/dashboard/matrix-view";
import { QuestionsTable } from "@/components/dashboard/questions-table";
import { RemediationPanel } from "@/components/dashboard/remediation-panel";
import { SectionCard } from "@/components/dashboard/section-card";
import { StatCard } from "@/components/dashboard/stat-card";
import { StudentsTable } from "@/components/dashboard/students-table";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription } from "@/components/ui/card";
import {
  ConceptAccuracy,
  DashboardPayload,
  ProcessSelectedQuestionsResponse,
} from "@/lib/dashboard-api";

type DashboardClientProps = {
  data: DashboardPayload;
};

function pct(numerator: number, denominator: number): number {
  if (denominator <= 0) {
    return 0;
  }
  return (numerator / denominator) * 100;
}

function formatQuestionRef(question: { questionId: number; classCorrectPct: number } | null): string {
  if (!question) {
    return "N/A";
  }
  return `Q${question.questionId} (${question.classCorrectPct.toFixed(2)}%)`;
}

function clampProbability(value: number): number {
  if (Number.isNaN(value)) {
    return 0;
  }
  if (value < 0) {
    return 0;
  }
  if (value > 1) {
    return 1;
  }
  return value;
}

export function DashboardClient({ data }: DashboardClientProps) {
  const [processedResult, setProcessedResult] = useState<ProcessSelectedQuestionsResponse | null>(null);
  const [testedQuestionIds, setTestedQuestionIds] = useState<number[]>([]);
  const [studentThresholdPct, setStudentThresholdPct] = useState<number>(
    data.settings.studentThresholdPct,
  );
  const [questionThresholdPct, setQuestionThresholdPct] = useState<number>(
    data.settings.questionThresholdPct,
  );
  const testedQuestionIdSet = useMemo(() => new Set(testedQuestionIds), [testedQuestionIds]);
  const probabilitiesByStudent = useMemo(() => {
    const probabilities = new Map<number, Map<number, number>>();
    if (!processedResult) {
      return probabilities;
    }
    for (const student of processedResult.students) {
      const perQuestion = new Map<number, number>();
      for (const summary of student.question_summaries) {
        const p = clampProbability(summary.avg_probability);
        perQuestion.set(summary.question_id, p);
      }
      probabilities.set(student.student_id, perQuestion);
    }
    return probabilities;
  }, [processedResult]);

  const displayedQuestions = useMemo(() => {
    if (!processedResult) {
      return data.questions;
    }

    return data.questions.map((question) => {
      if (!testedQuestionIdSet.has(question.questionId)) {
        return question;
      }

      const probabilities = processedResult.students
        .map((student) => probabilitiesByStudent.get(student.student_id)?.get(question.questionId))
        .filter((value): value is number => value !== undefined);

      const attempts = probabilities.length;
      const expectedCorrect = probabilities.reduce((sum, value) => sum + value, 0);
      const classCorrectPct = attempts > 0 ? Number(((expectedCorrect / attempts) * 100).toFixed(2)) : 0;

      return {
        ...question,
        classCorrectPct,
        correctCount: Number(expectedCorrect.toFixed(2)),
        attempts,
        flagged: classCorrectPct < questionThresholdPct,
      };
    });
  }, [
    data.questions,
    processedResult,
    probabilitiesByStudent,
    testedQuestionIdSet,
    questionThresholdPct,
  ]);

  const displayedMatrix = useMemo(() => {
    if (!processedResult) {
      return data.matrix;
    }

    const processedByStudent = new Map(processedResult.students.map((student) => [student.student_id, student]));
    return {
      questionIds: data.matrix.questionIds,
      rows: data.matrix.rows.map((row) => {
        const processed = processedByStudent.get(row.studentId);
        const probabilities = probabilitiesByStudent.get(row.studentId);
        const baseCellsByQuestion = new Map(row.cells.map((cell) => [cell.questionId, cell]));
        return {
          ...row,
          cells: data.matrix.questionIds.map((questionId) => {
            const baseCell = baseCellsByQuestion.get(questionId);
            if (!testedQuestionIdSet.has(questionId)) {
              return baseCell || { questionId, state: "unanswered" as const };
            }
            if (!processed || !probabilities) {
              return { questionId, state: "unanswered" as const };
            }
            const probability = probabilities.get(questionId);
            if (probability === undefined) {
              return { questionId, state: "unanswered" as const };
            }
            return {
              questionId,
              state: probability >= 0.5 ? ("correct" as const) : ("wrong" as const),
              probability,
            };
          }),
        };
      }),
    };
  }, [data.matrix, processedResult, probabilitiesByStudent, testedQuestionIdSet]);

  const displayedStudents = useMemo(() => {
    return data.students.map((student) => {
      const total = displayedMatrix.questionIds.length;
      const matrixRow = displayedMatrix.rows.find((row) => row.studentId === student.studentId);
      const cells = matrixRow?.cells || [];

      const expectedCorrect = cells.reduce((sum, cell) => {
        if (cell.probability !== undefined) {
          return sum + cell.probability;
        }
        if (cell.state === "correct") {
          return sum + 1;
        }
        return sum;
      }, 0);
      const answered = cells.filter(
        (cell) => cell.probability !== undefined || cell.state !== "unanswered",
      ).length;
      return {
        ...student,
        totalQuestions: total,
        correctCount: Number(expectedCorrect.toFixed(2)),
        answeredCount: answered,
        scorePct: Number(pct(expectedCorrect, total).toFixed(2)),
        completionPct: Number(pct(answered, total).toFixed(2)),
        status: answered === total ? ("Complete" as const) : ("Incomplete" as const),
      };
    });
  }, [data.students, displayedMatrix]);

  const displayedSummary = useMemo(() => {
    const scores = displayedStudents.map((student) => student.scorePct);
    const sortedQuestions = [...displayedQuestions].sort(
      (left, right) => left.classCorrectPct - right.classCorrectPct,
    );
    const classAverage =
      scores.length > 0 ? scores.reduce((sum, value) => sum + value, 0) / scores.length : 0;
    const meanScore = classAverage;
    const medianScore =
      scores.length > 0
        ? [...scores].sort((a, b) => a - b)[Math.floor(scores.length / 2)]
        : 0;
    const belowThresholdCount =
      testedQuestionIds.length === 0
        ? 0
        : displayedStudents.filter((student) => student.scorePct < studentThresholdPct).length;
    const hardest = sortedQuestions[0] || null;
    const easiest = sortedQuestions[sortedQuestions.length - 1] || null;

    return {
      classAveragePct: classAverage,
      meanScorePct: meanScore,
      medianScorePct: medianScore,
      studentsBelowThresholdPct:
        testedQuestionIds.length === 0 ? 0 : pct(belowThresholdCount, displayedStudents.length),
      hardestQuestion: hardest
        ? { questionId: hardest.questionId, classCorrectPct: hardest.classCorrectPct }
        : null,
      easiestQuestion: easiest
        ? { questionId: easiest.questionId, classCorrectPct: easiest.classCorrectPct }
        : null,
    };
  }, [displayedStudents, displayedQuestions, studentThresholdPct, testedQuestionIds.length]);

  const mostMissedConcepts: ConceptAccuracy[] = useMemo(() => {
    return data.conceptAccuracy
      .filter((c) => c.correctPct < questionThresholdPct)
      .slice(0, 5);
  }, [data.conceptAccuracy, questionThresholdPct]);

  const belowThresholdValueClass =
    displayedSummary.studentsBelowThresholdPct <= 20
      ? "text-emerald-700"
      : displayedSummary.studentsBelowThresholdPct <= 40
        ? "text-amber-600"
        : "text-rose-700";

  return (
    <>
      <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        <StatCard
          label="Class average score"
          value={`${displayedSummary.classAveragePct.toFixed(2)}%`}
        />
        <StatCard
          label="Median score"
          value={`${displayedSummary.medianScorePct.toFixed(2)}%`}
        />
        <StatCard
          label="Mean score"
          value={`${displayedSummary.meanScorePct.toFixed(2)}%`}
        />
        <StatCard
          label="Hardest question"
          value={formatQuestionRef(displayedSummary.hardestQuestion)}
        />
        <StatCard
          label="Easiest question"
          value={formatQuestionRef(displayedSummary.easiestQuestion)}
        />
        <StatCard
          label="Students below threshold"
          value={`${displayedSummary.studentsBelowThresholdPct.toFixed(2)}%`}
          valueClassName={belowThresholdValueClass}
        />
      </section>

      {mostMissedConcepts.length > 0 && (
        <SectionCard
          title="Most Missed Concepts"
          subtitle={
            <span className="text-sm text-zinc-500">
              Concepts below {questionThresholdPct}% class accuracy (top 5)
            </span>
          }
        >
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {mostMissedConcepts.map((concept) => (
              <Card key={concept.tag} className="bg-zinc-50">
                <CardContent className="flex items-center justify-between gap-3 p-3">
                  <span className="text-sm font-medium text-zinc-800">{concept.tag}</span>
                  <Badge
                    variant={concept.correctPct < 30 ? "destructive" : "outline"}
                    className="shrink-0"
                  >
                    {concept.correctPct.toFixed(1)}%
                  </Badge>
                </CardContent>
              </Card>
            ))}
          </div>
        </SectionCard>
      )}

      <SectionCard
        title="Students Table"
        subtitle={
          <label className="inline-flex items-center gap-2 text-sm text-zinc-500">
            Flag students below
            <input
              type="number"
              min={0}
              max={100}
              step={1}
              value={studentThresholdPct}
              onChange={(event) => {
                const next = Number(event.target.value);
                if (Number.isNaN(next)) {
                  setStudentThresholdPct(0);
                  return;
                }
                setStudentThresholdPct(Math.min(100, Math.max(0, next)));
              }}
              className="w-16 rounded-md border border-zinc-200 bg-white px-2 py-1 text-right text-sm text-zinc-900"
              aria-label="Student threshold percent"
            />
            %
          </label>
        }
      >
        <StudentsTable students={displayedStudents} />
      </SectionCard>

      <SectionCard
        title="Question Analysis"
        subtitle={
          <label className="inline-flex items-center gap-2 text-sm text-zinc-500">
            Flag questions below
            <input
              type="number"
              min={0}
              max={100}
              step={1}
              value={questionThresholdPct}
              onChange={(event) => {
                const next = Number(event.target.value);
                if (Number.isNaN(next)) {
                  setQuestionThresholdPct(0);
                  return;
                }
                setQuestionThresholdPct(Math.min(100, Math.max(0, next)));
              }}
              className="w-16 rounded-md border border-zinc-200 bg-white px-2 py-1 text-right text-sm text-zinc-900"
              aria-label="Question threshold percent"
            />
            % class correct
          </label>
        }
      >
        <QuestionsTable
          questions={displayedQuestions}
          onProcessedSelection={(questionIds, result) => {
            setTestedQuestionIds(questionIds);
            setProcessedResult(result);
          }}
        />
      </SectionCard>

      <SectionCard title="Student x Question Matrix">
        <MatrixView questionIds={displayedMatrix.questionIds} rows={displayedMatrix.rows} />
      </SectionCard>

      <SectionCard title="Remediation Test-Set Builder">
        <RemediationPanel data={data} />
      </SectionCard>
    </>
  );
}
