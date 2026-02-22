"use client";

import { useMemo, useState } from "react";

import { MatrixView } from "@/components/dashboard/matrix-view";
import { QuestionsTable } from "@/components/dashboard/questions-table";
import { RemediationPanel } from "@/components/dashboard/remediation-panel";
import { SectionCard } from "@/components/dashboard/section-card";
import { StatCard } from "@/components/dashboard/stat-card";
import { StudentsTable } from "@/components/dashboard/students-table";
import {
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
  const [activeQuestionIds, setActiveQuestionIds] = useState<number[]>(data.matrix.questionIds);
  const [processedResult, setProcessedResult] = useState<ProcessSelectedQuestionsResponse | null>(null);

  const activeQuestionIdSet = useMemo(() => new Set(activeQuestionIds), [activeQuestionIds]);
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
    const filteredBaseQuestions = data.questions.filter((question) =>
      activeQuestionIdSet.has(question.questionId),
    );
    if (!processedResult) {
      return filteredBaseQuestions;
    }

    const skillTagsByQuestion = new Map(
      data.questions.map((question) => [question.questionId, question.skillTags]),
    );

    return activeQuestionIds.map((questionId) => {
      const probabilities = processedResult.students
        .map((student) => probabilitiesByStudent.get(student.student_id)?.get(questionId))
        .filter((value): value is number => value !== undefined);

      const attempts = probabilities.length;
      const expectedCorrect = probabilities.reduce((sum, value) => sum + value, 0);
      const classCorrectPct = attempts > 0 ? Number(((expectedCorrect / attempts) * 100).toFixed(2)) : 0;

      return {
        questionId,
        classCorrectPct,
        correctCount: Number(expectedCorrect.toFixed(2)),
        attempts,
        skillTags: skillTagsByQuestion.get(questionId) || [],
        flagged: classCorrectPct < data.settings.questionThresholdPct,
      };
    });
  }, [
    data.questions,
    data.settings.questionThresholdPct,
    activeQuestionIdSet,
    activeQuestionIds,
    processedResult,
    probabilitiesByStudent,
  ]);

  const displayedMatrix = useMemo(() => {
    if (!processedResult) {
      return {
        questionIds: activeQuestionIds,
        rows: data.matrix.rows.map((row) => ({
          ...row,
          cells: row.cells.filter((cell) => activeQuestionIdSet.has(cell.questionId)),
        })),
      };
    }

    const processedByStudent = new Map(processedResult.students.map((student) => [student.student_id, student]));
    return {
      questionIds: activeQuestionIds,
      rows: data.matrix.rows.map((row) => {
        const processed = processedByStudent.get(row.studentId);
        const probabilities = probabilitiesByStudent.get(row.studentId);
        return {
          ...row,
          cells: activeQuestionIds.map((questionId) => {
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
  }, [data.matrix.rows, activeQuestionIds, activeQuestionIdSet, processedResult, probabilitiesByStudent]);

  const displayedStudents = useMemo(() => {
    const byStudentId = new Map(data.matrix.rows.map((row) => [row.studentId, row]));
    const processedByStudent = new Map(
      (processedResult?.students || []).map((student) => [student.student_id, student]),
    );

    return data.students.map((student) => {
      const total = activeQuestionIds.length;
      const processed = processedByStudent.get(student.studentId);
      const probabilities = probabilitiesByStudent.get(student.studentId);

      if (processed && probabilities) {
        const activeProbabilities = activeQuestionIds
          .map((questionId) => probabilities.get(questionId))
          .filter((value): value is number => value !== undefined);
        const expectedCorrect = activeProbabilities.reduce((sum, value) => sum + value, 0);
        const answered = activeProbabilities.length;
        return {
          ...student,
          totalQuestions: total,
          correctCount: Number(expectedCorrect.toFixed(2)),
          answeredCount: answered,
          scorePct: Number(pct(expectedCorrect, total).toFixed(2)),
          completionPct: Number(pct(answered, total).toFixed(2)),
          status: answered === total ? ("Complete" as const) : ("Incomplete" as const),
        };
      }

      const matrixRow = byStudentId.get(student.studentId);
      const cells = (matrixRow?.cells || []).filter((cell) => activeQuestionIdSet.has(cell.questionId));
      const correct = cells.filter((cell) => cell.state === "correct").length;
      const answered = cells.filter((cell) => cell.state !== "unanswered").length;
      return {
        ...student,
        totalQuestions: total,
        correctCount: correct,
        answeredCount: answered,
        scorePct: Number(pct(correct, total).toFixed(2)),
        completionPct: Number(pct(answered, total).toFixed(2)),
        status: answered === total ? ("Complete" as const) : ("Incomplete" as const),
      };
    });
  }, [
    data.students,
    data.matrix.rows,
    activeQuestionIdSet,
    activeQuestionIds,
    processedResult,
    probabilitiesByStudent,
  ]);

  const displayedSummary = useMemo(() => {
    const scores = displayedStudents.map((student) => student.scorePct);
    const completion = displayedStudents.map((student) => student.completionPct);
    const sortedQuestions = [...displayedQuestions].sort(
      (left, right) => left.classCorrectPct - right.classCorrectPct,
    );
    const classAverage =
      scores.length > 0 ? scores.reduce((sum, value) => sum + value, 0) / scores.length : 0;
    const medianScore =
      scores.length > 0
        ? [...scores].sort((a, b) => a - b)[Math.floor(scores.length / 2)]
        : 0;
    const completionRate =
      completion.length > 0
        ? completion.reduce((sum, value) => sum + value, 0) / completion.length
        : 0;
    const belowThresholdCount = displayedStudents.filter(
      (student) => student.scorePct < data.settings.studentThresholdPct,
    ).length;
    const hardest = sortedQuestions[0] || null;
    const easiest = sortedQuestions[sortedQuestions.length - 1] || null;

    return {
      classAveragePct: classAverage,
      medianScorePct: medianScore,
      studentsBelowThresholdPct: pct(belowThresholdCount, displayedStudents.length),
      completionRatePct: completionRate,
      hardestQuestion: hardest
        ? { questionId: hardest.questionId, classCorrectPct: hardest.classCorrectPct }
        : null,
      easiestQuestion: easiest
        ? { questionId: easiest.questionId, classCorrectPct: easiest.classCorrectPct }
        : null,
    };
  }, [displayedStudents, displayedQuestions, data.settings.studentThresholdPct]);

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
          label="Students below threshold"
          value={`${displayedSummary.studentsBelowThresholdPct.toFixed(2)}%`}
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
          label="Completion rate"
          value={`${displayedSummary.completionRatePct.toFixed(2)}%`}
        />
      </section>

      <SectionCard
        title="Students Table"
        subtitle={`Flag students below ${data.settings.studentThresholdPct.toFixed(0)}%`}
      >
        <StudentsTable students={displayedStudents} />
      </SectionCard>

      <SectionCard
        title="Question Analysis"
        subtitle={`Flag questions below ${data.settings.questionThresholdPct.toFixed(0)}% class correct`}
      >
        <QuestionsTable
          questions={displayedQuestions}
          questionThresholdPct={data.settings.questionThresholdPct}
          onProcessedSelection={(questionIds, result) => {
            setActiveQuestionIds(questionIds);
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
