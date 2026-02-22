"use client";

import { useMemo, useState } from "react";

import {
  ProcessSelectedQuestionsResponse,
  QuestionRow,
  processSelectedQuestions,
} from "@/lib/dashboard-api";

type QuestionsTableProps = {
  questions: QuestionRow[];
  questionThresholdPct: number;
  onProcessedSelection?: (questionIds: number[], result: ProcessSelectedQuestionsResponse) => void;
};

export function QuestionsTable({
  questions,
  questionThresholdPct,
  onProcessedSelection,
}: QuestionsTableProps) {
  const defaultSelected = useMemo(
    () => new Set(questions.map((question) => question.questionId)),
    [questions],
  );
  const [selected, setSelected] = useState<Set<number>>(defaultSelected);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ProcessSelectedQuestionsResponse | null>(null);

  const toggle = (questionId: number) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(questionId)) {
        next.delete(questionId);
      } else {
        next.add(questionId);
      }
      return next;
    });
  };

  const processCheckedQuestions = async () => {
    setError(null);
    setResult(null);
    const checked = questions
      .map((question) => question.questionId)
      .filter((questionId) => selected.has(questionId));
    if (checked.length === 0) {
      setError("Select at least one question before processing.");
      return;
    }

    setIsProcessing(true);
    try {
      const response = await processSelectedQuestions(checked);
      setResult(response);
      onProcessedSelection?.(checked, response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown processing error.");
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-sm text-zinc-600 dark:text-zinc-300">
          Checked questions are sent to backend for all discovered students.
        </p>
        <button
          type="button"
          onClick={processCheckedQuestions}
          disabled={isProcessing}
          className="rounded-md bg-zinc-900 px-3 py-2 text-sm font-medium text-white hover:bg-zinc-700 disabled:cursor-not-allowed disabled:opacity-60 dark:bg-zinc-200 dark:text-zinc-900 dark:hover:bg-zinc-400"
        >
          {isProcessing ? "Processing..." : "Process selected questions"}
        </button>
      </div>

      {error ? (
        <p className="rounded-md border border-rose-300 bg-rose-50 px-3 py-2 text-sm text-rose-900 dark:border-rose-900 dark:bg-rose-950/30 dark:text-rose-200">
          {error}
        </p>
      ) : null}

      {result ? (
        <p className="text-sm text-emerald-700 dark:text-emerald-300">
          {result.errors && result.errors.length > 0
            ? `Processed with partial errors: ${result.students.length} students succeeded, ${result.errors.length} failed.`
            : `Success: processed ${result.students.length} students.`}
        </p>
      ) : null}

      <div className="overflow-x-auto">
        <table className="min-w-full text-left text-sm">
          <thead>
            <tr className="border-b border-zinc-200 text-zinc-500 dark:border-zinc-800 dark:text-zinc-400">
              <th className="px-2 py-2 font-medium">Process</th>
              <th className="px-2 py-2 font-medium">Question</th>
              <th className="px-2 py-2 font-medium">Class % Correct</th>
              <th className="px-2 py-2 font-medium">Correct / Attempts</th>
              <th className="px-2 py-2 font-medium">Skill Tags</th>
              <th className="px-2 py-2 font-medium">Flag</th>
            </tr>
          </thead>
          <tbody>
            {questions.map((question) => (
              <tr
                key={question.questionId}
                className="border-b border-zinc-100 text-zinc-800 dark:border-zinc-900 dark:text-zinc-200"
              >
                <td className="px-2 py-2">
                  <input
                    type="checkbox"
                    checked={selected.has(question.questionId)}
                    onChange={() => toggle(question.questionId)}
                    aria-label={`Process question ${question.questionId}`}
                  />
                </td>
                <td className="px-2 py-2">Q{question.questionId}</td>
                <td className="px-2 py-2">{question.classCorrectPct.toFixed(2)}%</td>
                <td className="px-2 py-2">
                  {question.correctCount} / {question.attempts}
                </td>
                <td className="px-2 py-2">
                  {question.skillTags.length > 0
                    ? question.skillTags.join(", ")
                    : "None"}
                </td>
                <td className="px-2 py-2">
                  {question.classCorrectPct < questionThresholdPct ? "Flagged" : "OK"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

    </div>
  );
}
