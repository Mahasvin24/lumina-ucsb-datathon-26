"use client";

import { useMemo, useState } from "react";

import {
  ProcessSelectedQuestionsResponse,
  QuestionRow,
  processSelectedQuestions,
} from "@/lib/dashboard-api";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

type QuestionsTableProps = {
  questions: QuestionRow[];
  onProcessedSelection?: (questionIds: number[], result: ProcessSelectedQuestionsResponse) => void;
};

export function QuestionsTable({
  questions,
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
  const [testedQuestionIds, setTestedQuestionIds] = useState<Set<number>>(new Set());

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
      setTestedQuestionIds(new Set(checked));
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
        <p className="text-sm text-zinc-600">
          Checked questions are sent to backend for all discovered students.
        </p>
        <Button
          type="button"
          onClick={processCheckedQuestions}
          disabled={isProcessing}
        >
          {isProcessing ? "Processing..." : "Process selected questions"}
        </Button>
      </div>

      {error ? (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      ) : null}

      {result ? (
        <Alert variant="success">
          <AlertDescription>
          {result.errors && result.errors.length > 0
            ? `Processed with partial errors: ${result.students.length} students succeeded, ${result.errors.length} failed.`
            : `Success: processed ${result.students.length} students.`}
          </AlertDescription>
        </Alert>
      ) : null}

      <Table className="min-w-full text-left">
        <TableHeader>
          <TableRow>
            <TableHead>Process</TableHead>
            <TableHead>Question</TableHead>
            <TableHead>Class % Correct</TableHead>
            <TableHead>Correct / Attempts</TableHead>
            <TableHead>Skill Tags</TableHead>
            <TableHead>Flag</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
            {questions.map((question) => (
              <TableRow key={question.questionId}>
                <TableCell>
                  <Checkbox
                    checked={selected.has(question.questionId)}
                    onCheckedChange={() => toggle(question.questionId)}
                    aria-label={`Process question ${question.questionId}`}
                  />
                </TableCell>
                <TableCell>Q{question.questionId}</TableCell>
                <TableCell>{question.classCorrectPct.toFixed(2)}%</TableCell>
                <TableCell>
                  {question.correctCount} / {question.attempts}
                </TableCell>
                <TableCell>
                  {question.skillTags.length > 0
                    ? question.skillTags.join(", ")
                    : "None"}
                </TableCell>
                <TableCell>
                  {question.flagged ? (
                    <Badge variant="destructive">Flagged</Badge>
                  ) : testedQuestionIds.has(question.questionId) ? (
                    <Badge variant="success">Tested</Badge>
                  ) : (
                    <Badge variant="secondary">Not Tested</Badge>
                  )}
                </TableCell>
              </TableRow>
            ))}
        </TableBody>
      </Table>

    </div>
  );
}
