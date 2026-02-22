"use client";

import { useEffect, useMemo, useState } from "react";

import {
  ProcessSelectedQuestionsResponse,
  QuestionRow,
  processSelectedQuestions,
} from "@/lib/dashboard-api";
import { useProcessedData } from "@/lib/processed-data-context";
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
  initialTestedQuestionIds?: number[];
  onProcessedSelection?: (questionIds: number[], result: ProcessSelectedQuestionsResponse) => void;
};

export function QuestionsTable({
  questions,
  initialTestedQuestionIds,
  onProcessedSelection,
}: QuestionsTableProps) {
  const { clearProcessedData } = useProcessedData();
  const defaultSelected = useMemo(
    () => new Set(questions.map((question) => question.questionId)),
    [questions],
  );
  const initialSelected = useMemo(() => {
    if (initialTestedQuestionIds?.length) {
      return new Set(initialTestedQuestionIds);
    }
    return defaultSelected;
  }, [defaultSelected, initialTestedQuestionIds]);
  const [selected, setSelected] = useState<Set<number>>(initialSelected);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ProcessSelectedQuestionsResponse | null>(null);
  const initialTestedSet = useMemo(
    () => (initialTestedQuestionIds?.length ? new Set(initialTestedQuestionIds) : new Set<number>()),
    [initialTestedQuestionIds],
  );
  const [testedQuestionIds, setTestedQuestionIds] = useState<Set<number>>(initialTestedSet);

  useEffect(() => {
    if (initialTestedQuestionIds?.length) {
      setSelected(new Set(initialTestedQuestionIds));
      setTestedQuestionIds(new Set(initialTestedQuestionIds));
    } else if (Array.isArray(initialTestedQuestionIds) && initialTestedQuestionIds.length === 0) {
      setResult(null);
      setTestedQuestionIds(new Set());
    }
  }, [initialTestedQuestionIds]);

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
          Select questions to test student knowledge.
        </p>
        <div className="flex items-center gap-2">
          <Button
            type="button"
            onClick={processCheckedQuestions}
            disabled={isProcessing}
          >
            {isProcessing ? "Processing..." : "Process Questions"}
          </Button>
          <Button
            type="button"
            variant="outline"
            onClick={clearProcessedData}
            disabled={isProcessing}
          >
            Reset
          </Button>
        </div>
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
            ? `Predictions ready for ${result.students.length} students (${result.errors.length} could not be analyzed).`
            : `Predictions ready — ${result.students.length} student${result.students.length === 1 ? "" : "s"} analyzed across ${result.selected_question_ids.length} question${result.selected_question_ids.length === 1 ? "" : "s"}.`}
          </AlertDescription>
        </Alert>
      ) : null}

      <Table className="min-w-full text-left">
        <TableHeader>
          <TableRow>
            <TableHead>Process</TableHead>
            <TableHead>Question</TableHead>
            <TableHead>Class % Correct</TableHead>
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
                  {question.skillTags.length > 0 ? (
                    <div className="flex flex-wrap gap-1">
                      {question.skillTags.map((tag) => (
                        <Badge key={tag} variant="outline" className="text-xs font-normal">
                          {tag}
                        </Badge>
                      ))}
                    </div>
                  ) : (
                    "None"
                  )}
                </TableCell>
                <TableCell>
                  {testedQuestionIds.has(question.questionId) ? (
                    question.flagged ? (
                      <Badge variant="destructive">Weak</Badge>
                    ) : (
                      <Badge variant="success">Strong</Badge>
                    )
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
