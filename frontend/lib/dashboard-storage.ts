"use client";

import type { ProcessSelectedQuestionsResponse } from "@/lib/dashboard-api";

const STORAGE_KEY = "lumina-dashboard-persisted";

export type PersistedDashboardState = {
  processedResult: ProcessSelectedQuestionsResponse | null;
  testedQuestionIds: number[];
  studentThresholdPct?: number;
  questionThresholdPct?: number;
};

function isClient(): boolean {
  return typeof window !== "undefined";
}

export function getPersistedDashboardState(): PersistedDashboardState | null {
  if (!isClient()) {
    return null;
  }
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw == null) return null;
    const parsed = JSON.parse(raw) as unknown;
    if (parsed == null || typeof parsed !== "object") return null;
    const obj = parsed as Record<string, unknown>;
    const state: PersistedDashboardState = {
      processedResult: (obj.processedResult as ProcessSelectedQuestionsResponse | null) ?? null,
      testedQuestionIds: Array.isArray(obj.testedQuestionIds)
        ? (obj.testedQuestionIds as number[]).filter((x) => typeof x === "number")
        : [],
    };
    if (typeof obj.studentThresholdPct === "number") {
      state.studentThresholdPct = obj.studentThresholdPct;
    }
    if (typeof obj.questionThresholdPct === "number") {
      state.questionThresholdPct = obj.questionThresholdPct;
    }
    return state;
  } catch {
    return null;
  }
}

export function setPersistedDashboardState(
  partial: Partial<PersistedDashboardState>,
): void {
  if (!isClient()) return;
  try {
    const current = getPersistedDashboardState();
    const next: PersistedDashboardState = {
      processedResult: partial.processedResult ?? current?.processedResult ?? null,
      testedQuestionIds:
        partial.testedQuestionIds ?? current?.testedQuestionIds ?? [],
    };
    if (partial.studentThresholdPct !== undefined) {
      next.studentThresholdPct = partial.studentThresholdPct;
    } else if (current?.studentThresholdPct !== undefined) {
      next.studentThresholdPct = current.studentThresholdPct;
    }
    if (partial.questionThresholdPct !== undefined) {
      next.questionThresholdPct = partial.questionThresholdPct;
    } else if (current?.questionThresholdPct !== undefined) {
      next.questionThresholdPct = current.questionThresholdPct;
    }
    localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  } catch {
    // ignore write errors
  }
}
