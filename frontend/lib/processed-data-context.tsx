"use client";

import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";
import type { ProcessSelectedQuestionsResponse } from "@/lib/dashboard-api";
import { getPersistedDashboardState, setPersistedDashboardState } from "@/lib/dashboard-storage";

type ProcessedDataContextValue = {
  processedResult: ProcessSelectedQuestionsResponse | null;
  testedQuestionIds: number[];
  setProcessedData: (ids: number[], result: ProcessSelectedQuestionsResponse) => void;
  clearProcessedData: () => void;
};

const ProcessedDataContext = createContext<ProcessedDataContextValue>({
  processedResult: null,
  testedQuestionIds: [],
  setProcessedData: () => {},
  clearProcessedData: () => {},
});

export function ProcessedDataProvider({ children }: { children: ReactNode }) {
  const [processedResult, setProcessedResult] =
    useState<ProcessSelectedQuestionsResponse | null>(null);
  const [testedQuestionIds, setTestedQuestionIds] = useState<number[]>([]);

  useEffect(() => {
    const stored = getPersistedDashboardState();
    if (stored?.processedResult != null && Array.isArray(stored.testedQuestionIds)) {
      setProcessedResult(stored.processedResult);
      setTestedQuestionIds(stored.testedQuestionIds);
    }
  }, []);

  const setProcessedData = useCallback(
    (ids: number[], result: ProcessSelectedQuestionsResponse) => {
      setTestedQuestionIds(ids);
      setProcessedResult(result);
      setPersistedDashboardState({ processedResult: result, testedQuestionIds: ids });
    },
    [],
  );

  const clearProcessedData = useCallback(() => {
    setTestedQuestionIds([]);
    setProcessedResult(null);
    setPersistedDashboardState({ processedResult: null, testedQuestionIds: [] });
  }, []);

  return (
    <ProcessedDataContext.Provider
      value={{ processedResult, testedQuestionIds, setProcessedData, clearProcessedData }}
    >
      {children}
    </ProcessedDataContext.Provider>
  );
}

export function useProcessedData() {
  return useContext(ProcessedDataContext);
}
