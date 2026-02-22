"use client";

import { createContext, useCallback, useContext, useState, type ReactNode } from "react";
import type { ProcessSelectedQuestionsResponse } from "@/lib/dashboard-api";

type ProcessedDataContextValue = {
  processedResult: ProcessSelectedQuestionsResponse | null;
  testedQuestionIds: number[];
  setProcessedData: (ids: number[], result: ProcessSelectedQuestionsResponse) => void;
};

const ProcessedDataContext = createContext<ProcessedDataContextValue>({
  processedResult: null,
  testedQuestionIds: [],
  setProcessedData: () => {},
});

export function ProcessedDataProvider({ children }: { children: ReactNode }) {
  const [processedResult, setProcessedResult] =
    useState<ProcessSelectedQuestionsResponse | null>(null);
  const [testedQuestionIds, setTestedQuestionIds] = useState<number[]>([]);

  const setProcessedData = useCallback(
    (ids: number[], result: ProcessSelectedQuestionsResponse) => {
      setTestedQuestionIds(ids);
      setProcessedResult(result);
    },
    [],
  );

  return (
    <ProcessedDataContext.Provider
      value={{ processedResult, testedQuestionIds, setProcessedData }}
    >
      {children}
    </ProcessedDataContext.Provider>
  );
}

export function useProcessedData() {
  return useContext(ProcessedDataContext);
}
