export type DashboardSettings = {
  studentThresholdPct: number;
  questionThresholdPct: number;
  excludeUnanswered: boolean;
  topNHardest: number;
  totalStudents: number;
  totalQuestions: number;
};

export type DashboardSummary = {
  classAveragePct: number;
  medianScorePct: number;
  studentsBelowThresholdPct: number;
  hardestQuestion: { questionId: number; classCorrectPct: number } | null;
  easiestQuestion: { questionId: number; classCorrectPct: number } | null;
  completionRatePct: number;
};

export type StudentRow = {
  studentId: number;
  name: string;
  userId: number;
  scorePct: number;
  correctCount: number;
  totalQuestions: number;
  answeredCount: number;
  completionPct: number;
  status: "Complete" | "Incomplete";
  timeSpentMinutes: number;
  medianSpacingMinutes: number;
  mostMissedConcept: string;
};

export type QuestionRow = {
  questionId: number;
  classCorrectPct: number;
  correctCount: number;
  attempts: number;
  skillTags: string[];
  flagged: boolean;
};

export type MatrixCell = {
  questionId: number;
  state: "correct" | "wrong" | "unanswered";
  probability?: number;
};

export type MatrixRow = {
  studentId: number;
  name: string;
  cells: MatrixCell[];
};

export type DashboardPayload = {
  settings: DashboardSettings;
  summary: DashboardSummary;
  students: StudentRow[];
  questions: QuestionRow[];
  matrix: {
    questionIds: number[];
    rows: MatrixRow[];
  };
  remediation: {
    flaggedStudentIds: number[];
    selectedQuestionIds: number[];
    questionCount: number;
    estimatedTimeMinutes: number;
    skillCoverage: string[];
    previewQuestions: QuestionRow[];
  };
};

export type ProcessedQuestionSummary = {
  question_id: number;
  rows: number;
  avg_probability: number;
  used_fallback?: boolean;
  used_concept_proxy?: boolean;
};

export type ProcessedStudentResult = {
  student_id: number;
  user_id: number;
  processed_rows: number;
  matched_rows?: number;
  proxy_rows?: number;
  question_summaries: ProcessedQuestionSummary[];
};

export type ProcessSelectedQuestionsError = {
  student_id: number;
  error: string;
};

export type ProcessSelectedQuestionsResponse = {
  selected_question_ids: number[];
  students: ProcessedStudentResult[];
  total_processed_rows: number;
  errors?: ProcessSelectedQuestionsError[];
};

type DashboardQuery = {
  studentThresholdPct?: number;
  questionThresholdPct?: number;
  excludeUnanswered?: boolean;
  topNHardest?: number;
};

function resolveBackendBaseUrl(): string {
  return (
    process.env.BACKEND_API_URL ||
    process.env.NEXT_PUBLIC_BACKEND_API_URL ||
    "http://127.0.0.1:8000"
  );
}

export async function getDashboardData(
  query: DashboardQuery = {},
): Promise<DashboardPayload> {
  const params = new URLSearchParams();
  if (query.studentThresholdPct !== undefined) {
    params.set("student_threshold", String(query.studentThresholdPct));
  }
  if (query.questionThresholdPct !== undefined) {
    params.set("question_threshold", String(query.questionThresholdPct));
  }
  if (query.excludeUnanswered !== undefined) {
    params.set("exclude_unanswered", String(query.excludeUnanswered));
  }
  if (query.topNHardest !== undefined) {
    params.set("top_n_hardest", String(query.topNHardest));
  }

  const baseUrl = resolveBackendBaseUrl();
  const url = `${baseUrl}/dashboard-data${params.toString() ? `?${params.toString()}` : ""}`;

  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    const details = await response.text();
    throw new Error(`Dashboard API failed (${response.status}): ${details}`);
  }
  return (await response.json()) as DashboardPayload;
}

export async function processSelectedQuestions(
  questionIds: number[],
): Promise<ProcessSelectedQuestionsResponse> {
  const baseUrl = resolveBackendBaseUrl();
  const response = await fetch(`${baseUrl}/process-selected-questions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question_ids: questionIds }),
  });
  if (!response.ok) {
    const details = await response.text();
    throw new Error(`Process API failed (${response.status}): ${details}`);
  }
  return (await response.json()) as ProcessSelectedQuestionsResponse;
}
