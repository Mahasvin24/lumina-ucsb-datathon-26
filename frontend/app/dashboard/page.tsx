import { MatrixView } from "@/components/dashboard/matrix-view";
import { QuestionsTable } from "@/components/dashboard/questions-table";
import { RemediationPanel } from "@/components/dashboard/remediation-panel";
import { SectionCard } from "@/components/dashboard/section-card";
import { StatCard } from "@/components/dashboard/stat-card";
import { StudentsTable } from "@/components/dashboard/students-table";
import { DashboardPayload, getDashboardData } from "@/lib/dashboard-api";

export const metadata = {
  title: "Test Set Dashboard",
};

function formatQuestionRef(
  question: { questionId: number; classCorrectPct: number } | null,
): string {
  if (!question) {
    return "N/A";
  }
  return `Q${question.questionId} (${question.classCorrectPct.toFixed(2)}%)`;
}

function DashboardErrorState({ message }: { message: string }) {
  return (
    <main className="mx-auto min-h-screen max-w-6xl p-6">
      <div className="rounded-xl border border-rose-300 bg-rose-50 p-6 text-rose-900 dark:border-rose-900 dark:bg-rose-950/40 dark:text-rose-200">
        <h1 className="text-xl font-semibold">Dashboard unavailable</h1>
        <p className="mt-2 text-sm">{message}</p>
        <p className="mt-2 text-sm">
          Start the backend API and refresh this page. Expected endpoint:{" "}
          <code>/dashboard-data</code>
        </p>
      </div>
    </main>
  );
}

export default async function DashboardPage() {
  let data: DashboardPayload;
  try {
    data = await getDashboardData();
  } catch (error) {
    return (
      <DashboardErrorState
        message={error instanceof Error ? error.message : "Unknown error"}
      />
    );
  }

  return (
    <main className="mx-auto min-h-screen max-w-7xl space-y-6 bg-zinc-50 p-6 dark:bg-black">
      <header className="space-y-2">
        <h1 className="text-3xl font-semibold text-zinc-900 dark:text-zinc-100">
          Test Set Summary
        </h1>
        <p className="text-sm text-zinc-600 dark:text-zinc-400">
          Real metrics from <code>students/student_*.csv</code> and{" "}
          <code>students/test_questions.csv</code>.
        </p>
      </header>

      <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        <StatCard
          label="Class average score"
          value={`${data.summary.classAveragePct.toFixed(2)}%`}
        />
        <StatCard
          label="Median score"
          value={`${data.summary.medianScorePct.toFixed(2)}%`}
        />
        <StatCard
          label="Students below threshold"
          value={`${data.summary.studentsBelowThresholdPct.toFixed(2)}%`}
        />
        <StatCard
          label="Hardest question"
          value={formatQuestionRef(data.summary.hardestQuestion)}
        />
        <StatCard
          label="Easiest question"
          value={formatQuestionRef(data.summary.easiestQuestion)}
        />
        <StatCard
          label="Completion rate"
          value={`${data.summary.completionRatePct.toFixed(2)}%`}
        />
      </section>

      <SectionCard
        title="Students Table"
        subtitle={`Flag students below ${data.settings.studentThresholdPct.toFixed(0)}%`}
      >
        <StudentsTable students={data.students} />
      </SectionCard>

      <SectionCard
        title="Question Analysis"
        subtitle={`Flag questions below ${data.settings.questionThresholdPct.toFixed(0)}% class correct`}
      >
        <QuestionsTable
          questions={data.questions}
          questionThresholdPct={data.settings.questionThresholdPct}
        />
      </SectionCard>

      <SectionCard title="Student x Question Matrix">
        <MatrixView questionIds={data.matrix.questionIds} rows={data.matrix.rows} />
      </SectionCard>

      <SectionCard title="Remediation Test-Set Builder">
        <RemediationPanel data={data} />
      </SectionCard>
    </main>
  );
}
