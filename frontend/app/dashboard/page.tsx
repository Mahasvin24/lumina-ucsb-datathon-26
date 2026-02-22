import { DashboardClient } from "@/components/dashboard/dashboard-client";
import { DashboardPayload, getDashboardData } from "@/lib/dashboard-api";

export const metadata = {
  title: "Test Set Dashboard",
};

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

      <DashboardClient data={data} />
    </main>
  );
}
