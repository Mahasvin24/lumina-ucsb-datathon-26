import { DashboardClient } from "@/components/dashboard/dashboard-client";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { DashboardPayload, getDashboardData } from "@/lib/dashboard-api";

export const metadata = {
  title: "Test Set Dashboard",
};

function DashboardErrorState({ message }: { message: string }) {
  return (
    <main className="mx-auto min-h-screen max-w-6xl p-6">
      <Alert variant="destructive" className="p-6">
        <AlertTitle className="text-xl">Dashboard unavailable</AlertTitle>
        <AlertDescription className="mt-2">{message}</AlertDescription>
        <AlertDescription className="mt-2">
          Start the backend API and refresh this page. Expected endpoint:{" "}
          <code>/dashboard-data</code>
        </AlertDescription>
      </Alert>
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
    <main className="mx-auto min-h-screen max-w-7xl space-y-6 bg-zinc-50 p-6">
      <header className="space-y-2">
        <h1 className="text-3xl font-semibold text-zinc-900">Lumina Dashboard</h1>
        <p className="text-sm text-zinc-600">
          Illuminating Student Learning Through Deep Knowledge Tracing
        </p>
      </header>

      <DashboardClient data={data} />
    </main>
  );
}
