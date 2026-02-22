import Link from "next/link";

import { StudentDashboard } from "@/components/dashboard/student-dashboard";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { StudentDetail, getStudentData } from "@/lib/dashboard-api";

type Params = { studentId: string };

export async function generateMetadata({ params }: { params: Promise<Params> }) {
  const { studentId } = await params;
  return { title: `Student ${studentId} — Dashboard` };
}

function ErrorState({ message }: { message: string }) {
  return (
    <main className="mx-auto min-h-screen max-w-6xl p-6">
      <Alert variant="destructive" className="p-6">
        <AlertTitle className="text-xl">Student data unavailable</AlertTitle>
        <AlertDescription className="mt-2">{message}</AlertDescription>
      </Alert>
    </main>
  );
}

export default async function StudentPage({ params }: { params: Promise<Params> }) {
  const { studentId } = await params;
  const id = Number(studentId);

  if (Number.isNaN(id) || id <= 0) {
    return <ErrorState message={`Invalid student ID: ${studentId}`} />;
  }

  let data: StudentDetail;
  try {
    data = await getStudentData(id);
  } catch (error) {
    return (
      <ErrorState message={error instanceof Error ? error.message : "Unknown error"} />
    );
  }

  return (
    <main className="mx-auto min-h-screen max-w-7xl space-y-6 bg-zinc-50 p-6">
      <header className="space-y-2">
        <Link
          href="/dashboard"
          className="inline-flex items-center gap-1 text-sm text-sky-600 hover:text-sky-800"
        >
          &larr; Back to Dashboard
        </Link>
        <h1 className="text-3xl font-semibold text-zinc-900">{data.name}</h1>
        <p className="text-sm text-zinc-600">
          User ID {data.userId}
        </p>
      </header>

      <StudentDashboard data={data} />
    </main>
  );
}
