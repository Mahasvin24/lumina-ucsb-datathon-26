import { DashboardPayload } from "@/lib/dashboard-api";

type RemediationPanelProps = {
  data: DashboardPayload;
};

export function RemediationPanel({ data }: RemediationPanelProps) {
  const settings = data.settings;
  const remediation = data.remediation;
  const flaggedNames = data.students
    .filter((student) => remediation.flaggedStudentIds.includes(student.studentId))
    .map((student) => student.name);

  return (
    <div className="space-y-4">
      <div className="grid gap-3 md:grid-cols-2">
        <div className="rounded-lg bg-zinc-50 p-3 dark:bg-zinc-900">
          <p className="text-xs uppercase tracking-wide text-zinc-500 dark:text-zinc-400">
            Include Rules
          </p>
          <p className="mt-2 text-sm text-zinc-700 dark:text-zinc-300">
            Questions below {settings.questionThresholdPct.toFixed(0)}% class
            correct, questions missed by students below{" "}
            {settings.studentThresholdPct.toFixed(0)}%, and top{" "}
            {settings.topNHardest} hardest.
          </p>
        </div>
        <div className="rounded-lg bg-zinc-50 p-3 dark:bg-zinc-900">
          <p className="text-xs uppercase tracking-wide text-zinc-500 dark:text-zinc-400">
            Student Group
          </p>
          <p className="mt-2 text-sm text-zinc-700 dark:text-zinc-300">
            {flaggedNames.length > 0 ? flaggedNames.join(", ") : "No flagged students"}
          </p>
        </div>
      </div>

      <div className="grid gap-3 md:grid-cols-3">
        <div className="rounded-lg border border-zinc-200 p-3 dark:border-zinc-800">
          <p className="text-xs text-zinc-500 dark:text-zinc-400">Questions selected</p>
          <p className="text-xl font-semibold">{remediation.questionCount}</p>
        </div>
        <div className="rounded-lg border border-zinc-200 p-3 dark:border-zinc-800">
          <p className="text-xs text-zinc-500 dark:text-zinc-400">Estimated time</p>
          <p className="text-xl font-semibold">
            {remediation.estimatedTimeMinutes} min
          </p>
        </div>
        <div className="rounded-lg border border-zinc-200 p-3 dark:border-zinc-800">
          <p className="text-xs text-zinc-500 dark:text-zinc-400">Skill coverage</p>
          <p className="text-sm font-medium">
            {remediation.skillCoverage.length > 0
              ? remediation.skillCoverage.join(", ")
              : "No concept tags"}
          </p>
        </div>
      </div>
    </div>
  );
}
