import { DashboardPayload } from "@/lib/dashboard-api";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription } from "@/components/ui/card";

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
        <Card className="bg-zinc-50 shadow-none">
          <CardContent className="p-3">
            <CardDescription className="text-xs uppercase tracking-wide">
            Include Rules
            </CardDescription>
            <p className="mt-2 text-sm text-zinc-700">
            Questions below {settings.questionThresholdPct.toFixed(0)}% class
            correct, questions missed by students below{" "}
            {settings.studentThresholdPct.toFixed(0)}%, and top{" "}
            {settings.topNHardest} hardest.
            </p>
          </CardContent>
        </Card>
        <Card className="bg-zinc-50 shadow-none">
          <CardContent className="p-3">
            <CardDescription className="text-xs uppercase tracking-wide">
            Student Group
            </CardDescription>
            <p className="mt-2 text-sm text-zinc-700">
            {flaggedNames.length > 0 ? flaggedNames.join(", ") : "No flagged students"}
            </p>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-3 md:grid-cols-3">
        <Card className="shadow-none">
          <CardContent className="p-3">
            <CardDescription className="text-xs">Questions selected</CardDescription>
            <p className="text-xl font-semibold">{remediation.questionCount}</p>
          </CardContent>
        </Card>
        <Card className="shadow-none">
          <CardContent className="p-3">
            <CardDescription className="text-xs">Estimated time</CardDescription>
            <p className="text-xl font-semibold">
            {remediation.estimatedTimeMinutes} min
            </p>
          </CardContent>
        </Card>
        <Card className="shadow-none">
          <CardContent className="p-3">
            <CardDescription className="text-xs">Skill coverage</CardDescription>
            {remediation.skillCoverage.length > 0 ? (
              <div className="mt-1 flex flex-wrap gap-1">
                {remediation.skillCoverage.map((skill) => (
                  <Badge key={skill} variant="outline" className="text-xs font-normal">
                    {skill}
                  </Badge>
                ))}
              </div>
            ) : (
              <p className="text-sm font-medium">No concept tags</p>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
