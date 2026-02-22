import { StudentRow } from "@/lib/dashboard-api";

type StudentsTableProps = {
  students: StudentRow[];
};

export function StudentsTable({ students }: StudentsTableProps) {
  return (
    <div className="overflow-x-auto">
      <table className="min-w-full text-left text-sm">
        <thead>
          <tr className="border-b border-zinc-200 text-zinc-500 dark:border-zinc-800 dark:text-zinc-400">
            <th className="px-2 py-2 font-medium">Student</th>
            <th className="px-2 py-2 font-medium">Score</th>
            <th className="px-2 py-2 font-medium">Correct / Total</th>
            <th className="px-2 py-2 font-medium">Completion</th>
            <th className="px-2 py-2 font-medium">Status</th>
            <th className="px-2 py-2 font-medium">Time Spent</th>
            <th className="px-2 py-2 font-medium">Most Missed Concept</th>
          </tr>
        </thead>
        <tbody>
          {students.map((student) => (
            <tr
              key={student.studentId}
              className="border-b border-zinc-100 text-zinc-800 dark:border-zinc-900 dark:text-zinc-200"
            >
              <td className="px-2 py-2">{student.name}</td>
              <td className="px-2 py-2">{student.scorePct.toFixed(2)}%</td>
              <td className="px-2 py-2">
                {student.correctCount} / {student.totalQuestions}
              </td>
              <td className="px-2 py-2">{student.completionPct.toFixed(2)}%</td>
              <td className="px-2 py-2">{student.status}</td>
              <td className="px-2 py-2">{student.timeSpentMinutes.toFixed(1)} min</td>
              <td className="px-2 py-2">{student.mostMissedConcept}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
