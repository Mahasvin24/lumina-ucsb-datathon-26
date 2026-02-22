import { StudentRow } from "@/lib/dashboard-api";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

type StudentsTableProps = {
  students: StudentRow[];
};

export function StudentsTable({ students }: StudentsTableProps) {
  return (
    <Table className="min-w-full text-left">
      <TableHeader>
        <TableRow>
          <TableHead>Student</TableHead>
          <TableHead>Score</TableHead>
          <TableHead>Correct / Total</TableHead>
          <TableHead>Completion</TableHead>
          <TableHead>Status</TableHead>
          <TableHead>Most Missed Concept</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
          {students.map((student) => (
            <TableRow key={student.studentId}>
              <TableCell>{student.name}</TableCell>
              <TableCell>{student.scorePct.toFixed(2)}%</TableCell>
              <TableCell>
                {student.correctCount} / {student.totalQuestions}
              </TableCell>
              <TableCell>{student.completionPct.toFixed(2)}%</TableCell>
              <TableCell>{student.status}</TableCell>
              <TableCell>{student.mostMissedConcept}</TableCell>
            </TableRow>
          ))}
      </TableBody>
    </Table>
  );
}
