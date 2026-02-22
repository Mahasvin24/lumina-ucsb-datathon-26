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
          <TableHead className="w-[65%]">Most Missed Concept</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
          {students.map((student) => (
            <TableRow key={student.studentId}>
              <TableCell>{student.name}</TableCell>
              <TableCell>{student.scorePct.toFixed(2)}%</TableCell>
              <TableCell className="max-w-0 whitespace-normal wrap-break-word">
                {student.mostMissedConcept || "None"}
              </TableCell>
            </TableRow>
          ))}
      </TableBody>
    </Table>
  );
}
