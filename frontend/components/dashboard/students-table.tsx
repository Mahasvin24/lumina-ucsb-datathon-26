import Link from "next/link";

import { StudentRow } from "@/lib/dashboard-api";
import { Badge } from "@/components/ui/badge";
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
  thresholdPct: number;
  hasProcessed: boolean;
};

export function StudentsTable({ students, thresholdPct, hasProcessed }: StudentsTableProps) {
  return (
    <Table className="min-w-full text-left">
      <TableHeader>
        <TableRow>
          <TableHead>Student</TableHead>
          <TableHead>Score</TableHead>
          <TableHead className="w-[65%]">Areas to Study Next</TableHead>
        </TableRow>
      </TableHeader> 
      <TableBody>
          {students.map((student) => (
            <TableRow
              key={student.studentId}
              className={hasProcessed && student.scorePct < thresholdPct ? "bg-red-50 hover:bg-red-100" : ""}
            >
              <TableCell>
                <Link
                  href={`/dashboard/student/${student.studentId}`}
                  className="text-sky-600 underline decoration-sky-300 hover:text-sky-800"
                >
                  {student.name}
                </Link>
              </TableCell>
              <TableCell>{student.scorePct.toFixed(2)}%</TableCell>
              <TableCell>
                {student.areasToStudy.length > 0 ? (
                  <div className="flex flex-wrap gap-1">
                    {student.areasToStudy.map((tag) => (
                      <Badge key={tag} variant="secondary" className="text-xs font-normal">
                        {tag}
                      </Badge>
                    ))}
                  </div>
                ) : (
                  <span className="text-zinc-400">—</span>
                )}
              </TableCell>
            </TableRow>
          ))}
      </TableBody>
    </Table>
  );
}
