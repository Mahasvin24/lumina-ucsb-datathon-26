import Link from "next/link";

import { MatrixRow } from "@/lib/dashboard-api";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

type MatrixViewProps = {
  questionIds: number[];
  rows: MatrixRow[];
};

function clampProbability(value: number): number {
  if (Number.isNaN(value)) {
    return 0;
  }
  if (value < 0) {
    return 0;
  }
  if (value > 1) {
    return 1;
  }
  return value;
}

function renderCell(state: "correct" | "wrong" | "unanswered", probability?: number) {
  if (probability !== undefined) {
    const p = clampProbability(probability);
    const curved = Math.pow(p, 2.2);
    const hue = Math.round(curved * 120);
    const saturation = 50 + Math.round(Math.abs(p - 0.5) * 50);
    const lightness = 48 - Math.round(Math.abs(p - 0.5) * 16);
    const style = {
      backgroundColor: `hsl(${hue} 92% 72%)`,
      color: `hsl(${hue} 12% 16%)`,
    };
    return (
      <Badge className="inline-block border-transparent px-2 py-1 tabular-nums" style={style}>
        {(p * 100).toFixed(0)}%
      </Badge>
    );
  }
  if (state === "correct") {
    return <Badge variant="success">C</Badge>;
  }
  if (state === "wrong") {
    return <Badge variant="destructive">W</Badge>;
  }
  return <Badge variant="secondary">-</Badge>;
}

function legendColor(pct: number): React.CSSProperties {
  const t = pct / 100;
  const curved = Math.pow(t, 2.2);
  const hue = Math.round(curved * 120);
  return {
    backgroundColor: `hsl(${hue} 92% 72%)`,
    color: `hsl(${hue} 12% 16%)`,
  };
}

export function MatrixView({ questionIds, rows }: MatrixViewProps) {
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2 text-xs text-zinc-500">
        <Badge className="border-transparent" style={legendColor(0)}>
          &lt; 40% Needs Work
        </Badge>
        <Badge className="border-transparent" style={legendColor(50)}>
          40–59% Developing
        </Badge>
        <Badge className="border-transparent" style={legendColor(70)}>
          60–79% Proficient
        </Badge>
        <Badge className="border-transparent" style={legendColor(100)}>
          ≥ 80% Mastered
        </Badge>
      </div>

      <Table className="min-w-full text-center text-xs">
        <TableHeader>
          <TableRow>
            <TableHead className="text-left">Student</TableHead>
              {questionIds.map((qid) => (
                <TableHead key={qid} className="text-center">
                  Q{qid}
                </TableHead>
              ))}
          </TableRow>
        </TableHeader>
        <TableBody>
            {rows.map((row) => (
              <TableRow key={row.studentId}>
                <TableCell className="text-left text-sm">
                  <Link
                    href={`/dashboard/student/${row.studentId}`}
                    className="text-sky-600 underline decoration-sky-300 hover:text-sky-800"
                  >
                    {row.name}
                  </Link>
                </TableCell>
                {row.cells.map((cell) => (
                  <TableCell key={`${row.studentId}-${cell.questionId}`} className="text-center">
                    {renderCell(cell.state, cell.probability)}
                  </TableCell>
                ))}
              </TableRow>
            ))}
        </TableBody>
      </Table>
    </div>
  );
}
