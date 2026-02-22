import { MatrixRow } from "@/lib/dashboard-api";

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
    const hue = Math.round(p * 120); // 0=red, 120=green
    const style = {
      backgroundColor: `hsl(${hue} 92% 72%)`,
      color: `hsl(${hue} 72% 16%)`,
    };
    return (
      <span className="inline-block rounded px-2 py-1 tabular-nums" style={style}>
        {(p * 100).toFixed(0)}%
      </span>
    );
  }
  if (state === "correct") {
    return (
      <span className="inline-block rounded bg-emerald-300 px-2 py-1 text-emerald-950 dark:bg-emerald-500/65 dark:text-emerald-50">
        C
      </span>
    );
  }
  if (state === "wrong") {
    return (
      <span className="inline-block rounded bg-rose-300 px-2 py-1 text-rose-950 dark:bg-rose-500/65 dark:text-rose-50">
        W
      </span>
    );
  }
  return (
    <span className="inline-block rounded bg-zinc-300 px-2 py-1 text-zinc-900 dark:bg-zinc-600 dark:text-zinc-50">
      -
    </span>
  );
}

export function MatrixView({ questionIds, rows }: MatrixViewProps) {
  return (
    <div className="overflow-x-auto">
      <table className="min-w-full text-center text-xs">
        <thead>
          <tr className="border-b border-zinc-200 text-zinc-500 dark:border-zinc-800 dark:text-zinc-400">
            <th className="px-2 py-2 text-left">Student</th>
            {questionIds.map((qid) => (
              <th key={qid} className="px-2 py-2 font-medium">
                Q{qid}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr
              key={row.studentId}
              className="border-b border-zinc-100 text-zinc-800 dark:border-zinc-900 dark:text-zinc-200"
            >
              <td className="px-2 py-2 text-left text-sm">{row.name}</td>
              {row.cells.map((cell) => (
                <td key={`${row.studentId}-${cell.questionId}`} className="px-2 py-2">
                  {renderCell(cell.state, cell.probability)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
