import { MatrixRow } from "@/lib/dashboard-api";

type MatrixViewProps = {
  questionIds: number[];
  rows: MatrixRow[];
};

function renderCell(state: "correct" | "wrong" | "unanswered") {
  if (state === "correct") {
    return (
      <span className="inline-block rounded bg-emerald-100 px-2 py-1 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300">
        C
      </span>
    );
  }
  if (state === "wrong") {
    return (
      <span className="inline-block rounded bg-rose-100 px-2 py-1 text-rose-800 dark:bg-rose-900/40 dark:text-rose-300">
        W
      </span>
    );
  }
  return (
    <span className="inline-block rounded bg-zinc-100 px-2 py-1 text-zinc-600 dark:bg-zinc-800 dark:text-zinc-300">
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
                  {renderCell(cell.state)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
