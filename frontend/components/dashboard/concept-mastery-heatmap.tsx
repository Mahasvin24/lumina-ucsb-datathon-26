"use client";

import { useMemo, useState } from "react";
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

type ConceptMasteryHeatmapProps = {
  matrixRows: MatrixRow[];
  skillTagsByQuestion: Map<number, string[]>;
};

type StudentConceptMastery = {
  studentId: number;
  name: string;
  conceptScores: Map<string, { correct: number; total: number }>;
};

function masteryColor(pct: number): React.CSSProperties {
  const t = pct / 100;
  const hue = Math.round(t * 120);
  return {
    backgroundColor: `hsl(${hue} 85% 78%)`,
    color: `hsl(${hue} 40% 20%)`,
  };
}

function masteryLabel(pct: number): string {
  if (pct < 40) return "Needs Work";
  if (pct < 60) return "Developing";
  if (pct < 80) return "Proficient";
  return "Mastered";
}

export function ConceptMasteryHeatmap({
  matrixRows,
  skillTagsByQuestion,
}: ConceptMasteryHeatmapProps) {
  const [sortBy, setSortBy] = useState<"concept" | "weakness">("weakness");

  const { studentMastery, allConcepts, classConceptAvg } = useMemo(() => {
    const students: StudentConceptMastery[] = [];

    const classConcept: Map<string, { correct: number; total: number }> = new Map();

    for (const row of matrixRows) {
      const conceptScores: Map<string, { correct: number; total: number }> = new Map();

      for (const cell of row.cells) {
        if (cell.state === "unanswered" && cell.probability === undefined) continue;

        const tags = skillTagsByQuestion.get(cell.questionId) ?? [];
        for (const tag of tags) {
          const existing = conceptScores.get(tag) ?? { correct: 0, total: 0 };
          existing.total += 1;

          if (cell.probability !== undefined) {
            existing.correct += cell.probability;
          } else if (cell.state === "correct") {
            existing.correct += 1;
          }
          conceptScores.set(tag, existing);

          const classEntry = classConcept.get(tag) ?? { correct: 0, total: 0 };
          classEntry.total += 1;
          if (cell.probability !== undefined) {
            classEntry.correct += cell.probability;
          } else if (cell.state === "correct") {
            classEntry.correct += 1;
          }
          classConcept.set(tag, classEntry);
        }
      }

      students.push({
        studentId: row.studentId,
        name: row.name,
        conceptScores,
      });
    }

    const classAvg = new Map<string, number>();
    for (const [tag, stats] of classConcept) {
      classAvg.set(tag, stats.total > 0 ? (stats.correct / stats.total) * 100 : 0);
    }

    const concepts = Array.from(classConcept.keys());

    return {
      studentMastery: students,
      allConcepts: concepts,
      classConceptAvg: classAvg,
    };
  }, [matrixRows, skillTagsByQuestion]);

  const sortedConcepts = useMemo(() => {
    const sorted = [...allConcepts];
    if (sortBy === "weakness") {
      sorted.sort((a, b) => (classConceptAvg.get(a) ?? 0) - (classConceptAvg.get(b) ?? 0));
    } else {
      sorted.sort((a, b) => a.localeCompare(b));
    }
    return sorted;
  }, [allConcepts, classConceptAvg, sortBy]);

  if (allConcepts.length === 0) {
    return (
      <p className="py-4 text-center text-sm text-zinc-400">
        No concept mastery data available. Process questions first to generate the heatmap.
      </p>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex flex-wrap items-center gap-2 text-xs text-zinc-500">
          <Badge className="border-transparent" style={masteryColor(0)}>
            &lt; 40% Needs Work
          </Badge>
          <Badge className="border-transparent" style={masteryColor(50)}>
            40–59% Developing
          </Badge>
          <Badge className="border-transparent" style={masteryColor(70)}>
            60–79% Proficient
          </Badge>
          <Badge className="border-transparent" style={masteryColor(100)}>
            ≥ 80% Mastered
          </Badge>
        </div>
        <button
          onClick={() => setSortBy(sortBy === "weakness" ? "concept" : "weakness")}
          className="rounded-md border border-zinc-200 bg-white px-3 py-1 text-xs text-zinc-600 hover:bg-zinc-50"
        >
          Sort: {sortBy === "weakness" ? "Weakest First" : "A-Z"}
        </button>
      </div>

      <div className="overflow-x-auto">
        <Table className="min-w-full text-center text-xs">
          <TableHeader>
            <TableRow>
              <TableHead className="sticky left-0 z-10 min-w-[120px] bg-white text-left">
                Student
              </TableHead>
              {sortedConcepts.map((tag) => (
                <TableHead
                  key={tag}
                  className="min-w-[100px] text-center"
                  title={tag}
                >
                  <span className="inline-block max-w-[90px] truncate align-middle">
                    {tag}
                  </span>
                </TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {studentMastery.map((student) => (
                <TableRow key={student.studentId}>
                  <TableCell className="sticky left-0 z-10 bg-white text-left text-sm font-medium">
                    <Link
                      href={`/dashboard/student/${student.studentId}`}
                      className="text-sky-600 underline decoration-sky-300 hover:text-sky-800"
                    >
                      {student.name}
                    </Link>
                  </TableCell>
                  {sortedConcepts.map((tag) => {
                    const stats = student.conceptScores.get(tag);
                    if (!stats || stats.total === 0) {
                      return (
                        <TableCell key={tag} className="text-center">
                          <span className="inline-block rounded bg-zinc-100 px-2 py-1 text-zinc-400">
                            —
                          </span>
                        </TableCell>
                      );
                    }
                    const pct = (stats.correct / stats.total) * 100;
                    return (
                      <TableCell key={tag} className="text-center">
                        <span
                          className="inline-block rounded-md px-2 py-1 tabular-nums font-medium"
                          style={masteryColor(pct)}
                          title={`${tag}: ${pct.toFixed(1)}% (${masteryLabel(pct)})`}
                        >
                          {pct.toFixed(0)}%
                        </span>
                      </TableCell>
                    );
                  })}
                </TableRow>
            ))}

            <TableRow className="bg-zinc-50 font-semibold">
              <TableCell className="sticky left-0 z-10 bg-zinc-50 text-left text-sm">
                Class Average
              </TableCell>
              {sortedConcepts.map((tag) => {
                const avg = classConceptAvg.get(tag) ?? 0;
                return (
                  <TableCell key={tag} className="text-center">
                    <span
                      className="inline-block rounded-md px-2 py-1 tabular-nums font-semibold"
                      style={masteryColor(avg)}
                      title={`Class avg: ${avg.toFixed(1)}%`}
                    >
                      {avg.toFixed(0)}%
                    </span>
                  </TableCell>
                );
              })}
            </TableRow>
          </TableBody>
        </Table>
      </div>
    </div>
  );
}
