import { ConceptAccuracy } from "@/lib/dashboard-api";
import { Badge } from "@/components/ui/badge";

type RemediationPanelProps = {
  conceptAccuracies: ConceptAccuracy[];
};

function accuracyColor(pct: number): { bg: string; text: string; border: string } {
  if (pct < 40) return { bg: "bg-rose-50", text: "text-rose-700", border: "border-rose-300" };
  if (pct < 60) return { bg: "bg-amber-50", text: "text-amber-700", border: "border-amber-300" };
  if (pct < 80) return { bg: "bg-sky-50", text: "text-sky-700", border: "border-sky-300" };
  return { bg: "bg-emerald-50", text: "text-emerald-700", border: "border-emerald-300" };
}

export function RemediationPanel({ conceptAccuracies }: RemediationPanelProps) {
  if (conceptAccuracies.length === 0) {
    return (
      <p className="py-4 text-center text-sm text-zinc-400">
        Process questions to view skill tag performance.
      </p>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2 text-xs text-zinc-500">
        <span className="font-medium">Legend:</span>
        <Badge className="border-rose-300 bg-rose-50 text-rose-700">&lt; 40%</Badge>
        <Badge className="border-amber-300 bg-amber-50 text-amber-700">40–59%</Badge>
        <Badge className="border-sky-300 bg-sky-50 text-sky-700">60–79%</Badge>
        <Badge className="border-emerald-300 bg-emerald-50 text-emerald-700">≥ 80%</Badge>
      </div>
      <div className="flex flex-wrap gap-2">
        {conceptAccuracies.map((concept) => {
          const c = accuracyColor(concept.correctPct);
          return (
            <Badge
              key={concept.tag}
              className={`${c.border} ${c.bg} ${c.text} text-xs font-normal`}
            >
              {concept.tag} — {concept.correctPct.toFixed(1)}%
            </Badge>
          );
        })}
      </div>
    </div>
  );
}
