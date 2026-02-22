import { Card, CardContent, CardDescription } from "@/components/ui/card";
import { cn } from "@/lib/utils";

type StatCardProps = {
  label: string;
  value: string;
  valueClassName?: string;
  compact?: boolean;
};

export function StatCard({ label, value, valueClassName, compact }: StatCardProps) {
  return (
    <Card className="bg-zinc-50">
      <CardContent className="p-4">
        <CardDescription className="text-xs uppercase tracking-wide">{label}</CardDescription>
        <p className={cn(
          "mt-2 font-semibold",
          compact ? "text-lg" : "text-2xl",
          valueClassName || "text-zinc-900",
        )}>
          {value}
        </p>
      </CardContent>
    </Card>
  );
}
