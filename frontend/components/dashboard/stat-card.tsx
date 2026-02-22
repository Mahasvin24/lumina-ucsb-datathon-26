import { Card, CardContent, CardDescription } from "@/components/ui/card";
import { cn } from "@/lib/utils";

type StatCardProps = {
  label: string;
  value: string;
  valueClassName?: string;
};

export function StatCard({ label, value, valueClassName }: StatCardProps) {
  return (
    <Card className="bg-zinc-50">
      <CardContent className="p-4">
        <CardDescription className="text-xs uppercase tracking-wide">{label}</CardDescription>
        <p className={cn("mt-2 text-2xl font-semibold", valueClassName || "text-zinc-900")}>
          {value}
        </p>
      </CardContent>
    </Card>
  );
}
