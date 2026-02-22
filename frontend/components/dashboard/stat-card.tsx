import { Card, CardContent, CardDescription } from "@/components/ui/card";

type StatCardProps = {
  label: string;
  value: string;
};

export function StatCard({ label, value }: StatCardProps) {
  return (
    <Card className="bg-zinc-50">
      <CardContent className="p-4">
        <CardDescription className="text-xs uppercase tracking-wide">{label}</CardDescription>
        <p className="mt-2 text-2xl font-semibold text-zinc-900">{value}</p>
      </CardContent>
    </Card>
  );
}
