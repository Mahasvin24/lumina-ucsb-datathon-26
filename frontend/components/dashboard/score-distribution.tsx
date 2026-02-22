"use client";

import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type ScoreDistributionProps = {
  scores: number[];
};

function gaussian(x: number, mean: number, sigma: number): number {
  const exp = -0.5 * ((x - mean) / sigma) ** 2;
  return Math.exp(exp) / (sigma * Math.sqrt(2 * Math.PI));
}

function kde(scores: number[], points: number[], bandwidth: number) {
  return points.map((x) => {
    const density =
      scores.reduce((sum, s) => sum + gaussian(x, s, bandwidth), 0) / scores.length;
    return { score: x, density: Number((density * 100).toFixed(4)) };
  });
}

export function ScoreDistribution({ scores }: ScoreDistributionProps) {
  if (scores.length === 0) {
    return (
      <p className="py-4 text-center text-sm text-zinc-400">
        Process questions to view score distribution.
      </p>
    );
  }

  const points = Array.from({ length: 101 }, (_, i) => i);
  const bandwidth = Math.max(3, Math.min(8, 100 / Math.sqrt(scores.length)));
  const data = kde(scores, points, bandwidth);

  return (
    <ResponsiveContainer width="100%" height={260}>
      <AreaChart data={data} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
        <defs>
          <linearGradient id="scoreFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#93c5fd" stopOpacity={0.5} />
            <stop offset="100%" stopColor="#93c5fd" stopOpacity={0.08} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e4e4e7" />
        <XAxis
          dataKey="score"
          type="number"
          domain={[0, 100]}
          tickFormatter={(v: number) => `${v}%`}
          tick={{ fontSize: 11, fill: "#71717a" }}
          tickLine={false}
          axisLine={{ stroke: "#d4d4d8" }}
        />
        <YAxis hide />
        <Tooltip
          formatter={(value: number) => [value.toFixed(2), "Density"]}
          labelFormatter={(label: number) => `Score: ${label}%`}
          contentStyle={{
            fontSize: 12,
            borderRadius: 6,
            border: "1px solid #e4e4e7",
            boxShadow: "0 1px 3px rgba(0,0,0,0.08)",
          }}
        />
        <Area
          type="monotone"
          dataKey="density"
          stroke="#60a5fa"
          strokeWidth={2}
          fill="url(#scoreFill)"
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
