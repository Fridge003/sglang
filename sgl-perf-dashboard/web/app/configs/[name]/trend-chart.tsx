"use client";

import Link from "next/link";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  TooltipProps,
  XAxis,
  YAxis,
} from "recharts";
import type { TrendPoint } from "@/lib/api";

type ChartRow = TrendPoint & { tsMs: number };

export default function TrendChart({ data }: { data: TrendPoint[] }) {
  const rows: ChartRow[] = data.map((p) => ({
    ...p,
    tsMs: Date.parse(p.started_at),
  }));

  return (
    <div className="h-80 w-full">
      <ResponsiveContainer>
        <LineChart data={rows} margin={{ top: 10, right: 24, bottom: 8, left: 0 }}>
          <CartesianGrid stroke="hsl(var(--border))" strokeDasharray="3 3" />
          <XAxis
            dataKey="tsMs"
            type="number"
            domain={["dataMin", "dataMax"]}
            tickFormatter={(ms) => new Date(ms).toLocaleDateString()}
            tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
            stroke="hsl(var(--border))"
          />
          <YAxis
            tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
            stroke="hsl(var(--border))"
            width={80}
          />
          <Tooltip content={<CustomTooltip />} />
          <Line
            type="monotone"
            dataKey="value"
            stroke="hsl(var(--primary))"
            strokeWidth={2}
            dot={{ r: 3, fill: "hsl(var(--primary))" }}
            activeDot={{ r: 5 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function CustomTooltip(props: TooltipProps<number, string>) {
  const { active, payload } = props;
  if (!active || !payload?.length) return null;
  const row = payload[0].payload as ChartRow;
  return (
    <div className="rounded-md border bg-background p-2 text-xs shadow-md">
      <p className="font-semibold tabular-nums">{row.value.toLocaleString()}</p>
      <p className="mt-1 text-muted-foreground">
        {new Date(row.tsMs).toLocaleString()}
      </p>
      {row.commit_short_sha && (
        <p className="mt-1 font-mono text-muted-foreground">
          {row.commit_short_sha}
          {row.commit_author ? ` · ${row.commit_author}` : ""}
        </p>
      )}
      <Link
        href={`/runs/${row.run_id}`}
        className="mt-1 inline-block text-primary hover:underline"
      >
        → run detail
      </Link>
    </div>
  );
}
