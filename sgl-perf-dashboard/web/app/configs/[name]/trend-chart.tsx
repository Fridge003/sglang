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
        <LineChart data={rows} margin={{ top: 16, right: 24, bottom: 8, left: 0 }}>
          <CartesianGrid
            stroke="hsl(var(--border))"
            strokeDasharray="2 4"
            vertical={false}
          />
          <XAxis
            dataKey="tsMs"
            type="number"
            domain={["dataMin", "dataMax"]}
            tickFormatter={(ms) =>
              new Date(ms).toLocaleDateString("en-US", { month: "short", day: "numeric" })
            }
            tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
            stroke="hsl(var(--border))"
            tickLine={false}
            axisLine={false}
          />
          <YAxis
            tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
            stroke="hsl(var(--border))"
            width={80}
            tickLine={false}
            axisLine={false}
            tickFormatter={(v) => v.toLocaleString()}
          />
          <Tooltip
            content={<CustomTooltip />}
            cursor={{ stroke: "hsl(var(--primary))", strokeWidth: 1, strokeOpacity: 0.3 }}
          />
          <Line
            type="monotone"
            dataKey="value"
            stroke="hsl(var(--primary))"
            strokeWidth={2}
            dot={{ r: 3, fill: "hsl(var(--primary))", strokeWidth: 0 }}
            activeDot={{ r: 5, strokeWidth: 2, stroke: "hsl(var(--background))" }}
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
    <div className="min-w-[160px] rounded-lg border border-border/80 bg-card/95 p-2.5 text-[12px] shadow-lg backdrop-blur">
      <p className="font-mono text-[15px] font-semibold tabular-numbers">
        {row.value.toLocaleString()}
      </p>
      <p className="mt-1 text-[11px] text-muted-foreground">
        {new Date(row.tsMs).toLocaleString()}
      </p>
      {row.commit_short_sha && (
        <p className="mt-1 font-mono text-[11px] text-muted-foreground">
          {row.commit_short_sha}
          {row.commit_author && (
            <span className="ml-1 text-muted-foreground/70">· {row.commit_author}</span>
          )}
        </p>
      )}
      <Link
        href={`/runs/${row.run_id}`}
        className="mt-2 inline-block text-[11px] text-primary transition hover:underline"
      >
        → run detail
      </Link>
    </div>
  );
}
