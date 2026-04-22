import Link from "next/link";
import { notFound } from "next/navigation";
import { api, type RunMetricDelta, type RunSummary } from "@/lib/api";
import { compactUnit, formatNumber, formatRelative } from "@/lib/format";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { CopyLinkButton } from "@/components/copy-link-button";

export const dynamic = "force-dynamic";

export default async function ComparePage({
  params,
}: {
  params: Promise<{ a: string; b: string }>;
}) {
  const { a, b } = await params;
  const aId = Number(a);
  const bId = Number(b);
  if (!Number.isFinite(aId) || !Number.isFinite(bId)) notFound();

  const result = await api.compare(aId, bId).catch(() => null);
  if (!result) notFound();

  const regressions = result.metric_deltas.filter((d) => d.is_regression === true);
  const improvements = result.metric_deltas.filter(
    (d) => d.is_regression === false && d.delta_percent !== null,
  );
  const rest = result.metric_deltas.filter(
    (d) => d.is_regression === null || (d.delta_percent === null),
  );

  return (
    <div className="space-y-8 animate-fade-in-up">
      <section className="flex flex-wrap items-start justify-between gap-3 border-b border-border/60 pb-6">
        <div className="space-y-1">
        <p className="text-[11px] uppercase tracking-wider text-muted-foreground">
          Compare runs
        </p>
        <h1 className="mt-1 text-xl font-semibold tracking-tight">
          {regressions.length > 0 && (
            <span className="text-destructive tabular-numbers">
              {regressions.length} regression{regressions.length === 1 ? "" : "s"}
            </span>
          )}
          {regressions.length > 0 && improvements.length > 0 && (
            <span className="mx-2 text-muted-foreground/60">·</span>
          )}
          {improvements.length > 0 && (
            <span className="text-success tabular-numbers">
              {improvements.length} improvement{improvements.length === 1 ? "" : "s"}
            </span>
          )}
          {regressions.length === 0 && improvements.length === 0 && (
            <span className="text-muted-foreground">No meaningful change</span>
          )}
        </h1>
        </div>
        <CopyLinkButton />
      </section>

      <section className="grid gap-4 md:grid-cols-2">
        <RunCard label="A (baseline)" run={result.a} />
        <RunCard label="B" run={result.b} />
      </section>

      {regressions.length > 0 && (
        <section className="space-y-3">
          <SectionHeader title="Regressions" count={regressions.length} tint="destructive" />
          <DeltaTable deltas={regressions} />
        </section>
      )}

      {improvements.length > 0 && (
        <section className="space-y-3">
          <SectionHeader title="Improvements" count={improvements.length} tint="success" />
          <DeltaTable deltas={improvements} />
        </section>
      )}

      {rest.length > 0 && (
        <section className="space-y-3">
          <SectionHeader title="All other metrics" count={rest.length} />
          <DeltaTable deltas={rest} />
        </section>
      )}
    </div>
  );
}

function SectionHeader({
  title,
  count,
  tint,
}: {
  title: string;
  count: number;
  tint?: "destructive" | "success";
}) {
  const color =
    tint === "destructive"
      ? "text-destructive"
      : tint === "success"
        ? "text-success"
        : "text-muted-foreground";
  return (
    <div className="flex items-baseline gap-2">
      <h2 className={`text-[11px] font-semibold uppercase tracking-[0.08em] ${color}`}>
        {title}
      </h2>
      <span className="text-[11px] text-muted-foreground/70">· {count}</span>
    </div>
  );
}

function RunCard({ label, run }: { label: string; run: RunSummary }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>{label}</span>
          <Badge variant="outline" className="font-mono">{run.trigger}</Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-1 text-[13px]">
        <p className="font-mono text-foreground/80">{run.config_name}</p>
        <p className="text-[12px] text-muted-foreground">
          conc{" "}
          <span className="tabular-numbers text-foreground/80">
            {run.concurrency.toLocaleString()}
          </span>
          <span className="mx-1.5">·</span>
          {formatRelative(run.started_at)}
        </p>
        {run.commit_short_sha && (
          <p className="text-[12px]">
            <Link
              href={`/commits/${run.commit_short_sha}`}
              className="font-mono text-primary hover:underline"
            >
              {run.commit_short_sha}
            </Link>
            {run.commit_author && (
              <span className="ml-1.5 text-muted-foreground">· {run.commit_author}</span>
            )}
          </p>
        )}
        <Link
          href={`/runs/${run.id}`}
          className="inline-block pt-2 text-[12px] text-muted-foreground hover:text-foreground"
        >
          → run detail
        </Link>
      </CardContent>
    </Card>
  );
}

function DeltaTable({ deltas }: { deltas: RunMetricDelta[] }) {
  return (
    <div className="overflow-hidden rounded-xl border border-border/60">
      <table className="w-full text-[13px]">
        <thead className="bg-muted/30 text-left text-[11px] font-medium uppercase tracking-[0.05em] text-muted-foreground">
          <tr>
            <th className="px-4 py-2.5">Metric</th>
            <th className="px-4 py-2.5 text-right">A</th>
            <th className="px-4 py-2.5 text-right">B</th>
            <th className="px-4 py-2.5 text-right">Δ</th>
          </tr>
        </thead>
        <tbody>
          {deltas.map((d) => (
            <tr key={d.name} className="border-t border-border/60">
              <td className="px-4 py-2.5 font-mono text-[12px] text-foreground/80">
                {d.name}
                {d.unit && (
                  <span className="ml-1.5 text-muted-foreground">{compactUnit(d.unit)}</span>
                )}
              </td>
              <td className="px-4 py-2.5 text-right font-mono tabular-numbers">
                {d.a_value !== null ? formatNumber(d.a_value) : "—"}
              </td>
              <td className="px-4 py-2.5 text-right font-mono tabular-numbers">
                {d.b_value !== null ? formatNumber(d.b_value) : "—"}
              </td>
              <td className="px-4 py-2.5 text-right font-mono tabular-numbers">
                {d.delta_percent !== null ? (
                  <span
                    className={
                      d.is_regression
                        ? "text-destructive"
                        : d.is_regression === false
                          ? "text-success"
                          : "text-muted-foreground"
                    }
                  >
                    {d.delta_percent > 0 ? "+" : ""}
                    {d.delta_percent.toFixed(2)}%
                  </span>
                ) : (
                  <span className="text-muted-foreground">—</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
