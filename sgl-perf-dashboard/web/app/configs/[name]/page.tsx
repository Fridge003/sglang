import { notFound } from "next/navigation";
import { api } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { CopyLinkButton } from "@/components/copy-link-button";
import TrendChart from "./trend-chart";

export const dynamic = "force-dynamic";

const DEFAULT_METRIC = "total_token_throughput";

export default async function ConfigDetailPage({
  params,
  searchParams,
}: {
  params: Promise<{ name: string }>;
  searchParams: Promise<{ metric?: string; concurrency?: string; window_days?: string }>;
}) {
  const { name } = await params;
  const sp = await searchParams;

  const configs = await api.listConfigs().catch(() => []);
  const config = configs.find((c) => c.config_name === name);
  if (!config) notFound();

  const metric = sp.metric ?? DEFAULT_METRIC;
  const concurrency = Number(sp.concurrency ?? config.concurrency_levels[0] ?? 1);
  const windowDays = Number(sp.window_days ?? 30);

  const trend = await api
    .configTrend(name, { metric, concurrency, window_days: windowDays })
    .catch(() => []);

  return (
    <div className="space-y-8 animate-fade-in-up">
      <section className="flex flex-wrap items-start justify-between gap-3 border-b border-border/60 pb-6">
        <div className="space-y-1">
          <h1 className="font-mono text-lg font-semibold leading-tight">{name}</h1>
          <p className="text-[13px] text-muted-foreground">
            {config.concurrency_levels.length} concurrency level
            {config.concurrency_levels.length === 1 ? "" : "s"} · last run{" "}
            {config.latest_started_at
              ? new Date(config.latest_started_at).toLocaleString()
              : "—"}
          </p>
        </div>
        <CopyLinkButton />
      </section>

      <section className="flex flex-wrap items-center gap-2">
        <span className="mr-1 text-[11px] uppercase tracking-wider text-muted-foreground">
          Concurrency
        </span>
        {config.concurrency_levels.map((c) => {
          const active = c === concurrency;
          const qs = new URLSearchParams({
            metric,
            concurrency: String(c),
            window_days: String(windowDays),
          });
          return (
            <a
              key={c}
              href={`?${qs.toString()}`}
              className={`rounded-md border px-2 py-0.5 font-mono text-[12px] transition ${
                active
                  ? "border-primary/60 bg-primary/10 text-primary"
                  : "border-border/60 text-muted-foreground hover:border-border hover:text-foreground"
              }`}
            >
              {c.toLocaleString()}
            </a>
          );
        })}
        <span className="ml-4 text-[11px] uppercase tracking-wider text-muted-foreground">
          Metric
        </span>
        <Badge variant="secondary" className="font-mono">{metric}</Badge>
        <span className="ml-auto text-[11px] text-muted-foreground">
          window: {windowDays}d
        </span>
      </section>

      <Card>
        <CardHeader>
          <CardTitle className="font-mono">
            {metric}{" "}
            <span className="font-sans text-muted-foreground">
              · concurrency {concurrency.toLocaleString()}
            </span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {trend.length === 0 ? (
            <p className="py-16 text-center text-[13px] text-muted-foreground">
              No data in the last {windowDays} days for this metric + concurrency.
            </p>
          ) : (
            <TrendChart data={trend} />
          )}
        </CardContent>
      </Card>
    </div>
  );
}
