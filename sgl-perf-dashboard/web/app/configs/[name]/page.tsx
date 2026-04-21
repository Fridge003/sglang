import { notFound } from "next/navigation";
import { api } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
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
    <div className="space-y-6">
      <section>
        <h1 className="font-mono text-xl font-semibold">{name}</h1>
        <p className="text-sm text-muted-foreground">
          {config.concurrency_levels.length} concurrency level
          {config.concurrency_levels.length === 1 ? "" : "s"}
        </p>
      </section>

      <section className="flex flex-wrap items-center gap-3">
        <span className="text-xs uppercase tracking-wider text-muted-foreground">
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
              className={`rounded-md border px-2.5 py-0.5 font-mono text-xs transition ${
                active
                  ? "border-primary bg-primary text-primary-foreground"
                  : "border-border hover:border-foreground/40"
              }`}
            >
              {c}
            </a>
          );
        })}
        <span className="ml-4 text-xs text-muted-foreground">metric:</span>
        <Badge variant="secondary" className="font-mono">{metric}</Badge>
        <span className="text-xs text-muted-foreground">window: {windowDays}d</span>
      </section>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm">
            {metric} · concurrency {concurrency}
          </CardTitle>
        </CardHeader>
        <CardContent>
          {trend.length === 0 ? (
            <p className="py-10 text-center text-sm text-muted-foreground">
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
