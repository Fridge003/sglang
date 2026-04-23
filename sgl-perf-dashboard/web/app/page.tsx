import Link from "next/link";
import { api, type ConfigSummary, type RunSummary } from "@/lib/api";
import { formatRelative } from "@/lib/format";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { AutoRefresh } from "@/components/auto-refresh";
import { StatusTooltip } from "@/components/status-tooltip";

export const dynamic = "force-dynamic";

async function loadHomeData() {
  try {
    const [recentRuns, configs, health] = await Promise.all([
      api.listRuns({ limit: 20 }),
      api.listConfigs(),
      api.health(),
    ]);
    return { recentRuns, configs, health, error: null };
  } catch (err) {
    return {
      recentRuns: [],
      configs: [],
      health: null,
      error: err instanceof Error ? err.message : String(err),
    };
  }
}

export default async function HomePage() {
  const { recentRuns, configs, health, error } = await loadHomeData();

  return (
    <div className="space-y-10 animate-fade-in-up">
      <AutoRefresh />
      {/* Header */}
      <section className="flex flex-wrap items-end justify-between gap-4">
        <div className="space-y-1">
          <h1 className="text-2xl font-semibold tracking-tight">GB200 Nightly</h1>
          <p className="text-[13px] text-muted-foreground">
            {health ? (
              <>
                <span className="tabular-numbers font-medium text-foreground">
                  {health.runs.toLocaleString()}
                </span>{" "}
                runs{" "}
                <span className="tabular-numbers text-success">
                  {health.runs_passed.toLocaleString()} passed
                </span>
                {health.runs_failed > 0 && (
                  <>
                    <span className="mx-1 text-muted-foreground/60">·</span>
                    <span className="tabular-numbers text-destructive">
                      {health.runs_failed.toLocaleString()} failed
                    </span>
                  </>
                )}
                {health.runs_partial > 0 && (
                  <>
                    <span className="mx-1 text-muted-foreground/60">·</span>
                    <span className="tabular-numbers text-warning">
                      {health.runs_partial.toLocaleString()} partial
                    </span>
                  </>
                )}
                <span className="mx-1.5 text-muted-foreground/60">·</span>
                new data {formatRelative(health.last_ingest_at)}
                <span className="mx-1.5 text-muted-foreground/60">·</span>
                <span title={`last scheduler tick: ${health.last_scheduler_run_at ?? "never"}`}>
                  sync {formatRelative(health.last_scheduler_run_at)}
                </span>
              </>
            ) : (
              "loading…"
            )}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {health && !health.github_enrichment && (
            <Badge variant="warning">GitHub enrichment disabled</Badge>
          )}
          {health && (
            <Badge variant="outline" className="font-mono">
              status: {health.status}
            </Badge>
          )}
        </div>
      </section>

      {error && <ErrorBanner message={error} />}

      {/* Configs */}
      <section className="space-y-3">
        <SectionHeader
          title="Configs"
          hint={`${configs.length} tracked`}
        />
        {configs.length === 0 ? (
          <EmptyState
            title="No configs yet"
            hint="Ingester hasn't seen any result JSONs — kick one off on the GB200 nightly workflow."
          />
        ) : (
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {configs.map((c) => (
              <ConfigCard key={c.config_name} config={c} />
            ))}
          </div>
        )}
      </section>

      {/* Recent runs */}
      <section className="space-y-3">
        <SectionHeader
          title="Recent runs"
          hint={recentRuns.length > 0 ? `${recentRuns.length} shown` : undefined}
          action={
            recentRuns.length > 0 ? (
              <Link
                href="/runs"
                className="text-[12px] text-muted-foreground transition hover:text-foreground"
              >
                view all →
              </Link>
            ) : null
          }
        />
        {recentRuns.length === 0 ? (
          <EmptyState title="No runs yet" hint="Runs appear here as they complete." />
        ) : (
          <RunsTable runs={recentRuns} />
        )}
      </section>
    </div>
  );
}

function SectionHeader({
  title,
  hint,
  action,
}: {
  title: string;
  hint?: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="flex items-baseline justify-between">
      <div className="flex items-baseline gap-2">
        <h2 className="text-[11px] font-semibold uppercase tracking-[0.08em] text-muted-foreground">
          {title}
        </h2>
        {hint && <span className="text-[11px] text-muted-foreground/70">· {hint}</span>}
      </div>
      {action}
    </div>
  );
}

function ConfigCard({ config }: { config: ConfigSummary }) {
  const isPassing = config.latest_status === "passed";
  const isPartial = config.latest_status === "partial";
  return (
    <Link href={`/configs/${config.config_name}`} className="group block">
      <Card className="h-full">
        <CardHeader>
          <div className="flex items-start justify-between gap-3">
            <CardTitle className="font-mono text-[13px] leading-tight text-foreground/90">
              {config.config_name}
            </CardTitle>
            <StatusTooltip status={config.latest_status ?? ""}>
              <Badge variant={isPassing ? "success" : isPartial ? "warning" : "destructive"}>
                <StatusDot passing={isPassing} />
                {config.latest_status ?? "—"}
              </Badge>
            </StatusTooltip>
          </div>
          <CardDescription>
            {config.concurrency_levels.length} concurrency level
            {config.concurrency_levels.length === 1 ? "" : "s"} · latest{" "}
            {formatRelative(config.latest_started_at)}
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-wrap gap-1">
          {config.concurrency_levels.map((c) => (
            <Badge key={c} variant="secondary" className="font-mono text-[11px]">
              {c}
            </Badge>
          ))}
        </CardContent>
      </Card>
    </Link>
  );
}

function RunsTable({ runs }: { runs: RunSummary[] }) {
  return (
    <div className="overflow-hidden rounded-xl border border-border/60">
      <table className="w-full text-[13px]">
        <thead className="bg-muted/30 text-left text-[11px] font-medium uppercase tracking-[0.05em] text-muted-foreground">
          <tr>
            <th className="px-4 py-2.5">Time</th>
            <th className="px-4 py-2.5">Trigger</th>
            <th className="px-4 py-2.5">Config</th>
            <th className="px-4 py-2.5 text-right">Conc.</th>
            <th className="px-4 py-2.5">Commit</th>
            <th className="px-4 py-2.5">Author</th>
            <th className="px-4 py-2.5">Status</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((r) => (
            <tr
              key={r.id}
              className="group border-t border-border/60 transition-colors hover:bg-muted/30"
            >
              <td className="px-4 py-2.5">
                <Link
                  href={`/runs/${r.id}`}
                  className="font-medium transition group-hover:text-primary"
                >
                  {formatRelative(r.started_at)}
                </Link>
              </td>
              <td className="px-4 py-2.5">
                <Badge variant="outline" className="font-mono lowercase">
                  {r.trigger}
                </Badge>
              </td>
              <td className="px-4 py-2.5 font-mono text-[12px] text-foreground/80">
                {r.config_name}
              </td>
              <td className="px-4 py-2.5 text-right font-mono tabular-numbers text-[12px]">
                {r.concurrency.toLocaleString()}
              </td>
              <td className="px-4 py-2.5 font-mono text-[12px] text-muted-foreground">
                {r.commit_short_sha ?? "—"}
              </td>
              <td className="px-4 py-2.5 text-muted-foreground">
                {r.commit_author ?? "—"}
              </td>
              <td className="px-4 py-2.5">
                <StatusTooltip status={r.status}>
                  <Badge
                    variant={
                      r.status === "passed"
                        ? "success"
                        : r.status === "partial"
                          ? "warning"
                          : "destructive"
                    }
                  >
                    <StatusDot passing={r.status === "passed"} />
                    {r.status}
                  </Badge>
                </StatusTooltip>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function StatusDot({ passing }: { passing: boolean }) {
  return (
    <span
      className={`inline-block h-1.5 w-1.5 rounded-full ${
        passing ? "bg-success" : "bg-destructive"
      }`}
      aria-hidden
    />
  );
}

function EmptyState({ title, hint }: { title: string; hint: string }) {
  return (
    <Card>
      <CardContent className="py-10 text-center">
        <p className="text-[14px] font-medium">{title}</p>
        <p className="mt-1 text-[13px] text-muted-foreground">{hint}</p>
      </CardContent>
    </Card>
  );
}

function ErrorBanner({ message }: { message: string }) {
  return (
    <div className="rounded-xl border border-destructive/40 bg-destructive/5 p-4">
      <p className="text-[13px] font-medium text-destructive">Dashboard is unreachable</p>
      <p className="mt-1 font-mono text-[11px] text-muted-foreground">{message}</p>
    </div>
  );
}
