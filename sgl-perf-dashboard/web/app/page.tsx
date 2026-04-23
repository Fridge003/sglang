import Link from "next/link";
import { api, type ConfigSummary, type RegressionSummary, type RunSummary } from "@/lib/api";
import { formatRelative } from "@/lib/format";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { AutoRefresh } from "@/components/auto-refresh";

export const dynamic = "force-dynamic";

async function loadHomeData() {
  try {
    const [recentRuns, configs, health, regressions, recentFailures] = await Promise.all([
      api.listRuns({ limit: 20 }),
      api.listConfigs(),
      api.health(),
      api.listRegressions("active"),
      api.listRuns({ limit: 10, status: "failed" }),
    ]);
    return { recentRuns, configs, health, regressions, recentFailures, error: null };
  } catch (err) {
    return {
      recentRuns: [],
      configs: [],
      health: null,
      regressions: [] as RegressionSummary[],
      recentFailures: [] as RunSummary[],
      error: err instanceof Error ? err.message : String(err),
    };
  }
}

export default async function HomePage() {
  const { recentRuns, configs, health, regressions, recentFailures, error } =
    await loadHomeData();

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

      {/* Needs attention */}
      {(regressions.length > 0 || recentFailures.length > 0) && (
        <section className="space-y-3">
          <SectionHeader
            title="Needs attention"
            hint={[
              regressions.length > 0
                ? `${regressions.length} regression${regressions.length === 1 ? "" : "s"}`
                : null,
              recentFailures.length > 0
                ? `${recentFailures.length} failed run${recentFailures.length === 1 ? "" : "s"}`
                : null,
            ]
              .filter(Boolean)
              .join(" · ")}
          />
          <div className="space-y-2">
            {regressions.map((r) => (
              <RegressionRow key={r.id} reg={r} />
            ))}
            {recentFailures.map((r) => (
              <FailedRunRow key={r.id} run={r} />
            ))}
          </div>
        </section>
      )}

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
            <Badge variant={isPassing ? "success" : isPartial ? "warning" : "destructive"}>
              <StatusDot passing={isPassing} />
              {config.latest_status ?? "—"}
            </Badge>
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
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function RegressionRow({ reg }: { reg: RegressionSummary }) {
  const severityVariant =
    reg.severity === "critical"
      ? "destructive"
      : reg.severity === "major"
        ? "destructive"
        : "warning";
  const delta = reg.delta_percent;
  return (
    <Link href={`/regressions/${reg.id}`} className="block">
      <Card className="transition-colors hover:border-destructive/60">
        <CardContent className="flex flex-wrap items-center gap-3 p-4">
          <Badge variant={severityVariant} className="uppercase tracking-wider">
            {reg.severity}
          </Badge>
          <div className="min-w-0 flex-1">
            <p className="truncate font-mono text-[13px]">
              <span className="text-foreground/80">{reg.config_name}</span>
              <span className="mx-1.5 text-muted-foreground">·</span>
              <span className="text-foreground">{reg.metric_name}</span>
              <span className="mx-1.5 text-muted-foreground">·</span>
              <span className="text-muted-foreground">conc {reg.concurrency.toLocaleString()}</span>
            </p>
            <p className="mt-0.5 text-[11px] text-muted-foreground">
              {delta !== null && (
                <span className="font-mono tabular-numbers text-destructive">
                  {delta > 0 ? "+" : ""}
                  {delta.toFixed(1)}%
                </span>
              )}
              {reg.z_score !== null && (
                <>
                  <span className="mx-1 text-muted-foreground/60">·</span>
                  <span className="font-mono tabular-numbers">z={reg.z_score.toFixed(1)}</span>
                </>
              )}
              <span className="mx-1 text-muted-foreground/60">·</span>
              detected {formatRelative(reg.detected_at)}
              {reg.commit_short_sha && (
                <>
                  <span className="mx-1 text-muted-foreground/60">·</span>
                  <span className="font-mono">{reg.commit_short_sha}</span>
                </>
              )}
              {reg.commit_author && (
                <>
                  <span className="mx-1 text-muted-foreground/60">·</span>
                  {reg.commit_author}
                </>
              )}
            </p>
          </div>
          <span className="text-[12px] text-muted-foreground transition group-hover:text-foreground">
            investigate →
          </span>
        </CardContent>
      </Card>
    </Link>
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

function FailedRunRow({ run }: { run: RunSummary }) {
  const isPartial = run.status === "partial";
  return (
    <Link href={`/runs/${run.id}`} className="block">
      <Card className="transition-colors hover:border-destructive/60">
        <CardContent className="flex flex-wrap items-center gap-3 p-4">
          <Badge variant={isPartial ? "warning" : "destructive"} className="uppercase tracking-wider">
            {run.status}
          </Badge>
          <div className="min-w-0 flex-1">
            <p className="truncate font-mono text-[13px]">
              <span className="text-foreground/80">{run.config_name}</span>
              <span className="mx-1.5 text-muted-foreground">·</span>
              <span className="text-muted-foreground">conc {run.concurrency.toLocaleString()}</span>
            </p>
            <p className="mt-0.5 text-[11px] text-muted-foreground">
              {run.failure_reason && (
                <>
                  <span>{run.failure_reason}</span>
                  <span className="mx-1 text-muted-foreground/60">·</span>
                </>
              )}
              {formatRelative(run.started_at)}
              {run.commit_short_sha && (
                <>
                  <span className="mx-1 text-muted-foreground/60">·</span>
                  <span className="font-mono">{run.commit_short_sha}</span>
                </>
              )}
              {run.commit_author && (
                <>
                  <span className="mx-1 text-muted-foreground/60">·</span>
                  {run.commit_author}
                </>
              )}
            </p>
          </div>
          <span className="text-[12px] text-muted-foreground transition group-hover:text-foreground">
            investigate →
          </span>
        </CardContent>
      </Card>
    </Link>
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
