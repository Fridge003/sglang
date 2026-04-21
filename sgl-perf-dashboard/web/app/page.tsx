import Link from "next/link";
import { api, type ConfigSummary, type RunSummary } from "@/lib/api";
import { formatRelative } from "@/lib/format";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

// Home page — server-rendered, one round-trip for recent runs + config summaries.
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
    <div className="space-y-8">
      <section className="flex items-baseline justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">GB200 Nightly</h1>
          <p className="text-sm text-muted-foreground">
            {health
              ? `${health.runs.toLocaleString()} runs tracked · last ingest ${formatRelative(
                  health.last_ingest_at,
                )}`
              : "dashboard loading"}
          </p>
        </div>
        {health && !health.github_enrichment && (
          <Badge variant="outline" className="text-amber-500 border-amber-500/50">
            GitHub enrichment disabled
          </Badge>
        )}
      </section>

      {error && <ErrorBanner message={error} />}

      <section>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-muted-foreground">
          Configs
        </h2>
        {configs.length === 0 ? (
          <EmptyState
            title="No configs yet"
            hint="Ingester hasn't seen any result JSONs — kick one off on the GB200 nightly workflow."
          />
        ) : (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {configs.map((c) => (
              <ConfigCard key={c.config_name} config={c} />
            ))}
          </div>
        )}
      </section>

      <section>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-muted-foreground">
          Recent runs
        </h2>
        {recentRuns.length === 0 ? (
          <EmptyState title="No runs yet" hint="Runs appear here as they complete." />
        ) : (
          <RunsTable runs={recentRuns} />
        )}
      </section>
    </div>
  );
}

function ConfigCard({ config }: { config: ConfigSummary }) {
  const isPassing = config.latest_status === "passed";
  return (
    <Link
      href={`/configs/${config.config_name}`}
      className="group block transition hover:-translate-y-0.5"
    >
      <Card className="h-full transition group-hover:border-foreground/40">
        <CardHeader>
          <div className="flex items-start justify-between gap-3">
            <CardTitle className="font-mono text-sm leading-tight">
              {config.config_name}
            </CardTitle>
            <Badge variant={isPassing ? "success" : "destructive"}>
              {config.latest_status ?? "—"}
            </Badge>
          </div>
          <CardDescription>
            {config.concurrency_levels.length} concurrency level
            {config.concurrency_levels.length === 1 ? "" : "s"} · latest{" "}
            {formatRelative(config.latest_started_at)}
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-wrap gap-1.5">
          {config.concurrency_levels.map((c) => (
            <Badge key={c} variant="secondary" className="font-mono">
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
    <div className="overflow-hidden rounded-xl border">
      <table className="w-full text-sm">
        <thead className="bg-muted/40 text-left text-xs uppercase tracking-wider text-muted-foreground">
          <tr>
            <th className="px-4 py-2">Time</th>
            <th className="px-4 py-2">Trigger</th>
            <th className="px-4 py-2">Config</th>
            <th className="px-4 py-2">Conc.</th>
            <th className="px-4 py-2">Commit</th>
            <th className="px-4 py-2">Author</th>
            <th className="px-4 py-2">Status</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((r) => (
            <tr
              key={r.id}
              className="border-t transition hover:bg-muted/30"
            >
              <td className="px-4 py-2">
                <Link
                  href={`/runs/${r.id}`}
                  className="font-medium hover:underline"
                >
                  {formatRelative(r.started_at)}
                </Link>
              </td>
              <td className="px-4 py-2">
                <Badge variant="outline">{r.trigger}</Badge>
              </td>
              <td className="px-4 py-2 font-mono text-xs">{r.config_name}</td>
              <td className="px-4 py-2 font-mono text-xs">{r.concurrency}</td>
              <td className="px-4 py-2 font-mono text-xs text-muted-foreground">
                {r.commit_short_sha ?? "—"}
              </td>
              <td className="px-4 py-2 text-muted-foreground">
                {r.commit_author ?? "—"}
              </td>
              <td className="px-4 py-2">
                <Badge variant={r.status === "passed" ? "success" : "destructive"}>
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

function EmptyState({ title, hint }: { title: string; hint: string }) {
  return (
    <Card>
      <CardContent className="py-10 text-center">
        <p className="font-medium">{title}</p>
        <p className="mt-1 text-sm text-muted-foreground">{hint}</p>
      </CardContent>
    </Card>
  );
}

function ErrorBanner({ message }: { message: string }) {
  return (
    <div className="rounded-lg border border-destructive/50 bg-destructive/10 p-4 text-sm">
      <p className="font-medium text-destructive">Dashboard is unreachable</p>
      <p className="mt-1 font-mono text-xs text-muted-foreground">{message}</p>
    </div>
  );
}
