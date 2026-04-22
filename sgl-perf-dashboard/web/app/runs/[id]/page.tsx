import { notFound } from "next/navigation";
import { api } from "@/lib/api";
import { compactUnit, formatNumber, formatRelative } from "@/lib/format";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { CopyLinkButton } from "@/components/copy-link-button";
import { RerunLink } from "@/components/rerun-link";

export const dynamic = "force-dynamic";

export default async function RunDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const runId = Number(id);
  if (!Number.isFinite(runId)) notFound();

  const run = await api.getRun(runId).catch(() => null);
  if (!run) notFound();

  return (
    <div className="space-y-8 animate-fade-in-up">
      {/* Header */}
      <section className="flex flex-wrap items-start justify-between gap-4 border-b border-border/60 pb-6">
        <div className="space-y-1">
          <h1 className="font-mono text-lg font-semibold leading-tight">{run.config_name}</h1>
          <p className="text-[13px] text-muted-foreground">
            concurrency{" "}
            <span className="tabular-numbers text-foreground/80">
              {run.concurrency.toLocaleString()}
            </span>{" "}
            · {formatRelative(run.started_at)}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-1.5">
          <Badge variant="outline" className="font-mono lowercase">{run.trigger}</Badge>
          <Badge variant={run.status === "passed" ? "success" : "destructive"}>
            <span
              className={`inline-block h-1.5 w-1.5 rounded-full ${
                run.status === "passed" ? "bg-success" : "bg-destructive"
              }`}
              aria-hidden
            />
            {run.status}
          </Badge>
          <RerunLink prNumber={run.pr_number} configName={run.config_name} />
          <CopyLinkButton />
        </div>
      </section>

      {/* Two-column meta */}
      <section className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Commit</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-[13px]">
            <div className="flex items-center gap-2">
              <span className="w-20 shrink-0 text-[11px] uppercase tracking-wider text-muted-foreground">
                SHA
              </span>
              {run.commit_sha ? (
                <a
                  className="font-mono text-primary transition hover:underline"
                  href={`https://github.com/sgl-project/sglang/commit/${run.commit_sha}`}
                  target="_blank"
                  rel="noreferrer"
                >
                  {run.commit_short_sha ?? run.commit_sha.slice(0, 7)}
                </a>
              ) : (
                <span className="text-muted-foreground">—</span>
              )}
            </div>
            {run.commit_message && (
              <p className="line-clamp-2 text-foreground/80">
                {run.commit_message.split("\n")[0]}
              </p>
            )}
            <div className="flex flex-wrap gap-x-3 gap-y-1 text-[12px] text-muted-foreground">
              <span>{run.commit_author ?? "unknown author"}</span>
              <span>·</span>
              <span>{run.commit_date ? formatRelative(run.commit_date) : "—"}</span>
            </div>
            {run.pr_number && (
              <div className="pt-1 text-[12px]">
                <a
                  className="text-primary transition hover:underline"
                  href={`https://github.com/sgl-project/sglang/pull/${run.pr_number}`}
                  target="_blank"
                  rel="noreferrer"
                >
                  PR #{run.pr_number}
                </a>
                {run.pr_title && (
                  <span className="text-foreground/70"> · {run.pr_title}</span>
                )}
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Run</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-[13px]">
            <KV label="GitHub Actions">
              <a
                className="text-primary transition hover:underline"
                href={run.github_run_url}
                target="_blank"
                rel="noreferrer"
              >
                <span className="font-mono">
                  {run.github_run_id}-{run.github_run_attempt}
                </span>
              </a>
            </KV>
            <KV label="Slurm job">
              <span className="font-mono text-foreground/80">{run.slurm_job_id ?? "—"}</span>
            </KV>
            <KV label="GPUs">
              <span className="font-mono tabular-numbers text-foreground/80">
                {run.num_gpus ?? "—"}
                <span className="mx-1.5 text-muted-foreground">·</span>
                prefill {run.prefill_gpus ?? "—"} / decode {run.decode_gpus ?? "—"}
              </span>
            </KV>
            <KV label="ISL / OSL">
              <span className="font-mono tabular-numbers text-foreground/80">
                {run.isl ?? "—"} / {run.osl ?? "—"}
              </span>
            </KV>
            <KV label="Logs">
              <a
                className="break-all font-mono text-[11px] text-primary transition hover:underline"
                href={`https://minio.34-93-45-118.nip.io/sglang-ci-logs/${run.s3_log_prefix}`}
                target="_blank"
                rel="noreferrer"
              >
                {run.s3_log_prefix}
              </a>
            </KV>
          </CardContent>
        </Card>
      </section>

      {/* Metrics */}
      <section className="space-y-3">
        <div className="flex items-baseline justify-between">
          <h2 className="text-[11px] font-semibold uppercase tracking-[0.08em] text-muted-foreground">
            Metrics
          </h2>
          <span className="text-[11px] text-muted-foreground/70">
            {run.metrics.length} captured
          </span>
        </div>
        {run.metrics.length === 0 ? (
          <Card>
            <CardContent className="py-10 text-center text-[13px] text-muted-foreground">
              No metrics parsed for this run.
            </CardContent>
          </Card>
        ) : (
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {run.metrics.map((m) => (
              <MetricCard key={m.name} name={m.name} value={m.value} unit={m.unit} />
            ))}
          </div>
        )}
      </section>
    </div>
  );
}

function KV({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-baseline gap-2">
      <span className="w-24 shrink-0 text-[11px] uppercase tracking-wider text-muted-foreground">
        {label}
      </span>
      <span className="min-w-0">{children}</span>
    </div>
  );
}

function MetricCard({
  name,
  value,
  unit,
}: {
  name: string;
  value: number;
  unit: string | null;
}) {
  return (
    <Card>
      <CardContent className="space-y-1 p-4">
        <p className="truncate font-mono text-[11px] uppercase tracking-wider text-muted-foreground">
          {name}
        </p>
        <p className="font-mono text-[17px] font-semibold tabular-numbers">
          {formatNumber(value)}
          {compactUnit(unit) && (
            <span className="ml-1 text-[12px] font-normal text-muted-foreground">
              {compactUnit(unit)}
            </span>
          )}
        </p>
      </CardContent>
    </Card>
  );
}
