import { notFound } from "next/navigation";
import { api } from "@/lib/api";
import { compactUnit, formatNumber, formatRelative } from "@/lib/format";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

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
    <div className="space-y-6">
      <section>
        <div className="flex items-baseline justify-between gap-4">
          <div>
            <h1 className="font-mono text-xl font-semibold">{run.config_name}</h1>
            <p className="text-sm text-muted-foreground">
              concurrency {run.concurrency} · {formatRelative(run.started_at)}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="outline">{run.trigger}</Badge>
            <Badge variant={run.status === "passed" ? "success" : "destructive"}>
              {run.status}
            </Badge>
          </div>
        </div>
      </section>

      <section className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Commit</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-sm">
            <div className="flex items-center gap-2">
              <span className="text-muted-foreground">SHA</span>
              {run.commit_sha ? (
                <a
                  className="font-mono text-primary hover:underline"
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
              <p className="text-foreground/80">{run.commit_message.split("\n")[0]}</p>
            )}
            <div className="flex gap-4 text-xs text-muted-foreground">
              <span>{run.commit_author ?? "unknown author"}</span>
              <span>·</span>
              <span>{run.commit_date ? formatRelative(run.commit_date) : "—"}</span>
            </div>
            {run.pr_number && (
              <div>
                <a
                  className="text-primary hover:underline"
                  href={`https://github.com/sgl-project/sglang/pull/${run.pr_number}`}
                  target="_blank"
                  rel="noreferrer"
                >
                  PR #{run.pr_number}
                </a>{" "}
                {run.pr_title && (
                  <span className="text-foreground/80">· {run.pr_title}</span>
                )}
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Run</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-sm">
            <KV k="GitHub Actions">
              <a
                className="text-primary hover:underline"
                href={run.github_run_url}
                target="_blank"
                rel="noreferrer"
              >
                {run.github_run_id}-{run.github_run_attempt}
              </a>
            </KV>
            <KV k="Slurm job">
              <span className="font-mono">{run.slurm_job_id ?? "—"}</span>
            </KV>
            <KV k="GPUs">
              <span className="font-mono">
                {run.num_gpus ?? "—"} (prefill {run.prefill_gpus ?? "—"} / decode{" "}
                {run.decode_gpus ?? "—"})
              </span>
            </KV>
            <KV k="ISL / OSL">
              <span className="font-mono">
                {run.isl ?? "—"} / {run.osl ?? "—"}
              </span>
            </KV>
            <KV k="Logs">
              <a
                className="font-mono text-primary hover:underline"
                href={`https://minio.34-93-45-118.nip.io/sglang-ci-logs/${run.s3_log_prefix}`}
                target="_blank"
                rel="noreferrer"
              >
                s3://sglang-ci-logs/{run.s3_log_prefix}
              </a>
            </KV>
          </CardContent>
        </Card>
      </section>

      <section>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-muted-foreground">
          Metrics
        </h2>
        {run.metrics.length === 0 ? (
          <Card>
            <CardContent className="py-8 text-center text-sm text-muted-foreground">
              No metrics parsed for this run.
            </CardContent>
          </Card>
        ) : (
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {run.metrics.map((m) => (
              <Card key={m.name}>
                <CardHeader className="p-4 pb-1">
                  <CardDescription className="font-mono text-xs">{m.name}</CardDescription>
                </CardHeader>
                <CardContent className="p-4 pt-0">
                  <p className="font-mono text-xl font-semibold tabular-nums">
                    {formatNumber(m.value, { unit: compactUnit(m.unit) })}
                  </p>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}

function KV({ k, children }: { k: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center gap-2">
      <span className="w-28 shrink-0 text-muted-foreground">{k}</span>
      <span className="min-w-0 break-all">{children}</span>
    </div>
  );
}
