import Link from "next/link";
import { notFound } from "next/navigation";
import { api, type RunSummary } from "@/lib/api";
import { formatRelative } from "@/lib/format";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

export const dynamic = "force-dynamic";

export default async function CommitPage({
  params,
}: {
  params: Promise<{ sha: string }>;
}) {
  const { sha } = await params;
  const result = await api.getCommit(sha).catch(() => null);
  if (!result) notFound();

  return (
    <div className="space-y-8 animate-fade-in-up">
      <section className="border-b border-border/60 pb-6">
        <p className="text-[11px] uppercase tracking-wider text-muted-foreground">
          Commit
        </p>
        <h1 className="mt-1 font-mono text-2xl font-semibold tracking-tight">
          <a
            className="hover:text-primary hover:underline"
            href={`https://github.com/sgl-project/sglang/commit/${result.sha}`}
            target="_blank"
            rel="noreferrer"
          >
            {result.short_sha}
          </a>
        </h1>
        {result.commit_message && (
          <p className="mt-2 text-[14px] text-foreground/80">
            {result.commit_message.split("\n")[0]}
          </p>
        )}
        <p className="mt-1 text-[12px] text-muted-foreground">
          {result.commit_author ?? "—"}
          {result.pr_number && (
            <>
              <span className="mx-1.5">·</span>
              <a
                className="text-primary hover:underline"
                href={`https://github.com/sgl-project/sglang/pull/${result.pr_number}`}
                target="_blank"
                rel="noreferrer"
              >
                PR #{result.pr_number}
              </a>
              {result.pr_title && <span className="text-foreground/70"> · {result.pr_title}</span>}
            </>
          )}
        </p>
      </section>

      <section className="space-y-3">
        <div className="flex items-baseline gap-2">
          <h2 className="text-[11px] font-semibold uppercase tracking-[0.08em] text-muted-foreground">
            Runs testing this commit
          </h2>
          <span className="text-[11px] text-muted-foreground/70">
            · {result.runs.length}
          </span>
        </div>

        <div className="overflow-hidden rounded-xl border border-border/60">
          <table className="w-full text-[13px]">
            <thead className="bg-muted/30 text-left text-[11px] font-medium uppercase tracking-[0.05em] text-muted-foreground">
              <tr>
                <th className="px-4 py-2.5">Time</th>
                <th className="px-4 py-2.5">Trigger</th>
                <th className="px-4 py-2.5">Config</th>
                <th className="px-4 py-2.5 text-right">Conc.</th>
                <th className="px-4 py-2.5">Status</th>
              </tr>
            </thead>
            <tbody>
              {result.runs.map((r: RunSummary) => (
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
                  <td className="px-4 py-2.5">
                    <Badge variant={r.status === "passed" ? "success" : "destructive"}>
                      {r.status}
                    </Badge>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
