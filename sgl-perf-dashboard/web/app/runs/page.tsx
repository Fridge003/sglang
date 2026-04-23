import Link from "next/link";
import { api } from "@/lib/api";
import { formatRelative } from "@/lib/format";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { AutoRefresh } from "@/components/auto-refresh";

export const dynamic = "force-dynamic";

export default async function RunsPage({
  searchParams,
}: {
  searchParams: Promise<{ config?: string; trigger?: string; status?: string }>;
}) {
  const params = await searchParams;
  const runs = await api.listRuns({ limit: 200, ...params }).catch(() => []);

  return (
    <div className="space-y-6 animate-fade-in-up">
      <AutoRefresh />
      <section className="flex items-baseline justify-between border-b border-border/60 pb-4">
        <h1 className="text-xl font-semibold tracking-tight">Runs</h1>
        <p className="text-[12px] text-muted-foreground">
          <span className="tabular-numbers font-medium text-foreground">
            {runs.length.toLocaleString()}
          </span>{" "}
          run{runs.length === 1 ? "" : "s"}
        </p>
      </section>

      {runs.length === 0 ? (
        <Card>
          <CardContent className="py-10 text-center text-[13px] text-muted-foreground">
            No runs match these filters yet.
          </CardContent>
        </Card>
      ) : (
        <div className="overflow-hidden rounded-xl border border-border/60">
          <table className="w-full text-[13px]">
            <thead className="bg-muted/30 text-left text-[11px] font-medium uppercase tracking-[0.05em] text-muted-foreground">
              <tr>
                <th className="px-4 py-2.5">Time</th>
                <th className="px-4 py-2.5">Trigger</th>
                <th className="px-4 py-2.5">Config</th>
                <th className="px-4 py-2.5 text-right">Conc.</th>
                <th className="px-4 py-2.5">Commit</th>
                <th className="px-4 py-2.5">PR</th>
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
                  <td className="px-4 py-2.5">
                    {r.pr_number ? (
                      <a
                        className="text-[12px] text-primary transition hover:underline"
                        href={`https://github.com/sgl-project/sglang/pull/${r.pr_number}`}
                        target="_blank"
                        rel="noreferrer"
                      >
                        #{r.pr_number}
                      </a>
                    ) : (
                      <span className="text-muted-foreground">—</span>
                    )}
                  </td>
                  <td className="px-4 py-2.5">
                    <Badge variant={r.status === "passed" ? "success" : "destructive"}>
                      <span
                        className={`inline-block h-1.5 w-1.5 rounded-full ${
                          r.status === "passed" ? "bg-success" : "bg-destructive"
                        }`}
                        aria-hidden
                      />
                      {r.status}
                    </Badge>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
