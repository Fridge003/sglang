import Link from "next/link";
import { api } from "@/lib/api";
import { formatRelative } from "@/lib/format";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

export const dynamic = "force-dynamic";

export default async function RunsPage({
  searchParams,
}: {
  searchParams: Promise<{ config?: string; trigger?: string; status?: string }>;
}) {
  const params = await searchParams;
  const runs = await api.listRuns({ limit: 200, ...params }).catch(() => []);

  return (
    <div className="space-y-6">
      <div className="flex items-baseline justify-between">
        <h1 className="text-2xl font-bold tracking-tight">Runs</h1>
        <p className="text-sm text-muted-foreground">
          {runs.length.toLocaleString()} run{runs.length === 1 ? "" : "s"}
        </p>
      </div>

      {runs.length === 0 ? (
        <Card>
          <CardContent className="py-10 text-center text-sm text-muted-foreground">
            No runs match these filters yet.
          </CardContent>
        </Card>
      ) : (
        <div className="overflow-hidden rounded-xl border">
          <table className="w-full text-sm">
            <thead className="bg-muted/40 text-left text-xs uppercase tracking-wider text-muted-foreground">
              <tr>
                <th className="px-4 py-2">Time</th>
                <th className="px-4 py-2">Trigger</th>
                <th className="px-4 py-2">Config</th>
                <th className="px-4 py-2">Conc.</th>
                <th className="px-4 py-2">Commit</th>
                <th className="px-4 py-2">PR</th>
                <th className="px-4 py-2">Status</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((r) => (
                <tr key={r.id} className="border-t transition hover:bg-muted/30">
                  <td className="px-4 py-2">
                    <Link href={`/runs/${r.id}`} className="font-medium hover:underline">
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
                  <td className="px-4 py-2">
                    {r.pr_number ? (
                      <a
                        className="text-primary hover:underline"
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
      )}
    </div>
  );
}
