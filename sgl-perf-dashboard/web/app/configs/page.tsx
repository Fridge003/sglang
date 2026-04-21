import Link from "next/link";
import { api } from "@/lib/api";
import { formatRelative } from "@/lib/format";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

export const dynamic = "force-dynamic";

export default async function ConfigsPage() {
  const configs = await api.listConfigs().catch(() => []);
  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold tracking-tight">Configs</h1>
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {configs.map((c) => (
          <Link key={c.config_name} href={`/configs/${c.config_name}`}>
            <Card className="h-full transition hover:border-foreground/40">
              <CardHeader>
                <CardTitle className="font-mono text-sm">{c.config_name}</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-sm">
                <p className="text-muted-foreground">
                  latest {formatRelative(c.latest_started_at)}
                </p>
                <div className="flex flex-wrap gap-1.5">
                  {c.concurrency_levels.map((conc) => (
                    <Badge key={conc} variant="secondary" className="font-mono text-xs">
                      {conc}
                    </Badge>
                  ))}
                </div>
              </CardContent>
            </Card>
          </Link>
        ))}
      </div>
    </div>
  );
}
