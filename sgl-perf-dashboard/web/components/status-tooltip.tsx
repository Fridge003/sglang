"use client";

import * as React from "react";
import { Tooltip } from "@/components/ui/tooltip";

const DEFINITIONS: Record<string, { title: string; body: string }> = {
  passed: {
    title: "Passed",
    body: "All expected concurrencies for this config produced valid result JSONs.",
  },
  partial: {
    title: "Partial",
    body: "This concurrency was expected to run but no JSON was uploaded. The matrix job may have OOM'd on this size after others finished, or an upload failed silently.",
  },
  failed: {
    title: "Failed",
    body: "The entire matrix job produced no output. Likely crashed before any benchmark ran. Check the GitHub Actions logs.",
  },
};

export function StatusTooltip({
  status,
  children,
  extraNote,
}: {
  status: string;
  children: React.ReactNode;
  extraNote?: string | null;
}) {
  const def = DEFINITIONS[status];
  if (!def) return <>{children}</>;
  return (
    <Tooltip
      content={
        <div className="space-y-1">
          <p className="font-medium">{def.title}</p>
          <p className="text-muted-foreground">{def.body}</p>
          {extraNote && (
            <p className="pt-0.5 text-[11px] font-mono text-muted-foreground">
              {extraNote}
            </p>
          )}
        </div>
      }
    >
      <span className="inline-flex">{children}</span>
    </Tooltip>
  );
}
