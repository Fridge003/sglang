"use client";

import { Check, Link as LinkIcon } from "lucide-react";
import * as React from "react";
import { cn } from "@/lib/utils";

/**
 * Small button that copies the current URL to the clipboard. Flash-checkmarks
 * on success. Used on any detail page where a dev might want to paste a link
 * into Slack / a PR comment.
 */
export function CopyLinkButton({ className }: { className?: string }) {
  const [copied, setCopied] = React.useState(false);

  const onClick = React.useCallback(async () => {
    try {
      await navigator.clipboard.writeText(window.location.href);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard write can fail in insecure contexts; fall back to alert so
      // the user knows something went wrong.
      alert(window.location.href);
    }
  }, []);

  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "inline-flex h-7 items-center gap-1.5 rounded-md border border-border/60 px-2 text-[12px] text-muted-foreground transition",
        "hover:border-border hover:text-foreground",
        className,
      )}
      title="Copy link to this page"
      aria-live="polite"
    >
      {copied ? (
        <>
          <Check size={12} aria-hidden />
          copied
        </>
      ) : (
        <>
          <LinkIcon size={12} aria-hidden />
          copy link
        </>
      )}
    </button>
  );
}
