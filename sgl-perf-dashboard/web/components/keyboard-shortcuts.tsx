"use client";

import { useRouter } from "next/navigation";
import * as React from "react";

/**
 * Global keyboard shortcuts using the vim-style chord pattern (`g h` → home).
 * Listens at window level; doesn't fire when focus is on an input/textarea.
 *
 * Bindings:
 *   g h  → /
 *   g r  → /runs
 *   g c  → /configs
 *   g x  → /regressions  (x because r is runs)
 *   ?    → toggle shortcut help
 */
export function KeyboardShortcuts() {
  const router = useRouter();
  const [helpOpen, setHelpOpen] = React.useState(false);
  const [gPending, setGPending] = React.useState(false);
  const gTimeoutRef = React.useRef<number | null>(null);

  React.useEffect(() => {
    const isEditable = (el: EventTarget | null): boolean => {
      if (!(el instanceof HTMLElement)) return false;
      const tag = el.tagName;
      return (
        tag === "INPUT" ||
        tag === "TEXTAREA" ||
        tag === "SELECT" ||
        el.isContentEditable
      );
    };

    const clearG = () => {
      setGPending(false);
      if (gTimeoutRef.current !== null) {
        window.clearTimeout(gTimeoutRef.current);
        gTimeoutRef.current = null;
      }
    };

    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (isEditable(e.target)) return;

      // `?` toggles help
      if (e.key === "?") {
        e.preventDefault();
        setHelpOpen((v) => !v);
        clearG();
        return;
      }

      // Esc closes help
      if (e.key === "Escape") {
        setHelpOpen(false);
        clearG();
        return;
      }

      if (gPending) {
        // Second key of the chord
        let dest: string | null = null;
        if (e.key === "h") dest = "/";
        else if (e.key === "r") dest = "/runs";
        else if (e.key === "c") dest = "/configs";
        else if (e.key === "x") dest = "/regressions";
        clearG();
        if (dest !== null) {
          e.preventDefault();
          router.push(dest);
        }
        return;
      }

      if (e.key === "g") {
        setGPending(true);
        // Cancel the chord if the user doesn't follow up within a second.
        gTimeoutRef.current = window.setTimeout(clearG, 1000);
      }
    };

    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      if (gTimeoutRef.current !== null) window.clearTimeout(gTimeoutRef.current);
    };
  }, [gPending, router]);

  if (!helpOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-background/60 backdrop-blur-sm animate-fade-in-up"
      role="dialog"
      aria-modal="true"
      aria-label="Keyboard shortcuts"
      onClick={() => setHelpOpen(false)}
    >
      <div
        className="w-full max-w-sm rounded-xl border border-border bg-card p-6 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-4 flex items-baseline justify-between">
          <h2 className="text-[13px] font-semibold">Keyboard shortcuts</h2>
          <span className="text-[11px] text-muted-foreground">press ? to close</span>
        </div>
        <dl className="space-y-2 text-[13px]">
          {[
            ["g h", "Home"],
            ["g r", "Runs"],
            ["g c", "Configs"],
            ["g x", "Regressions"],
            ["?", "Toggle this help"],
            ["Esc", "Close dialogs"],
          ].map(([keys, desc]) => (
            <div key={keys} className="flex items-center justify-between gap-4">
              <dt className="text-muted-foreground">{desc}</dt>
              <dd className="flex gap-1">
                {keys.split(" ").map((k, i) => (
                  <kbd
                    key={i}
                    className="inline-flex h-5 min-w-[20px] items-center justify-center rounded border border-border/80 bg-muted px-1.5 font-mono text-[11px]"
                  >
                    {k}
                  </kbd>
                ))}
              </dd>
            </div>
          ))}
        </dl>
      </div>
    </div>
  );
}
