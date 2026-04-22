"use client";

import { Monitor, Moon, Sun } from "lucide-react";
import { useTheme } from "next-themes";
import * as React from "react";
import { cn } from "@/lib/utils";

const MODES = [
  { value: "light", icon: Sun, label: "Light" },
  { value: "system", icon: Monitor, label: "System" },
  { value: "dark", icon: Moon, label: "Dark" },
] as const;

export function ThemeToggle() {
  const { theme, setTheme, resolvedTheme } = useTheme();
  const [mounted, setMounted] = React.useState(false);
  React.useEffect(() => setMounted(true), []);

  if (!mounted) {
    // Placeholder keeps layout stable during hydration
    return <div className="h-7 w-20 rounded-md border border-border/60" aria-hidden />;
  }

  const current = theme ?? "system";

  return (
    <div
      role="radiogroup"
      aria-label="Theme"
      className="inline-flex h-7 items-center rounded-md border border-border/60 p-0.5"
    >
      {MODES.map(({ value, icon: Icon, label }) => (
        <button
          key={value}
          type="button"
          role="radio"
          aria-checked={current === value}
          aria-label={label}
          title={label}
          onClick={() => setTheme(value)}
          className={cn(
            "flex h-6 w-6 items-center justify-center rounded-sm transition",
            current === value
              ? "bg-muted text-foreground"
              : "text-muted-foreground hover:text-foreground",
          )}
        >
          <Icon size={12} aria-hidden />
        </button>
      ))}
    </div>
  );
}
