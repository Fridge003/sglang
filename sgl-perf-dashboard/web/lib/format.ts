import { formatDistanceToNowStrict, parseISO } from "date-fns";

export function formatRelative(iso: string | null | undefined): string {
  if (!iso) return "—";
  try {
    return formatDistanceToNowStrict(parseISO(iso), { addSuffix: true });
  } catch {
    return iso;
  }
}

export function formatNumber(
  value: number | null | undefined,
  opts?: { decimals?: number; unit?: string | null },
): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  const decimals = opts?.decimals ?? (Math.abs(value) >= 100 ? 0 : 2);
  const formatted = value.toLocaleString("en-US", {
    maximumFractionDigits: decimals,
    minimumFractionDigits: decimals,
  });
  return opts?.unit ? `${formatted} ${opts.unit}` : formatted;
}

export function compactUnit(unit: string | null | undefined): string | null {
  if (!unit) return null;
  if (unit === "milliseconds") return "ms";
  if (unit === "seconds") return "s";
  if (unit === "tokens/sec") return "tok/s";
  return unit;
}
