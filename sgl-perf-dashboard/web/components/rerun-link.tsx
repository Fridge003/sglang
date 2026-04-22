import { Play } from "lucide-react";

const REPO = "sgl-project/sglang";
const WORKFLOW_FILE = "nightly-72-gpu-gb200.yml";

/**
 * Link to the GitHub Actions workflow dispatch form, pre-filled with
 * `pr_number` / `configs` matching this run. User still has to confirm the
 * form submission on GitHub — we can't dispatch directly (no write perms).
 */
export function RerunLink({
  prNumber,
  configName,
}: {
  prNumber: number | null;
  configName: string;
}) {
  const url = new URL(
    `https://github.com/${REPO}/actions/workflows/${WORKFLOW_FILE}`,
  );
  if (prNumber) url.searchParams.set("inputs%5Bpr_number%5D", String(prNumber));
  url.searchParams.set("inputs%5Bconfigs%5D", configName);

  return (
    <a
      className="inline-flex h-7 items-center gap-1.5 rounded-md border border-border/60 px-2 text-[12px] text-muted-foreground transition hover:border-primary/60 hover:text-primary"
      href={url.toString()}
      target="_blank"
      rel="noreferrer"
      title="Re-run this config on GB200 (opens the GitHub Actions dispatch form)"
    >
      <Play size={12} aria-hidden />
      re-run
    </a>
  );
}
