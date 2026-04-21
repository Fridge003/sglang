/**
 * Thin typed API client. Same-origin calls; Next.js rewrites /api/* to the
 * backend container. SSR-friendly: works from both server and client components.
 */

// Server components (SSR) run in Node, where fetch requires absolute URLs.
// Client components run in the browser, where relative URLs resolve against
// the current origin. Next.js rewrites `/api/*` to the backend container.
const BASE =
  typeof window === "undefined"
    ? `${process.env.API_INTERNAL_URL ?? "http://dashboard-api:8000"}/api`
    : process.env.NEXT_PUBLIC_API_URL ?? "/api";

export interface Metric {
  name: string;
  value: number;
  unit: string | null;
}

export interface RunSummary {
  id: number;
  github_run_id: string;
  github_run_attempt: number;
  github_run_url: string;
  commit_sha: string | null;
  commit_short_sha: string | null;
  commit_author: string | null;
  pr_number: number | null;
  pr_title: string | null;
  trigger: string;
  config_name: string;
  model_prefix: string | null;
  precision: string | null;
  seq_len: string | null;
  concurrency: number;
  started_at: string;
  status: string;
}

export interface RunDetail extends RunSummary {
  commit_message: string | null;
  commit_date: string | null;
  isl: number | null;
  osl: number | null;
  recipe: string | null;
  num_gpus: number | null;
  prefill_gpus: number | null;
  decode_gpus: number | null;
  s3_log_prefix: string;
  slurm_job_id: string | null;
  ingested_at: string;
  metrics: Metric[];
}

export interface ConfigSummary {
  config_name: string;
  latest_run_id: number | null;
  latest_started_at: string | null;
  latest_status: string | null;
  concurrency_levels: number[];
}

export interface TrendPoint {
  run_id: number;
  github_run_id: string;
  commit_short_sha: string | null;
  commit_author: string | null;
  started_at: string;
  value: number;
}

export interface HealthStatus {
  status: string;
  runs: number;
  metrics: number;
  last_ingest_at: string | null;
  github_enrichment: boolean;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const url = `${BASE}${path}`;
  const resp = await fetch(url, {
    ...init,
    cache: "no-store",
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!resp.ok) {
    throw new Error(`${resp.status} ${resp.statusText} — ${url}`);
  }
  return resp.json() as Promise<T>;
}

export const api = {
  health: () => request<HealthStatus>("/health"),
  listRuns: (params?: {
    limit?: number;
    offset?: number;
    config?: string;
    trigger?: string;
    status?: string;
  }) => {
    const qs = new URLSearchParams();
    Object.entries(params ?? {}).forEach(([k, v]) => {
      if (v !== undefined && v !== null && v !== "") qs.set(k, String(v));
    });
    const q = qs.toString();
    return request<RunSummary[]>(`/runs${q ? `?${q}` : ""}`);
  },
  getRun: (id: number) => request<RunDetail>(`/runs/${id}`),
  listConfigs: () => request<ConfigSummary[]>("/configs"),
  configTrend: (
    config: string,
    params: { metric: string; concurrency: number; window_days?: number },
  ) => {
    const qs = new URLSearchParams({
      metric: params.metric,
      concurrency: String(params.concurrency),
    });
    if (params.window_days) qs.set("window_days", String(params.window_days));
    return request<TrendPoint[]>(`/configs/${config}/trend?${qs.toString()}`);
  },
};
