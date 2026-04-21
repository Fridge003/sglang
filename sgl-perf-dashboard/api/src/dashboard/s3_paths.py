"""Parse MinIO object keys into structured run metadata.

The ingester relies on this to recover context (GitHub run ID, config name,
concurrency, etc.) from S3 paths. Must stay in sync with
`scripts/ci/slurm/launch_gb200.sh` and srt-slurm's postprocess stage — any
change to the prefix layout there requires updating `parse_result_key()` here.

Expected layout (as of 2026-04):

    <trigger>/<run_id>-<attempt>/<seq_len>/<config>/<date>/<slurm_job_id>/<bench_subdir>/
    results_concurrency_<N>_gpus_<G>_ctx_<P>_gen_<D>.json

`<bench_subdir>` is added by srt-slurm (sweep's per-benchmark folder —
e.g., `sa-bench_isl_1024_osl_1024`, future `gsm8k_...` for evals).

Example:
    manual/24591826053-1/1k1k/dsr1-fp8-1k1k-max-tpt/2026-04-18/4665/
    sa-bench_isl_1024_osl_1024/
    results_concurrency_1024_gpus_48_ctx_16_gen_32.json
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Filename pattern — must be present, hence required fields become mandatory.
_RESULT_RE = re.compile(
    r"^results_concurrency_(?P<conc>\d+)"
    r"_gpus_(?P<gpus>\d+)"
    r"_ctx_(?P<prefill>\d+)"
    r"_gen_(?P<decode>\d+)\.json$"
)

# Config name encodes model/precision/seq-len/recipe, e.g. dsr1-fp8-1k1k-max-tpt.
_CONFIG_RE = re.compile(
    r"^(?P<model_prefix>[a-z0-9]+)"
    r"-(?P<precision>fp4|fp8|bf16|fp16|fp32)"
    r"-(?P<seq_len>[0-9]+k[0-9]+k|[0-9]+k[0-9]+|[0-9]+[0-9]+)"
    r"-(?P<recipe>.+)$"
)

# Seq-len like "1k1k" -> (isl=1024, osl=1024). Accepts Nk or N.
_SEQ_LEN_RE = re.compile(r"^(?P<isl>[0-9]+k?)(?P<osl>[0-9]+k?)$")


@dataclass(frozen=True)
class ParsedPath:
    trigger: str  # "cron" | "manual" | other
    github_run_id: str
    github_run_attempt: int
    seq_len: str  # "1k1k" (verbatim from path)
    config_name: str  # "dsr1-fp8-1k1k-max-tpt"
    date: str  # "YYYY-MM-DD" from srt-slurm
    slurm_job_id: str
    bench_subdir: str  # e.g. "sa-bench_isl_1024_osl_1024"
    concurrency: int
    num_gpus: int
    prefill_gpus: int
    decode_gpus: int
    # Derived from config_name:
    model_prefix: str | None
    precision: str | None
    recipe: str | None
    # Derived from seq_len:
    isl: int | None
    osl: int | None


def _parse_seq_len_component(component: str) -> int | None:
    """Turn '1k' -> 1024, '8' -> 8, '1024' -> 1024. Returns None on malformed."""
    if not component:
        return None
    if component.endswith("k"):
        try:
            return int(component[:-1]) * 1024
        except ValueError:
            return None
    try:
        return int(component)
    except ValueError:
        return None


def parse_seq_len(seq_len: str) -> tuple[int | None, int | None]:
    """Turn '1k1k' -> (1024, 1024). Handles '8k1k', '8k8', '1024', etc."""
    m = _SEQ_LEN_RE.match(seq_len)
    if not m:
        return None, None
    return _parse_seq_len_component(m.group("isl")), _parse_seq_len_component(
        m.group("osl")
    )


def parse_config_name(config_name: str) -> tuple[str | None, str | None, str | None]:
    """Decompose 'dsr1-fp8-1k1k-max-tpt' -> (model_prefix, precision, recipe)."""
    m = _CONFIG_RE.match(config_name)
    if not m:
        return None, None, None
    return m.group("model_prefix"), m.group("precision"), m.group("recipe")


def parse_result_key(key: str) -> ParsedPath | None:
    """Parse a MinIO object key like `manual/24.../.../sa-bench_.../results_...json`.

    Returns None if the key doesn't match the expected schema (e.g., it's a
    top-level index file, an agg_*.json, or some non-result artifact).
    """
    parts = key.split("/")
    if len(parts) < 8:
        return None

    # Layout (from right):
    #   -1 filename (results_concurrency_*.json)
    #   -2 bench_subdir (sa-bench_isl_..._osl_...)
    #   -3 slurm_job_id
    #   -4 date (YYYY-MM-DD)
    #   -5 config_name (e.g. dsr1-fp8-1k1k-max-tpt)
    #   -6 seq_len (e.g. 1k1k)
    #   -7 run-attempt (e.g. 24591826053-1)
    #   -8 trigger (cron | manual)
    filename = parts[-1]
    bench_subdir = parts[-2]
    slurm_job_id = parts[-3]
    date = parts[-4]
    config_name = parts[-5]
    seq_len = parts[-6]
    run_attempt = parts[-7]
    trigger = parts[-8]

    # Filename must match the sa-bench result pattern
    fm = _RESULT_RE.match(filename)
    if not fm:
        return None

    # Run-attempt like "24591826053-1"
    if "-" not in run_attempt:
        return None
    try:
        github_run_id, attempt_str = run_attempt.rsplit("-", 1)
        github_run_attempt = int(attempt_str)
    except ValueError:
        return None

    model_prefix, precision, recipe = parse_config_name(config_name)
    isl, osl = parse_seq_len(seq_len)

    return ParsedPath(
        trigger=trigger,
        github_run_id=github_run_id,
        github_run_attempt=github_run_attempt,
        seq_len=seq_len,
        config_name=config_name,
        date=date,
        slurm_job_id=slurm_job_id,
        bench_subdir=bench_subdir,
        concurrency=int(fm.group("conc")),
        num_gpus=int(fm.group("gpus")),
        prefill_gpus=int(fm.group("prefill")),
        decode_gpus=int(fm.group("decode")),
        model_prefix=model_prefix,
        precision=precision,
        recipe=recipe,
        isl=isl,
        osl=osl,
    )
