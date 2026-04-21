"""Tests for S3 path parsing. These pin the contract between the ingester
and `launch_gb200.sh` — if they break, one side moved without the other.
"""

from __future__ import annotations

from dashboard.s3_paths import parse_config_name, parse_result_key, parse_seq_len


def test_parse_result_key_manual_run():
    key = (
        "manual/24591826053-1/1k1k/dsr1-fp8-1k1k-max-tpt/"
        "2026-04-18/4665/results_concurrency_1024_gpus_48_ctx_16_gen_32.json"
    )
    parsed = parse_result_key(key)
    assert parsed is not None
    assert parsed.trigger == "manual"
    assert parsed.github_run_id == "24591826053"
    assert parsed.github_run_attempt == 1
    assert parsed.seq_len == "1k1k"
    assert parsed.config_name == "dsr1-fp8-1k1k-max-tpt"
    assert parsed.date == "2026-04-18"
    assert parsed.slurm_job_id == "4665"
    assert parsed.concurrency == 1024
    assert parsed.num_gpus == 48
    assert parsed.prefill_gpus == 16
    assert parsed.decode_gpus == 32
    assert parsed.model_prefix == "dsr1"
    assert parsed.precision == "fp8"
    assert parsed.recipe == "max-tpt"
    assert parsed.isl == 1024
    assert parsed.osl == 1024


def test_parse_result_key_cron_fp4():
    key = (
        "cron/24500000000-2/8k1k/dsr1-fp4-8k1k-mid-curve/"
        "2026-04-19/5001/results_concurrency_4096_gpus_72_ctx_24_gen_48.json"
    )
    parsed = parse_result_key(key)
    assert parsed is not None
    assert parsed.trigger == "cron"
    assert parsed.github_run_attempt == 2
    assert parsed.seq_len == "8k1k"
    assert parsed.precision == "fp4"
    assert parsed.isl == 8 * 1024
    assert parsed.osl == 1 * 1024
    assert parsed.concurrency == 4096


def test_parse_result_key_rejects_non_result_files():
    assert parse_result_key("cron/24.../.../2026-04-19/5001/server.log") is None
    assert parse_result_key("cron/24.../.../2026-04-19/5001/agg_foo.json") is None
    assert parse_result_key("") is None
    assert parse_result_key("just/a/shallow/path.json") is None


def test_parse_seq_len_variants():
    assert parse_seq_len("1k1k") == (1024, 1024)
    assert parse_seq_len("8k1k") == (8192, 1024)
    assert parse_seq_len("1k8k") == (1024, 8192)


def test_parse_config_name():
    assert parse_config_name("dsr1-fp8-1k1k-max-tpt") == ("dsr1", "fp8", "max-tpt")
    assert parse_config_name("dsr1-fp4-8k1k-mid-curve") == ("dsr1", "fp4", "mid-curve")
    # Malformed — parser returns all-None rather than raising
    assert parse_config_name("totally-not-valid") == (None, None, None)
