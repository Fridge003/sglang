#!/usr/bin/env python3
"""
Count ALL CI tests by parsing GitHub workflow files as the source of truth.

This script parses:
  - .github/workflows/pr-test.yml (CUDA CI)
  - .github/workflows/pr-test-amd.yml (AMD CI)

And extracts test information from each job's run commands.

Usage:
    python3 count_tests.py              # Count all tests
    python3 count_tests.py --hw cuda    # CUDA tests only
    python3 count_tests.py --hw amd     # AMD tests only
    python3 count_tests.py --summary    # Show only summary
    python3 count_tests.py --csv output.csv  # Export test files to CSV
"""

import argparse
import ast
import csv
import fnmatch
import re
import sys
from collections import defaultdict
from pathlib import Path

try:
    import yaml
except ImportError:
    print("Error: PyYAML is required. Install with: pip install pyyaml")
    sys.exit(1)

from tabulate import tabulate

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Workflow files to parse
WORKFLOW_FILES = {
    "CUDA": PROJECT_ROOT / ".github/workflows/pr-test.yml",
    "AMD": PROJECT_ROOT / ".github/workflows/pr-test-amd.yml",
}

# NV Exclusive Tests - manually maintained
# These tests are fundamentally incompatible with AMD and will never be ported.
# Format: "testfile_pattern": "reason"
# Supports glob patterns like "sgl-kernel/benchmark/*"
NV_EXCLUSIVE_TESTS = {
    # ===== sgl-kernel: NVIDIA-only =====
    "sgl-kernel/tests/test_cutlass_*": "Cutlass is NVIDIA-only",
    "sgl-kernel/tests/test_flash*": "FlashAttention is NVIDIA-only",
    "sgl-kernel/tests/test_fp4_*": "FP4 requires NVIDIA B200/GB200",
    "sgl-kernel/tests/test_mscclpp*": "MSCCL++ is NVIDIA-only",
    "python/sglang/jit_kernel/tests/*": "jit-kernel tests are CUDA-specific",
    # ===== CUDA Graph: weak_ref_tensor not supported on ROCm =====
    "test/registered/cuda_graph/test_piecewise_cuda_graph_small_1_gpu.py": "weak_ref_tensor is CUDA/NPU only",
    "test/registered/cuda_graph/test_piecewise_cuda_graph_large_1_gpu.py": "weak_ref_tensor is CUDA/NPU only",
    # ===== MLA: private model or FlashMLA/FlashInfer =====
    "test/registered/mla/test_flashmla.py": "FlashMLA is NVIDIA-only + private model",
    "test/registered/mla/test_mla_flashinfer.py": "FlashInfer is NVIDIA-only + private model",
    "test/registered/mla/test_mla_int8_deepseek_v3.py": "private model",
    # ===== FlashInfer/FA3 dependencies =====
    "test/registered/models/test_encoder_embedding_models.py": "FlashInfer required",
    "test/registered/attention/test_fa3.py": "FA3/FlashInfer required",
    "test/registered/attention/test_flash_attention_4.py": "FA3/FlashInfer required",
    "test/registered/attention/test_hybrid_attn_backend.py": "FA3/FlashInfer required",
    "test/registered/spec/test_ngram_speculative_decoding.py": "FA3/FlashInfer required",
    "test/registered/spec/test_standalone_speculative_decoding.py": "FA3/FlashInfer required",
    "test/registered/moe/test_cutedsl_moe.py": "FlashInfer required",
    # ===== EAGLE: NVIDIA-only =====
    "test/registered/spec/eagle/test_eagle_infer_a.py": "EAGLE is NVIDIA-only",
    "test/registered/spec/eagle/test_eagle_infer_b.py": "EAGLE is NVIDIA-only",
    "test/registered/spec/eagle/test_eagle_infer_beta.py": "EAGLE is NVIDIA-only",
    "test/registered/spec/eagle/test_eagle_dp_attention.py": "EAGLE/FA3 is NVIDIA-only",
    "test/registered/spec/eagle/test_deepseek_v3_fp4_mtp_small.py": "Requires 4-gpu-b200",
    # ===== Quantization: not supported on ROCm =====
    "test/registered/quant/test_fp8_utils.py": "deepGEMM required",
    "test/registered/quant/test_autoround.py": "gptq_marlin_repack not available on ROCm",
    "test/registered/model_loading/test_modelopt_loader.py": "modelopt_fp8 not supported on ROCm",
    "test/srt/quant/test_awq.py": "awq_marlin not supported on ROCm",
    "test/srt/quant/test_marlin_moe.py": "moe_wna16_marlin_gemm not available on ROCm",
    "test/srt/test_bnb.py": "bitsandbytes not supported on ROCm",
    "test/srt/test_gptqmodel_dynamic.py": "GPTQ bfloat16 not supported on ROCm",
    "test/srt/test_quantization.py": "gptq_shuffle not available on ROCm",
    "test/srt/test_gguf.py": "GGUF not supported on ROCm",
    "test/srt/quant/test_w4a8_deepseek_v3.py": "w4afp8 not supported on ROCm",
    # ===== VLM: compilation or assertion issues on ROCm =====
    "test/registered/vlm/test_vision_openai_server_a.py": "ROCm compilation error in deepseek_ocr",
    "test/registered/vlm/test_vlm_input_format.py": "assertion error on ROCm",
    # ===== Disaggregation: private model or failures =====
    "test/srt/test_disaggregation_different_tp.py": "failed on ROCm",
    "test/srt/test_disaggregation_pp.py": "timeout on ROCm",
    "test/srt/test_disaggregation_dp_attention.py": "private model",
    # ===== Other failures =====
    "test/srt/models/test_mimo_models.py": "failed on ROCm",
    # ===== Multi-GPU tests requiring specific NV hardware =====
    "test/srt/models/test_qwen3_next_models.py": "Requires 4-gpu-h100",
    "test/srt/test_gpt_oss_4gpu.py": "Requires 4-gpu-h100/b200",
    "test/srt/test_multi_instance_release_memory_occupation.py": "Requires 4-gpu-h100",
    "test/srt/test_pp_single_node.py": "Requires 4-gpu-h100",
    "test/srt/test_epd_disaggregation.py": "Requires 4-gpu-h100",
    "test/srt/ep/test_deepep_small.py": "Requires 4-gpu-h100",
    "test/srt/test_deepseek_v3_fp4_4gpu.py": "Requires 4-gpu-b200",
    "test/srt/test_fp8_blockwise_gemm.py": "Requires 4-gpu-b200",
    "test/srt/test_llama31_fp4.py": "Requires 4-gpu-b200",
    "test/srt/test_deepseek_v3_cutedsl_4gpu.py": "Requires 4-gpu-gb200",
}


def is_nv_exclusive(testfile: str) -> tuple:
    """
    Check if a test file is marked as NV exclusive.
    Returns (is_exclusive, reason) tuple.
    """
    # Check exact match first
    if testfile in NV_EXCLUSIVE_TESTS:
        return (True, NV_EXCLUSIVE_TESTS[testfile])

    # Check glob patterns
    for pattern, reason in NV_EXCLUSIVE_TESTS.items():
        if "*" in pattern or "?" in pattern:
            if fnmatch.fnmatch(testfile, pattern):
                return (True, reason)

    return (False, "")


def parse_run_command(run_cmd: str) -> list:
    """
    Parse a run command to extract test information.
    Returns a list of test_info dicts (one per command in the step).
    """
    if not run_cmd:
        return []

    results = []

    # Pre-process: join lines with backslash continuation
    processed_cmd = run_cmd.replace("\\\n", " ")
    lines = processed_cmd.strip().split("\n")

    # Track current working directory from cd commands
    cwd = ""

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("echo"):
            continue

        # Track cd commands for path resolution
        cd_match = re.search(r"^cd\s+([^\s&;]+)", line)
        if cd_match:
            cwd = cd_match.group(1)

        info = None

        # Pattern 1: python3 run_suite.py --hw X --suite Y
        match = re.search(
            r"python3\s+run_suite\.py\s+--hw\s+(\w+)\s+--suite\s+([\w-]+)", line
        )
        if match:
            hw = match.group(1).upper()
            if hw == "CUDA":
                hw = "CUDA"
            elif hw == "CPU":
                hw = "CPU"
            elif hw == "AMD":
                hw = "AMD"
            info = {
                "type": "run_suite",
                "hw": hw,
                "suite": match.group(2),
                "tests": [],
                "details": [line],
            }
            # Check for partition
            part_match = re.search(r"--auto-partition-size\s+(\d+)", line)
            if part_match:
                info["partitions"] = int(part_match.group(1))
            results.append(info)
            continue

        # Pattern 2: python3 run_suite.py --suite X (test/srt style)
        match = re.search(r"python3\s+run_suite\.py\s+--suite\s+([\w-]+)", line)
        if match:
            info = {
                "type": "run_suite_srt",
                "suite": match.group(1),
                "tests": [],
                "details": [line],
            }
            part_match = re.search(r"--auto-partition-size\s+(\d+)", line)
            if part_match:
                info["partitions"] = int(part_match.group(1))
            results.append(info)
            continue

        # Pattern 3: pytest
        if "pytest" in line:
            test_path = None

            # Handle docker exec with -w flag (AMD style)
            docker_match = re.search(
                r"docker\s+exec\s+-w\s+(/[^\s]+)\s+\S+\s+.*pytest\s+([^\s|]+)", line
            )
            if docker_match:
                workdir = docker_match.group(1)
                test_file = docker_match.group(2)
                workdir = re.sub(r"^/sglang-checkout/", "", workdir)
                test_path = (
                    f"{workdir}/{test_file}"
                    if not test_file.startswith("/")
                    else test_file
                )
            else:
                match = re.search(r"pytest\s+([^\s|]+)", line)
                if match:
                    pytest_arg = match.group(1)
                    if cwd and not pytest_arg.startswith("/"):
                        test_path = f"{cwd}/{pytest_arg}"
                    else:
                        test_path = pytest_arg

            if test_path:
                info = {
                    "type": "pytest",
                    "tests": [test_path],
                    "details": [line],
                }
                results.append(info)
            continue

        # Pattern 4: python3 -m unittest
        match = re.search(r"python3\s+-m\s+unittest\s+([\w.]+)", line)
        if match:
            info = {
                "type": "unittest",
                "tests": [match.group(1)],
                "details": [line],
            }
            results.append(info)
            continue

        # Pattern 5: python3 -m sglang.test.ci.run_with_retry test_*.py
        match = re.search(
            r"python3\s+-m\s+sglang\.test\.ci\.run_with_retry\s+([^\s]+\.py)", line
        )
        if match:
            info = {
                "type": "run_with_retry",
                "tests": [match.group(1)],
                "details": [line],
            }
            results.append(info)
            continue

        # Pattern 6: python3 test_*.py or python3 path/to/test_*.py
        match = re.search(r"python3\s+([^\s]*test_[\w]+\.py)", line)
        if match:
            info = {
                "type": "python_direct",
                "tests": [match.group(1)],
                "details": [line],
            }
            results.append(info)
            continue

        # Pattern 7: multimodal_gen run_suite.py (various path formats)
        match = re.search(
            r"multimodal_gen/test/run_suite\.py.*--suite\s+([\w-]+)", line
        )
        if match:
            info = {
                "type": "multimodal_gen",
                "suite": match.group(1),
                "tests": [],
                "details": [line],
            }
            part_match = re.search(r"--total-partitions\s+(\d+)", line)
            if part_match:
                info["partitions"] = int(part_match.group(1))
            results.append(info)
            continue

        # Pattern 8: bash for loop running python files (e.g., sgl-kernel benchmark)
        match = re.search(r"for\s+\w+\s+in\s+([\w*_.]+\.py)", line)
        if match:
            info = {
                "type": "bash_loop",
                "tests": [match.group(1)],
                "details": [line],
            }
            results.append(info)
            continue

    return results


def resolve_runner(runs_on, matrix: dict) -> str:
    """Resolve runner string, expanding matrix references."""
    if not runs_on:
        return "unknown"

    runner_str = str(runs_on)

    # Handle ${{ matrix.runner }} pattern
    if "matrix.runner" in runner_str:
        if matrix and "runner" in matrix:
            runners = matrix["runner"]
            if isinstance(runners, list):
                return ", ".join(runners)
            return str(runners)
        return "matrix-runner"

    # Handle ${{ needs.*.outputs.* }} pattern
    match = re.search(r"\$\{\{\s*needs\.[^}]+\.outputs\.(\w+)\s*\}\}", runner_str)
    if match:
        output_name = match.group(1)
        # Map known outputs to actual values
        if output_name == "b200_runner":
            return "4-gpu-b200"
        return f"dynamic:{output_name}"

    return runner_str


def extract_jobs_from_workflow(workflow_path: Path, hw_name: str) -> list:
    """Extract test jobs from a workflow file."""
    if not workflow_path.exists():
        print(f"Warning: Workflow file not found: {workflow_path}")
        return []

    with open(workflow_path, "r") as f:
        workflow = yaml.safe_load(f)

    jobs = []

    for job_name, job_config in workflow.get("jobs", {}).items():
        # Skip non-test jobs
        skip_patterns = ["check-changes", "call-gate", "finish", "build"]
        if any(p in job_name.lower() for p in skip_patterns):
            continue

        # Determine actual HW for this job (some CUDA workflow jobs are actually CPU)
        actual_hw = hw_name
        if job_name == "stage-a-cpu-only":
            actual_hw = "CPU"

        job_info = {
            "name": job_name,
            "hw": actual_hw,  # Actual hardware (may differ from workflow file)
            "runner": None,
            "timeout": None,
            "matrix": None,
            "tests": [],
            "needs": job_config.get("needs", []),
            "if_condition": job_config.get("if", ""),
        }

        # Extract matrix first (needed for runner resolution)
        strategy = job_config.get("strategy", {})
        matrix = strategy.get("matrix", {})
        if matrix:
            job_info["matrix"] = matrix
            # Count matrix combinations
            if "part" in matrix:
                job_info["matrix_size"] = len(matrix["part"])
            elif "partition" in matrix:
                job_info["matrix_size"] = len(matrix["partition"])
            elif "runner" in matrix and "part" in matrix:
                job_info["matrix_size"] = len(matrix["runner"]) * len(matrix["part"])
            elif "runner" in matrix:
                job_info["matrix_size"] = len(matrix["runner"])
            else:
                job_info["matrix_size"] = 1

        # Extract runner (with matrix resolution)
        runs_on = job_config.get("runs-on", "")
        job_info["runner"] = resolve_runner(runs_on, matrix)

        # Extract steps and find test commands
        for step in job_config.get("steps", []):
            step_name = step.get("name", "")
            run_cmd = step.get("run", "")
            timeout = step.get("timeout-minutes")

            if timeout:
                job_info["timeout"] = timeout

            # Skip non-test steps
            if any(
                s in step_name.lower()
                for s in [
                    "checkout",
                    "install",
                    "download",
                    "setup",
                    "cleanup",
                    "diagnose",
                    "warmup",
                ]
            ):
                continue

            if run_cmd:
                test_infos = parse_run_command(run_cmd)
                for test_info in test_infos:
                    test_info["step_name"] = step_name
                    test_info["timeout"] = timeout
                    job_info["tests"].append(test_info)

        if job_info["tests"]:
            jobs.append(job_info)

    return jobs


def count_tests_in_suite(suite_name: str, source: str = "registered") -> tuple:
    """
    Count tests for a suite by looking at the actual test files.
    Returns (enabled_count, disabled_count).
    """
    import glob

    if source == "registered":
        # Count from test/registered/
        sys.path.insert(0, str(PROJECT_ROOT / "python"))
        try:
            from sglang.test.ci.ci_register import collect_tests

            files = glob.glob(
                str(PROJECT_ROOT / "test/registered/**/*.py"), recursive=True
            )
            if not files:
                return (1, 0)

            all_tests = collect_tests(files, sanity_check=False)
            # Filter by suite
            enabled = sum(
                1 for t in all_tests if t.suite == suite_name and t.disabled is None
            )
            disabled = sum(
                1 for t in all_tests if t.suite == suite_name and t.disabled is not None
            )
            return (enabled if enabled > 0 else 0, disabled)
        except Exception:
            return (1, 0)

    elif source == "srt":
        # Count from test/srt/run_suite.py
        srt_path = PROJECT_ROOT / "test/srt"
        sys.path.insert(0, str(srt_path))
        try:
            from run_suite import suites

            suite_tests = suites.get(suite_name, [])
            # srt suites don't have disabled concept in the same way
            return (len(suite_tests) if suite_tests else 0, 0)
        except Exception:
            return (1, 0)

    elif source == "multimodal_gen":
        # Dynamically parse SUITES from python/sglang/multimodal_gen/test/run_suite.py using AST
        try:
            multimodal_suite_file = (
                PROJECT_ROOT / "python/sglang/multimodal_gen/test/run_suite.py"
            )
            if not multimodal_suite_file.exists():
                return (1, 0)

            with open(multimodal_suite_file, "r") as f:
                tree = ast.parse(f.read())

            # Find the SUITES assignment
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == "SUITES":
                            # Evaluate the dict literal
                            suites_dict = ast.literal_eval(node.value)
                            suite_files = suites_dict.get(suite_name, [])
                            return (len(suite_files), 0)
            return (1, 0)
        except Exception as e:
            print(f"Warning: Failed to parse multimodal_gen SUITES: {e}")
            return (1, 0)

    elif source == "bash_loop":
        # Count files matching the glob pattern
        import glob as glob_module

        try:
            # For sgl-kernel benchmark
            pattern = str(PROJECT_ROOT / "sgl-kernel/benchmark/bench_*.py")
            files = glob_module.glob(pattern)
            return (len(files) if files else 1, 0)
        except Exception:
            return (1, 0)

    return (1, 0)


def get_test_files_from_suite(suite_name: str, source: str = "registered") -> list:
    """
    Get list of test files for a suite.
    Returns list of tuples: (filepath, is_enabled, backend, is_nightly)
    """
    import glob

    results = []

    if source == "registered":
        sys.path.insert(0, str(PROJECT_ROOT / "python"))
        try:
            from sglang.test.ci.ci_register import collect_tests

            files = glob.glob(
                str(PROJECT_ROOT / "test/registered/**/*.py"), recursive=True
            )
            if not files:
                return []

            all_tests = collect_tests(files, sanity_check=False)
            for t in all_tests:
                if t.suite == suite_name:
                    # Make path relative to PROJECT_ROOT
                    rel_path = str(Path(t.filename).relative_to(PROJECT_ROOT))
                    is_enabled = t.disabled is None
                    backend = (
                        t.backend.name if hasattr(t.backend, "name") else str(t.backend)
                    )
                    is_nightly = getattr(t, "nightly", False)
                    results.append((rel_path, is_enabled, backend, is_nightly))
        except Exception:
            pass
        return results

    elif source == "srt":
        srt_path = PROJECT_ROOT / "test/srt"
        sys.path.insert(0, str(srt_path))
        try:
            from run_suite import suites

            suite_tests = suites.get(suite_name, [])
            for t in suite_tests:
                # TestFile has .name attribute
                filepath = f"test/srt/{t.name}"
                # srt tests don't have nightly field, default to False
                results.append((filepath, True, "CUDA", False))
        except Exception:
            pass
        return results

    elif source == "srt_amd":
        srt_path = PROJECT_ROOT / "test/srt"
        sys.path.insert(0, str(srt_path))
        try:
            from run_suite import suite_amd

            suite_tests = suite_amd.get(suite_name, [])
            for t in suite_tests:
                filepath = f"test/srt/{t.name}"
                results.append((filepath, True, "AMD", False))
        except Exception:
            pass
        return results

    elif source == "multimodal_gen":
        try:
            multimodal_suite_file = (
                PROJECT_ROOT / "python/sglang/multimodal_gen/test/run_suite.py"
            )
            if not multimodal_suite_file.exists():
                return []

            with open(multimodal_suite_file, "r") as f:
                tree = ast.parse(f.read())

            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == "SUITES":
                            suites_dict = ast.literal_eval(node.value)
                            suite_files = suites_dict.get(suite_name, [])
                            for f in suite_files:
                                filepath = f"python/sglang/multimodal_gen/test/{f}"
                                results.append((filepath, True, "CUDA", False))
        except Exception:
            pass
        return results

    elif source == "bash_loop":
        import glob as glob_module

        try:
            pattern = str(PROJECT_ROOT / "sgl-kernel/benchmark/bench_*.py")
            files = glob_module.glob(pattern)
            for f in files:
                rel_path = str(Path(f).relative_to(PROJECT_ROOT))
                results.append((rel_path, True, "CUDA", False))
        except Exception:
            pass
        return results

    elif source == "direct":
        # For direct test file paths (pytest, unittest, python_direct, run_with_retry)
        # suite_name here is actually the file path
        if suite_name:
            results.append((suite_name, True, "unknown", False))
        return results

    return results


def expand_pytest_path(test_path: str) -> list:
    """
    Expand pytest path to individual test files.
    If path is a directory (ends with /), list all test_*.py files in it.
    Returns list of file paths.
    """
    import glob as glob_module

    # If it's a specific file, return as-is
    if test_path.endswith(".py"):
        return [test_path]

    # If it's a directory, expand to test files
    if test_path.endswith("/"):
        dir_path = PROJECT_ROOT / test_path
        if dir_path.exists():
            # Find all test_*.py files recursively
            pattern = str(dir_path / "**/test_*.py")
            files = glob_module.glob(pattern, recursive=True)
            return [str(Path(f).relative_to(PROJECT_ROOT)) for f in files]

    # Return as-is if can't expand
    return [test_path]


def collect_all_test_files(all_jobs: dict) -> list:
    """
    Collect all test files from all jobs.
    Returns list of dicts with test file info.
    """
    test_files = []

    for hw_name, jobs in all_jobs.items():
        for job in jobs:
            job_name = job["name"]
            job_hw = job.get("hw", hw_name)
            runner = job["runner"] or "unknown"

            for test_info in job["tests"]:
                test_type = test_info["type"]
                # Use test_info's hw if available (from --hw flag), otherwise use job_hw
                actual_hw = test_info.get("hw", job_hw)
                files = []

                if test_type == "run_suite":
                    suite = test_info.get("suite", "")
                    files = get_test_files_from_suite(suite, "registered")
                elif test_type == "run_suite_srt":
                    suite = test_info.get("suite", "")
                    # Check if this is AMD workflow
                    if actual_hw == "AMD":
                        files = get_test_files_from_suite(suite, "srt_amd")
                    else:
                        files = get_test_files_from_suite(suite, "srt")
                elif test_type == "pytest":
                    # Expand pytest paths (directories -> individual files)
                    for test_path in test_info.get("tests", []):
                        expanded = expand_pytest_path(test_path)
                        for fp in expanded:
                            files.append((fp, True, actual_hw, False))
                elif test_type in ["unittest", "python_direct", "run_with_retry"]:
                    for test_path in test_info.get("tests", []):
                        files.append((test_path, True, actual_hw, False))
                elif test_type == "multimodal_gen":
                    suite = test_info.get("suite", "")
                    files = get_test_files_from_suite(suite, "multimodal_gen")
                elif test_type == "bash_loop":
                    files = get_test_files_from_suite("", "bash_loop")

                for filepath, is_enabled, backend, is_nightly in files:
                    test_files.append(
                        {
                            "testfile": filepath,
                            "hw": actual_hw,
                            "job_name": job_name,
                            "runner": runner,
                            "test_type": test_type,
                            "enabled": is_enabled,
                            "nightly": is_nightly,
                        }
                    )

    return test_files


def export_test_files_csv(all_jobs: dict, output_path: str):
    """
    Export unique test files to CSV (one row per testfile).
    Columns:
      - In NV: test is in CUDA CI (GPU)
      - In CPU: test is in CPU-only CI
      - In AMD: test is in AMD CI
      - NV Exclusive: test is fundamentally incompatible with AMD (manually marked)
    """
    # Collect all test files
    test_files = collect_all_test_files(all_jobs)

    # Aggregate info per testfile
    file_info = {}
    for t in test_files:
        tf = t["testfile"]
        if tf not in file_info:
            file_info[tf] = {
                "nv_jobs": [],
                "cpu_jobs": [],
                "amd_jobs": [],
                "test_type": t["test_type"],
                "enabled": t["enabled"],
                "nightly": t["nightly"],
            }
        if t["hw"] == "CUDA":
            file_info[tf]["nv_jobs"].append(t["job_name"])
        elif t["hw"] == "CPU":
            file_info[tf]["cpu_jobs"].append(t["job_name"])
        elif t["hw"] == "AMD":
            file_info[tf]["amd_jobs"].append(t["job_name"])
        # Keep nightly=True if any entry is nightly
        if t["nightly"]:
            file_info[tf]["nightly"] = True

    # Write CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "testfile",
                "In NV",
                "In CPU",
                "In AMD",
                "NV Exclusive",
                "NV Exclusive Reason",
                "NV Jobs",
                "CPU Jobs",
                "AMD Jobs",
                "Test Type",
                "Nightly",
            ]
        )

        for tf in sorted(file_info.keys()):
            info = file_info[tf]
            in_nv = len(info["nv_jobs"]) > 0
            in_cpu = len(info["cpu_jobs"]) > 0
            in_amd = len(info["amd_jobs"]) > 0
            nv_exclusive, nv_reason = is_nv_exclusive(tf)
            writer.writerow(
                [
                    tf,
                    "Yes" if in_nv else "No",
                    "Yes" if in_cpu else "No",
                    "Yes" if in_amd else "No",
                    "Yes" if nv_exclusive else "No",
                    nv_reason,
                    "; ".join(sorted(set(info["nv_jobs"]))) if info["nv_jobs"] else "",
                    (
                        "; ".join(sorted(set(info["cpu_jobs"])))
                        if info["cpu_jobs"]
                        else ""
                    ),
                    (
                        "; ".join(sorted(set(info["amd_jobs"])))
                        if info["amd_jobs"]
                        else ""
                    ),
                    info["test_type"],
                    "Yes" if info["nightly"] else "No",
                ]
            )

    # Calculate summary
    nv_files = set(tf for tf, info in file_info.items() if info["nv_jobs"])
    cpu_files = set(tf for tf, info in file_info.items() if info["cpu_jobs"])
    amd_files = set(tf for tf, info in file_info.items() if info["amd_jobs"])
    only_nv = nv_files - amd_files - cpu_files
    only_cpu = cpu_files - nv_files - amd_files
    only_amd = amd_files - nv_files - cpu_files
    nv_and_amd = nv_files & amd_files
    nv_exclusive_count = sum(1 for tf in file_info if is_nv_exclusive(tf)[0])
    pending_amd_count = len(only_nv) - nv_exclusive_count
    nightly_count = sum(1 for tf, info in file_info.items() if info["nightly"])

    print(f"Exported {len(file_info)} unique test files to {output_path}")
    print(f"  In NV (CUDA GPU): {len(nv_files)}")
    print(f"  In CPU only: {len(cpu_files)}")
    print(f"  In AMD: {len(amd_files)}")
    print(f"  In both NV and AMD: {len(nv_and_amd)}")
    print(f"  Only in NV: {len(only_nv)}")
    print(f"    - NV Exclusive (cannot port): {nv_exclusive_count}")
    print(f"    - Pending AMD port: {pending_amd_count}")
    print(f"  Nightly tests: {nightly_count}")


def main():
    parser = argparse.ArgumentParser(
        description="Count CI tests by parsing GitHub workflow files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 count_tests.py                    # All tests
    python3 count_tests.py --hw cuda          # CUDA tests only
    python3 count_tests.py --hw amd           # AMD tests only
    python3 count_tests.py --summary          # Summary only
    python3 count_tests.py --list-commands    # Show actual run commands
        """,
    )
    parser.add_argument(
        "--hw",
        type=str,
        choices=["cuda", "amd", "all"],
        default="all",
        help="Filter by hardware backend",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show only summary (no per-job breakdown)",
    )
    parser.add_argument(
        "--list-commands",
        action="store_true",
        help="Show actual run commands for each job",
    )
    parser.add_argument(
        "--csv",
        type=str,
        metavar="FILE",
        help="Export unique test files to CSV (one row per file, shows In NV, In AMD, NV Exclusive, Jobs)",
    )
    args = parser.parse_args()

    all_jobs = {}

    # Parse workflow files
    for hw_name, workflow_path in WORKFLOW_FILES.items():
        # For CSV export, always parse all workflows to determine NV exclusive
        if args.csv is None and args.hw != "all" and args.hw.upper() != hw_name:
            continue

        jobs = extract_jobs_from_workflow(workflow_path, hw_name)
        all_jobs[hw_name] = jobs

    # Export to CSV if requested
    if args.csv:
        export_test_files_csv(all_jobs, args.csv)
        return

    # Build summary table
    rows = []
    total_enabled = 0
    total_disabled = 0
    total_jobs = 0

    for hw_name, jobs in all_jobs.items():
        for job in jobs:
            job_name = job["name"]
            job_hw = job.get("hw", hw_name)  # Use job-specific HW if available
            runner = job["runner"] or "unknown"
            timeout = job.get("timeout", "?")
            matrix_size = job.get("matrix_size", 1)

            # Count tests in this job
            enabled_count = 0
            disabled_count = 0
            test_types = []

            for test_info in job["tests"]:
                test_type = test_info["type"]
                test_types.append(test_type)

                if test_type == "run_suite":
                    suite = test_info.get("suite", "")
                    enabled, disabled = count_tests_in_suite(suite, "registered")
                    enabled_count += enabled
                    disabled_count += disabled
                elif test_type == "run_suite_srt":
                    suite = test_info.get("suite", "")
                    enabled, disabled = count_tests_in_suite(suite, "srt")
                    enabled_count += enabled
                    disabled_count += disabled
                elif test_type in ["unittest", "python_direct", "run_with_retry"]:
                    enabled_count += len(test_info.get("tests", []))
                elif test_type == "pytest":
                    # Estimate pytest tests
                    enabled_count += max(1, len(test_info.get("tests", [])))
                elif test_type == "multimodal_gen":
                    # Count from actual suite definition
                    suite = test_info.get("suite", "")
                    enabled, disabled = count_tests_in_suite(suite, "multimodal_gen")
                    enabled_count += enabled
                    disabled_count += disabled
                elif test_type == "bash_loop":
                    # Count actual matching files (e.g., sgl-kernel benchmark)
                    enabled, disabled = count_tests_in_suite("", "bash_loop")
                    enabled_count += enabled
                    disabled_count += disabled

            enabled_count = (
                max(enabled_count, 1)
                if enabled_count == 0 and disabled_count == 0
                else enabled_count
            )

            # Multiply by matrix size for parallel jobs
            effective_runs = matrix_size if matrix_size else 1

            rows.append(
                [
                    job_hw,  # Use job-specific HW
                    job_name,
                    runner,
                    ", ".join(set(test_types)) or "unknown",
                    enabled_count,
                    disabled_count,
                    effective_runs,
                    f"{timeout}min" if timeout else "?",
                ]
            )

            total_enabled += enabled_count
            total_disabled += disabled_count
            total_jobs += 1

    if not args.summary:
        headers = [
            "HW",
            "Job Name",
            "Runner",
            "Test Type",
            "Enabled",
            "Disabled",
            "Matrix",
            "Timeout",
        ]
        print(tabulate(rows, headers=headers, tablefmt="psql"))
        print()

    # Summary by HW (group by actual HW from rows, not workflow file)
    print("=" * 90)
    print("Summary by Hardware:")
    print("=" * 90)
    hw_set = sorted(set(r[0] for r in rows))
    for hw in hw_set:
        hw_rows = [r for r in rows if r[0] == hw]
        hw_enabled = sum(r[4] for r in hw_rows)
        hw_disabled = sum(r[5] for r in hw_rows)
        hw_jobs = len(hw_rows)
        print(
            f"  [{hw:6}] {hw_jobs:3} jobs, {hw_enabled:4} enabled, {hw_disabled:3} disabled"
        )

    print("─" * 90)
    print(
        f"  {'TOTAL':8} {total_jobs:3} jobs, {total_enabled:4} enabled, {total_disabled:3} disabled"
    )
    print()

    # Summary by test type
    print("=" * 90)
    print("Summary by Test Type:")
    print("=" * 90)
    type_counts = defaultdict(lambda: {"enabled": 0, "disabled": 0})
    for row in rows:
        for t in row[3].split(", "):
            if t:
                type_counts[t]["enabled"] += row[4]
                type_counts[t]["disabled"] += row[5]
    for test_type, counts in sorted(type_counts.items()):
        print(
            f"  [{test_type:20}] {counts['enabled']:4} enabled, {counts['disabled']:3} disabled"
        )
    print()

    # Show commands if requested
    if args.list_commands:
        print("=" * 90)
        print("Run Commands by Job:")
        print("=" * 90)
        for hw_name, jobs in all_jobs.items():
            print(f"\n[{hw_name}]")
            for job in jobs:
                print(f"\n  Job: {job['name']}")
                print(f"  Runner: {job['runner']}")
                for test_info in job["tests"]:
                    print(f"    Type: {test_info['type']}")
                    for detail in test_info.get("details", []):
                        print(f"      {detail[:100]}...")


if __name__ == "__main__":
    main()
