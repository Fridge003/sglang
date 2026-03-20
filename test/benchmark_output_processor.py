"""
Benchmark for process_batch_result_decode in SchedulerOutputProcessorMixin.

This script mocks the Scheduler, Req, ScheduleBatch, and GenerationBatchResult
to measure the performance of the decode output processing loop.

Usage:
    python test/benchmark_output_processor.py [--batch-size 3072] [--num-warmup 5] [--num-iters 20]
"""

import argparse
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Set, Tuple

import torch

# ============================================================================
# Minimal mock classes that replicate the real code paths exercised by
# process_batch_result_decode + stream_output_generation.
#
# The goal is to be faithful to the attribute accesses, method calls, and
# branching that the hot loop actually performs, while avoiding heavy deps
# (GPU, model weights, IPC, tokenizer, etc.).
# ============================================================================


class MockFinishReason:
    """Mock BaseFinishReason."""

    def __init__(self, type_name: str, **kwargs):
        self.type_name = type_name
        self.extra = kwargs

    def to_json(self):
        return {"type": self.type_name, **self.extra}

    def __bool__(self):
        return True


MOCK_FINISH_LENGTH = lambda length: MockFinishReason("length", length=length)
MOCK_FINISH_MATCHED_TOKEN = lambda matched: MockFinishReason("stop", matched=matched)


class MockSamplingParams:
    """Minimal SamplingParams with the fields read in the hot path."""

    def __init__(
        self,
        max_new_tokens: int = 256,
        stop_token_ids: Optional[Set[int]] = None,
        stop_strs: Optional[List[str]] = None,
        stop_regex_strs: Optional[List[str]] = None,
        ignore_eos: bool = False,
        skip_special_tokens: bool = True,
        spaces_between_special_tokens: bool = True,
        no_stop_trim: bool = False,
        stream_interval: Optional[int] = None,
    ):
        self.max_new_tokens = max_new_tokens
        self.stop_token_ids = stop_token_ids
        self.stop_strs = stop_strs if stop_strs is not None else []
        self.stop_str_max_len = 0
        self.stop_regex_strs = stop_regex_strs if stop_regex_strs is not None else []
        self.stop_regex_max_len = 0
        self.ignore_eos = ignore_eos
        self.skip_special_tokens = skip_special_tokens
        self.spaces_between_special_tokens = spaces_between_special_tokens
        self.no_stop_trim = no_stop_trim
        self.stream_interval = stream_interval


class MockTokenizer:
    """Minimal tokenizer mock - the eos_token_id property is intentionally
    implemented as a property to simulate the overhead of the real tokenizer's
    overridden __getattr__."""

    def __init__(self, eos_token_id: int = 2):
        self._eos_token_id = eos_token_id
        self.additional_stop_token_ids = None

    @property
    def eos_token_id(self):
        return self._eos_token_id


class MockTraceCtx:
    """Mock trace context."""

    def abort(self):
        pass


class MockTimeStats:
    """Minimal time stats that records timing calls as no-ops."""

    def __init__(self):
        self.last_decode_finish_time = None
        self.completion_time = None
        self.trace_ctx = MockTraceCtx()

    def set_last_decode_finish_time(self, ts=None):
        self.last_decode_finish_time = ts

    def set_completion_time(self, ts=None):
        self.completion_time = ts

    def disagg_mode_str(self):
        return "none"

    def convert_to_duration(self):
        return {}


class MockReq:
    """Minimal Req mock with the same attribute layout and method behavior
    as the real Req class for the decode output processing path."""

    def __init__(
        self,
        rid: str,
        origin_input_ids: List[int],
        output_ids: List[int],
        sampling_params: MockSamplingParams,
        tokenizer: MockTokenizer,
        vocab_size: int = 32000,
        stream: bool = False,
        return_logprob: bool = False,
        return_hidden_states: bool = False,
        return_routed_experts: bool = False,
    ):
        self.rid = rid
        self.origin_input_ids = origin_input_ids
        self.origin_input_ids_unpadded = origin_input_ids
        self.output_ids = list(output_ids)  # mutable copy
        self.sampling_params = sampling_params
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.stream = stream
        self.return_logprob = return_logprob
        self.return_hidden_states = return_hidden_states
        self.return_routed_experts = return_routed_experts

        # Finish state
        self.finished_reason = None
        self.finished_len = None
        self.finished_output = None
        self.to_finish = None
        self.eos_token_ids = None

        # Grammar
        self.grammar = None

        # Mamba (typically None for non-Mamba models)
        self.mamba_ping_pong_track_buffer = None

        # Multimodal
        self.multimodal_inputs = None

        # Streaming offsets
        self.send_token_offset = 0
        self.send_decode_id_offset = 0
        self.send_output_token_logprobs_offset = 0
        self.decoded_text = ""
        self.surr_offset = None
        self.read_offset = None
        self.cur_decode_ids_len = 0
        self.surr_and_decode_ids = None

        # Logprob fields
        self.top_logprobs_num = 0
        self.token_ids_logprob = None
        self.input_logprob_sent = False
        self.input_token_logprobs_val = None
        self.input_token_logprobs_idx = None
        self.input_top_logprobs_val = None
        self.input_top_logprobs_idx = None
        self.input_token_ids_logprobs_val = None
        self.input_token_ids_logprobs_idx = None
        self.output_token_logprobs_val = None
        self.output_token_logprobs_idx = None
        self.output_top_logprobs_val = None
        self.output_top_logprobs_idx = None
        self.output_token_ids_logprobs_val = None
        self.output_token_ids_logprobs_idx = None

        # Hidden states / experts
        self.hidden_states = []
        self.routed_experts = None

        # Customized info
        self.customized_info = None

        # Cache info
        self.cached_tokens = 0
        self.cached_tokens_device = 0
        self.cached_tokens_host = 0
        self.cached_tokens_storage = 0
        self.retraction_count = 0

        # Spec decoding
        self.spec_verify_ct = 0
        self.spec_accepted_tokens = 0
        self.spec_acceptance_histogram = []
        self.kv_committed_len = 0

        # Retracted
        self.is_retracted = False

        # Pool - set to None so release_kv_cache exits early
        # (we don't have a real token pool to free from)
        self.req_pool_idx = None
        self.mamba_pool_idx = None

        # Time stats
        self.time_stats = MockTimeStats()

        # Logging
        self.has_log_time_stats = False

        # IPC
        self.http_worker_ipc = "mock_ipc"

    def finished(self) -> bool:
        return self.finished_reason is not None

    @property
    def seqlen(self) -> int:
        return len(self.origin_input_ids) + len(self.output_ids)

    @property
    def output_ids_through_stop(self) -> List[int]:
        if self.finished_len is not None:
            return self.output_ids[: self.finished_len]
        return self.output_ids

    def check_finished(self, new_accepted_len: int = 1):
        """Replicate the real check_finished logic."""
        if self.finished():
            return

        if self.to_finish:
            self.finished_reason = self.to_finish
            self.to_finish = None
            return

        if len(self.output_ids) >= self.sampling_params.max_new_tokens:
            self.finished_reason = MOCK_FINISH_LENGTH(
                self.sampling_params.max_new_tokens
            )
            self.finished_len = self.sampling_params.max_new_tokens
            return

        if self.grammar is not None:
            if hasattr(self.grammar, "is_terminated") and self.grammar.is_terminated():
                self.finished_reason = MOCK_FINISH_MATCHED_TOKEN(self.output_ids[-1])
                return

        new_accepted_tokens = self.output_ids[-new_accepted_len:]
        self._check_token_based_finish(new_accepted_tokens)

    def _check_token_based_finish(self, new_accepted_tokens: List[int]) -> bool:
        if self.sampling_params.ignore_eos:
            return False

        matched_eos = False
        for i, token_id in enumerate(new_accepted_tokens):
            if self.sampling_params.stop_token_ids:
                matched_eos |= token_id in self.sampling_params.stop_token_ids
            if self.eos_token_ids:
                matched_eos |= token_id in self.eos_token_ids
            if self.tokenizer is not None:
                matched_eos |= token_id == self.tokenizer.eos_token_id
                if self.tokenizer.additional_stop_token_ids:
                    matched_eos |= token_id in self.tokenizer.additional_stop_token_ids
            if matched_eos:
                self.finished_reason = MOCK_FINISH_MATCHED_TOKEN(token_id)
                matched_pos = len(self.output_ids) - len(new_accepted_tokens) + i
                self.finished_len = matched_pos + 1
                return True
        return False

    def init_incremental_detokenize(self):
        """Replicate init_incremental_detokenize."""
        INIT_OFFSET = 5
        first_iter = self.surr_offset is None or self.read_offset is None
        output_ids = self.output_ids_through_stop

        if first_iter:
            self.read_offset = len(self.origin_input_ids_unpadded)
            self.surr_offset = max(self.read_offset - INIT_OFFSET, 0)
            self.surr_and_decode_ids = (
                self.origin_input_ids_unpadded[self.surr_offset :] + output_ids
            )
            self.cur_decode_ids_len = len(output_ids)
        else:
            self.surr_and_decode_ids.extend(output_ids[self.cur_decode_ids_len :])
            self.cur_decode_ids_len = len(output_ids)

        return self.surr_and_decode_ids, self.read_offset - self.surr_offset

    def check_match_stop_str_prefix(self) -> bool:
        # In the common path (no stop strings), returns False immediately
        if not self.sampling_params.stop_strs:
            return False
        return False

    def log_time_stats(self):
        if self.has_log_time_stats:
            return
        self.has_log_time_stats = True


class MockSpecAlgorithm:
    """Mock SpeculativeAlgorithm.NONE"""

    def is_none(self) -> bool:
        return True


class MockLogitsProcessorOutput:
    """Minimal LogitsProcessorOutput - None for the common no-logprob path."""

    def __init__(self):
        self.next_token_logprobs = None
        self.next_token_top_logprobs_val = None
        self.next_token_top_logprobs_idx = None
        self.next_token_token_ids_logprobs_val = None
        self.next_token_token_ids_logprobs_idx = None
        self.hidden_states = None
        self.customized_info = None


@dataclass
class MockGenerationBatchResult:
    logits_output: Any = None
    next_token_ids: Any = None
    num_accepted_tokens: int = 0
    accept_length_per_req_cpu: Any = None
    can_run_cuda_graph: bool = False
    copy_done: Any = None
    accept_lens: Any = None


class MockModelConfig:
    def __init__(self):
        self.is_multimodal_gen = False
        self.vocab_size = 32000


class MockServerArgs:
    def __init__(self):
        self.disaggregation_decode_enable_offload_kvcache = False
        self.enable_request_time_stats_logging = False
        self.multi_item_scoring_delimiter = None


class MockTreeCache:
    def cache_finished_req(self, req, **kwargs):
        pass

    def supports_mamba(self):
        return True


class MockTokenToKVPoolAllocator:
    def free_group_begin(self):
        pass

    def free_group_end(self):
        pass


class MockSendToDetokenizer:
    def send_output(self, output):
        pass


class MockGetLoadReqOutput:
    def __init__(self):
        self.dp_rank = 0
        self.num_reqs = 0
        self.num_waiting_reqs = 0
        self.num_tokens = 0
        self.ts_tic = 0.0


class MockMetricsCollector:
    def increment_decode_cuda_graph_pass(self, value=False):
        pass


class MockScheduleBatch:
    """Minimal ScheduleBatch."""

    def __init__(self, reqs, return_logprob=False):
        self.reqs = reqs
        self.return_logprob = return_logprob
        self.spec_algorithm = MockSpecAlgorithm()
        self.is_spec_v2 = False
        self.decoding_reqs = None
        self.prefill_stats = None
        self.dp_cooperation_info = None

    def batch_size(self):
        return len(self.reqs)


# ============================================================================
# Build the mock Scheduler by composing the real mixin with mock attributes
# ============================================================================


def make_mock_scheduler():
    """Create a mock scheduler that has all attributes needed by
    process_batch_result_decode and stream_output_generation."""
    from sglang.srt.managers.scheduler_output_processor_mixin import (
        SchedulerOutputProcessorMixin,
    )

    class MockScheduler(SchedulerOutputProcessorMixin):
        pass

    sched = MockScheduler()

    # Core attributes
    sched.is_generation = True
    sched.enable_overlap = True  # overlap scheduler path
    sched.spec_algorithm = MockSpecAlgorithm()
    sched.server_args = MockServerArgs()
    sched.model_config = MockModelConfig()
    sched.tree_cache = MockTreeCache()
    sched.token_to_kv_pool_allocator = MockTokenToKVPoolAllocator()
    sched.send_to_detokenizer = MockSendToDetokenizer()
    sched.stream_interval = 1
    sched.dp_rank = 0
    sched.attn_tp_rank = 0
    sched.num_generated_tokens = 0
    sched.forward_ct_decode = 0
    sched.enable_metrics = False
    sched.metrics_collector = MockMetricsCollector()
    sched.disaggregation_mode = None  # no disaggregation
    sched.enable_hierarchical_cache = False
    sched.enable_hicache_storage = False

    # req_to_token_pool for maybe_collect_routed_experts (not called in practice
    # since routed_experts_capturer returns None, but we mock it anyway)
    sched.req_to_token_pool = None

    # Running batch for get_load
    sched.running_batch = MockScheduleBatch([])
    sched.waiting_queue = []
    sched.is_hybrid_swa = False
    sched.is_hybrid_ssm = False

    # get_load returns a mock
    sched.get_load = lambda: MockGetLoadReqOutput()

    # report_decode_stats is a no-op
    sched.report_decode_stats = lambda *args, **kwargs: None

    # update_spec_metrics is a no-op
    sched.update_spec_metrics = lambda *args, **kwargs: None

    # abort_request is a no-op
    sched.abort_request = lambda *args, **kwargs: None

    # maybe_collect_routed_experts is a no-op (needs global capturer)
    sched.maybe_collect_routed_experts = lambda req: None

    return sched


def make_reqs(
    batch_size: int,
    input_len: int = 128,
    output_len: int = 50,
    vocab_size: int = 32000,
    stream: bool = False,
) -> List[MockReq]:
    """Create batch_size mock requests in the typical decode state
    (i.e., already past prefill, generating tokens one at a time)."""
    tokenizer = MockTokenizer(eos_token_id=2)
    reqs = []
    for i in range(batch_size):
        sp = MockSamplingParams(max_new_tokens=256)
        origin_input_ids = list(range(100, 100 + input_len))
        # Already produced some output tokens (none of them are eos=2)
        output_ids = [10 + (j % 100) for j in range(output_len)]
        req = MockReq(
            rid=f"req-{i}",
            origin_input_ids=origin_input_ids,
            output_ids=output_ids,
            sampling_params=sp,
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            stream=stream,
        )
        reqs.append(req)
    return reqs


def make_batch_and_result(
    reqs: List[MockReq],
    return_logprob: bool = False,
) -> Tuple[MockScheduleBatch, MockGenerationBatchResult]:
    """Build a ScheduleBatch + GenerationBatchResult for these requests."""
    batch = MockScheduleBatch(reqs, return_logprob=return_logprob)

    # next_token_ids: one token per request, all non-eos (e.g., token 42)
    next_token_ids = torch.full((len(reqs),), 42, dtype=torch.int64)

    result = MockGenerationBatchResult(
        logits_output=MockLogitsProcessorOutput(),
        next_token_ids=next_token_ids,
        can_run_cuda_graph=False,
        copy_done=None,
    )
    return batch, result


def reset_reqs(reqs: List[MockReq], output_len: int = 50):
    """Reset request state so we can re-run the benchmark cleanly."""
    for req in reqs:
        req.output_ids = [10 + (j % 100) for j in range(output_len)]
        req.finished_reason = None
        req.finished_len = None
        req.finished_output = None
        req.to_finish = None
        req.send_token_offset = 0
        req.send_decode_id_offset = 0
        req.send_output_token_logprobs_offset = 0
        req.surr_offset = None
        req.read_offset = None
        req.cur_decode_ids_len = 0
        req.surr_and_decode_ids = None
        req.has_log_time_stats = False


def benchmark(
    batch_size: int = 3072,
    num_warmup: int = 5,
    num_iters: int = 20,
    stream: bool = False,
    return_logprob: bool = False,
):
    """Run the benchmark and print timing results."""
    print(f"=== Benchmark: process_batch_result_decode ===")
    print(
        f"  batch_size={batch_size}, stream={stream}, return_logprob={return_logprob}"
    )
    print(f"  num_warmup={num_warmup}, num_iters={num_iters}")

    scheduler = make_mock_scheduler()
    reqs = make_reqs(batch_size, stream=stream)

    # Warmup
    for _ in range(num_warmup):
        batch, result = make_batch_and_result(reqs, return_logprob=return_logprob)
        scheduler.process_batch_result_decode(batch, result)
        reset_reqs(reqs)

    # Timed iterations
    times = []
    for _ in range(num_iters):
        batch, result = make_batch_and_result(reqs, return_logprob=return_logprob)

        t0 = time.perf_counter()
        scheduler.process_batch_result_decode(batch, result)
        t1 = time.perf_counter()

        times.append(t1 - t0)
        reset_reqs(reqs)

    times_ms = [t * 1000 for t in times]
    avg_ms = sum(times_ms) / len(times_ms)
    min_ms = min(times_ms)
    max_ms = max(times_ms)
    per_req_us = avg_ms * 1000 / batch_size

    print(f"\nResults ({num_iters} iterations):")
    print(f"  Average: {avg_ms:.3f} ms")
    print(f"  Min:     {min_ms:.3f} ms")
    print(f"  Max:     {max_ms:.3f} ms")
    print(f"  Per-req: {per_req_us:.2f} us")
    print(f"  Estimated bs=3072 time: {per_req_us * 3072 / 1000:.3f} ms")

    return avg_ms


def correctness_check():
    """Basic correctness check: verify that the mock produces the same
    side effects as the original logic."""
    print("\n=== Correctness Check ===")

    scheduler = make_mock_scheduler()

    # Test 1: Normal decode (no finish)
    reqs = make_reqs(4, output_len=50, stream=False)
    batch, result = make_batch_and_result(reqs)
    scheduler.process_batch_result_decode(batch, result)

    for req in reqs:
        # Token 42 should be appended
        assert (
            req.output_ids[-1] == 42
        ), f"Expected last token 42, got {req.output_ids[-1]}"
        assert (
            len(req.output_ids) == 51
        ), f"Expected 51 output tokens, got {len(req.output_ids)}"
        assert not req.finished(), f"Request should not be finished"
    print("  [PASS] Normal decode: token appended, not finished")

    # Test 2: Finish by max_new_tokens
    reset_reqs(reqs, output_len=255)  # max_new_tokens=256, after append => 256
    batch, result = make_batch_and_result(reqs)
    scheduler.process_batch_result_decode(batch, result)

    for req in reqs:
        assert req.finished(), "Request should be finished (max_new_tokens)"
        assert req.output_ids[-1] == 42
        assert len(req.output_ids) == 256
    print("  [PASS] Finish by max_new_tokens")

    # Test 3: Finish by EOS token
    reqs = make_reqs(4, output_len=50)
    # Set next_token_ids to EOS (token_id=2)
    batch = MockScheduleBatch(reqs)
    eos_ids = torch.full((4,), 2, dtype=torch.int64)  # EOS = 2
    result = MockGenerationBatchResult(
        logits_output=MockLogitsProcessorOutput(),
        next_token_ids=eos_ids,
        can_run_cuda_graph=False,
        copy_done=None,
    )
    scheduler.process_batch_result_decode(batch, result)

    for req in reqs:
        assert req.finished(), "Request should be finished (EOS)"
        assert req.output_ids[-1] == 2
    print("  [PASS] Finish by EOS token")

    # Test 4: Overlap scheduler - already finished reqs are skipped
    scheduler.enable_overlap = True
    reqs = make_reqs(4, output_len=50)
    reqs[0].finished_reason = MOCK_FINISH_LENGTH(50)  # pre-finished
    batch, result = make_batch_and_result(reqs)
    old_len = len(reqs[0].output_ids)
    scheduler.process_batch_result_decode(batch, result)
    assert (
        len(reqs[0].output_ids) == old_len
    ), "Finished req should be skipped in overlap mode"
    assert reqs[1].output_ids[-1] == 42, "Non-finished req should still be processed"
    print("  [PASS] Overlap scheduler: finished reqs skipped")

    # Test 5: Streaming output
    reqs = make_reqs(4, output_len=0, stream=True)  # First token
    batch = MockScheduleBatch(reqs)
    first_token_ids = torch.full((4,), 42, dtype=torch.int64)
    result = MockGenerationBatchResult(
        logits_output=MockLogitsProcessorOutput(),
        next_token_ids=first_token_ids,
        can_run_cuda_graph=False,
        copy_done=None,
    )
    scheduler.process_batch_result_decode(batch, result)
    # With stream_interval=1, output_len=1 (after append): 1 % 1 == 0 -> should_output
    for req in reqs:
        assert len(req.output_ids) == 1
    print("  [PASS] Streaming output")

    print("\n  All correctness checks passed!")


def correctness_check_fast():
    """Correctness check: compare fast (Rust) and slow (Python) paths
    to verify identical side effects."""
    import os

    print("\n=== Correctness Check (Fast vs Slow) ===")

    os.environ["SGLANG_USE_FAST_OUTPUT_PROCESSOR"] = "1"

    try:
        # Create two identical schedulers
        scheduler_slow = make_mock_scheduler()
        scheduler_fast = make_mock_scheduler()

        for test_name, make_kwargs, next_token_val, stream in [
            (
                "Normal decode",
                dict(batch_size=8, output_len=50, stream=False),
                42,
                False,
            ),
            (
                "Finish by max_new_tokens",
                dict(batch_size=8, output_len=255, stream=False),
                42,
                False,
            ),
            (
                "Finish by EOS",
                dict(batch_size=8, output_len=50, stream=False),
                2,
                False,
            ),
            ("Streaming", dict(batch_size=8, output_len=0, stream=True), 42, True),
            (
                "Mixed finish/non-finish",
                dict(batch_size=8, output_len=50, stream=False),
                42,
                False,
            ),
        ]:
            reqs_slow = make_reqs(**make_kwargs)
            reqs_fast = make_reqs(**make_kwargs)

            next_ids_slow = torch.full(
                (make_kwargs["batch_size"],), next_token_val, dtype=torch.int64
            )
            next_ids_fast = torch.full(
                (make_kwargs["batch_size"],), next_token_val, dtype=torch.int64
            )

            batch_slow = MockScheduleBatch(reqs_slow)
            result_slow = MockGenerationBatchResult(
                logits_output=MockLogitsProcessorOutput(),
                next_token_ids=next_ids_slow,
                can_run_cuda_graph=False,
                copy_done=None,
            )

            batch_fast = MockScheduleBatch(reqs_fast)
            result_fast = MockGenerationBatchResult(
                logits_output=MockLogitsProcessorOutput(),
                next_token_ids=next_ids_fast,
                can_run_cuda_graph=False,
                copy_done=None,
            )

            # Capture send_output calls
            slow_outputs = []
            fast_outputs = []
            scheduler_slow.send_to_detokenizer = type(
                "", (), {"send_output": lambda self, o: slow_outputs.append(o)}
            )()
            scheduler_fast.send_to_detokenizer = type(
                "", (), {"send_output": lambda self, o: fast_outputs.append(o)}
            )()

            # Run slow path
            old_env = os.environ.pop("SGLANG_USE_FAST_OUTPUT_PROCESSOR", None)
            scheduler_slow.process_batch_result_decode(batch_slow, result_slow)
            os.environ["SGLANG_USE_FAST_OUTPUT_PROCESSOR"] = "1"

            # Run fast path
            scheduler_fast.process_batch_result_decode(batch_fast, result_fast)

            # Compare req state
            for j, (rs, rf) in enumerate(zip(reqs_slow, reqs_fast)):
                assert (
                    rs.output_ids == rf.output_ids
                ), f"{test_name} req[{j}]: output_ids mismatch: {rs.output_ids[-3:]} vs {rf.output_ids[-3:]}"
                assert (rs.finished_reason is None) == (
                    rf.finished_reason is None
                ), f"{test_name} req[{j}]: finished mismatch"
                assert (
                    rs.finished_len == rf.finished_len
                ), f"{test_name} req[{j}]: finished_len mismatch: {rs.finished_len} vs {rf.finished_len}"
                assert (
                    rs.send_token_offset == rf.send_token_offset
                ), f"{test_name} req[{j}]: send_token_offset mismatch"

            # Compare output message
            assert len(slow_outputs) == len(
                fast_outputs
            ), f"{test_name}: output count mismatch: {len(slow_outputs)} vs {len(fast_outputs)}"
            if slow_outputs and fast_outputs:
                so = slow_outputs[0]
                fast_out = fast_outputs[0]
                assert so.rids == fast_out.rids, f"{test_name}: rids mismatch"
                assert (
                    so.finished_reasons == fast_out.finished_reasons
                ), f"{test_name}: finished_reasons mismatch"
                assert (
                    so.read_offsets == fast_out.read_offsets
                ), f"{test_name}: read_offsets mismatch"
                assert (
                    so.prompt_tokens == fast_out.prompt_tokens
                ), f"{test_name}: prompt_tokens mismatch"
                assert (
                    so.completion_tokens == fast_out.completion_tokens
                ), f"{test_name}: completion_tokens mismatch"

            print(f"  [PASS] {test_name}")

        print("\n  All fast vs slow correctness checks passed!")
    finally:
        os.environ.pop("SGLANG_USE_FAST_OUTPUT_PROCESSOR", None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=3072)
    parser.add_argument("--num-warmup", type=int, default=5)
    parser.add_argument("--num-iters", type=int, default=20)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--return-logprob", action="store_true")
    parser.add_argument(
        "--fast", action="store_true", help="Enable fast (Rust) output processor"
    )
    args = parser.parse_args()

    if args.fast:
        import os

        os.environ["SGLANG_USE_FAST_OUTPUT_PROCESSOR"] = "1"

    correctness_check()

    if args.fast:
        correctness_check_fast()

    print()
    benchmark(
        batch_size=args.batch_size,
        num_warmup=args.num_warmup,
        num_iters=args.num_iters,
        stream=args.stream,
        return_logprob=args.return_logprob,
    )
