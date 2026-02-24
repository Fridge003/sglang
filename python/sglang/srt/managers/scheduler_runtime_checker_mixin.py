from __future__ import annotations

import logging
import time
import warnings
from typing import TYPE_CHECKING

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.utils.common import ceil_align, raise_error_or_warn
from sglang.srt.utils.request_logger import disable_request_logging
from sglang.srt.utils.watchdog import WatchdogRaw

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


class SchedulerRuntimeCheckerMixin:
    def _get_token_info(self: Scheduler):
        available_size = self.token_to_kv_pool_allocator.available_size()
        evictable_size = self.tree_cache.evictable_size()
        num_used = self.max_total_num_tokens - (available_size + evictable_size)
        token_usage = num_used / self.max_total_num_tokens
        return num_used, token_usage, available_size, evictable_size

    def _get_mamba_token_info(self: Scheduler):
        is_mamba_radix_cache = (
            self.tree_cache.supports_mamba() and self.tree_cache.is_tree_cache()
        )
        full_available_size = self.token_to_kv_pool_allocator.available_size()
        full_evictable_size = (
            self.tree_cache.full_evictable_size() if is_mamba_radix_cache else 0
        )
        mamba_available_size = self.req_to_token_pool.mamba_pool.available_size()
        mamba_evictable_size = (
            self.tree_cache.mamba_evictable_size() if is_mamba_radix_cache else 0
        )
        full_num_used = self.token_to_kv_pool_allocator.size - (
            full_available_size + full_evictable_size
        )
        mamba_num_used = self.req_to_token_pool.mamba_pool.size - (
            mamba_available_size + mamba_evictable_size
        )
        full_token_usage = full_num_used / self.token_to_kv_pool_allocator.size
        mamba_usage = mamba_num_used / self.req_to_token_pool.mamba_pool.size
        return (
            full_num_used,
            mamba_num_used,
            full_token_usage,
            mamba_usage,
            full_available_size,
            full_evictable_size,
            mamba_available_size,
            mamba_evictable_size,
        )

    def _get_swa_token_info(self: Scheduler):
        full_available_size = self.token_to_kv_pool_allocator.full_available_size()
        full_evictable_size = self.tree_cache.full_evictable_size()
        swa_available_size = self.token_to_kv_pool_allocator.swa_available_size()
        swa_evictable_size = self.tree_cache.swa_evictable_size()
        full_num_used = self.full_tokens_per_layer - (
            full_available_size + full_evictable_size
        )
        swa_num_used = self.swa_tokens_per_layer - (
            swa_available_size + swa_evictable_size
        )
        full_token_usage = full_num_used / self.full_tokens_per_layer
        swa_token_usage = swa_num_used / self.swa_tokens_per_layer
        return (
            full_num_used,
            swa_num_used,
            full_token_usage,
            swa_token_usage,
            full_available_size,
            full_evictable_size,
            swa_available_size,
            swa_evictable_size,
        )

    def _check_hybrid_memory(self: Scheduler):
        (
            full_num_used,
            swa_num_used,
            _,
            _,
            full_available_size,
            full_evictable_size,
            swa_available_size,
            swa_evictable_size,
        ) = self._get_swa_token_info()
        memory_leak = full_num_used != 0 or swa_num_used != 0
        token_msg = (
            f"{self.full_tokens_per_layer=}, {full_available_size=}, {full_evictable_size=}, {self.tree_cache.full_protected_size()=}\n"
            f"{self.swa_tokens_per_layer=}, {swa_available_size=}, {swa_evictable_size=}, {self.tree_cache.swa_protected_size()=}\n"
        )
        return memory_leak, token_msg

    def _check_mamba_memory(self: Scheduler):
        (
            full_num_used,
            mamba_num_used,
            _,
            _,
            full_available_size,
            full_evictable_size,
            mamba_available_size,
            mamba_evictable_size,
        ) = self._get_mamba_token_info()
        memory_leak = (
            full_num_used != self.tree_cache.full_protected_size()
            or mamba_num_used != self.tree_cache.mamba_protected_size()
        )
        if memory_leak:
            free_full_pages = set(
                self.token_to_kv_pool_allocator.free_pages.tolist()
                + self.token_to_kv_pool_allocator.release_pages.tolist()
            )
            cached_full_pages = set(self.tree_cache.all_values_flatten().tolist())
            expected_full_pages = set(
                range(1, self.token_to_kv_pool_allocator.size + 1)
            )
            leaked_full_pages = (
                expected_full_pages - free_full_pages - cached_full_pages
            )
            free_mamba_pages = set(
                self.req_to_token_pool.mamba_pool.free_slots.tolist()
            )
            cached_mamba_pages = set(
                self.tree_cache.all_mamba_values_flatten().tolist()
            )
            expected_mamba_pages = set(range(self.req_to_token_pool.mamba_pool.size))
            leaked_mamba_pages = (
                expected_mamba_pages - free_mamba_pages - cached_mamba_pages
            )
            token_msg = (
                f"{full_available_size=}, {full_evictable_size=}, {self.token_to_kv_pool_allocator.size=}, {self.tree_cache.full_protected_size()=}\n"
                f"{mamba_available_size=}, {mamba_evictable_size=}, {self.req_to_token_pool.mamba_pool.size=}, {self.tree_cache.mamba_protected_size()=}, leaked_full_pages={leaked_full_pages if len(leaked_full_pages) > 0 else None}, leaked_mamba_pages={leaked_mamba_pages if len(leaked_mamba_pages) > 0 else None}\n"
            )
        else:
            token_msg = (
                f"{full_available_size=}, {full_evictable_size=}, {self.token_to_kv_pool_allocator.size=}, {self.tree_cache.full_protected_size()=}\n"
                f"{mamba_available_size=}, {mamba_evictable_size=}, {self.req_to_token_pool.mamba_pool.size=}, {self.tree_cache.mamba_protected_size()=}\n"
            )
        return memory_leak, token_msg

    def _check_radix_cache_memory(self: Scheduler):
        _, _, available_size, evictable_size = self._get_token_info()
        protected_size = self.tree_cache.protected_size()
        expected = self.max_total_num_tokens - protected_size
        actual = available_size + evictable_size
        # With paged allocation (page_size > 1), partial pages in the radix
        # tree cause a token-level mismatch of up to page_size per partial
        # page.  Allow a tolerance proportional to the number of tree pages
        # that could be partially filled.
        tolerance = 0
        if self.page_size > 1:
            tree_tokens = evictable_size + protected_size
            # Worst case: every page_size-aligned boundary holds a partial page
            max_partial_pages = (tree_tokens + self.page_size - 1) // self.page_size
            tolerance = max_partial_pages * (self.page_size - 1)
        memory_leak = abs(actual - expected) > tolerance
        token_msg = f"{self.max_total_num_tokens=}, {available_size=}, {evictable_size=}, {protected_size=}\n"
        import logging as _logging
        import time as _time
        _now = _time.monotonic()
        if not hasattr(self, '_last_idle_chk') or _now - self._last_idle_chk > 10:
            self._last_idle_chk = _now
            _logging.getLogger(__name__).warning(
                f"[IDLE-CHK] leak={memory_leak} avail={available_size} evict={evictable_size} prot={protected_size} max={self.max_total_num_tokens}"
            )
        if memory_leak:
            import torch as _torch
            tree_total = self.tree_cache.total_size()
            alloc = self.token_to_kv_pool_allocator
            n_free = len(alloc.free_pages)
            n_release = len(alloc.release_pages)
            ps = alloc.page_size

            # Collect all page indices from tree nodes
            tree_pages = set()
            stack = [self.tree_cache.root_node]
            while stack:
                nd = stack.pop()
                if nd.value is not None and len(nd.value) > 0:
                    for idx in (nd.value // ps).tolist():
                        tree_pages.add(idx)
                for ch in nd.children.values():
                    if not ch.evicted:
                        stack.append(ch)
            n_tree_pages = len(tree_pages)

            # Collect free + release page indices
            free_set = set()
            if n_free > 0:
                free_set.update(alloc.free_pages.tolist())
            if n_release > 0:
                free_set.update(alloc.release_pages.tolist())
            n_free_unique = len(free_set)

            overlap = tree_pages & free_set
            total_accounted = n_tree_pages + n_free_unique - len(overlap)
            num_pages = alloc.num_pages  # excludes page 0 (padding)
            leaked = num_pages - total_accounted

            # Find leaked page indices (sample up to 20)
            all_pages = set(range(1, num_pages + 1))
            leaked_pages = all_pages - tree_pages - free_set
            sample = sorted(leaked_pages)[:20]

            # Check req_to_token_pool for leaked pages
            rtp = self.req_to_token_pool
            leaked_in_rtp = 0
            leaked_in_rtp_slots = []
            if leaked_pages:
                leaked_set = leaked_pages
                free_slots_set = set(int(s) for s in rtp.free_slots)
                for slot in range(rtp.size):
                    if slot in free_slots_set:
                        continue
                    slot_indices = rtp.req_to_token[slot].tolist()
                    slot_pages = set(idx // ps for idx in slot_indices if idx >= 0)
                    hit = slot_pages & leaked_set
                    if hit:
                        leaked_in_rtp += len(hit)
                        if len(leaked_in_rtp_slots) < 5:
                            leaked_in_rtp_slots.append((slot, len(slot_pages), sorted(hit)[:5]))

            token_msg += (
                f"  DIAG: tree_total={tree_total}, counter_drift={tree_total - (evictable_size + protected_size)}, "
                f"tree_pages={n_tree_pages}, free_pages={n_free}+{n_release}={n_free_unique}u, "
                f"overlap={len(overlap)}, total_accounted={total_accounted}, "
                f"num_pages={num_pages}, leaked={leaked}, "
                f"leaked_in_rtp={leaked_in_rtp}, sample={sample}, "
                f"rtp_slots={leaked_in_rtp_slots}\n"
            )
        return memory_leak, token_msg

    def _get_batch_uncached_size(self: Scheduler, batch: ScheduleBatch) -> int:
        ret = 0
        for req in batch.reqs:
            assert req.kv_committed_freed == req.kv_overallocated_freed
            uncached_len = 0
            if not req.kv_committed_freed:
                allocated_len = req.kv_allocated_len
                if self.page_size > 1:
                    allocated_len = ceil_align(allocated_len, self.page_size)
                    assert req.cache_protected_len % self.page_size == 0
                uncached_len = allocated_len - req.cache_protected_len

            ret += uncached_len

        return ret

    def self_check_during_busy(self: Scheduler):
        current_batch: ScheduleBatch = self.last_batch

        if current_batch is None:
            return

        spec_topk = self.server_args.speculative_eagle_topk or 1
        if spec_topk > 1:
            warnings.warn(
                "Runtime memory check (busy) is not supported when speculation topk > 1."
            )
            return

        _, _, available_size, evictable_size = self._get_token_info()
        protected_size = self.tree_cache.protected_size()

        uncached_size = self._get_batch_uncached_size(current_batch)

        if (
            current_batch.forward_mode.is_extend()
            and self.running_batch is not None
            and not self.running_batch.is_empty()
        ):
            uncached_size += self._get_batch_uncached_size(self.running_batch)

        if envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get() > 1:
            log_msg = f"[Mem Check (BUSY)] {available_size=}, {evictable_size=}, {protected_size=}, {uncached_size=}"
            logger.info(log_msg)

        total_tokens = available_size + evictable_size + protected_size + uncached_size
        assert (
            total_tokens == self.max_total_num_tokens
        ), f"Mem Leak Detected! {total_tokens=} vs {self.max_total_num_tokens=}"

    def _check_req_pool(self: Scheduler):
        if self.disaggregation_mode == DisaggregationMode.DECODE:
            req_total_size = (
                self.req_to_token_pool.size + self.req_to_token_pool.pre_alloc_size
            )
        else:
            req_total_size = self.req_to_token_pool.size

        if len(self.req_to_token_pool.free_slots) != req_total_size:
            msg = (
                "req_to_token_pool memory leak detected!"
                f"available_size={len(self.req_to_token_pool.free_slots)}, "
                f"total_size={self.req_to_token_pool.size}\n"
            )
            raise_error_or_warn(
                self,
                envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE.get(),
                "count_req_pool_leak_warnings",
                msg,
            )

    def check_memory(self: Scheduler):
        if self.is_hybrid_swa:
            memory_leak, token_msg = self._check_hybrid_memory()
        elif self.is_hybrid_ssm and self.tree_cache.supports_mamba():
            memory_leak, token_msg = self._check_mamba_memory()
        else:
            memory_leak, token_msg = self._check_radix_cache_memory()

        if memory_leak:
            msg = "token_to_kv_pool_allocator memory leak detected! " f"{token_msg}"
            raise_error_or_warn(
                self,
                envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE.get(),
                "count_memory_leak_warnings",
                msg,
            )

        self._check_req_pool()

        if (
            self.enable_metrics
            and self.current_scheduler_metrics_enabled
            and time.perf_counter() > self.metrics_collector.last_log_time + 30
        ):
            # During idle time, also collect metrics every 30 seconds.
            if self.is_hybrid_swa:
                (
                    full_num_used,
                    swa_num_used,
                    full_token_usage,
                    swa_token_usage,
                    _,
                    _,
                    _,
                    _,
                ) = self._get_swa_token_info()
                num_used = max(full_num_used, swa_num_used)
                token_usage = max(full_token_usage, swa_token_usage)
            elif self.is_hybrid_ssm:
                (
                    num_used,
                    _,
                    token_usage,
                    _,
                    _,
                    _,
                    _,
                    _,
                ) = self._get_mamba_token_info()
            else:
                num_used, token_usage, _, _ = self._get_token_info()
            num_running_reqs = len(self.running_batch.reqs)
            self.stats.num_running_reqs = num_running_reqs
            self.stats.num_used_tokens = num_used
            self.stats.token_usage = round(token_usage, 2)
            self.stats.gen_throughput = 0
            self.stats.num_queue_reqs = len(self.waiting_queue)
            self.stats.num_grammar_queue_reqs = len(self.grammar_manager)
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                self.stats.num_prefill_prealloc_queue_reqs = len(
                    self.disagg_prefill_bootstrap_queue.queue
                )
                self.stats.num_prefill_inflight_queue_reqs = len(
                    self.disagg_prefill_inflight_queue
                )
            if self.disaggregation_mode == DisaggregationMode.DECODE:
                self.stats.num_decode_prealloc_queue_reqs = len(
                    self.disagg_decode_prealloc_queue.queue
                )
                self.stats.num_decode_transfer_queue_reqs = len(
                    self.disagg_decode_transfer_queue.queue
                )
            self.metrics_collector.log_stats(self.stats)
        self._publish_kv_events()

    def check_tree_cache(self: Scheduler):
        if (
            self.tree_cache.is_tree_cache()
            and (self.is_hybrid_swa and self.tree_cache.supports_swa())
            or (self.is_hybrid_ssm and self.tree_cache.supports_mamba())
        ):
            self.tree_cache.sanity_check()

    def self_check_during_idle(self: Scheduler):
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            if len(self.disagg_prefill_inflight_queue) > 0:
                return
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            # Running requests hold decode pages (from alloc_decode) that are
            # not yet in the radix tree.  The idle memory check formula does not
            # account for these pages, so skip the check when requests are
            # still running to avoid false-positive leak detection.
            if self.running_batch is not None and not self.running_batch.is_empty():
                return
            queue_size = (
                len(self.waiting_queue)
                + len(self.disagg_decode_transfer_queue.queue)
                + len(self.disagg_decode_prealloc_queue.queue)
            )
            if self.server_args.disaggregation_decode_enable_offload_kvcache:
                queue_size += len(self.decode_offload_manager.ongoing_offload)
            if queue_size:
                return

        self.check_memory()
        self.check_tree_cache()
        self.new_token_ratio = self.init_new_token_ratio
        self.maybe_sleep_on_idle()


def create_scheduler_watchdog(
    scheduler: Scheduler, watchdog_timeout: float, soft: bool = False
) -> WatchdogRaw:
    def dump_info() -> str:
        if scheduler.is_initializing or disable_request_logging():
            return ""
        if scheduler.is_hybrid_swa:
            _, info_msg = scheduler._check_hybrid_memory()
        elif scheduler.is_hybrid_ssm and scheduler.tree_cache.supports_mamba():
            _, info_msg = scheduler._check_mamba_memory()
        else:
            _, info_msg = scheduler._check_radix_cache_memory()
        return (
            f"{scheduler.cur_batch.batch_size()=}\n"
            f"{scheduler.cur_batch.reqs=}\n"
            f"{info_msg}"
        )

    return WatchdogRaw(
        debug_name="Scheduler",
        get_counter=lambda: getattr(scheduler, "forward_ct", 0),
        is_active=lambda: scheduler.is_initializing
        or getattr(scheduler, "cur_batch", None) is not None,
        watchdog_timeout=watchdog_timeout,
        soft=soft,
        dump_info=dump_info,
    )
