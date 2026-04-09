import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.observability.decode_observability import DecodeObservabilityState


class TestDecodeObservabilityState(unittest.TestCase):
    def _build_scheduler(self):
        req = SimpleNamespace(
            rid="rid-1",
            origin_input_ids=[1, 2, 3],
            output_ids=[4, 5],
            fill_ids=[1, 2, 3, 4, 5],
            prefix_indices=[10, 11],
            cache_protected_len=2,
            kv_committed_len=4,
            kv_allocated_len=5,
        )
        prealloc_entry = SimpleNamespace(req=req)
        scheduler = SimpleNamespace(
            forward_ct_decode=8,
            disaggregation_mode=DisaggregationMode.DECODE,
            waiting_queue=[req],
            running_batch=SimpleNamespace(reqs=[req]),
            disagg_decode_prealloc_queue=SimpleNamespace(
                queue=[prealloc_entry],
                retracted_queue=[],
                num_tokens_pre_allocated=32,
            ),
            disagg_decode_transfer_queue=SimpleNamespace(queue=[prealloc_entry]),
            tree_cache=SimpleNamespace(protected_size=lambda: 16),
            max_total_num_tokens=128,
            _get_token_info=lambda: (80, 0.625, 32, 16),
            _session_held_tokens=lambda: 0,
        )
        return scheduler, req

    def test_basic_capture_without_request_samples(self):
        scheduler, _ = self._build_scheduler()
        state = DecodeObservabilityState(
            level="basic", sample_interval=10, request_sample_size=2
        )
        state.capture_iteration(scheduler, scheduler.running_batch)
        self.assertIsNotNone(state.last_snapshot)
        self.assertEqual(state.last_snapshot.running_reqs, 1)
        self.assertEqual(state.last_snapshot.prealloc_reqs, 1)
        self.assertEqual(state.last_snapshot.transfer_reqs, 1)
        self.assertEqual(state.last_snapshot.request_samples, [])

    def test_debug_capture_with_request_samples(self):
        scheduler, req = self._build_scheduler()
        state = DecodeObservabilityState(
            level="debug", sample_interval=4, request_sample_size=2
        )
        state.capture_iteration(scheduler, scheduler.running_batch)
        self.assertIsNotNone(state.last_snapshot)
        self.assertGreaterEqual(len(state.last_snapshot.request_samples), 1)
        sample = state.last_snapshot.request_samples[0]
        self.assertEqual(sample.rid, req.rid)
        self.assertEqual(sample.cache_protected_len, req.cache_protected_len)

    def test_record_invariant_breach_uses_fresh_snapshot(self):
        scheduler, _ = self._build_scheduler()
        state = DecodeObservabilityState(
            level="invariant", sample_interval=100, request_sample_size=1
        )
        breach = state.record_invariant_breach(
            scheduler,
            name="token_leak",
            message="available_size=1",
        )
        self.assertIsNotNone(breach)
        self.assertEqual(breach.name, "token_leak")
        self.assertEqual(breach.snapshot.iteration, scheduler.forward_ct_decode)
        self.assertIs(state.last_breach, breach)


if __name__ == "__main__":
    unittest.main()
