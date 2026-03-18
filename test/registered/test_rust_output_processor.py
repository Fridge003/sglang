"""
Tests for the Rust output processor serialization roundtrip.

These tests verify that:
1. BatchTokenIDOutput can be serialized to msgpack and deserialized back
2. BatchEmbeddingOutput can be serialized to msgpack and deserialized back
3. The format prefix bytes are correctly applied
4. The dual-format receiver can handle both pickle and msgpack messages
"""

import pickle
import unittest

import msgspec

from sglang.srt.managers.io_struct import (
    BatchEmbeddingOutput,
    BatchTokenIDOutput,
    GetLoadReqOutput,
)
from sglang.srt.managers.rust_output_processor import (
    FORMAT_PREFIX_MSGPACK_EMBEDDING_PY,
    FORMAT_PREFIX_MSGPACK_TOKEN_PY,
    _serialize_batch_embedding_output,
    _serialize_batch_token_id_output,
    deserialize_rust_output,
)


class TestSerializationRoundtrip(unittest.TestCase):
    """Test msgpack serialization roundtrip for batch outputs."""

    def _make_token_output(self, with_logprobs=False):
        """Create a BatchTokenIDOutput for testing."""
        load = GetLoadReqOutput(
            rid=None,
            dp_rank=0,
            num_reqs=2,
            num_waiting_reqs=0,
            num_tokens=100,
            ts_tic=1.0,
        )

        logprob_fields = {}
        if with_logprobs:
            logprob_fields = {
                "input_token_logprobs_val": [[-0.5, -1.0], [-0.3]],
                "input_token_logprobs_idx": [[1, 2], [3]],
                "output_token_logprobs_val": [[-0.1], [-0.2, -0.4]],
                "output_token_logprobs_idx": [[10], [20, 30]],
                "input_top_logprobs_val": [[], []],
                "input_top_logprobs_idx": [[], []],
                "output_top_logprobs_val": [[], []],
                "output_top_logprobs_idx": [[], []],
                "input_token_ids_logprobs_val": [[], []],
                "input_token_ids_logprobs_idx": [[], []],
                "output_token_ids_logprobs_val": [[], []],
                "output_token_ids_logprobs_idx": [[], []],
            }
        else:
            logprob_fields = {
                "input_token_logprobs_val": None,
                "input_token_logprobs_idx": None,
                "output_token_logprobs_val": None,
                "output_token_logprobs_idx": None,
                "input_top_logprobs_val": None,
                "input_top_logprobs_idx": None,
                "output_top_logprobs_val": None,
                "output_top_logprobs_idx": None,
                "input_token_ids_logprobs_val": None,
                "input_token_ids_logprobs_idx": None,
                "output_token_ids_logprobs_val": None,
                "output_token_ids_logprobs_idx": None,
            }

        return BatchTokenIDOutput(
            rids=["req-001", "req-002"],
            http_worker_ipcs=[None, "ipc:///tmp/worker-1"],
            finished_reasons=[None, '{"type": "stop", "matched": "\\n"}'],
            decoded_texts=["Hello", "World"],
            decode_ids=[[1, 2, 3], [4, 5]],
            read_offsets=[0, 2],
            output_ids=[[10, 20], [30]],
            skip_special_tokens=[True, True],
            spaces_between_special_tokens=[True, False],
            no_stop_trim=[False, False],
            prompt_tokens=[5, 10],
            completion_tokens=[2, 1],
            cached_tokens=[3, 7],
            cached_tokens_details=[None, {"device": 5, "host": 2}],
            spec_verify_ct=[],
            spec_accepted_tokens=[],
            spec_acceptance_histogram=[],
            retraction_counts=[0, 1],
            load=load,
            dp_ranks=[0, 0],
            output_token_entropy_val=None,
            output_hidden_states=None,
            routed_experts=None,
            customized_info={},
            placeholder_tokens_idx=None,
            placeholder_tokens_val=None,
            time_stats=None,
            token_steps=None,
            **logprob_fields,
        )

    def _make_embedding_output(self):
        """Create a BatchEmbeddingOutput for testing."""
        return BatchEmbeddingOutput(
            rids=["emb-001"],
            http_worker_ipcs=[None],
            finished_reasons=['{"type": "stop"}'],
            embeddings=[[0.1, 0.2, 0.3]],
            prompt_tokens=[5],
            cached_tokens=[2],
            cached_tokens_details=[None],
            retraction_counts=[0],
            placeholder_tokens_idx=None,
            placeholder_tokens_val=None,
            time_stats=None,
        )

    def test_token_output_roundtrip_no_logprobs(self):
        """Test BatchTokenIDOutput roundtrip without logprobs."""
        original = self._make_token_output(with_logprobs=False)
        data = _serialize_batch_token_id_output(original)

        # Simulate what Rust does: prepend format prefix
        msg = bytes([FORMAT_PREFIX_MSGPACK_TOKEN_PY]) + data
        result = deserialize_rust_output(msg)

        self.assertIsInstance(result, BatchTokenIDOutput)
        self.assertEqual(result.rids, original.rids)
        self.assertEqual(result.http_worker_ipcs, original.http_worker_ipcs)
        self.assertEqual(result.finished_reasons, original.finished_reasons)
        self.assertEqual(result.decoded_texts, original.decoded_texts)
        self.assertEqual(result.decode_ids, original.decode_ids)
        self.assertEqual(result.read_offsets, original.read_offsets)
        self.assertEqual(result.output_ids, original.output_ids)
        self.assertEqual(result.skip_special_tokens, original.skip_special_tokens)
        self.assertEqual(
            result.spaces_between_special_tokens,
            original.spaces_between_special_tokens,
        )
        self.assertEqual(result.no_stop_trim, original.no_stop_trim)
        self.assertEqual(result.prompt_tokens, original.prompt_tokens)
        self.assertEqual(result.completion_tokens, original.completion_tokens)
        self.assertEqual(result.cached_tokens, original.cached_tokens)
        self.assertEqual(result.retraction_counts, original.retraction_counts)
        self.assertEqual(result.dp_ranks, original.dp_ranks)
        self.assertIsNone(result.input_token_logprobs_val)
        self.assertIsNone(result.output_token_logprobs_val)

    def test_token_output_roundtrip_with_logprobs(self):
        """Test BatchTokenIDOutput roundtrip with logprobs."""
        original = self._make_token_output(with_logprobs=True)
        data = _serialize_batch_token_id_output(original)

        msg = bytes([FORMAT_PREFIX_MSGPACK_TOKEN_PY]) + data
        result = deserialize_rust_output(msg)

        self.assertIsInstance(result, BatchTokenIDOutput)
        self.assertEqual(result.input_token_logprobs_val, original.input_token_logprobs_val)
        self.assertEqual(result.input_token_logprobs_idx, original.input_token_logprobs_idx)
        self.assertEqual(result.output_token_logprobs_val, original.output_token_logprobs_val)
        self.assertEqual(result.output_token_logprobs_idx, original.output_token_logprobs_idx)

    def test_embedding_output_roundtrip(self):
        """Test BatchEmbeddingOutput roundtrip."""
        original = self._make_embedding_output()
        data = _serialize_batch_embedding_output(original)

        msg = bytes([FORMAT_PREFIX_MSGPACK_EMBEDDING_PY]) + data
        result = deserialize_rust_output(msg)

        self.assertIsInstance(result, BatchEmbeddingOutput)
        self.assertEqual(result.rids, original.rids)
        self.assertEqual(result.finished_reasons, original.finished_reasons)
        self.assertEqual(result.prompt_tokens, original.prompt_tokens)
        self.assertEqual(result.cached_tokens, original.cached_tokens)
        self.assertEqual(result.retraction_counts, original.retraction_counts)
        # Embeddings may have float precision differences
        for orig_emb, result_emb in zip(original.embeddings, result.embeddings):
            for o, r in zip(orig_emb, result_emb):
                self.assertAlmostEqual(o, r, places=5)

    def test_load_roundtrip(self):
        """Test that GetLoadReqOutput survives serialization."""
        original = self._make_token_output()
        data = _serialize_batch_token_id_output(original)

        msg = bytes([FORMAT_PREFIX_MSGPACK_TOKEN_PY]) + data
        result = deserialize_rust_output(msg)

        self.assertIsNotNone(result.load)
        self.assertEqual(result.load.dp_rank, 0)
        self.assertEqual(result.load.num_reqs, 2)
        self.assertEqual(result.load.num_waiting_reqs, 0)
        self.assertEqual(result.load.num_tokens, 100)

    def test_format_prefix_detection(self):
        """Test that format prefix bytes don't collide with pickle."""
        # Pickle protocol 2+ starts with 0x80 (128)
        # Our prefixes are 0x01 and 0x02 — no collision
        token_output = self._make_token_output()
        pickle_bytes = pickle.dumps(token_output)
        self.assertNotEqual(pickle_bytes[0], FORMAT_PREFIX_MSGPACK_TOKEN_PY)
        self.assertNotEqual(pickle_bytes[0], FORMAT_PREFIX_MSGPACK_EMBEDDING_PY)

    def test_empty_batch_roundtrip(self):
        """Test roundtrip with empty batch (idle batch)."""
        empty_output = BatchTokenIDOutput(
            rids=[],
            http_worker_ipcs=[],
            finished_reasons=[],
            decoded_texts=[],
            decode_ids=[],
            read_offsets=[],
            output_ids=[],
            skip_special_tokens=[],
            spaces_between_special_tokens=[],
            no_stop_trim=[],
            prompt_tokens=[],
            completion_tokens=[],
            cached_tokens=[],
            cached_tokens_details=[],
            spec_verify_ct=[],
            spec_accepted_tokens=[],
            spec_acceptance_histogram=[],
            retraction_counts=[],
            load=None,
            dp_ranks=None,
            input_token_logprobs_val=None,
            input_token_logprobs_idx=None,
            output_token_logprobs_val=None,
            output_token_logprobs_idx=None,
            input_top_logprobs_val=None,
            input_top_logprobs_idx=None,
            output_top_logprobs_val=None,
            output_top_logprobs_idx=None,
            input_token_ids_logprobs_val=None,
            input_token_ids_logprobs_idx=None,
            output_token_ids_logprobs_val=None,
            output_token_ids_logprobs_idx=None,
            output_token_entropy_val=None,
            output_hidden_states=None,
            routed_experts=None,
            customized_info={},
            placeholder_tokens_idx=None,
            placeholder_tokens_val=None,
            time_stats=None,
            token_steps=None,
        )

        data = _serialize_batch_token_id_output(empty_output)
        msg = bytes([FORMAT_PREFIX_MSGPACK_TOKEN_PY]) + data
        result = deserialize_rust_output(msg)

        self.assertEqual(result.rids, [])
        self.assertEqual(result.output_ids, [])

    def test_unknown_prefix_raises(self):
        """Test that unknown format prefix raises ValueError."""
        with self.assertRaises(ValueError):
            deserialize_rust_output(bytes([0xFF, 0x00]))


class TestFeatureGate(unittest.TestCase):
    """Test the feature gating logic."""

    def test_default_disabled(self):
        """Without env var, Rust output should be disabled."""
        import os

        # Save and clear the env var
        old_val = os.environ.pop("SGLANG_USE_RUST_OUTPUT_PROCESSOR", None)
        try:
            # Re-import to get fresh state
            import importlib

            import sglang.srt.managers.rust_output_processor as mod

            importlib.reload(mod)
            self.assertFalse(mod.is_rust_output_enabled())
        finally:
            if old_val is not None:
                os.environ["SGLANG_USE_RUST_OUTPUT_PROCESSOR"] = old_val


if __name__ == "__main__":
    unittest.main()
