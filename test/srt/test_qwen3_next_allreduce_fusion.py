import unittest
from unittest.mock import patch

import torch

from sglang.srt.models import qwen3_next


class FakeSparseMoe:
    def __init__(self):
        self.calls = []

    def __call__(
        self,
        hidden_states,
        forward_batch,
        use_reduce_scatter,
        should_allreduce_fusion,
    ):
        self.calls.append(
            (forward_batch, use_reduce_scatter, should_allreduce_fusion)
        )
        return hidden_states + 1


class FakeCommunicator:
    def __init__(self, should_fuse, use_reduce_scatter=False):
        self.should_fuse = should_fuse
        self.use_reduce_scatter = use_reduce_scatter
        self.postprocess_called = False

    def prepare_mlp(self, hidden_states, residual, forward_batch):
        return hidden_states + 2, residual

    def should_use_reduce_scatter(self, forward_batch):
        return self.use_reduce_scatter

    def should_fuse_mlp_allreduce_with_next_layer(self, forward_batch):
        return self.should_fuse

    def postprocess_layer(self, hidden_states, residual, forward_batch):
        self.postprocess_called = True
        return hidden_states + 3, residual


class FakeLayer:
    def __init__(self, should_fuse, use_reduce_scatter=False):
        self.layer_communicator = FakeCommunicator(
            should_fuse=should_fuse,
            use_reduce_scatter=use_reduce_scatter,
        )
        self.mlp = FakeSparseMoe()


class TestQwen3NextAllReduceFusion(unittest.TestCase):
    def test_marks_moe_output_for_next_layer_fusion(self):
        layer = FakeLayer(should_fuse=True, use_reduce_scatter=False)
        hidden_states = torch.zeros((2, 4), dtype=torch.float32)
        forward_batch = object()

        with patch.object(qwen3_next, "Qwen2MoeSparseMoeBlock", FakeSparseMoe):
            output, residual = qwen3_next._apply_qwen3_next_mlp(
                layer, hidden_states, None, forward_batch
            )

        self.assertIsNone(residual)
        self.assertTrue(output._sglang_needs_allreduce_fusion)
        self.assertFalse(layer.layer_communicator.postprocess_called)
        self.assertEqual(layer.mlp.calls, [(forward_batch, False, True)])

    def test_non_fusion_path_postprocesses_moe_output(self):
        layer = FakeLayer(should_fuse=False, use_reduce_scatter=True)
        hidden_states = torch.zeros((2, 4), dtype=torch.float32)
        forward_batch = object()

        with patch.object(qwen3_next, "Qwen2MoeSparseMoeBlock", FakeSparseMoe):
            output, residual = qwen3_next._apply_qwen3_next_mlp(
                layer, hidden_states, None, forward_batch
            )

        self.assertIsNone(residual)
        self.assertFalse(hasattr(output, "_sglang_needs_allreduce_fusion"))
        self.assertTrue(layer.layer_communicator.postprocess_called)
        self.assertEqual(layer.mlp.calls, [(forward_batch, True, False)])
        self.assertTrue(torch.equal(output, torch.full((2, 4), 6.0)))


if __name__ == "__main__":
    unittest.main()
