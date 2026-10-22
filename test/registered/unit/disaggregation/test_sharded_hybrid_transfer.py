# SPDX-License-Identifier: Apache-2.0
"""Page-striped DSA and TP-sharded draft KV must land in separate address spaces."""

import concurrent.futures
import ctypes
import unittest
from types import SimpleNamespace

import numpy as np

from sglang.srt.disaggregation.base.conn import KVPoll, StateType
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestShardedHybridTransfer(CustomTestCase):
    def setUp(self):
        from sglang.srt.runtime_context import create_context, use_context
        from sglang.srt.server_args import ServerArgs

        self.enterContext(
            use_context(create_context(ServerArgs(model_path="dummy"), role="test"))
        )

    def test_all_dsa_owners_and_only_matching_draft_heads(self):
        self._check_kv_transfer([2, 5, 6], [7, 2, 4])

    def test_empty_dsa_owner_still_transfers_its_draft_heads(self):
        self._check_kv_transfer([5], [7])

    def _check_kv_transfer(self, pages, dst_pages):
        # Ordinary TP draft rows deliberately differ by rank. A DSA-only chunk
        # on rank 1 still requires rank 0's complete draft chunk to reach D0.
        pages = np.array(pages, dtype=np.int32)
        dst_pages = np.array(dst_pages, dtype=np.int32)
        destinations = [np.zeros((3, 12, 8), dtype=np.uint8) for _ in range(2)]
        sources = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for rank in range(2):
                src = np.zeros((3, 12, 8), dtype=np.uint8)
                sources.append(src)
                for page in pages:
                    if page % 2 == rank:
                        src[0, page // 2] = page + 10
                    src[1, page] = rank * 40 + page + 30
                    src[2, page] = rank * 40 + page + 50
                manager = MooncakeKVManager.__new__(MooncakeKVManager)
                manager.kv_shard_rank = rank
                manager.kv_shard_size = manager.state_tp_size = 2
                manager.is_hybrid_mla_backend = True
                manager.is_mla_backend = False
                manager.pp_size = 1
                manager.enable_custom_mem_pool = False
                manager.max_transfer_batch_indices = 0
                manager.enable_deferred_decode_kv_release = False
                manager.kv_args = SimpleNamespace(
                    engine_rank=rank,
                    num_draft_entries=2,
                    kv_data_ptrs=[row.ctypes.data for row in src],
                    kv_item_lens=[8] * 3,
                    kv_layer_ids=[3, 45, 45],
                )

                def transfer(session, blocks):
                    for source, dest, size in blocks:
                        ctypes.memmove(dest, source, size)
                    return 0

                manager._transfer_data = transfer
                for dst_rank, dst in enumerate(destinations):
                    self.assertEqual(
                        manager._send_sharded_hybrid_kvcache(
                            mooncake_session_id="test",
                            prefill_kv_indices=pages,
                            dst_kv_ptrs=[row.ctypes.data for row in dst],
                            dst_kv_indices=dst_pages,
                            executor=executor,
                            dst_layer_ids=[3, 45, 45],
                            dst_tp_rank=dst_rank,
                            dst_attn_tp_size=2,
                        ),
                        0,
                    )
        for rank, dst in enumerate(destinations):
            for page, dst_page in zip(pages, dst_pages):
                np.testing.assert_array_equal(dst[0, dst_page], np.full(8, page + 10))
                np.testing.assert_array_equal(
                    dst[1:, dst_page], sources[rank][1:, page]
                )
            self.assertFalse(dst[:, 0].any())

    def test_index_pages_fan_out_but_kda_and_tail_keep_tp_ownership(self):
        destinations = [np.zeros((3, 12, 8), dtype=np.uint8) for _ in range(2)]
        sources = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for rank in range(2):
                src = np.zeros((3, 12, 8), dtype=np.uint8)
                sources.append(src)
                # DSA index pages have different owners; KDA and compression
                # tails have rank-specific values in a single request slot.
                src[0, (2 + rank) // 2] = 20 + rank
                src[1, 4] = 50 + rank
                src[2, 4] = 80 + rank
                manager = MooncakeKVManager.__new__(MooncakeKVManager)
                manager.kv_shard_rank = rank
                manager.kv_shard_size = manager.state_tp_size = 2
                manager.attn_tp_size = 1
                manager.is_hybrid_mla_backend = True
                manager.is_mla_backend = False
                manager.pp_size = 1
                manager.enable_custom_mem_pool = False
                manager.max_transfer_batch_indices = 0
                manager.enable_deferred_decode_kv_release = False
                manager.request_status = {1: KVPoll.Transferring}
                manager.state_strides_validated = set()
                manager.kv_args = SimpleNamespace(
                    engine_rank=rank,
                    state_types=[StateType.DSA, StateType.MAMBA, StateType.DSA_TAIL],
                    state_data_ptrs=[[row.ctypes.data] for row in src],
                    state_item_lens=[[8]] * 3,
                    state_dim_per_tensor=[[], [], []],
                    state_layer_ids=[[3], [0], [3]],
                    prefill_start_layer=0,
                    prefill_end_layer=45,
                )

                def transfer(session, blocks):
                    for source, dest, size in blocks:
                        ctypes.memmove(dest, source, size)
                    return 0

                manager._transfer_data = transfer
                for dst_rank, dst in enumerate(destinations):
                    peer = SimpleNamespace(
                        dst_tp_rank=dst_rank,
                        dst_attn_tp_size=2,
                        dst_state_data_ptrs=[[row.ctypes.data] for row in dst],
                        dst_state_item_lens=[[8]] * 3,
                        dst_state_dim_per_tensor=[[], [], []],
                        dst_state_layer_ids=[[3], [0], [3]],
                    )
                    request = SimpleNamespace(
                        mooncake_session_id="test",
                        room=1,
                        dst_state_indices=[[6, 7], [9], [9, 0, 4, 0, 0, 4]],
                    )
                    self.assertEqual(
                        manager.maybe_send_extra(
                            request, [[2, 3], [4], [4, 0, 4, 0, 0, 4]], executor, peer
                        ),
                        0,
                    )
        for rank, dst in enumerate(destinations):
            np.testing.assert_array_equal(dst[0, 6], np.full(8, 20))
            np.testing.assert_array_equal(dst[0, 7], np.full(8, 21))
            np.testing.assert_array_equal(dst[1:, 9], sources[rank][1:, 4])
            self.assertFalse(dst[:, 0].any())


if __name__ == "__main__":
    unittest.main()
