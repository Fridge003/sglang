# SPDX-License-Identifier: Apache-2.0
"""KPool4 persisted pages and gathered prefixes match unsharded bytes."""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase, publish_build_topology

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


def check_rank(rank, world, rendezvous):
    from sglang.srt.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
    from sglang.srt.mem_cache.page_interleave import PageShardSpec
    from sglang.srt.mem_cache.page_interleave_pool import PageInterleaveDSATokenToKVPool
    from sglang.srt.runtime_context import get_parallel

    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world))
    torch.cuda.set_device(rank)
    init_distributed_environment(
        world_size=world,
        rank=rank,
        local_rank=rank,
        distributed_init_method=rendezvous,
        backend="nccl",
    )
    publish_build_topology(tp_size=world, attn_cp_size=1, world_rank=rank)
    initialize_model_parallel()
    group = get_parallel().attn_tp_group
    device = group.device
    page = 256
    kwargs = dict(
        page_size=page,
        kv_lora_rank=512,
        qk_rope_head_dim=0,
        dtype=torch.bfloat16,
        layer_num=2,
        device=str(device),
        index_head_dim=128,
        enable_memory_saver=False,
        kv_cache_dim=512,
        index_kpool=4,
        index_kpool_compress=True,
        max_running_requests=4,
    )
    pool = PageInterleaveDSATokenToKVPool(
        8 * page,
        **kwargs,
        shard_group=group,
        shard_spec=PageShardSpec(rank, world, page, world * 4 * page, 4 * page),
    )
    ref = DSATokenToKVPool(10 * world * page, **kwargs)

    def locs(pages, length):
        return (
            torch.tensor(pages, device=device)[:, None] * page
            + torch.arange(page, device=device)
        ).flatten()[:length]

    write_plans = {}

    def write(layer, locations, generation):
        values = (
            ((locations[:, None] + torch.arange(512, device=device) + generation) % 31)
            .bfloat16()
            .unsqueeze(1)
        )
        # The prefix and chunks start on 256-token pages. Only closed pools
        # write compressed keys; an incomplete 4-token tail is not persisted.
        closed = locations.numel() // 4
        key_locs = locations[: closed * 4 : 4] // 4
        keys = (
            (key_locs[:, None] + torch.arange(128, device=device) + generation) % 16
        ).to(torch.float8_e4m3fn)
        scales = torch.ones((closed, 1), device=device, dtype=torch.float32)
        for target in (pool, ref):
            if generation == 3:
                from sglang.srt.layers.attention.dsa.dsa_indexer_kpool import (
                    IndexerKPool,
                )
                from sglang.srt.layers.attention.dsa.kpool_plan import (
                    PoolWriteRows,
                    TailWriteRows,
                )

                indexer = IndexerKPool.__new__(IndexerKPool)
                torch.nn.Module.__init__(indexer)
                indexer.head_dim = 128
                indexer.index_kpool = 4
                indexer.scale_fmt = None
                indexer.index_kpool_compress_ape = torch.zeros((4, 128), device=device)
                zeros = torch.zeros(closed, device=device, dtype=torch.int32)
                empty = torch.empty(0, device=device, dtype=torch.int32)
                plan = SimpleNamespace(
                    writes=PoolWriteRows(
                        req=torch.ones_like(zeros),
                        pool_id=torch.arange(closed, device=device),
                        n_from_tail=zeros,
                        chunk_src=torch.arange(closed, device=device, dtype=torch.int32)
                        * 4,
                        tail_logical_base=zeros,
                        write_loc=key_locs,
                    ),
                    tails=TailWriteRows(empty, empty, empty, empty),
                )
                # Real prefill metadata is shared by every DSA layer. Reuse
                # the plan so this checks the cached owner/scratch mapping
                # against independently written unsharded bytes on the GPU.
                plan = write_plans.setdefault((id(target), generation), plan)
                # Only the thread-local pool lookup is replaced. Compression,
                # quantization and both cache writes execute real CUDA kernels.
                with patch(
                    "sglang.srt.layers.attention.dsa.dsa_indexer_kpool.get_token_to_kv_pool",
                    return_value=target,
                ):
                    indexer._compress_write_extend(
                        key=values[:, 0, :128].contiguous(),
                        gate_score=torch.zeros_like(values[:, 0, :128]),
                        positions=None,
                        forward_batch=SimpleNamespace(
                            seq_lens_cpu=[locations.numel()],
                            extend_seq_lens_cpu=[locations.numel()],
                        ),
                        layer_id=layer,
                        metadata=SimpleNamespace(
                            attn_metadata=SimpleNamespace(kpool_extend_plan=plan)
                        ),
                    )
            else:
                target.set_index_k_scale_buffer(layer, key_locs, keys, scales)
            target.set_mla_kv_buffer(
                SimpleNamespace(layer_id=layer), locations, values, values[..., :0]
            )

    # A prefix spanning all owners, then a page plus an incomplete pooled tail.
    prefix_pages = list(range(world, 2 * world))
    prefix = locs(prefix_pages, world * page)
    for layer in range(2):
        write(layer, prefix, 1)
    for epoch, chunk_pages in enumerate(
        ([3 * world, 3 * world + 1], [4 * world + 1, 4 * world + 2]), start=2
    ):
        chunk = locs(chunk_pages, page + 3)
        row = torch.cat((prefix, chunk))
        table = torch.zeros((2, row.numel()), device=device, dtype=torch.int64)
        table[1] = row
        pool.begin_shard_extend(
            table, torch.tensor([1], device=device), [prefix.numel()], [row.numel()]
        )
        for layer in range(2):
            write(layer, chunk, epoch)
            scratch = pool.get_index_k_with_scale_buffer(layer)
            read_pages = pool._page_pos[
                torch.tensor(prefix_pages + chunk_pages[:1], device=device)
            ].long()
            expected_pages = torch.tensor(prefix_pages + chunk_pages[:1], device=device)
            torch.testing.assert_close(
                scratch[read_pages],
                ref.index_k_with_scale_buffer[layer][expected_pages],
                rtol=0,
                atol=0,
            )
            got = pool.get_key_buffer(layer)[pool.translate_loc_to_scratch(row)]
            torch.testing.assert_close(got, ref.kv_buffer[layer][row], rtol=0, atol=0)
            local = pool.index_k_with_scale_buffer[layer]
            logical = torch.arange(local.shape[0], device=device) * world + rank
            torch.testing.assert_close(
                local, ref.index_k_with_scale_buffer[layer][logical], rtol=0, atol=0
            )
        pool.end_shard_extend()
    # Exercise the real pinned-host backup/load kernels with new physical
    # pages but the same owner sequence, as HiCache eviction/reload does.
    from sglang.srt.mem_cache.pool_host.dsa import (
        DSAIndexerPoolHost,
        make_dsa_indexer_pool_decl,
    )
    from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost

    host = MLATokenToKVPoolHost(
        pool.get_device_pool_infos()[0].buffer_info,
        host_to_device_ratio=2,
        host_size=0,
        page_size=page,
        layout="page_first",
        device_capacity_tokens=pool.size * world,
        start_layer=0,
        end_layer=2,
        target_layer_num=2,
    )
    index_host = DSAIndexerPoolHost(make_dsa_indexer_pool_decl(pool), host)
    original = row[: row.numel() // page * page]
    host_rows = torch.arange(page, page + original.numel(), device=device)
    relocated = original + world * page
    owned = pool.placement.local_mask(original, rank)
    original_local = pool.placement.local_index(original[owned])
    relocated_local = pool.placement.local_index(relocated[owned])
    expected_kv = [buf[original_local].clone() for buf in pool.kv_buffer]
    expected_index = [
        buf[original_local[::page] // page].clone()
        for buf in pool.index_k_with_scale_buffer
    ]
    # HybridCacheController keeps host indices on CPU for staged page-first
    # D2H; H2D and the direct-write kernels consume device indices instead.
    for host_pool in (host, index_host):
        backup_host_rows = (
            host_rows.cpu() if host_pool.can_use_write_back_jit else host_rows
        )
        host_pool.backup_from_device_all_layer(
            pool, backup_host_rows, original, "kernel"
        )
    torch.cuda.synchronize()
    for buf in pool.kv_buffer + pool.index_k_with_scale_buffer:
        buf.zero_()
    for layer in range(2):
        host.load_to_device_per_layer(pool, host_rows, relocated, layer, "kernel")
        index_host.load_to_device_per_layer(pool, host_rows, relocated, layer, "kernel")
        torch.testing.assert_close(
            pool.kv_buffer[layer][relocated_local], expected_kv[layer], rtol=0, atol=0
        )
        torch.testing.assert_close(
            pool.index_k_with_scale_buffer[layer][relocated_local[::page] // page],
            expected_index[layer],
            rtol=0,
            atol=0,
        )
    torch.cuda.synchronize()
    print(
        f"rank {rank}: KPool4 bytes and prefix gather matched, peak "
        f"{torch.cuda.max_memory_allocated() / 2**20:.1f} MiB",
        flush=True,
    )
    destroy_model_parallel()
    destroy_distributed_environment()


class TestPageShardedKPoolCUDA(CustomTestCase):
    def test_two_rank_partial_tail_and_new_epoch(self):
        if torch.cuda.device_count() < 2:
            self.skipTest("requires two CUDA GPUs")
        with tempfile.TemporaryDirectory(prefix="kpool-shard-") as tmp:
            mp.spawn(check_rank, args=(2, f"file://{tmp}/init"), nprocs=2, join=True)

    def test_four_rank_partial_tail_and_new_epoch(self):
        if torch.cuda.device_count() < 4:
            self.skipTest("requires four CUDA GPUs")
        with tempfile.TemporaryDirectory(prefix="kpool-shard-") as tmp:
            mp.spawn(check_rank, args=(4, f"file://{tmp}/init"), nprocs=4, join=True)


if __name__ == "__main__":
    unittest.main()
