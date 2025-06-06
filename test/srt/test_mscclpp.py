"""For Now, MSCCL is only supported on TP16 and TP8 case

if [[ $RANK -eq 0  ]]; then
    ray start --block --head --port=6379 &
    python3 test_mscclpp.py;
else
    ray start --block --address=${MASTER_ADDR}:6379;
fi
"""

import itertools
import os
import random
import socket
import unittest
from contextlib import contextmanager, nullcontext
from typing import Any, List, Optional, Union

import ray
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup, ReduceOp

from sglang.srt.distributed import init_distributed_environment
from sglang.srt.distributed.communication_op import (  # noqa
    tensor_model_parallel_all_reduce,
)
from sglang.srt.distributed.device_communicators.custom_all_reduce import (
    CustomAllreduce,
)
from sglang.srt.distributed.device_communicators.pymscclpp import PyMscclppCommunicator
from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator
from sglang.srt.distributed.parallel_state import (
    get_tensor_model_parallel_group,
    graph_capture,
    initialize_model_parallel,
    set_custom_all_reduce,
    set_mscclpp_all_reduce,
)
from sglang.srt.distributed.utils import StatelessProcessGroup
from sglang.test.test_utils import CustomTestCase


def get_open_port() -> int:
    # try ipv4
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]
    except OSError:
        # try ipv6
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]


def multi_process_parallel(
    world_size: int,
    master_addr: str,
    cls: Any,
    test_target: Any,
) -> None:

    # Using ray helps debugging the error when it failed
    # as compared to multiprocessing.
    # NOTE: We need to set working_dir for distributed tests,
    # otherwise we may get import errors on ray workers

    ray.init(log_to_driver=True)

    distributed_init_port = get_open_port()
    refs = []
    for rank in range(world_size):
        refs.append(
            test_target.remote(
                cls, world_size, master_addr, rank, distributed_init_port
            )
        )
    ray.get(refs)

    ray.shutdown()


class TestMSCCLAllReduce(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        random.seed(42)
        # 1KB to 1MB
        cls.test_sizes = [512, 4096, 32768, 262144, 524288]
        cls.world_sizes = [8]
        TEST_TP16 = int(os.getenv("SGL_MSCCLPP_TEST_TP16", "0"))
        if TEST_TP16:
            cls.world_sizes = [16]
        cls.test_loop = 10

    def test_graph_allreduce(self):
        TEST_MASTER_ADDR = os.getenv("SGL_MSCCLPP_TEST_MASTER_ADDR", "localhost")
        for world_size in self.world_sizes:
            if world_size not in [8, 16]:
                continue
            multi_process_parallel(
                world_size, TEST_MASTER_ADDR, self, self.graph_allreduce
            )

    def test_eager_allreduce(self):
        TEST_MASTER_ADDR = os.getenv("SGL_MSCCLPP_TEST_MASTER_ADDR", "localhost")
        for world_size in self.world_sizes:
            if world_size not in [8, 16]:
                continue
            multi_process_parallel(
                world_size, TEST_MASTER_ADDR, self, self.eager_allreduce
            )

    @ray.remote(num_gpus=1, max_calls=1)
    def graph_allreduce(self, world_size, master_addr, rank, distributed_init_port):
        del os.environ["CUDA_VISIBLE_DEVICES"]
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        torch.cuda.set_device(device)
        distributed_init_method = f"tcp://{master_addr}:{distributed_init_port}"
        set_mscclpp_all_reduce(True)
        set_custom_all_reduce(False)
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            distributed_init_method=distributed_init_method,
            local_rank=rank % torch.cuda.device_count(),
        )
        initialize_model_parallel(tensor_model_parallel_size=world_size)
        group = get_tensor_model_parallel_group().device_group

        # A small all_reduce for warmup.
        # this is needed because device communicators might be created lazily
        # (e.g. NCCL). This will ensure that the communicator is initialized
        # before any communication happens, so that this group can be used for
        # graph capture immediately.
        data = torch.zeros(1)
        data = data.to(device=device)
        torch.distributed.all_reduce(data, group=group)
        torch.cuda.synchronize()
        del data

        for sz in self.test_sizes:
            for dtype in [torch.float32, torch.float16, torch.bfloat16]:
                for _ in range(self.test_loop):
                    with graph_capture() as graph_capture_context:
                        # use integers so result matches NCCL exactly
                        inp1 = torch.randint(
                            1,
                            16,
                            (sz,),
                            dtype=dtype,
                            device=torch.cuda.current_device(),
                        )
                        inp2 = torch.randint(
                            1,
                            16,
                            (sz,),
                            dtype=dtype,
                            device=torch.cuda.current_device(),
                        )
                        torch.cuda.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(
                            graph, stream=graph_capture_context.stream
                        ):
                            out1 = tensor_model_parallel_all_reduce(inp1)
                            # the input buffer is immediately modified to test
                            # synchronization
                            dist.all_reduce(inp1, group=group)
                            out2 = tensor_model_parallel_all_reduce(inp2)
                            dist.all_reduce(inp2, group=group)
                    graph.replay()
                    torch.testing.assert_close(out1, inp1)
                    torch.testing.assert_close(out2, inp2)

    @ray.remote(num_gpus=1, max_calls=1)
    def eager_allreduce(self, world_size, master_addr, rank, distributed_init_port):
        del os.environ["CUDA_VISIBLE_DEVICES"]
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        torch.cuda.set_device(device)
        distributed_init_method = f"tcp://{master_addr}:{distributed_init_port}"
        set_mscclpp_all_reduce(True)
        set_custom_all_reduce(False)
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            distributed_init_method=distributed_init_method,
            local_rank=rank,
        )
        initialize_model_parallel(tensor_model_parallel_size=world_size)
        group = get_tensor_model_parallel_group().device_group

        for sz in self.test_sizes:
            for dtype in [torch.float32, torch.float16, torch.bfloat16]:
                for _ in range(self.test_loop):
                    inp1 = torch.randint(
                        1, 16, (sz,), dtype=dtype, device=torch.cuda.current_device()
                    )
                    out1 = tensor_model_parallel_all_reduce(inp1)
                    dist.all_reduce(inp1, group=group)
                    torch.testing.assert_close(out1, inp1)


if __name__ == "__main__":
    unittest.main()
