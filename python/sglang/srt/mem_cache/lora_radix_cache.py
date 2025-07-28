import logging
import threading
from typing import Optional

import torch

from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MLATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.mem_cache.memory_pool_host import (
    MHATokenToKVPoolHost,
    MLATokenToKVPoolHost,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, TreeNode

logger = logging.getLogger(__name__)


class LoRARadixCache(RadixCache):

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        page_size: int,
        disable: bool = False,
    ):
        if page_size > 1:
            raise ValueError("LoRARadixCache only supports page_size = 1")
        super().__init__(
            req_to_token_pool, token_to_kv_pool_allocator, page_size, disable
        )
