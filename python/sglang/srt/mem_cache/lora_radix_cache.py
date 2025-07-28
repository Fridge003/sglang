import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, MatchResult
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

logger = logging.getLogger(__name__)


@dataclass
class LoRAKey:
    lora_id: str  # lora_id of adaptor, should be hash value of adaptor path
    token_ids: List[int]  # token_ids of the key


class LoRATreeNode:

    counter = 0

    def __init__(self, id: Optional[int] = None):
        self.children = defaultdict(LoRATreeNode)
        self.parent: LoRATreeNode = None
        self.key: LoRAKey = None
        self.value: Optional[torch.Tensor] = None
        self.lock_ref = 0
        self.last_access_time = time.monotonic()

        self.id = LoRATreeNode.counter if id is None else id
        LoRATreeNode.counter += 1

    @property
    def evicted(self):
        return self.value is None

    def __lt__(self, other: "LoRATreeNode"):
        return self.last_access_time < other.last_access_time


def _key_match_page_size1(key0: LoRAKey, key1: LoRAKey):
    if key0.lora_id != key1.lora_id:
        return 0
    i = 0
    for k0, k1 in zip(key0.token_ids, key1.token_ids):
        if k0 != k1:
            break
        i += 1
    return i


class LoRARadixCache(BasePrefixCache):

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        page_size: int,
        disable: bool = False,
    ):
        if page_size > 1:
            raise ValueError("LoRARadixCache currently only supports page_size = 1")

        if token_to_kv_pool_allocator is None:
            raise ValueError(
                "token_to_kv_pool_allocator is required to run LoraRadixCache"
            )

        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.page_size = page_size
        self.disable = disable
        self.device = self.token_to_kv_pool_allocator.device

        self.key_match_fn = _key_match_page_size1
        self.get_child_key_fn = lambda key: key[0]
        self.reset()

    def reset(self):
        self.root_node = LoRATreeNode()
        self.root_node.key = LoRAKey(lora_id="", token_ids=[])
        self.root_node.value = None
        self.evictable_size_ = 0
        self.protected_size_ = 0
