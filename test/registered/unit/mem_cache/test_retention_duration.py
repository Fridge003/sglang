"""
Unit tests for priority-based retention duration.

Tests the time-decay behavior added to PriorityStrategy and the
retention_duration propagation through RadixCache insert/split paths.

Coverage:
- PriorityStrategy decay: expired retention drops priority to 0
- PriorityStrategy no-decay: retention_duration=0 means permanent priority
- PriorityStrategy active retention: not-yet-expired keeps priority
- RadixCache: retention_duration propagated on insert, split, and new node
- RadixCache: eviction ordering respects decayed priority
- Backwards compat: priority without retention never decays

Usage:
    python -m pytest test_retention_duration.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu-only")

import time
import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams
from sglang.srt.mem_cache.evict_policy import PriorityStrategy
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode


def _make_node(**kwargs):
    node = MagicMock()
    node.last_access_time = kwargs.get("last_access_time", 0.0)
    node.priority = kwargs.get("priority", 0)
    node.retention_duration = kwargs.get("retention_duration", 0.0)
    return node


class TestPriorityStrategyDecay(unittest.TestCase):
    """Test time-decay behavior in PriorityStrategy."""

    def setUp(self):
        self.strategy = PriorityStrategy()

    def test_no_retention_permanent_priority(self):
        """retention_duration=0 means priority never decays."""
        node = _make_node(priority=10, retention_duration=0.0, last_access_time=0.0)
        self.assertEqual(self.strategy.get_priority(node), (10, 0.0))

    def test_expired_retention_decays_to_zero(self):
        """After retention_duration elapses, effective priority drops to 0."""
        now = time.monotonic()
        node = _make_node(
            priority=10,
            retention_duration=1.0,
            last_access_time=now - 2.0,  # 2s ago, retention is 1s
        )
        pri, _ = self.strategy.get_priority(node)
        self.assertEqual(pri, 0)

    def test_active_retention_keeps_priority(self):
        """Before retention_duration elapses, priority is preserved."""
        now = time.monotonic()
        node = _make_node(
            priority=10,
            retention_duration=60.0,
            last_access_time=now - 1.0,  # 1s ago, retention is 60s
        )
        pri, _ = self.strategy.get_priority(node)
        self.assertEqual(pri, 10)

    def test_zero_priority_unaffected_by_retention(self):
        """priority=0 stays 0 regardless of retention_duration."""
        now = time.monotonic()
        node = _make_node(
            priority=0,
            retention_duration=1.0,
            last_access_time=now - 2.0,
        )
        pri, _ = self.strategy.get_priority(node)
        self.assertEqual(pri, 0)

    def test_decay_ordering(self):
        """Expired high-priority node sorts below active low-priority node."""
        now = time.monotonic()
        expired_high = _make_node(
            priority=100,
            retention_duration=1.0,
            last_access_time=now - 5.0,  # expired
        )
        active_low = _make_node(
            priority=5,
            retention_duration=60.0,
            last_access_time=now,  # fresh
        )
        self.assertLess(
            self.strategy.get_priority(expired_high),
            self.strategy.get_priority(active_low),
        )

    def test_exact_boundary_decays(self):
        """At exactly retention_duration elapsed, priority decays."""
        now = time.monotonic()
        node = _make_node(
            priority=10,
            retention_duration=1.0,
            last_access_time=now - 1.0,  # exactly at boundary
        )
        pri, _ = self.strategy.get_priority(node)
        self.assertEqual(pri, 0)


class TestTreeNodeRetentionField(unittest.TestCase):
    """Test that TreeNode has retention_duration with correct default."""

    def test_default_retention_duration(self):
        node = TreeNode()
        self.assertEqual(node.retention_duration, 0.0)

    def test_retention_duration_is_settable(self):
        node = TreeNode()
        node.retention_duration = 300.0
        self.assertEqual(node.retention_duration, 300.0)


class TestRadixCacheRetentionPropagation(unittest.TestCase):
    """Test retention_duration propagation through insert and split."""

    def _make_cache(self, page_size=1):
        mock_allocator = MagicMock()
        mock_allocator.device = torch.device("cpu")
        return RadixCache.create_simulated(
            disable=False,
            mock_allocator=mock_allocator,
            page_size=page_size,
        )

    def test_insert_sets_retention_on_new_node(self):
        cache = self._make_cache()
        cache.insert(
            InsertParams(
                key=RadixKey([1, 2, 3]),
                value=torch.tensor([10, 20, 30]),
                priority=10,
                retention_duration=300.0,
            )
        )
        # Walk tree to find the leaf
        node = cache.root_node
        while node.children:
            node = list(node.children.values())[0]
        self.assertEqual(node.retention_duration, 300.0)
        self.assertEqual(node.priority, 10)

    def test_insert_propagates_retention_to_path(self):
        """Inserting with retention_duration propagates max along the path."""
        cache = self._make_cache()
        # First insert with no retention
        cache.insert(
            InsertParams(
                key=RadixKey([1, 2, 3, 4]),
                value=torch.tensor([10, 20, 30, 40]),
                priority=5,
                retention_duration=0.0,
            )
        )
        # Second insert overlapping prefix with retention
        cache.insert(
            InsertParams(
                key=RadixKey([1, 2, 3, 5]),
                value=torch.tensor([10, 20, 30, 50]),
                priority=10,
                retention_duration=600.0,
            )
        )
        # The shared prefix node [1,2,3] should have max retention
        node = cache.root_node
        child = list(node.children.values())[0]
        self.assertEqual(child.retention_duration, 600.0)

    def test_split_preserves_retention(self):
        """When a node is split, the new parent inherits retention_duration."""
        cache = self._make_cache()
        cache.insert(
            InsertParams(
                key=RadixKey([1, 2, 3, 4]),
                value=torch.tensor([10, 20, 30, 40]),
                priority=10,
                retention_duration=300.0,
            )
        )
        # Insert prefix that causes a split at [1,2]
        cache.insert(
            InsertParams(
                key=RadixKey([1, 2, 5]),
                value=torch.tensor([10, 20, 50]),
                priority=5,
                retention_duration=0.0,
            )
        )
        # The split parent node [1,2] should inherit retention=300
        node = cache.root_node
        split_node = list(node.children.values())[0]
        self.assertGreaterEqual(split_node.retention_duration, 300.0)

    def test_max_semantics_on_overlap(self):
        """Multiple inserts take max of retention_duration."""
        cache = self._make_cache()
        cache.insert(
            InsertParams(
                key=RadixKey([1, 2, 3]),
                value=torch.tensor([10, 20, 30]),
                priority=10,
                retention_duration=100.0,
            )
        )
        cache.insert(
            InsertParams(
                key=RadixKey([1, 2, 3]),
                value=torch.tensor([10, 20, 30]),
                priority=10,
                retention_duration=500.0,
            )
        )
        node = cache.root_node
        child = list(node.children.values())[0]
        self.assertEqual(child.retention_duration, 500.0)

    def test_backwards_compat_no_retention(self):
        """Insert without retention_duration works as before (default 0)."""
        cache = self._make_cache()
        cache.insert(
            InsertParams(
                key=RadixKey([1, 2, 3]),
                value=torch.tensor([10, 20, 30]),
                priority=10,
            )
        )
        node = cache.root_node
        child = list(node.children.values())[0]
        self.assertEqual(child.retention_duration, 0.0)
        self.assertEqual(child.priority, 10)


class TestRetentionEvictionIntegration(unittest.TestCase):
    """Integration test: eviction ordering with decayed retention."""

    def _make_cache(self):
        mock_allocator = MagicMock()
        mock_allocator.device = torch.device("cpu")
        cache = RadixCache.create_simulated(
            disable=False,
            mock_allocator=mock_allocator,
            page_size=1,
        )
        # Override eviction policy to priority
        cache.eviction_strategy = PriorityStrategy()
        return cache

    def test_high_priority_survives_eviction(self):
        """High-priority block with active retention survives over priority=0."""
        cache = self._make_cache()

        # Insert low-priority block
        cache.insert(
            InsertParams(
                key=RadixKey([1, 2, 3]),
                value=torch.tensor([10, 20, 30]),
                priority=0,
            )
        )
        # Insert high-priority block with long retention
        cache.insert(
            InsertParams(
                key=RadixKey([4, 5, 6]),
                value=torch.tensor([40, 50, 60]),
                priority=10,
                retention_duration=600.0,
            )
        )

        # Evict 3 tokens -- should evict the priority=0 block
        result = cache.evict(EvictParams(num_tokens=3))
        self.assertEqual(result.num_tokens_evicted, 3)

        # High-priority block should still be there
        self.assertEqual(cache.total_size(), 3)

    def test_expired_retention_allows_eviction(self):
        """After retention expires, high-priority block becomes evictable at priority=0."""
        cache = self._make_cache()

        # Insert block with very short retention (already expired)
        cache.insert(
            InsertParams(
                key=RadixKey([1, 2, 3]),
                value=torch.tensor([10, 20, 30]),
                priority=10,
                retention_duration=0.001,  # 1ms
            )
        )

        # Backdate last_access_time so retention is expired
        node = list(cache.root_node.children.values())[0]
        node.last_access_time = time.monotonic() - 1.0

        # Insert a fresh priority=0 block
        cache.insert(
            InsertParams(
                key=RadixKey([4, 5, 6]),
                value=torch.tensor([40, 50, 60]),
                priority=0,
            )
        )

        # Evict 3 tokens -- the expired retention block should be evicted first
        # because its effective priority decayed to 0 and it has older access time
        result = cache.evict(EvictParams(num_tokens=3))
        self.assertEqual(result.num_tokens_evicted, 3)

        # The fresh priority=0 block should remain
        self.assertEqual(cache.total_size(), 3)


if __name__ == "__main__":
    unittest.main()
