# SPDX-License-Identifier: Apache-2.0

import enum
import importlib
import importlib.util
import logging
import time
from typing import List

import requests

logger = logging.getLogger(__name__)


class RemoteInstanceWeightLoaderBackend(str, enum.Enum):
    NCCL = "nccl"
    TRANSFER_ENGINE = "transfer_engine"
    MODEL_EXPRESS = "model_express"


def trigger_init_weights_send_group_for_remote_instance_request(
    remote_instance_weight_loader_seed_instance_ip: str,
    remote_instance_weight_loader_seed_instance_service_port: int,
    remote_instance_weight_loader_send_weights_group_ports: List[int],
    remote_instance_weight_loader_client_id: str,
):
    seed_instance_service_url = f"http://{remote_instance_weight_loader_seed_instance_ip}:{remote_instance_weight_loader_seed_instance_service_port}"
    # Only support loading weights from instance with same parallelism strategy.
    # Per TP rank pair between seed and dst instances will build a communication group for sending weights.
    # i.e. seed TP 0 <-> dst TP 0, seed TP 1 <-> dst TP 1, etc.
    # Each communication group will have a world size 2.
    try:
        requests.post(
            f"{seed_instance_service_url}/init_weights_send_group_for_remote_instance",
            json={
                "master_address": remote_instance_weight_loader_seed_instance_ip,
                "ports": (
                    ",".join(
                        str(p)
                        for p in remote_instance_weight_loader_send_weights_group_ports
                    )
                ),
                "group_rank": 0,
                "world_size": 2,
                "group_name": f"send_weights_{remote_instance_weight_loader_client_id}",
                "backend": "nccl",
            },
        )
    except Exception as e:
        logger.error(
            f"Failed to trigger init_weights_send_group_for_remote_instance_request to seed instance {seed_instance_service_url}: {e}."
        )
        raise


def trigger_transferring_weights_request(
    remote_instance_weight_loader_seed_instance_ip: str,
    remote_instance_weight_loader_seed_instance_service_port: int,
    remote_instance_weight_loader_send_weights_group_ports: List[int],
    remote_instance_weight_loader_client_id: str,
):
    seed_instance_service_url = f"http://{remote_instance_weight_loader_seed_instance_ip}:{remote_instance_weight_loader_seed_instance_service_port}"
    try:
        requests.post(
            f"{seed_instance_service_url}/send_weights_to_remote_instance",
            json={
                "master_address": remote_instance_weight_loader_seed_instance_ip,
                "ports": (
                    ",".join(
                        str(p)
                        for p in remote_instance_weight_loader_send_weights_group_ports
                    )
                ),
                "group_name": f"send_weights_{remote_instance_weight_loader_client_id}",
            },
        )
    except Exception as e:
        logger.error(f"Failed to trigger send weights to remote instance request: {e}")
        raise


def get_remote_instance_transfer_engine_info_per_rank(seed_url: str, rank: int):
    try:
        response = requests.get(
            f"{seed_url}/get_remote_instance_transfer_engine_info",
            params={
                "rank": rank,
            },
        )

        if response.status_code == 200:
            data = response.json()

            if "remote_instance_transfer_engine_info" in data:
                return data["remote_instance_transfer_engine_info"]
            else:
                logger.error(
                    "Failed to get `remote_instance_transfer_engine_info` in response."
                )
                return None, None
        else:
            logger.error(f"request.get failed: {response.status_code}")
            return None, None
    except Exception as e:
        logger.error(f"Exception: {e}")
        return None, None


def parse_remote_instance_transfer_engine_info_from_scheduler_infos(scheduler_infos):
    remote_instance_transfer_engine_info = {}
    for data in scheduler_infos:
        if (
            "tp_rank" in data
            and "remote_instance_transfer_engine_session_id" in data
            and "remote_instance_transfer_engine_weights_info_dict" in data
        ):
            remote_instance_transfer_engine_info[data["tp_rank"]] = (
                data["remote_instance_transfer_engine_session_id"],
                data["remote_instance_transfer_engine_weights_info_dict"],
            )
    return remote_instance_transfer_engine_info


def register_memory_region(model, transfer_engine):
    # Always use v1 (per-param registration) for correctness.
    # v2's block-merging approach has issues with small tensors
    # that are sub-allocated within CUDA blocks.
    return register_memory_region_v1(model, transfer_engine)


def register_memory_region_v1(model, transfer_engine):
    start_tic = time.time()

    weight_mr_dict = {}
    for name, weight in model.named_parameters():
        ret = transfer_engine.register_memory(
            weight.data_ptr(), weight.numel() * weight.element_size()
        )
        if ret != 0:
            raise RuntimeError(
                f"register memory failed for weight {name}, error: {ret}"
            )
        weight_mr_dict[name] = (
            weight.data_ptr(),
            weight.numel(),
            weight.element_size(),
        )

    end_tic = time.time()
    logger.debug(f"Register memory region time: {(end_tic - start_tic):.4f}s")
    return weight_mr_dict


def register_memory_region_v2(model, transfer_engine):
    start_tic = time.time()

    weight_mr_dict = {}
    weight_addr_set = set()
    weight_ranges = []
    for name, weight in model.named_parameters():
        ptr = weight.data_ptr()
        nbytes = weight.numel() * weight.element_size()
        weight_mr_dict[name] = (ptr, weight.numel(), weight.element_size())
        weight_addr_set.add(ptr)
        weight_ranges.append((ptr, ptr + nbytes))

    import torch

    memory_snapshot = torch.cuda.memory.memory_snapshot()

    # Collect all active_allocated blocks with their addresses
    all_blocks = []
    for segment in memory_snapshot:
        for block in segment.get("blocks", []):
            address = block.get("address", -1)
            size = block.get("size", -1)
            state = block.get("state", "")
            if address >= 0 and size > 0 and state == "active_allocated":
                all_blocks.append((address, size))

    # Build set of block addresses that contain at least one weight.
    # First try exact match (fast), then check range containment.
    blocks_with_weights = set()
    for addr, sz in all_blocks:
        if addr in weight_addr_set:
            blocks_with_weights.add(addr)

    # For weights whose data_ptr doesn't match any block start,
    # find the enclosing block via range check.
    unmatched = [
        (w_start, w_end) for w_start, w_end in weight_ranges
        if w_start not in weight_addr_set or w_start not in blocks_with_weights
    ]
    if unmatched:
        # Also check all weights that matched addr_set but might not be in blocks
        # (shouldn't happen, but be safe)
        for w_start, w_end in weight_ranges:
            for b_addr, b_sz in all_blocks:
                if b_addr <= w_start < b_addr + b_sz:
                    blocks_with_weights.add(b_addr)
                    break

    weight_blocks_for_reg_mr = []
    # Blocks in each segment have continuous physical addresses,
    # so they can be merged for memory registration.
    for segment in memory_snapshot:
        current_weight_block = None
        blocks = segment.get("blocks", [])
        for block in blocks:
            address = block.get("address", -1)
            size = block.get("size", -1)
            state = block.get("state", "")
            if address < 0 or size < 0 or state == "":
                continue
            if state == "active_allocated" and address in blocks_with_weights:
                if current_weight_block is None:
                    current_weight_block = (address, size)
                elif current_weight_block[0] + current_weight_block[1] == address:
                    current_weight_block = (
                        current_weight_block[0],
                        current_weight_block[1] + size,
                    )
                else:
                    weight_blocks_for_reg_mr.append(current_weight_block)
                    current_weight_block = (address, size)
        if current_weight_block is not None:
            weight_blocks_for_reg_mr.append(current_weight_block)

    # Verify all weights are covered by a registered block
    uncovered = []
    for name, (ptr, numel, elem_size) in weight_mr_dict.items():
        nbytes = numel * elem_size
        covered = False
        for b_addr, b_sz in weight_blocks_for_reg_mr:
            if b_addr <= ptr and ptr + nbytes <= b_addr + b_sz:
                covered = True
                break
        if not covered:
            uncovered.append((name, ptr, nbytes))
    if uncovered:
        logger.warning(
            "register_memory_region_v2: %d weights NOT covered by any block: %s",
            len(uncovered),
            [(n, hex(a), s) for n, a, s in uncovered[:5]],
        )

    # Register merged memory blocks that hold weights.
    total_reg_bytes = 0
    for weight_block in weight_blocks_for_reg_mr:
        address, size = weight_block
        ret = transfer_engine.register_memory(address, size)
        if ret != 0:
            raise RuntimeError(
                f"register memory failed for weight block at address {address} with size {size}, error: {ret}"
            )
        total_reg_bytes += size

    end_tic = time.time()
    logger.info(
        "register_memory_region_v2: registered %d blocks (%d MB), "
        "%d weights, %d uncovered, took %.2fs",
        len(weight_blocks_for_reg_mr),
        total_reg_bytes // (1024 * 1024),
        len(weight_mr_dict),
        len(uncovered),
        end_tic - start_tic,
    )
    return weight_mr_dict
