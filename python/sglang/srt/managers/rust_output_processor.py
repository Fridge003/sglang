"""
Python wrapper for the Rust output processor.

Gated by SGLANG_USE_RUST_OUTPUT_PROCESSOR=1 env var.
Falls back to Python if the Rust module is not available.
"""

import logging
import os
from typing import List, Optional

import msgspec

logger = logging.getLogger(__name__)

_USE_RUST = os.environ.get("SGLANG_USE_RUST_OUTPUT_PROCESSOR", "0") == "1"
_rust_mod = None

if _USE_RUST:
    try:
        from sglang.srt.sgl_scheduler import (
            FORMAT_PREFIX_MSGPACK_EMBEDDING,
            FORMAT_PREFIX_MSGPACK_TOKEN,
            RustOutputSender,
        )

        _rust_mod = True
        logger.info("Rust output processor loaded successfully")
    except ImportError:
        import warnings

        warnings.warn(
            "SGLANG_USE_RUST_OUTPUT_PROCESSOR=1 but Rust module not found, "
            "falling back to Python"
        )
        _rust_mod = False

# Format prefix constants (also defined in Rust, mirrored here for the receiver)
FORMAT_PREFIX_PICKLE = 0x00
FORMAT_PREFIX_MSGPACK_TOKEN_PY = 0x01
FORMAT_PREFIX_MSGPACK_EMBEDDING_PY = 0x02

# msgspec encoder for efficient serialization
_msgpack_encoder = msgspec.msgpack.Encoder()
_msgpack_decoder = msgspec.msgpack.Decoder()


def is_rust_output_enabled():
    """Check if the Rust output processor is enabled and available."""
    return _USE_RUST and _rust_mod


def _load_to_dict(load):
    """Convert a GetLoadReqOutput to a dict for serialization."""
    if load is None:
        return None
    import dataclasses

    if dataclasses.is_dataclass(load):
        return dataclasses.asdict(load)
    return load


def _time_stats_to_list(time_stats):
    """Convert time_stats to serializable form.

    SchedulerReqTimeStats contains non-serializable fields (metrics_collector,
    trace_ctx), so we pass None. The detokenizer forwards time_stats to
    BatchStrOutput, and the tokenizer_manager has its own time_stats.
    """
    # Cannot serialize SchedulerReqTimeStats with msgpack due to complex fields.
    # The detokenizer just passes these through — setting to None is safe.
    return None


def _serialize_batch_token_id_output(batch_output) -> bytes:
    """Serialize a BatchTokenIDOutput to msgpack bytes.

    We convert the dataclass to a dict and serialize with msgspec.
    The Rust background thread prepends the format prefix byte.
    """
    d = {
        "rids": batch_output.rids,
        "http_worker_ipcs": batch_output.http_worker_ipcs,
        "finished_reasons": batch_output.finished_reasons,
        "decoded_texts": batch_output.decoded_texts,
        "decode_ids": batch_output.decode_ids,
        "read_offsets": batch_output.read_offsets,
        "output_ids": batch_output.output_ids,
        "skip_special_tokens": batch_output.skip_special_tokens,
        "spaces_between_special_tokens": batch_output.spaces_between_special_tokens,
        "no_stop_trim": batch_output.no_stop_trim,
        "prompt_tokens": batch_output.prompt_tokens,
        "completion_tokens": batch_output.completion_tokens,
        "cached_tokens": batch_output.cached_tokens,
        "cached_tokens_details": batch_output.cached_tokens_details,
        "spec_verify_ct": batch_output.spec_verify_ct,
        "spec_accepted_tokens": batch_output.spec_accepted_tokens,
        "spec_acceptance_histogram": batch_output.spec_acceptance_histogram,
        "retraction_counts": batch_output.retraction_counts,
        "load": _load_to_dict(batch_output.load),
        "dp_ranks": batch_output.dp_ranks,
        "input_token_logprobs_val": batch_output.input_token_logprobs_val,
        "input_token_logprobs_idx": batch_output.input_token_logprobs_idx,
        "output_token_logprobs_val": batch_output.output_token_logprobs_val,
        "output_token_logprobs_idx": batch_output.output_token_logprobs_idx,
        "input_top_logprobs_val": batch_output.input_top_logprobs_val,
        "input_top_logprobs_idx": batch_output.input_top_logprobs_idx,
        "output_top_logprobs_val": batch_output.output_top_logprobs_val,
        "output_top_logprobs_idx": batch_output.output_top_logprobs_idx,
        "input_token_ids_logprobs_val": batch_output.input_token_ids_logprobs_val,
        "input_token_ids_logprobs_idx": batch_output.input_token_ids_logprobs_idx,
        "output_token_ids_logprobs_val": batch_output.output_token_ids_logprobs_val,
        "output_token_ids_logprobs_idx": batch_output.output_token_ids_logprobs_idx,
        "output_token_entropy_val": batch_output.output_token_entropy_val,
        "output_hidden_states": batch_output.output_hidden_states,
        "routed_experts": None,  # Tensor can't be serialized; handled Python-side
        "customized_info": batch_output.customized_info,
        "placeholder_tokens_idx": batch_output.placeholder_tokens_idx,
        "placeholder_tokens_val": batch_output.placeholder_tokens_val,
        "time_stats": _time_stats_to_list(batch_output.time_stats),
        "token_steps": getattr(batch_output, "token_steps", None),
    }
    return _msgpack_encoder.encode(d)


def _serialize_batch_embedding_output(batch_output) -> bytes:
    """Serialize a BatchEmbeddingOutput to msgpack bytes."""
    d = {
        "rids": batch_output.rids,
        "http_worker_ipcs": batch_output.http_worker_ipcs,
        "finished_reasons": batch_output.finished_reasons,
        "embeddings": batch_output.embeddings,
        "prompt_tokens": batch_output.prompt_tokens,
        "cached_tokens": batch_output.cached_tokens,
        "cached_tokens_details": batch_output.cached_tokens_details,
        "retraction_counts": batch_output.retraction_counts,
        "placeholder_tokens_idx": batch_output.placeholder_tokens_idx,
        "placeholder_tokens_val": batch_output.placeholder_tokens_val,
        "time_stats": None,
    }
    return _msgpack_encoder.encode(d)


class RustSenderWrapper:
    """Drop-in replacement for SenderWrapper that uses the Rust background
    ZMQ sender thread.

    The send path is:
    1. Python serializes BatchTokenIDOutput/BatchEmbeddingOutput to msgpack bytes
    2. Bytes are enqueued to Rust via crossbeam channel (releases GIL)
    3. Rust background thread prepends format prefix byte and sends via ZMQ

    For non-batch messages (e.g. FreezeGCReq), falls back to pickle via a
    separate Python-side ZMQ socket.
    """

    def __init__(self, zmq_endpoint: str, sndbuf_size: int = -1):
        self.sender = RustOutputSender(zmq_endpoint, sndbuf_size)
        # Fallback Python ZMQ socket for non-batch messages
        import zmq

        self._fallback_ctx = zmq.Context(1)
        self._fallback_socket = self._fallback_ctx.socket(zmq.PUSH)
        self._fallback_socket.connect(zmq_endpoint)

    def send_output(self, output, recv_obj=None):
        """Send a batch output to the detokenizer.

        Matches the SenderWrapper.send_output interface.
        """
        from sglang.srt.managers.io_struct import (
            BaseBatchReq,
            BaseReq,
            BatchEmbeddingOutput,
            BatchTokenIDOutput,
        )

        if (
            isinstance(recv_obj, BaseReq)
            and recv_obj.http_worker_ipc is not None
            and output.http_worker_ipc is None
        ):
            output.http_worker_ipc = recv_obj.http_worker_ipc

        if isinstance(output, BatchTokenIDOutput):
            data = _serialize_batch_token_id_output(output)
            self.sender.send_token_batch_bytes(data)
        elif isinstance(output, BatchEmbeddingOutput):
            data = _serialize_batch_embedding_output(output)
            self.sender.send_embedding_batch_bytes(data)
        else:
            # For other message types (e.g. FreezeGCReq), fall back to pickle
            self._fallback_socket.send_pyobj(output)

    def shutdown(self):
        """Shut down the background sender thread."""
        self.sender.shutdown()


def _dict_to_load(d):
    """Reconstruct GetLoadReqOutput from a dict."""
    if d is None:
        return None
    from sglang.srt.managers.io_struct import GetLoadReqOutput

    return GetLoadReqOutput(**d)


def _list_to_time_stats(lst):
    """Reconstruct time_stats list from serialized form.

    SchedulerReqTimeStats has complex non-serializable fields (metrics_collector,
    trace_ctx), so we pass them through as dicts. The detokenizer just forwards
    time_stats to BatchStrOutput anyway.
    """
    return lst


def deserialize_rust_output(data: bytes):
    """Deserialize a msgpack-encoded output message from the Rust sender.

    Called by the detokenizer to decode messages with msgpack format prefix.

    Returns:
        A reconstructed BatchTokenIDOutput or BatchEmbeddingOutput object.
    """
    from sglang.srt.managers.io_struct import BatchEmbeddingOutput, BatchTokenIDOutput

    prefix = data[0]
    payload = data[1:]
    d = _msgpack_decoder.decode(payload)

    if prefix == FORMAT_PREFIX_MSGPACK_TOKEN_PY:
        return BatchTokenIDOutput(
            rids=d["rids"],
            http_worker_ipcs=d["http_worker_ipcs"],
            finished_reasons=d["finished_reasons"],
            decoded_texts=d["decoded_texts"],
            decode_ids=d["decode_ids"],
            read_offsets=d["read_offsets"],
            output_ids=d["output_ids"],
            skip_special_tokens=d["skip_special_tokens"],
            spaces_between_special_tokens=d["spaces_between_special_tokens"],
            no_stop_trim=d["no_stop_trim"],
            prompt_tokens=d["prompt_tokens"],
            completion_tokens=d["completion_tokens"],
            cached_tokens=d["cached_tokens"],
            cached_tokens_details=d.get("cached_tokens_details"),
            spec_verify_ct=d["spec_verify_ct"],
            spec_accepted_tokens=d["spec_accepted_tokens"],
            spec_acceptance_histogram=d["spec_acceptance_histogram"],
            retraction_counts=d["retraction_counts"],
            load=_dict_to_load(d.get("load")),
            dp_ranks=d.get("dp_ranks"),
            input_token_logprobs_val=d["input_token_logprobs_val"],
            input_token_logprobs_idx=d["input_token_logprobs_idx"],
            output_token_logprobs_val=d["output_token_logprobs_val"],
            output_token_logprobs_idx=d["output_token_logprobs_idx"],
            input_top_logprobs_val=d["input_top_logprobs_val"],
            input_top_logprobs_idx=d["input_top_logprobs_idx"],
            output_top_logprobs_val=d["output_top_logprobs_val"],
            output_top_logprobs_idx=d["output_top_logprobs_idx"],
            input_token_ids_logprobs_val=d["input_token_ids_logprobs_val"],
            input_token_ids_logprobs_idx=d["input_token_ids_logprobs_idx"],
            output_token_ids_logprobs_val=d["output_token_ids_logprobs_val"],
            output_token_ids_logprobs_idx=d["output_token_ids_logprobs_idx"],
            output_token_entropy_val=d.get("output_token_entropy_val"),
            output_hidden_states=d.get("output_hidden_states"),
            routed_experts=d.get("routed_experts"),
            customized_info=d.get("customized_info", {}),
            placeholder_tokens_idx=d.get("placeholder_tokens_idx"),
            placeholder_tokens_val=d.get("placeholder_tokens_val"),
            time_stats=_list_to_time_stats(d.get("time_stats")),
            token_steps=d.get("token_steps"),
        )
    elif prefix == FORMAT_PREFIX_MSGPACK_EMBEDDING_PY:
        return BatchEmbeddingOutput(
            rids=d["rids"],
            http_worker_ipcs=d["http_worker_ipcs"],
            finished_reasons=d["finished_reasons"],
            embeddings=d["embeddings"],
            prompt_tokens=d["prompt_tokens"],
            cached_tokens=d["cached_tokens"],
            cached_tokens_details=d.get("cached_tokens_details"),
            retraction_counts=d["retraction_counts"],
            placeholder_tokens_idx=d.get("placeholder_tokens_idx"),
            placeholder_tokens_val=d.get("placeholder_tokens_val"),
            time_stats=_list_to_time_stats(d.get("time_stats")),
        )
    else:
        raise ValueError(f"Unknown msgpack format prefix: 0x{prefix:02x}")
