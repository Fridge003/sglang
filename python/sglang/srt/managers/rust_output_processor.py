"""
Python wrapper for the Rust output processor.

Gated by SGLANG_USE_RUST_OUTPUT_PROCESSOR env var (via environ.py).
Falls back to Python if the Rust module is not available.
"""

import dataclasses
import logging
import warnings

import msgspec
import zmq

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_USE_RUST = envs.SGLANG_USE_RUST_OUTPUT_PROCESSOR.get()
_rust_mod = None

if _USE_RUST:
    try:
        from sglang.srt.sgl_scheduler import (
            RustOutputSender,
            check_finished_rust,
        )

        _rust_mod = True
        logger.info("Rust output processor loaded successfully")
    except ImportError:
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
        self._fallback_ctx = zmq.Context(1)
        self._fallback_socket = self._fallback_ctx.socket(zmq.PUSH)
        self._fallback_socket.connect(zmq_endpoint)

    def send_output(self, output, recv_obj=None):
        """Send a batch output to the detokenizer.

        Matches the SenderWrapper.send_output interface.
        """
        from sglang.srt.managers.io_struct import (
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

    def send_token_output_dict(self, output_dict: dict):
        """Phase 2: Send a token output dict directly via Rust serialization.

        Bypasses Python BatchTokenIDOutput construction and msgspec serialization.
        Rust serializes the dict to msgpack via rmp-serde and sends via background thread.
        """
        self.sender.serialize_and_send_token_output(output_dict)

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


# ---------------------------------------------------------------------------
# Phase 2: stream_output_generation with Rust serialization
# ---------------------------------------------------------------------------

DEFAULT_FORCE_STREAM_INTERVAL = 50


def stream_output_generation_rust(
    scheduler, reqs, return_logprob, skip_req=None, is_idle_batch=False
):
    """Phase 2 replacement for stream_output_generation.

    Same logic as the Python version, but builds a plain dict and passes it
    to Rust for msgpack serialization + background ZMQ send, bypassing
    the intermediate BatchTokenIDOutput dataclass and Python msgspec.encode.
    """
    from sglang.srt.disaggregation.utils import DisaggregationMode

    rids = []
    http_worker_ipcs = []
    finished_reasons = []

    decoded_texts = []
    decode_ids_list = []
    read_offsets = []
    output_ids = []

    skip_special_tokens = []
    spaces_between_special_tokens = []
    no_stop_trim = []
    prompt_tokens = []
    completion_tokens = []
    cached_tokens = []
    cached_tokens_details = []
    spec_verify_ct = []
    spec_accepted_tokens = []
    spec_acceptance_histogram = []
    retraction_counts = []
    output_hidden_states = None
    load = scheduler.get_load()
    routed_experts = None
    customized_info = {}
    time_stats = []

    if return_logprob:
        input_token_logprobs_val = []
        input_token_logprobs_idx = []
        output_token_logprobs_val = []
        output_token_logprobs_idx = []
        input_top_logprobs_val = []
        input_top_logprobs_idx = []
        output_top_logprobs_val = []
        output_top_logprobs_idx = []
        input_token_ids_logprobs_val = []
        input_token_ids_logprobs_idx = []
        output_token_ids_logprobs_val = []
        output_token_ids_logprobs_idx = []
    else:
        input_token_logprobs_val = input_token_logprobs_idx = (
            output_token_logprobs_val
        ) = output_token_logprobs_idx = input_top_logprobs_val = (
            input_top_logprobs_idx
        ) = output_top_logprobs_val = output_top_logprobs_idx = (
            input_token_ids_logprobs_val
        ) = input_token_ids_logprobs_idx = output_token_ids_logprobs_val = (
            output_token_ids_logprobs_idx
        ) = None

    for req in reqs:
        if req is skip_req:
            continue

        if scheduler.model_config.is_multimodal_gen and req.to_finish:
            continue

        if req.finished():
            if req.finished_output:
                continue
            req.finished_output = True
            if req.finished_len is None:
                req.finished_len = len(req.output_ids)
            should_output = True
        else:
            if req.stream:
                stream_interval = (
                    req.sampling_params.stream_interval or scheduler.stream_interval
                )
                should_output = (
                    len(req.output_ids) % stream_interval == 1
                    if not scheduler.model_config.is_multimodal_gen
                    and stream_interval > 1
                    else len(req.output_ids) % stream_interval == 0
                )
                if should_output:
                    should_output &= not req.check_match_stop_str_prefix()
            else:
                should_output = (
                    len(req.output_ids) % DEFAULT_FORCE_STREAM_INTERVAL == 0
                    if not scheduler.model_config.is_multimodal_gen
                    else False
                )

        if should_output:
            send_token_offset = req.send_token_offset
            send_output_token_logprobs_offset = req.send_output_token_logprobs_offset

            rids.append(req.rid)
            http_worker_ipcs.append(req.http_worker_ipc)
            finished_reasons.append(
                req.finished_reason.to_json() if req.finished_reason else None
            )
            decoded_texts.append(req.decoded_text)
            decode_ids, read_offset = req.init_incremental_detokenize()

            if scheduler.model_config.is_multimodal_gen:
                decode_ids_list.append(decode_ids)
            else:
                decode_ids_list.append(decode_ids[req.send_decode_id_offset :])

            output_ids_ = req.output_ids_through_stop

            req.send_decode_id_offset = len(decode_ids)
            read_offsets.append(read_offset)
            output_ids.append(output_ids_[send_token_offset:])
            req.send_token_offset = len(output_ids_)
            skip_special_tokens.append(req.sampling_params.skip_special_tokens)
            spaces_between_special_tokens.append(
                req.sampling_params.spaces_between_special_tokens
            )
            no_stop_trim.append(req.sampling_params.no_stop_trim)
            prompt_tokens.append(len(req.origin_input_ids))
            completion_tokens.append(len(output_ids_))
            cached_tokens.append(req.cached_tokens)
            cached_tokens_details.append(scheduler._get_cached_tokens_details(req))
            retraction_counts.append(req.retraction_count)
            time_stats.append(req.time_stats)

            if not scheduler.spec_algorithm.is_none():
                spec_verify_ct.append(req.spec_verify_ct)
                spec_accepted_tokens.append(req.spec_accepted_tokens)
                spec_acceptance_histogram.append(req.spec_acceptance_histogram)

            if return_logprob:
                if (
                    req.return_logprob
                    and not req.input_logprob_sent
                    and scheduler.disaggregation_mode != DisaggregationMode.DECODE
                ):
                    input_token_logprobs_val.append(req.input_token_logprobs_val)
                    input_token_logprobs_idx.append(req.input_token_logprobs_idx)
                    input_top_logprobs_val.append(req.input_top_logprobs_val)
                    input_top_logprobs_idx.append(req.input_top_logprobs_idx)
                    input_token_ids_logprobs_val.append(
                        req.input_token_ids_logprobs_val
                    )
                    input_token_ids_logprobs_idx.append(
                        req.input_token_ids_logprobs_idx
                    )
                    req.input_logprob_sent = True
                else:
                    input_token_logprobs_val.append([])
                    input_token_logprobs_idx.append([])
                    input_top_logprobs_val.append([])
                    input_top_logprobs_idx.append([])
                    input_token_ids_logprobs_val.append([])
                    input_token_ids_logprobs_idx.append([])

                if req.return_logprob:
                    output_token_logprobs_val.append(
                        req.output_token_logprobs_val[
                            send_output_token_logprobs_offset:
                        ]
                    )
                    output_token_logprobs_idx.append(
                        req.output_token_logprobs_idx[
                            send_output_token_logprobs_offset:
                        ]
                    )
                    output_top_logprobs_val.append(
                        req.output_top_logprobs_val[send_output_token_logprobs_offset:]
                    )
                    output_top_logprobs_idx.append(
                        req.output_top_logprobs_idx[send_output_token_logprobs_offset:]
                    )
                    output_token_ids_logprobs_val.append(
                        req.output_token_ids_logprobs_val[
                            send_output_token_logprobs_offset:
                        ]
                    )
                    output_token_ids_logprobs_idx.append(
                        req.output_token_ids_logprobs_idx[
                            send_output_token_logprobs_offset:
                        ]
                    )
                    req.send_output_token_logprobs_offset = len(
                        req.output_token_logprobs_val
                    )
                else:
                    output_token_logprobs_val.append([])
                    output_token_logprobs_idx.append([])
                    output_top_logprobs_val.append([])
                    output_top_logprobs_idx.append([])
                    output_token_ids_logprobs_val.append([])
                    output_token_ids_logprobs_idx.append([])

            if req.return_hidden_states:
                if output_hidden_states is None:
                    output_hidden_states = []
                output_hidden_states.append(req.hidden_states)
            if req.return_routed_experts:
                if routed_experts is None:
                    routed_experts = []
                routed_experts.append(req.routed_experts)

            if req.customized_info is not None:
                for k, v in req.customized_info.items():
                    if k not in customized_info:
                        customized_info[k] = []
                    customized_info[k].append(v[send_token_offset:])

        if (
            req.finished()
            and scheduler.attn_tp_rank == 0
            and scheduler.server_args.enable_request_time_stats_logging
        ):
            req.log_time_stats()

    dp_ranks = [scheduler.dp_rank] * len(rids) if rids else None

    # Send to detokenizer — build dict and pass to Rust for serialization
    if reqs or is_idle_batch:
        if scheduler.model_config.is_multimodal_gen:
            return
        output_dict = {
            "rids": rids,
            "http_worker_ipcs": http_worker_ipcs,
            "spec_verify_ct": spec_verify_ct,
            "spec_accepted_tokens": spec_accepted_tokens,
            "spec_acceptance_histogram": spec_acceptance_histogram,
            "time_stats": _time_stats_to_list(time_stats),
            "finished_reasons": finished_reasons,
            "decoded_texts": decoded_texts,
            "decode_ids": decode_ids_list,
            "read_offsets": read_offsets,
            "output_ids": output_ids,
            "skip_special_tokens": skip_special_tokens,
            "spaces_between_special_tokens": spaces_between_special_tokens,
            "no_stop_trim": no_stop_trim,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cached_tokens": cached_tokens,
            "cached_tokens_details": cached_tokens_details,
            "input_token_logprobs_val": input_token_logprobs_val,
            "input_token_logprobs_idx": input_token_logprobs_idx,
            "output_token_logprobs_val": output_token_logprobs_val,
            "output_token_logprobs_idx": output_token_logprobs_idx,
            "input_top_logprobs_val": input_top_logprobs_val,
            "input_top_logprobs_idx": input_top_logprobs_idx,
            "output_top_logprobs_val": output_top_logprobs_val,
            "output_top_logprobs_idx": output_top_logprobs_idx,
            "input_token_ids_logprobs_val": input_token_ids_logprobs_val,
            "input_token_ids_logprobs_idx": input_token_ids_logprobs_idx,
            "output_token_ids_logprobs_val": output_token_ids_logprobs_val,
            "output_token_ids_logprobs_idx": output_token_ids_logprobs_idx,
            "output_token_entropy_val": None,
            "output_hidden_states": output_hidden_states,
            "routed_experts": None,  # Tensor can't cross to Rust
            "customized_info": customized_info,
            "placeholder_tokens_idx": None,
            "placeholder_tokens_val": None,
            "retraction_counts": retraction_counts,
            "load": _load_to_dict(load),
            "dp_ranks": dp_ranks,
            "token_steps": None,
        }
        scheduler.send_to_detokenizer.send_token_output_dict(output_dict)


# ---------------------------------------------------------------------------
# Phase 3: check_finished in Rust
# ---------------------------------------------------------------------------


def _gather_all_stop_token_ids(req):
    """Collect all stop token IDs into a single list for the Rust check."""
    ids = set()
    if req.sampling_params.stop_token_ids:
        ids.update(req.sampling_params.stop_token_ids)
    if req.eos_token_ids:
        ids.update(req.eos_token_ids)
    if req.tokenizer is not None:
        if req.tokenizer.eos_token_id is not None:
            ids.add(req.tokenizer.eos_token_id)
        if getattr(req.tokenizer, "additional_stop_token_ids", None):
            ids.update(req.tokenizer.additional_stop_token_ids)
    return list(ids)


def _get_first_stop_token_id(req):
    """Get a fallback token ID for vocab boundary replacement."""
    if req.sampling_params.stop_token_ids:
        return next(iter(req.sampling_params.stop_token_ids))
    if req.eos_token_ids:
        return next(iter(req.eos_token_ids))
    return 0


def apply_check_finished_rust(req, new_accepted_len=1):
    """Rust-accelerated check_finished for a single Req.

    Handles the fast paths (max_new_tokens, stop tokens, stop strings,
    vocab boundary) in Rust. Falls back to Python for grammar and regex.
    """
    import re

    from sglang.srt.managers.schedule_batch import (
        FINISH_LENGTH,
        FINISH_MATCHED_STR,
        FINISH_MATCHED_TOKEN,
        FINISHED_MATCHED_REGEX,
    )

    if req.finished():
        return

    # Handle to_finish in Python (it's a Python object)
    if req.to_finish:
        req.finished_reason = req.to_finish
        req.to_finish = None
        return

    # Grammar check in Python (requires Python grammar object)
    if req.grammar is not None:
        if req.grammar.is_terminated():
            req.finished_reason = FINISH_MATCHED_TOKEN(matched=req.output_ids[-1])
            return

    has_stop_regex = len(req.sampling_params.stop_regex_strs) > 0

    # Pre-compute tail_str for stop string checking
    tail_str = ""
    if req.sampling_params.stop_strs or has_stop_regex:
        tail_str = req.tail_str()

    new_accepted_tokens = req.output_ids[-new_accepted_len:]

    result = check_finished_rust(
        output_ids_len=len(req.output_ids),
        max_new_tokens=req.sampling_params.max_new_tokens,
        new_accepted_tokens=new_accepted_tokens,
        ignore_eos=req.sampling_params.ignore_eos,
        stop_token_ids=_gather_all_stop_token_ids(req),
        vocab_size=req.vocab_size,
        first_stop_token_id=_get_first_stop_token_id(req),
        stop_strs=list(req.sampling_params.stop_strs),
        tail_str=tail_str,
        decoded_text=req.decoded_text or "",
        has_grammar=False,  # Already checked above
        has_to_finish=False,  # Already checked above
        has_stop_regex=has_stop_regex,
    )

    if result.finish_type == "none":
        # Rust didn't find a finish condition.
        # Check stop regex in Python (requires re module).
        if has_stop_regex and tail_str:
            for stop_regex_str in req.sampling_params.stop_regex_strs:
                if re.search(stop_regex_str, tail_str):
                    req.finished_reason = FINISHED_MATCHED_REGEX(matched=stop_regex_str)
                    return
        return

    if result.finish_type == "length":
        req.finished_reason = FINISH_LENGTH(length=int(result.match_int))
        req.finished_len = int(result.finished_len)
    elif result.finish_type == "matched_token":
        req.finished_reason = FINISH_MATCHED_TOKEN(matched=int(result.match_int))
        if result.finished_len >= 0:
            req.finished_len = int(result.finished_len)
    elif result.finish_type == "matched_str":
        req.finished_reason = FINISH_MATCHED_STR(matched=result.match_str)
    elif result.finish_type == "vocab_boundary":
        if result.modified_output_id_offset >= 0:
            req.output_ids[int(result.modified_output_id_offset)] = int(
                result.modified_output_id_value
            )
        req.finished_reason = FINISH_MATCHED_STR(matched=result.match_str)
        if result.finished_len >= 0:
            req.finished_len = int(result.finished_len)
