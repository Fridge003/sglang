"""Python-side bridge between the Rust gRPC server and TokenizerManager.

The RuntimeHandle runs an asyncio event loop in a dedicated thread and exposes
synchronous methods that Rust can call via PyO3 (with a brief GIL acquisition).
Response chunks are pushed into per-request crossbeam channels on the Rust side
via a callback object, keeping the GIL hold time minimal.
"""

import asyncio
import dataclasses
import json
import logging
import threading
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class RuntimeHandle:
    """Thin Python handle that the Rust gRPC server calls into.

    Provides synchronous ``submit_generate``, ``submit_embed``, ``abort``,
    ``get_model_info``, ``get_server_info``, and ``health_check`` methods.
    Each submit method receives a ``chunk_callback`` (a Rust-side PyO3 object)
    that it invokes with ``(chunk_dict, finished, error)`` for each response
    chunk produced by TokenizerManager.
    """

    def __init__(
        self,
        tokenizer_manager,
        template_manager,
        server_args,
        scheduler_info: Optional[Dict] = None,
    ):
        self.tokenizer_manager = tokenizer_manager
        self.template_manager = template_manager
        self.server_args = server_args
        self.scheduler_info = scheduler_info or {}

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, name="grpc-bridge-loop", daemon=True
        )
        self._thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    # ------------------------------------------------------------------
    # Generation (text or tokenized)
    # ------------------------------------------------------------------

    def submit_generate(
        self,
        *,
        rid: str,
        chunk_callback,
        text: Optional[str] = None,
        input_ids=None,
        sampling_params_json: str = "{}",
        stream: bool = False,
        return_logprob: bool = False,
        top_logprobs_num: int = 0,
        logprob_start_len: int = -1,
        return_text_in_logprobs: bool = False,
        lora_path: Optional[str] = None,
        routing_key: Optional[str] = None,
        routed_dp_rank: Optional[int] = None,
        trace_headers: Optional[Dict[str, str]] = None,
    ):
        """Submit a generate request (text or tokenized) to TokenizerManager."""
        from sglang.srt.managers.io_struct import GenerateReqInput

        sampling_params = json.loads(sampling_params_json) if sampling_params_json else {}

        obj = GenerateReqInput(
            text=text,
            input_ids=input_ids,
            rid=rid,
            sampling_params=sampling_params,
            stream=stream,
            return_logprob=return_logprob,
            top_logprobs_num=top_logprobs_num,
            logprob_start_len=logprob_start_len,
            return_text_in_logprobs=return_text_in_logprobs,
            lora_path=lora_path,
            routing_key=routing_key,
            routed_dp_rank=routed_dp_rank,
            external_trace_header=trace_headers,
            received_time=time.time(),
        )

        asyncio.run_coroutine_threadsafe(
            self._run_generate(obj, chunk_callback, stream), self._loop
        )

    async def _run_generate(self, obj, chunk_callback, stream: bool):
        try:
            gen = self.tokenizer_manager.generate_request(obj, request=None)
            if stream:
                async for chunk in gen:
                    finished = chunk.get("meta_info", {}).get("finish_reason") is not None
                    chunk_callback(chunk, finished=finished)
                    if finished:
                        return
            else:
                result = await gen.__anext__()
                chunk_callback(result, finished=True)
        except StopAsyncIteration:
            chunk_callback({}, finished=True)
        except Exception as e:
            logger.error("gRPC generate error for rid=%s: %s", obj.rid, e)
            chunk_callback({}, finished=True, error=str(e))

    # ------------------------------------------------------------------
    # Embedding (text or tokenized)
    # ------------------------------------------------------------------

    def submit_embed(
        self,
        *,
        rid: str,
        chunk_callback,
        text: Optional[str] = None,
        input_ids=None,
        routing_key: Optional[str] = None,
        trace_headers: Optional[Dict[str, str]] = None,
    ):
        """Submit an embed request (text or tokenized) to TokenizerManager."""
        from sglang.srt.managers.io_struct import EmbeddingReqInput

        obj = EmbeddingReqInput(
            text=text,
            input_ids=input_ids,
            rid=rid,
            routing_key=routing_key,
            external_trace_header=trace_headers,
            received_time=time.time(),
        )

        asyncio.run_coroutine_threadsafe(
            self._run_embed(obj, chunk_callback), self._loop
        )

    async def _run_embed(self, obj, chunk_callback):
        try:
            gen = self.tokenizer_manager.generate_request(obj, request=None)
            result = await gen.__anext__()
            chunk_callback(result, finished=True)
        except StopAsyncIteration:
            chunk_callback({}, finished=True)
        except Exception as e:
            logger.error("gRPC embed error for rid=%s: %s", obj.rid, e)
            chunk_callback({}, finished=True, error=str(e))

    # ------------------------------------------------------------------
    # Abort
    # ------------------------------------------------------------------

    def abort(self, rid: str):
        """Abort a request by its request ID."""
        self.tokenizer_manager.abort_request(rid=rid)

    # ------------------------------------------------------------------
    # Info RPCs (synchronous, small data)
    # ------------------------------------------------------------------

    def get_model_info(self) -> str:
        """Return model info as a JSON string."""
        model_config = self.tokenizer_manager.model_config
        result = {
            "model_path": self.tokenizer_manager.model_path,
            "tokenizer_path": self.server_args.tokenizer_path,
            "is_generation": self.tokenizer_manager.is_generation,
            "weight_version": self.server_args.weight_version,
            "model_type": getattr(model_config.hf_config, "model_type", None),
            "architectures": getattr(model_config.hf_config, "architectures", None),
        }
        return json.dumps(result, default=str)

    def get_server_info(self) -> str:
        """Return server info as a JSON string."""
        result: Dict[str, Any] = {}
        try:
            sa = self.server_args
            if hasattr(sa, "model_config"):
                sa = dataclasses.replace(sa)
                if hasattr(sa, "model_config"):
                    delattr(sa, "model_config")
            result.update(dataclasses.asdict(sa))
        except Exception:
            pass
        result.update(self.scheduler_info)
        return json.dumps(result, default=str)

    def health_check(self) -> bool:
        """Return True if the server is healthy."""
        from sglang.srt.managers.tokenizer_manager import ServerStatus

        if self.tokenizer_manager.gracefully_exit:
            return False
        if self.tokenizer_manager.server_status == ServerStatus.Starting:
            return False
        return True
