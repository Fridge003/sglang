from typing import Dict, Optional, Tuple, Type

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.parser.harmony_parser import HarmonyParser


class StreamingParseResult:
    """Result of streaming incremental parsing."""

    def __init__(
        self,
        normal_text: Optional[str] = None,
        reasoning_text: Optional[str] = None,
    ):
        self.normal_text = normal_text or ""
        self.reasoning_text = reasoning_text or ""


class BaseReasoningFormatDetector:
    """Base class providing two sets of interfaces: one-time and streaming incremental."""

    def __init__(
        self,
        think_start_token: str,
        think_end_token: str,
        force_reasoning: bool = False,
        stream_reasoning: bool = True,
        tool_start_token: Optional[str] = None,
        continue_final_message: bool = False,
        previous_content: str = "",
    ):
        self.think_start_token = think_start_token
        self.think_end_token = think_end_token
        self.tool_start_token = tool_start_token
        self._in_reasoning = force_reasoning
        self.stream_reasoning = stream_reasoning

        self._buffer = ""
        self.stripped_think_start = False
        self.think_start_self_label = ""
        # When True we expect the next characters in the buffer to *optionally*
        # begin with ``think_start_token + think_start_self_label`` and must be
        # consumed before any reasoning content is surfaced. This covers the
        # case where ``force_reasoning=True`` but the model still emits an
        # explicit start token (e.g. Gemma-4 with ``enable_thinking=True``,
        # DeepSeek-R1-0528).
        self._awaiting_start_strip = force_reasoning
        # Reasoning content buffered when ``stream_reasoning=False``; flushed
        # when a reasoning block ends.
        self._pending_reasoning = ""

        self.continue_final_message = continue_final_message
        if self.continue_final_message:
            self.previous_content = previous_content
            self.previous_count = len(previous_content)
        else:
            self.previous_content = ""
            self.previous_count = 0

        if self.think_start_token in self.previous_content:
            self._in_reasoning = True
        if self.think_end_token in self.previous_content:
            self._in_reasoning = False

    def detect_and_parse(self, text: str) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses reasoning sections in the provided text.
        Returns both reasoning content and normal text separately.
        """
        in_reasoning = self._in_reasoning or self.think_start_token in text

        if not in_reasoning:
            return StreamingParseResult(normal_text=text)

        # The text is considered to be in a reasoning block.
        processed_text = text.replace(
            self.think_start_token + self.think_start_self_label, ""
        ).strip()

        if (
            self.think_end_token not in processed_text
            and self.think_end_token not in self.previous_content
        ):
            # Check for tool_start_token interruption
            if (
                in_reasoning
                and self.tool_start_token is not None
                and self.tool_start_token in processed_text
            ):
                # Find the first occurrence of tool_start_token and split there
                tool_idx = processed_text.find(self.tool_start_token)
                reasoning_text = processed_text[:tool_idx].strip()
                # Preserve tool_start_token in normal text
                normal_text = processed_text[tool_idx:]
                return StreamingParseResult(
                    normal_text=normal_text, reasoning_text=reasoning_text
                )
            # Assume reasoning was truncated before end token
            return StreamingParseResult(reasoning_text=processed_text)

        # Extract reasoning content
        if self.think_end_token in processed_text:
            splits = processed_text.split(self.think_end_token, maxsplit=1)
            reasoning_text = splits[0]
            normal_text = splits[1].strip()

            return StreamingParseResult(
                normal_text=normal_text, reasoning_text=reasoning_text
            )
        else:
            # think_end_token is in self.previous_content for continue_final_message=True case
            return StreamingParseResult(normal_text=processed_text)

    @staticmethod
    def _split_trailing_partial_token(
        text: str, tokens: Tuple[str, ...]
    ) -> Tuple[str, str]:
        """Split ``text`` into processable content and a trailing partial-token suffix.

        Streaming chunks can end with the beginning of a control token, e.g.
        ``"...<chan"`` before the rest of ``"<channel|>"`` arrives. Returning the
        ``<chan`` as output would leak a delimiter fragment to the caller.
        This helper finds the longest suffix of ``text`` that is a STRICT
        prefix of any of ``tokens`` (never the full token — full matches are
        handled by the caller via ``find``). The returned pair is
        ``(head, partial)`` such that ``head + partial == text``.

        Example (tokens include ``"<channel|>"``):
            ``_split_trailing_partial_token("hello<chan", (..., "<channel|>"))``
            → ``("hello", "<chan")``
        """
        best_partial_len = 0
        for token in tokens:
            if not token:
                continue
            # Strict prefixes only: ``prefix_len < len(token)``. Full-token
            # matches must be consumed by ``find`` so the state machine can
            # actually transition.
            max_len = min(len(text), len(token) - 1)
            for prefix_len in range(max_len, best_partial_len, -1):
                if token.startswith(text[-prefix_len:]):
                    best_partial_len = prefix_len
                    break
        if best_partial_len == 0:
            return text, ""
        return text[:-best_partial_len], text[-best_partial_len:]

    def parse_streaming_increment(self, new_text: str) -> StreamingParseResult:
        """Streaming incremental parse of reasoning content.

        Implements an explicit two-state machine ({outside, inside}-reasoning)
        that loops until it either consumes the full buffer or needs more
        input. This guarantees:

        * Delimiter strings (``<|channel>``, ``<channel|>``, ``<think>``, etc.)
          never leak into emitted text, even when split across chunk
          boundaries — only the longest trailing *strict* prefix of a control
          token is retained in the internal buffer.
        * Normal text preceding a ``think_start_token`` in the same chunk is
          correctly routed to ``normal_text`` (not absorbed into reasoning).
        * Multiple reasoning blocks in a single response are handled
          idempotently; each block's boundaries are stripped.
        * With ``force_reasoning=True`` we start inside reasoning but still
          consume an optional leading start token if the model emits one.

        If ``stream_reasoning`` is False, reasoning content is buffered in
        ``_pending_reasoning`` and flushed (rstripped) only when the block's
        end token is seen.
        """
        self._buffer += new_text

        think_start_text = self.think_start_token + self.think_start_self_label
        reasoning_parts = []
        normal_parts = []

        while self._buffer:
            if not self._in_reasoning:
                # Look for a complete start sequence.
                idx = self._buffer.find(think_start_text)
                if idx >= 0:
                    if idx > 0:
                        normal_parts.append(self._buffer[:idx])
                    self._buffer = self._buffer[idx + len(think_start_text) :]
                    self._in_reasoning = True
                    self.stripped_think_start = True
                    self._awaiting_start_strip = False
                    continue

                # No full start token. Flush everything except a trailing
                # partial start-token (or partial end-token, which is unlikely
                # outside reasoning but handled for symmetry).
                head, partial = self._split_trailing_partial_token(
                    self._buffer, (think_start_text, self.think_end_token)
                )
                if head:
                    normal_parts.append(head)
                self._buffer = partial
                break

            # Inside a reasoning block.
            if self._awaiting_start_strip:
                # ``force_reasoning=True`` entry: the model may still emit a
                # leading start token (e.g. Gemma-4 with enable_thinking=True
                # emits ``<|channel>thought\\n``). Consume it if present or
                # partially present; otherwise just clear the flag and
                # proceed.
                if self._buffer.startswith(think_start_text):
                    self._buffer = self._buffer[len(think_start_text) :]
                    self.stripped_think_start = True
                    self._awaiting_start_strip = False
                    continue
                if think_start_text.startswith(self._buffer):
                    # Buffer is a strict prefix of the start sequence; wait
                    # for more data so we don't mis-classify it as reasoning.
                    break
                # Buffer does not start with (nor is a prefix of) the start
                # sequence — treat as raw reasoning (DeepSeek-R1 style).
                self._awaiting_start_strip = False

            # Find the first boundary: end-token or tool-token (if any).
            end_idx = self._buffer.find(self.think_end_token)
            tool_idx = (
                self._buffer.find(self.tool_start_token)
                if self.tool_start_token
                else -1
            )
            # Smallest non-negative index wins.
            candidate = -1
            candidate_kind = None
            if end_idx >= 0:
                candidate, candidate_kind = end_idx, "end"
            if tool_idx >= 0 and (candidate < 0 or tool_idx < candidate):
                candidate, candidate_kind = tool_idx, "tool"

            if candidate >= 0:
                if candidate > 0:
                    piece = self._buffer[:candidate]
                    # Match historical behavior: rstrip the reasoning chunk
                    # that closes the block.
                    if candidate_kind == "end":
                        piece = piece.rstrip()
                    if piece:
                        reasoning_parts.append(piece)
                if candidate_kind == "end":
                    self._buffer = self._buffer[candidate + len(self.think_end_token) :]
                    self._in_reasoning = False
                else:  # tool
                    # Tool delimiter is preserved in normal text so downstream
                    # function-call parsers can see it.
                    normal_parts.append(self._buffer[candidate:])
                    self._buffer = ""
                    self._in_reasoning = False
                continue

            # No boundary in buffer. Flush reasoning content minus any
            # trailing partial delimiter.
            partial_tokens = [self.think_end_token]
            if self.tool_start_token:
                partial_tokens.append(self.tool_start_token)
            head, partial = self._split_trailing_partial_token(
                self._buffer, tuple(partial_tokens)
            )
            # Also retain trailing whitespace in the buffer: it either gets
            # rstripped away when the end-token actually arrives, or it will
            # be flushed alongside the next non-whitespace reasoning content.
            # Without this, per-chunk arrivals of ``"...\n"`` followed by
            # ``"</think>"`` would emit a stray newline that single-shot
            # parsing correctly strips.
            trailing_ws_len = len(head) - len(head.rstrip())
            if trailing_ws_len:
                partial = head[-trailing_ws_len:] + partial
                head = head[:-trailing_ws_len]
            if head:
                reasoning_parts.append(head)
            self._buffer = partial
            break

        reasoning_text = "".join(reasoning_parts)
        normal_text = "".join(normal_parts)

        if not self.stream_reasoning:
            # Accumulate reasoning until a boundary closes the block.
            self._pending_reasoning += reasoning_text
            if not self._in_reasoning and self._pending_reasoning:
                reasoning_text = self._pending_reasoning.rstrip()
                self._pending_reasoning = ""
            else:
                reasoning_text = ""

        return StreamingParseResult(
            normal_text=normal_text, reasoning_text=reasoning_text
        )


class DeepSeekR1Detector(BaseReasoningFormatDetector):
    """
    Detector for DeepSeek-R1 model.
    Assumes reasoning format:
      (<think>)*(.*)</think>
    Returns all the text before the </think> tag as `reasoning_text`
    and the rest of the text as `normal_text`.

    Supported models:
      - DeepSeek-R1: Always generates thinking content without <think> start tag
      - DeepSeek-R1-0528: Generates thinking content with <think> start tag

    Format patterns:
      - DeepSeek-R1: "I need to think about this...</think>The answer is 42."
      - DeepSeek-R1-0528: "<think>I need to think about this...</think>The answer is 42."

    Args:
        stream_reasoning (bool): If False, accumulates reasoning content until the end tag.
            If True, streams reasoning content as it arrives.
    """

    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = True,
        continue_final_message: bool = False,
        previous_content: str = "",
    ):
        # DeepSeek-R1 is assumed to be reasoning until `</think>` token
        super().__init__(
            "<think>",
            "</think>",
            force_reasoning=True,
            stream_reasoning=stream_reasoning,
            continue_final_message=continue_final_message,
            previous_content=previous_content,
        )
        # https://github.com/sgl-project/sglang/pull/3202#discussion_r1950153599


class Qwen3Detector(BaseReasoningFormatDetector):
    """
    Detector for Qwen3 models (e.g., Qwen/Qwen3-235B-A22B).
    Assumes reasoning format:
      (<think>)*(.*)</think>

    Qwen3 models released before 07/2025 supports switching between thinking mode and normal
    mode using `enable_thinking` parameter in the request parameter.
      - enable_thinking=True: "<think>reasoning content</think>The answer is 42."
      - enable_thinking=False: "The answer is 42." (no thinking tokens)

    Args:
        stream_reasoning (bool): If False, accumulates reasoning content until the end tag.
            If True, streams reasoning content as it arrives.
    """

    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = False,
        continue_final_message: bool = False,
        previous_content: str = "",
    ):
        super().__init__(
            "<think>",
            "</think>",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            continue_final_message=continue_final_message,
            previous_content=previous_content,
        )


class KimiDetector(BaseReasoningFormatDetector):
    """
    Detector for Kimi Thinking model.
    Assumes reasoning format:
      ◁think▷*(.*)◁/think▷
    Returns all the text before the ◁/think▷ tag as `reasoning_text`
    and the rest of the text as `normal_text`.
    """

    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = False,
        continue_final_message: bool = False,
        previous_content: str = "",
    ):
        super().__init__(
            "◁think▷",
            "◁/think▷",
            force_reasoning=False,
            stream_reasoning=stream_reasoning,
            continue_final_message=continue_final_message,
            previous_content=previous_content,
        )


class KimiK2Detector(BaseReasoningFormatDetector):
    """
    Detector for Kimi K2 models.
    Assumes reasoning format:
      (<think>)*(.*)</think>

    Kimi K2 can switch from reasoning to tool-call section with
    `<|tool_calls_section_begin|>` before emitting `</think>`.
    """

    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = False,
        continue_final_message: bool = False,
        previous_content: str = "",
    ):
        super().__init__(
            "<think>",
            "</think>",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            tool_start_token="<|tool_calls_section_begin|>",
            continue_final_message=continue_final_message,
            previous_content=previous_content,
        )


class Glm45Detector(BaseReasoningFormatDetector):
    """
    Detector for GLM-4.5 models.
    Assumes reasoning format:
      (<think>)*(.*)</think>

    GLM-4.5 uses `<tool_call>` as the tool start token to switch from reasoning mode to normal mode.

    Args:
        stream_reasoning (bool): If False, accumulates reasoning content until the end tag.
            If True, streams reasoning content as it arrives.
    """

    def __init__(self, stream_reasoning: bool = True, force_reasoning: bool = False):
        super().__init__(
            "<think>",
            "</think>",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            tool_start_token="<tool_call>",
        )


class GptOssDetector(BaseReasoningFormatDetector):
    """
    Detector for T4-style reasoning format (GPT-OSS), using the HarmonyParser.
    """

    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = True,
        continue_final_message: bool = False,
        previous_content: str = "",
    ):
        super().__init__(
            "<|channel|>analysis<|message|>",
            "<|end|>",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            continue_final_message=continue_final_message,
            previous_content=previous_content,
        )
        self.parser = HarmonyParser()

    def detect_and_parse(self, text: str) -> StreamingParseResult:
        events = self.parser.parse(text)
        # Flush the buffer for one-shot parsing
        events += self.parser.parse("")

        reasoning_text = "".join(
            [e.content for e in events if e.event_type == "reasoning"]
        )
        normal_parts = []
        for e in events:
            if e.event_type == "normal":
                normal_parts.append(e.content)
            elif e.event_type == "tool_call":
                # Use raw_text to preserve structural markers for function call detector
                normal_parts.append(e.raw_text if e.raw_text else e.content)
        normal_text = "".join(normal_parts)
        # Tool call events preserve raw text with structural markers

        return StreamingParseResult(
            normal_text=normal_text,
            reasoning_text=reasoning_text,
        )

    def parse_streaming_increment(self, new_text: str) -> StreamingParseResult:
        events = self.parser.parse(new_text)

        reasoning_text = "".join(
            [e.content for e in events if e.event_type == "reasoning"]
        )
        normal_parts = []
        for e in events:
            if e.event_type == "normal":
                normal_parts.append(e.content)
            elif e.event_type == "tool_call":
                # Use raw_text to preserve structural markers for function call detector
                normal_parts.append(e.raw_text if e.raw_text else e.content)
        normal_text = "".join(normal_parts)

        return StreamingParseResult(
            normal_text=normal_text,
            reasoning_text=reasoning_text,
        )


class MiniMaxAppendThinkDetector(BaseReasoningFormatDetector):
    """
    Append `<think>` token to the beginning of the text.
    """

    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = False,
        continue_final_message: bool = False,
        previous_content: str = "",
    ):
        # scheduler.py need `reasoning_parser.detector.think_end_token`
        super().__init__(
            "<think>",
            "</think>",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            continue_final_message=continue_final_message,
            previous_content=previous_content,
        )
        self.is_first_chunk = False

    def parse_streaming_increment(self, new_text: str) -> StreamingParseResult:
        if not self.is_first_chunk:
            self.is_first_chunk = True
            new_text = self.think_start_token + new_text
        return StreamingParseResult(normal_text=new_text)

    def detect_and_parse(self, text: str) -> StreamingParseResult:
        return StreamingParseResult(normal_text=self.think_start_token + text)


class Nemotron3Detector(BaseReasoningFormatDetector):
    """
    Detector for Nemotron3 model.
    Uses the same reasoning format as DeepSeek-R1: (<think>)*(.*)</think>

    """

    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = False,
        continue_final_message: bool = False,
        previous_content: str = "",
        force_nonempty_content: bool = False,
    ):
        super().__init__(
            "<think>",
            "</think>",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            continue_final_message=continue_final_message,
            previous_content=previous_content,
        )
        self._force_nonempty_content = force_nonempty_content

    def detect_and_parse(self, text: str) -> StreamingParseResult:
        ret = super().detect_and_parse(text)
        if self._force_nonempty_content and not ret.normal_text:
            ret.normal_text, ret.reasoning_text = ret.reasoning_text, ret.normal_text
        return ret


class MistralDetector(BaseReasoningFormatDetector):
    """
    Detector for Mistral models with reasoning (e.g., Mistral-Small-4-119B-2603).
    Assumes reasoning format:
      [THINK]reasoning content[/THINK]answer

    Reasoning is optional — it only appears when reasoning_effort="high" is set.
    When reasoning_effort="none", the model outputs directly without thinking tokens.
    """

    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = False,
        continue_final_message: bool = False,
        previous_content: str = "",
    ):
        super().__init__(
            "[THINK]",
            "[/THINK]",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            continue_final_message=continue_final_message,
            previous_content=previous_content,
        )


class Gemma4Detector(BaseReasoningFormatDetector):
    """Gemma4 reasoning detector."""

    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = False,
        continue_final_message: bool = False,
        previous_content: str = "",
    ):
        super().__init__(
            "<|channel>",
            "<channel|>",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            continue_final_message=continue_final_message,
            previous_content=previous_content,
        )
        self.think_start_self_label = "thought\n"


class ReasoningParser:
    """
    Parser that handles both streaming and non-streaming scenarios for extracting
    reasoning content from model outputs.

    Args:
        model_type (str): Type of model to parse reasoning from
        stream_reasoning (bool): If False, accumulates reasoning content until complete.
            If True, streams reasoning content as it arrives.
    """

    DetectorMap: Dict[str, Type[BaseReasoningFormatDetector]] = {
        "deepseek-r1": DeepSeekR1Detector,
        "deepseek-v3": Qwen3Detector,
        "glm45": Glm45Detector,
        "gpt-oss": GptOssDetector,
        "kimi": KimiDetector,
        "kimi_k2": KimiK2Detector,
        "mimo": Qwen3Detector,
        "qwen3": Qwen3Detector,
        "qwen3-thinking": Qwen3Detector,
        "minimax": Qwen3Detector,
        "minimax-append-think": MiniMaxAppendThinkDetector,
        "step3": DeepSeekR1Detector,
        "step3p5": DeepSeekR1Detector,
        "mistral": MistralDetector,
        "nemotron_3": Nemotron3Detector,
        "interns1": Qwen3Detector,
        "gemma4": Gemma4Detector,
    }

    def __init__(
        self,
        model_type: Optional[str] = None,
        stream_reasoning: bool = True,
        force_reasoning: Optional[bool] = None,
        request: ChatCompletionRequest = None,
    ):
        if not model_type:
            raise ValueError("Model type must be specified")

        detector_class = self.DetectorMap.get(model_type.lower())
        if not detector_class:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Special cases where we override force_reasoning
        if model_type.lower() in {"qwen3-thinking", "gpt-oss", "minimax"}:
            force_reasoning = True

        # Only pass force_reasoning if explicitly set, let detectors use their defaults
        kwargs = {"stream_reasoning": stream_reasoning}
        if force_reasoning is not None:
            kwargs["force_reasoning"] = force_reasoning

        if (
            request is not None
            and isinstance(request, ChatCompletionRequest)
            and request.continue_final_message
            and request.messages[-1].role == "assistant"
        ):
            kwargs["continue_final_message"] = True
            kwargs["previous_content"] = request.messages[-1].content

        chat_template_kwargs = getattr(request, "chat_template_kwargs", None) or {}
        if chat_template_kwargs.get("force_nonempty_content") is True:
            kwargs["force_nonempty_content"] = True

        self.detector = detector_class(**kwargs)

    def parse_non_stream(self, full_text: str) -> Tuple[Optional[str], Optional[str]]:
        """Non-streaming call: one-time parsing"""
        ret = self.detector.detect_and_parse(full_text)
        return ret.reasoning_text, ret.normal_text

    def parse_stream_chunk(
        self, chunk_text: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Streaming call: incremental parsing"""
        ret = self.detector.parse_streaming_increment(chunk_text)
        return ret.reasoning_text, ret.normal_text
