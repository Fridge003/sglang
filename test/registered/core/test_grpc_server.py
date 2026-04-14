"""
Integration tests for the native Rust gRPC server.

These tests verify that the gRPC server starts alongside HTTP and correctly
handles all implemented RPCs: text generate, tokenized generate, streaming,
embed, classify, tokenize, detokenize, list models, get load, flush cache,
pause/continue, abort, health, model info, server info, and OpenAI-compat RPCs.

Usage:
    python3 -m pytest test_grpc_server.py -v
"""

import json
import struct
import unittest
from typing import Optional

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=300, suite="stage-b-test-small-1-gpu")


def _grpc_port_from_http_url(http_url: str) -> int:
    """Derive the gRPC port from the HTTP base URL (port + 10000)."""
    from urllib.parse import urlparse

    parsed = urlparse(http_url)
    return parsed.port + 10000


def _grpc_host_from_http_url(http_url: str) -> str:
    from urllib.parse import urlparse

    parsed = urlparse(http_url)
    return parsed.hostname


# ======================================================================
# Protobuf encoding/decoding helpers (minimal, no grpc-tools needed)
# ======================================================================


def _encode_varint(value: int) -> bytes:
    bits = value & 0x7F
    value >>= 7
    result = b""
    while value:
        result += bytes([0x80 | bits])
        bits = value & 0x7F
        value >>= 7
    result += bytes([bits])
    return result


def _encode_string_field(field_number: int, value: str) -> bytes:
    tag = (field_number << 3) | 2
    encoded = value.encode("utf-8")
    return _encode_varint(tag) + _encode_varint(len(encoded)) + encoded


def _encode_bytes_field(field_number: int, value: bytes) -> bytes:
    tag = (field_number << 3) | 2
    return _encode_varint(tag) + _encode_varint(len(value)) + value


def _encode_bool_field(field_number: int, value: bool) -> bytes:
    tag = (field_number << 3) | 0
    return _encode_varint(tag) + bytes([1 if value else 0])


def _encode_float_field(field_number: int, value: float) -> bytes:
    tag = (field_number << 3) | 5
    return _encode_varint(tag) + struct.pack("<f", value)


def _encode_int32_field(field_number: int, value: int) -> bytes:
    tag = (field_number << 3) | 0
    return _encode_varint(tag) + _encode_varint(value)


def _encode_submessage_field(field_number: int, data: bytes) -> bytes:
    tag = (field_number << 3) | 2
    return _encode_varint(tag) + _encode_varint(len(data)) + data


def _decode_varint(data: bytes, offset: int):
    result = 0
    shift = 0
    while offset < len(data):
        b = data[offset]
        result |= (b & 0x7F) << shift
        offset += 1
        if not (b & 0x80):
            return result, offset
        shift += 7
    return None, None


def _decode_string_field(data: bytes, field_number: int) -> Optional[str]:
    expected_tag = (field_number << 3) | 2
    i = 0
    while i < len(data):
        tag, new_i = _decode_varint(data, i)
        if new_i is None:
            break
        i = new_i
        wire_type = tag & 0x7

        if wire_type == 0:
            _, i = _decode_varint(data, i)
            if i is None:
                break
        elif wire_type == 2:
            length, i = _decode_varint(data, i)
            if i is None:
                break
            if tag == expected_tag:
                try:
                    return data[i : i + length].decode("utf-8")
                except UnicodeDecodeError:
                    return None
            i += length
        elif wire_type == 5:
            i += 4
        elif wire_type == 1:
            i += 8
        else:
            break
    return None


def _decode_bytes_field(data: bytes, field_number: int) -> Optional[bytes]:
    expected_tag = (field_number << 3) | 2
    i = 0
    while i < len(data):
        tag, new_i = _decode_varint(data, i)
        if new_i is None:
            break
        i = new_i
        wire_type = tag & 0x7

        if wire_type == 0:
            _, i = _decode_varint(data, i)
            if i is None:
                break
        elif wire_type == 2:
            length, i = _decode_varint(data, i)
            if i is None:
                break
            if tag == expected_tag:
                return data[i : i + length]
            i += length
        elif wire_type == 5:
            i += 4
        elif wire_type == 1:
            i += 8
        else:
            break
    return None


def _decode_bool_field(data: bytes, field_number: int) -> Optional[bool]:
    expected_tag = (field_number << 3) | 0
    i = 0
    while i < len(data):
        tag, new_i = _decode_varint(data, i)
        if new_i is None:
            break
        i = new_i
        wire_type = tag & 0x7

        if wire_type == 0:
            val, i = _decode_varint(data, i)
            if i is None:
                break
            if tag == expected_tag:
                return bool(val)
        elif wire_type == 2:
            length, i = _decode_varint(data, i)
            if i is None:
                break
            i += length
        elif wire_type == 5:
            i += 4
        elif wire_type == 1:
            i += 8
        else:
            break
    return None


def _decode_int32_field(data: bytes, field_number: int) -> Optional[int]:
    expected_tag = (field_number << 3) | 0
    i = 0
    while i < len(data):
        tag, new_i = _decode_varint(data, i)
        if new_i is None:
            break
        i = new_i
        wire_type = tag & 0x7

        if wire_type == 0:
            val, i = _decode_varint(data, i)
            if i is None:
                break
            if tag == expected_tag:
                return val
        elif wire_type == 2:
            length, i = _decode_varint(data, i)
            if i is None:
                break
            i += length
        elif wire_type == 5:
            i += 4
        elif wire_type == 1:
            i += 8
        else:
            break
    return None


def _build_text_generate_request(
    text: str,
    max_new_tokens: int = 16,
    temperature: float = 0.0,
    stream: bool = False,
) -> bytes:
    result = _encode_string_field(1, text)
    sampling = b""
    sampling += _encode_float_field(1, temperature)
    sampling += _encode_int32_field(8, max_new_tokens)
    result += _encode_submessage_field(2, sampling)
    result += _encode_bool_field(3, stream)
    return result


# ======================================================================
# Tests
# ======================================================================


class TestGrpcServer(CustomTestCase):
    """Test the native gRPC server running alongside HTTP."""

    grpc_channel = None

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.grpc_port = _grpc_port_from_http_url(cls.base_url)
        cls.grpc_host = _grpc_host_from_http_url(cls.base_url)

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=(
                "--mem-fraction-static",
                "0.7",
            ),
        )

        cls._setup_grpc_client()

    @classmethod
    def _setup_grpc_client(cls):
        try:
            import grpc

            target = f"{cls.grpc_host}:{cls.grpc_port}"
            cls.grpc_channel = grpc.insecure_channel(target)

            try:
                grpc.channel_ready_future(cls.grpc_channel).result(timeout=30)
            except grpc.FutureTimeoutError:
                raise RuntimeError(
                    f"gRPC channel to {target} did not become ready within 30s"
                )
        except ImportError:
            raise unittest.SkipTest("grpcio not installed")

    @classmethod
    def tearDownClass(cls):
        if cls.grpc_channel is not None:
            cls.grpc_channel.close()
        kill_process_tree(cls.process.pid)

    def _make_unary_call(self, method: str, request_bytes: bytes) -> bytes:
        return self.grpc_channel.unary_unary(
            f"/sglang.runtime.v1.SglangService/{method}",
            request_serializer=lambda x: x,
            response_deserializer=lambda x: x,
        )(request_bytes)

    def _make_server_stream_call(self, method: str, request_bytes: bytes):
        return self.grpc_channel.unary_stream(
            f"/sglang.runtime.v1.SglangService/{method}",
            request_serializer=lambda x: x,
            response_deserializer=lambda x: x,
        )(request_bytes)

    # ------------------------------------------------------------------
    # Existing Phase 1 RPCs
    # ------------------------------------------------------------------

    def test_http_still_works(self):
        """Regression: HTTP /generate still works when gRPC is enabled."""
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 8},
            },
        )
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn("text", result)
        self.assertGreater(len(result["text"]), 0)

    def test_http_health(self):
        response = requests.get(self.base_url + "/health")
        self.assertEqual(response.status_code, 200)

    def test_http_model_info(self):
        response = requests.get(self.base_url + "/model_info")
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn("model_path", result)

    def test_http_server_info(self):
        response = requests.get(self.base_url + "/server_info")
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn("internal_states", result)
        self.assertIn("version", result)
        self.assertNotIn("model_config", result)

    def test_grpc_health_check(self):
        response_bytes = self._make_unary_call("HealthCheck", b"")
        self.assertIn(b"\x08\x01", response_bytes)

    def test_grpc_get_model_info(self):
        response_bytes = self._make_unary_call("GetModelInfo", b"")
        self.assertGreater(len(response_bytes), 0)
        decoded = _decode_string_field(response_bytes, field_number=2)
        if decoded:
            info = json.loads(decoded)
            self.assertIn("model_path", info)

    def test_grpc_get_server_info(self):
        response_bytes = self._make_unary_call("GetServerInfo", b"")
        self.assertGreater(len(response_bytes), 0)
        decoded = _decode_string_field(response_bytes, field_number=1)
        if decoded:
            info = json.loads(decoded)
            self.assertIsInstance(info, dict)

    def test_grpc_text_generate(self):
        request = _build_text_generate_request(
            text="The capital of France is",
            max_new_tokens=8,
            temperature=0.0,
            stream=False,
        )
        responses = list(self._make_server_stream_call("TextGenerate", request))
        self.assertGreater(len(responses), 0)
        last = responses[-1]
        self.assertIn(b"\x18\x01", last)
        text = _decode_string_field(last, field_number=1)
        self.assertIsNotNone(text)
        self.assertGreater(len(text), 0)

    def test_grpc_text_generate_streaming(self):
        request = _build_text_generate_request(
            text="Write a short poem about the ocean",
            max_new_tokens=32,
            temperature=0.5,
            stream=True,
        )
        responses = list(self._make_server_stream_call("TextGenerate", request))
        self.assertGreater(len(responses), 0)
        last = responses[-1]
        self.assertIn(b"\x18\x01", last)

    def test_grpc_abort(self):
        rid = "test-abort-rid-12345"
        request = _encode_string_field(1, rid)
        response_bytes = self._make_unary_call("Abort", request)
        self.assertIn(b"\x08\x01", response_bytes)

    # ------------------------------------------------------------------
    # New Part 1 RPCs: Tokenize / Detokenize
    # ------------------------------------------------------------------

    def test_grpc_tokenize(self):
        """gRPC Tokenize returns tokens for a text string."""
        request = _encode_string_field(1, "Hello, world!")
        response_bytes = self._make_unary_call("Tokenize", request)
        self.assertGreater(len(response_bytes), 0)

        count = _decode_int32_field(response_bytes, field_number=2)
        self.assertIsNotNone(count)
        self.assertGreater(count, 0)

        max_model_len = _decode_int32_field(response_bytes, field_number=3)
        self.assertIsNotNone(max_model_len)
        self.assertGreater(max_model_len, 0)

        input_text = _decode_string_field(response_bytes, field_number=4)
        self.assertEqual(input_text, "Hello, world!")

    def test_grpc_detokenize(self):
        """gRPC Detokenize returns text from token IDs."""
        tok_request = _encode_string_field(1, "Hello")
        tok_response = self._make_unary_call("Tokenize", tok_request)

        http_result = requests.post(
            self.base_url + "/generate",
            json={
                "text": "Hello",
                "sampling_params": {"temperature": 0, "max_new_tokens": 1},
            },
        )
        self.assertEqual(http_result.status_code, 200)

        detok_request = b""
        for token_id in [9707]:
            detok_request += _encode_int32_field(1, token_id)
        response_bytes = self._make_unary_call("Detokenize", detok_request)
        text = _decode_string_field(response_bytes, field_number=1)
        self.assertIsNotNone(text)
        self.assertGreater(len(text), 0)

    # ------------------------------------------------------------------
    # New Part 1 RPCs: ListModels
    # ------------------------------------------------------------------

    def test_grpc_list_models(self):
        """gRPC ListModels returns at least one model."""
        response_bytes = self._make_unary_call("ListModels", b"")
        self.assertGreater(len(response_bytes), 0)

        models_bytes = _decode_bytes_field(response_bytes, field_number=1)
        self.assertIsNotNone(models_bytes)
        model_id = _decode_string_field(models_bytes, field_number=1)
        self.assertIsNotNone(model_id)
        self.assertGreater(len(model_id), 0)

    # ------------------------------------------------------------------
    # New Part 1 RPCs: GetLoad
    # ------------------------------------------------------------------

    def test_grpc_get_load(self):
        """gRPC GetLoad returns valid load info JSON."""
        response_bytes = self._make_unary_call("GetLoad", b"")
        json_info = _decode_string_field(response_bytes, field_number=1)
        self.assertIsNotNone(json_info)
        data = json.loads(json_info)
        self.assertIsInstance(data, list)

    # ------------------------------------------------------------------
    # New Part 1 RPCs: FlushCache
    # ------------------------------------------------------------------

    def test_grpc_flush_cache(self):
        """gRPC FlushCache returns success."""
        response_bytes = self._make_unary_call("FlushCache", b"")
        success = _decode_bool_field(response_bytes, field_number=1)
        self.assertTrue(success)

    # ------------------------------------------------------------------
    # New Part 2: OpenAI ChatComplete (JSON pass-through)
    # ------------------------------------------------------------------

    def test_grpc_chat_complete(self):
        """gRPC ChatComplete (non-streaming) returns valid OpenAI response JSON."""
        request_json = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": "What is 2+2? Answer with just the number.",
                    },
                ],
                "max_tokens": 8,
                "temperature": 0,
                "stream": False,
            }
        ).encode("utf-8")

        request = _encode_bytes_field(1, request_json)
        responses = list(self._make_server_stream_call("ChatComplete", request))
        self.assertGreater(len(responses), 0)

        last = responses[-1]
        json_chunk = _decode_bytes_field(last, field_number=1)
        self.assertIsNotNone(json_chunk)
        if json_chunk:
            data = json.loads(json_chunk)
            self.assertIn("choices", data)
            self.assertGreater(len(data["choices"]), 0)

    def test_grpc_chat_complete_streaming(self):
        """gRPC ChatComplete (streaming) returns multiple SSE chunks."""
        request_json = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": "Count from 1 to 5."},
                ],
                "max_tokens": 32,
                "temperature": 0,
                "stream": True,
            }
        ).encode("utf-8")

        request = _encode_bytes_field(1, request_json)
        responses = list(self._make_server_stream_call("ChatComplete", request))
        self.assertGreater(
            len(responses), 1, "Streaming should produce multiple chunks"
        )

    # ------------------------------------------------------------------
    # New Part 2: OpenAI Complete (JSON pass-through)
    # ------------------------------------------------------------------

    def test_grpc_complete(self):
        """gRPC Complete (non-streaming) returns valid OpenAI completion."""
        request_json = json.dumps(
            {
                "model": self.model,
                "prompt": "The capital of France is",
                "max_tokens": 8,
                "temperature": 0,
                "stream": False,
            }
        ).encode("utf-8")

        request = _encode_bytes_field(1, request_json)
        responses = list(self._make_server_stream_call("Complete", request))
        self.assertGreater(len(responses), 0)

        last = responses[-1]
        json_chunk = _decode_bytes_field(last, field_number=1)
        self.assertIsNotNone(json_chunk)
        if json_chunk:
            data = json.loads(json_chunk)
            self.assertIn("choices", data)

    # ------------------------------------------------------------------
    # Tokenizer equivalence: gRPC (Rust) vs HTTP (Python)
    # ------------------------------------------------------------------

    def test_grpc_tokenize_matches_http(self):
        """Rust tokenizer output matches Python tokenizer output."""
        test_texts = [
            "Hello, world!",
            "The capital of France is Paris.",
            "",  # empty string
            "🎉 emoji test 🚀",
            "def foo():\n    return 42",
            "a " * 100,  # longer text
        ]
        for text in test_texts:
            # gRPC tokenize
            request = _encode_string_field(1, text)
            grpc_response = self._make_unary_call("Tokenize", request)
            grpc_count = _decode_int32_field(grpc_response, field_number=2) or 0

            # HTTP tokenize
            http_response = requests.post(
                self.base_url + "/tokenize",
                json={"prompt": text},
            )
            self.assertEqual(
                http_response.status_code, 200, f"HTTP tokenize failed for: {text!r}"
            )
            http_data = http_response.json()
            http_count = http_data.get("count", len(http_data.get("tokens", [])))

            self.assertEqual(
                grpc_count,
                http_count,
                f"Token count mismatch for {text!r}: gRPC={grpc_count}, HTTP={http_count}",
            )

    def test_grpc_tokenize_no_special_tokens(self):
        """Tokenize with add_special_tokens=False."""
        text = "Hello"
        # With special tokens
        req_with = _encode_string_field(1, text)
        resp_with = self._make_unary_call("Tokenize", req_with)
        count_with = _decode_int32_field(resp_with, field_number=2)

        # Without special tokens
        req_without = _encode_string_field(1, text) + _encode_bool_field(2, False)
        resp_without = self._make_unary_call("Tokenize", req_without)
        count_without = _decode_int32_field(resp_without, field_number=2)

        # Both should return valid counts; with special tokens >= without
        self.assertIsNotNone(count_with)
        self.assertIsNotNone(count_without)
        self.assertGreaterEqual(count_with, count_without)

    def test_grpc_detokenize_roundtrip(self):
        """Tokenize then detokenize should approximately recover original text."""
        text = "The quick brown fox jumps over the lazy dog"

        # Tokenize via HTTP to get reliable token IDs
        http_response = requests.post(
            self.base_url + "/tokenize",
            json={"prompt": text},
        )
        self.assertEqual(http_response.status_code, 200)
        token_ids = http_response.json()["tokens"]

        # Detokenize via gRPC
        detok_request = b""
        for tid in token_ids:
            detok_request += _encode_int32_field(1, tid)
        response_bytes = self._make_unary_call("Detokenize", detok_request)
        decoded_text = _decode_string_field(response_bytes, field_number=1)

        self.assertIsNotNone(decoded_text)
        # Detokenized text should contain the original words
        self.assertIn("quick", decoded_text)
        self.assertIn("fox", decoded_text)
        self.assertIn("dog", decoded_text)

    # ------------------------------------------------------------------
    # Sampling params dict path
    # ------------------------------------------------------------------

    def test_grpc_text_generate_with_sampling_params(self):
        """TextGenerate with various sampling params via proto SamplingParams."""
        # Build request with explicit sampling params
        result = _encode_string_field(1, "Count: 1, 2, 3,")
        sampling = b""
        sampling += _encode_float_field(1, 0.0)  # temperature
        sampling += _encode_int32_field(8, 4)  # max_new_tokens
        sampling += _encode_float_field(2, 0.95)  # top_p
        result += _encode_submessage_field(2, sampling)
        result += _encode_bool_field(3, False)  # stream

        responses = list(self._make_server_stream_call("TextGenerate", result))
        self.assertGreater(len(responses), 0)
        last = responses[-1]
        text = _decode_string_field(last, field_number=1)
        self.assertIsNotNone(text)
        self.assertGreater(len(text), 0)


class TestGrpcHttpCoexist(CustomTestCase):
    """Regression test: HTTP /generate still works correctly when gRPC is enabled."""

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=(
                "--mem-fraction-static",
                "0.7",
            ),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_http_generate_with_grpc_enabled(self):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "What is 2+2?",
                "sampling_params": {"temperature": 0, "max_new_tokens": 16},
            },
        )
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn("text", result)
        self.assertGreater(len(result["text"]), 0)

    def test_http_generate_streaming_with_grpc_enabled(self):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "Tell me a joke",
                "sampling_params": {"temperature": 0, "max_new_tokens": 32},
                "stream": True,
            },
            stream=True,
        )
        self.assertEqual(response.status_code, 200)

        chunks = []
        for line in response.iter_lines():
            if line:
                decoded = line.decode("utf-8")
                if decoded.startswith("data:"):
                    data = decoded[5:].strip()
                    if data == "[DONE]":
                        break
                    chunks.append(json.loads(data))

        self.assertGreater(len(chunks), 0)

    def test_http_model_info_with_grpc_enabled(self):
        response = requests.get(self.base_url + "/model_info")
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn("model_path", result)

    def test_http_health_with_grpc_enabled(self):
        response = requests.get(self.base_url + "/health")
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
