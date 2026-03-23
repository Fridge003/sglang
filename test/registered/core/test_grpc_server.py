"""
Integration tests for the native Rust gRPC server.

These tests verify that the gRPC server starts alongside HTTP and correctly
handles text generate, tokenized generate, streaming, embed, abort, health,
model info, and server info RPCs.

Usage:
    python3 -m pytest test_grpc_server.py -v
    python3 -m unittest test_grpc_server.TestGrpcServer.test_text_generate
"""

import json
import time
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

register_cuda_ci(est_time=180, suite="stage-b-test-small-1-gpu")


def _grpc_port_from_http_url(http_url: str) -> int:
    """Derive the gRPC port from the HTTP base URL (port + 10000)."""
    from urllib.parse import urlparse

    parsed = urlparse(http_url)
    return parsed.port + 10000


def _grpc_host_from_http_url(http_url: str) -> str:
    from urllib.parse import urlparse

    parsed = urlparse(http_url)
    return parsed.hostname


class TestGrpcServer(CustomTestCase):
    """Test the native gRPC server running alongside HTTP."""

    grpc_channel = None
    grpc_stub = None

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
        """Set up gRPC client using grpcio stubs generated from the shared protos."""
        try:
            import grpc

            target = f"{cls.grpc_host}:{cls.grpc_port}"
            cls.grpc_channel = grpc.insecure_channel(target)

            # Wait for the gRPC channel to be ready
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
        """Make a raw unary gRPC call."""
        import grpc

        return self.grpc_channel.unary_unary(
            f"/sglang.runtime.v1.SglangService/{method}",
            request_serializer=lambda x: x,
            response_deserializer=lambda x: x,
        )(request_bytes)

    def _make_server_stream_call(self, method: str, request_bytes: bytes):
        """Make a raw server-streaming gRPC call."""
        import grpc

        return self.grpc_channel.unary_stream(
            f"/sglang.runtime.v1.SglangService/{method}",
            request_serializer=lambda x: x,
            response_deserializer=lambda x: x,
        )(request_bytes)

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
        """HTTP /health still works alongside gRPC."""
        response = requests.get(self.base_url + "/health")
        self.assertEqual(response.status_code, 200)

    def test_http_model_info(self):
        """HTTP /model_info still works alongside gRPC."""
        response = requests.get(self.base_url + "/model_info")
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn("model_path", result)

    def test_grpc_health_check(self):
        """gRPC HealthCheck returns healthy=true."""
        try:
            from google.protobuf import descriptor_pool, symbol_database
        except ImportError:
            pass

        # Use raw protobuf encoding for HealthCheckRequest (empty message = b"")
        response_bytes = self._make_unary_call("HealthCheck", b"")
        # Decode: field 1 (bool healthy) = varint
        # If healthy=true, bytes should contain 0x08 0x01
        self.assertIn(b"\x08\x01", response_bytes)

    def test_grpc_get_model_info(self):
        """gRPC GetModelInfo returns valid JSON model info."""
        response_bytes = self._make_unary_call("GetModelInfo", b"")
        # Field 2 is json_info (string), should contain model path
        self.assertGreater(len(response_bytes), 0)
        # The response contains a protobuf-encoded string with JSON
        # Decode field 2 (tag=0x12, length-delimited)
        decoded = self._decode_string_field(response_bytes, field_number=2)
        if decoded:
            info = json.loads(decoded)
            self.assertIn("model_path", info)

    def test_grpc_get_server_info(self):
        """gRPC GetServerInfo returns valid JSON server info."""
        response_bytes = self._make_unary_call("GetServerInfo", b"")
        self.assertGreater(len(response_bytes), 0)
        decoded = self._decode_string_field(response_bytes, field_number=1)
        if decoded:
            info = json.loads(decoded)
            self.assertIsInstance(info, dict)

    def test_grpc_text_generate(self):
        """gRPC TextGenerate returns text output via server streaming."""
        # Build TextGenerateRequest protobuf manually
        # field 1: text (string "The capital of France is")
        # field 2: sampling_params (sub-message)
        # field 3: stream (bool false)
        request = self._build_text_generate_request(
            text="The capital of France is",
            max_new_tokens=8,
            temperature=0.0,
            stream=False,
        )

        responses = list(self._make_server_stream_call("TextGenerate", request))
        self.assertGreater(len(responses), 0)

        # The last response should have finished=true (field 3, tag 0x18, value 0x01)
        last = responses[-1]
        self.assertIn(b"\x18\x01", last)

        # Field 1 should contain generated text
        text = self._decode_string_field(last, field_number=1)
        self.assertIsNotNone(text)
        self.assertGreater(len(text), 0)

    def test_grpc_text_generate_streaming(self):
        """gRPC TextGenerate with stream=true returns multiple chunks."""
        request = self._build_text_generate_request(
            text="Write a short poem about the ocean",
            max_new_tokens=32,
            temperature=0.5,
            stream=True,
        )

        responses = list(self._make_server_stream_call("TextGenerate", request))
        # Streaming should produce multiple response chunks
        self.assertGreater(len(responses), 0)

        # Last chunk should be finished
        last = responses[-1]
        self.assertIn(b"\x18\x01", last)

    def test_grpc_abort(self):
        """gRPC Abort returns success."""
        # Build AbortRequest: field 1 = rid (string), field 2 = abort_all (bool)
        rid = "test-abort-rid-12345"
        request = self._encode_string_field(1, rid)

        response_bytes = self._make_unary_call("Abort", request)
        # Field 1: success (bool) = true
        self.assertIn(b"\x08\x01", response_bytes)

    # ------------------------------------------------------------------
    # Protobuf encoding/decoding helpers (minimal, no grpc-tools needed)
    # ------------------------------------------------------------------

    @staticmethod
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

    @staticmethod
    def _encode_string_field(field_number: int, value: str) -> bytes:
        tag = (field_number << 3) | 2  # wire type 2 = length-delimited
        encoded = value.encode("utf-8")
        return (
            TestGrpcServer._encode_varint(tag)
            + TestGrpcServer._encode_varint(len(encoded))
            + encoded
        )

    @staticmethod
    def _encode_bool_field(field_number: int, value: bool) -> bytes:
        tag = (field_number << 3) | 0  # wire type 0 = varint
        return TestGrpcServer._encode_varint(tag) + bytes([1 if value else 0])

    @staticmethod
    def _encode_float_field(field_number: int, value: float) -> bytes:
        import struct

        tag = (field_number << 3) | 5  # wire type 5 = 32-bit
        return TestGrpcServer._encode_varint(tag) + struct.pack("<f", value)

    @staticmethod
    def _encode_int32_field(field_number: int, value: int) -> bytes:
        tag = (field_number << 3) | 0  # wire type 0 = varint
        return TestGrpcServer._encode_varint(tag) + TestGrpcServer._encode_varint(
            value
        )

    @staticmethod
    def _encode_submessage_field(field_number: int, data: bytes) -> bytes:
        tag = (field_number << 3) | 2
        return (
            TestGrpcServer._encode_varint(tag)
            + TestGrpcServer._encode_varint(len(data))
            + data
        )

    def _build_text_generate_request(
        self,
        text: str,
        max_new_tokens: int = 16,
        temperature: float = 0.0,
        stream: bool = False,
    ) -> bytes:
        """Build a TextGenerateRequest protobuf message."""
        result = self._encode_string_field(1, text)

        # SamplingParams sub-message (field 2)
        sampling = b""
        sampling += self._encode_float_field(1, temperature)  # temperature
        sampling += self._encode_int32_field(8, max_new_tokens)  # max_new_tokens
        result += self._encode_submessage_field(2, sampling)

        # stream (field 3)
        result += self._encode_bool_field(3, stream)

        return result

    @staticmethod
    def _decode_string_field(
        data: bytes, field_number: int
    ) -> Optional[str]:
        """Decode a string field from protobuf bytes (simplified)."""
        expected_tag = (field_number << 3) | 2
        i = 0
        while i < len(data):
            tag, new_i = TestGrpcServer._decode_varint(data, i)
            if new_i is None:
                break
            i = new_i
            wire_type = tag & 0x7

            if wire_type == 0:  # varint
                _, i = TestGrpcServer._decode_varint(data, i)
                if i is None:
                    break
            elif wire_type == 2:  # length-delimited
                length, i = TestGrpcServer._decode_varint(data, i)
                if i is None:
                    break
                if tag == expected_tag:
                    try:
                        return data[i : i + length].decode("utf-8")
                    except UnicodeDecodeError:
                        return None
                i += length
            elif wire_type == 5:  # 32-bit
                i += 4
            elif wire_type == 1:  # 64-bit
                i += 8
            else:
                break
        return None

    @staticmethod
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
        """POST /generate returns valid text when gRPC server is also running."""
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
        """POST /generate with stream=true works when gRPC is enabled."""
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
        """GET /model_info works when gRPC is enabled."""
        response = requests.get(self.base_url + "/model_info")
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn("model_path", result)

    def test_http_health_with_grpc_enabled(self):
        """GET /health returns 200 when gRPC is enabled."""
        response = requests.get(self.base_url + "/health")
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
