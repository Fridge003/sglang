import os
import pickle
import shutil
import tempfile
import threading
import time
import unittest

import zmq

from sglang.srt.utils.common import SafeUnpickler, safe_pickle_load
from sglang.srt.utils.gen_zmq_keys import generate_certificates
from sglang.srt.utils.network import (
    CurveConfig,
    apply_curve_client,
    apply_curve_server,
    connect_with_curve,
    get_curve_config,
    get_zmq_socket,
    get_zmq_socket_on_host,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")

CURVE_AVAILABLE = zmq.has("curve")


@unittest.skipUnless(CURVE_AVAILABLE, "libzmq built without CURVE support")
class TestGenZmqKeys(CustomTestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_generates_key_files(self):
        generate_certificates(self.tmp_dir)
        self.assertTrue(os.path.isfile(os.path.join(self.tmp_dir, "cluster.key")))
        self.assertTrue(
            os.path.isfile(os.path.join(self.tmp_dir, "cluster.key_secret"))
        )

    def test_key_files_are_loadable(self):
        generate_certificates(self.tmp_dir)
        pub, sec = zmq.auth.load_certificate(
            os.path.join(self.tmp_dir, "cluster.key_secret")
        )
        self.assertIsNotNone(pub)
        self.assertIsNotNone(sec)
        self.assertGreater(len(pub), 0)
        self.assertGreater(len(sec), 0)


@unittest.skipUnless(CURVE_AVAILABLE, "libzmq built without CURVE support")
class TestCurveConfig(CustomTestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        generate_certificates(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_from_keys_dir(self):
        cfg = CurveConfig.from_keys_dir(self.tmp_dir)
        self.assertIsNotNone(cfg.public_key)
        self.assertIsNotNone(cfg.secret_key)
        self.assertGreater(len(cfg.public_key), 0)
        self.assertGreater(len(cfg.secret_key), 0)

    def test_from_keys_dir_missing_file(self):
        empty_dir = tempfile.mkdtemp()
        try:
            with self.assertRaises(Exception):
                CurveConfig.from_keys_dir(empty_dir)
        finally:
            shutil.rmtree(empty_dir, ignore_errors=True)


@unittest.skipUnless(CURVE_AVAILABLE, "libzmq built without CURVE support")
class TestGetCurveConfigDisabled(CustomTestCase):
    def test_returns_none_when_env_unset(self):
        import sglang.srt.utils.network as net_mod

        old_cache = net_mod._curve_config_cache
        old_loaded = net_mod._curve_config_loaded
        try:
            net_mod._curve_config_cache = None
            net_mod._curve_config_loaded = False
            env_backup = os.environ.pop("SGLANG_ZMQ_CURVE_KEYS_DIR", None)
            try:
                result = get_curve_config()
                self.assertIsNone(result)
            finally:
                if env_backup is not None:
                    os.environ["SGLANG_ZMQ_CURVE_KEYS_DIR"] = env_backup
        finally:
            net_mod._curve_config_cache = old_cache
            net_mod._curve_config_loaded = old_loaded


@unittest.skipUnless(CURVE_AVAILABLE, "libzmq built without CURVE support")
class TestCurveZMQConnection(CustomTestCase):
    """Verify that CURVE-authenticated PUSH/PULL sockets can communicate,
    and that an unauthenticated client is rejected."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        generate_certificates(self.tmp_dir)
        self.curve = CurveConfig.from_keys_dir(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_authenticated_push_pull(self):
        ctx = zmq.Context()
        try:
            server = ctx.socket(zmq.PULL)
            apply_curve_server(server, self.curve)
            port = server.bind_to_random_port("tcp://127.0.0.1")

            client = ctx.socket(zmq.PUSH)
            apply_curve_client(client, self.curve)
            client.connect(f"tcp://127.0.0.1:{port}")

            client.send(b"hello-curve")
            self.assertTrue(server.poll(timeout=3000))
            msg = server.recv()
            self.assertEqual(msg, b"hello-curve")

            client.close()
            server.close()
        finally:
            ctx.term()

    def test_unauthenticated_client_rejected(self):
        ctx = zmq.Context()
        try:
            server = ctx.socket(zmq.PULL)
            apply_curve_server(server, self.curve)
            port = server.bind_to_random_port("tcp://127.0.0.1")

            # Plain (non-CURVE) client trying to connect
            plain_client = ctx.socket(zmq.PUSH)
            plain_client.setsockopt(zmq.LINGER, 0)
            plain_client.setsockopt(zmq.SNDTIMEO, 500)
            plain_client.connect(f"tcp://127.0.0.1:{port}")

            try:
                plain_client.send(b"should-fail")
            except zmq.Again:
                pass

            # The server should NOT receive the message
            self.assertFalse(server.poll(timeout=1000))

            plain_client.close()
            server.close()
        finally:
            ctx.term()

    def test_wrong_key_client_rejected(self):
        """A client with a different keypair cannot talk to the server."""
        wrong_dir = tempfile.mkdtemp()
        try:
            generate_certificates(wrong_dir)
            wrong_curve = CurveConfig.from_keys_dir(wrong_dir)

            ctx = zmq.Context()
            try:
                server = ctx.socket(zmq.PULL)
                apply_curve_server(server, self.curve)
                port = server.bind_to_random_port("tcp://127.0.0.1")

                client = ctx.socket(zmq.PUSH)
                client.setsockopt(zmq.LINGER, 0)
                client.setsockopt(zmq.SNDTIMEO, 500)
                apply_curve_client(client, wrong_curve)
                client.connect(f"tcp://127.0.0.1:{port}")

                try:
                    client.send(b"wrong-key-msg")
                except zmq.Again:
                    pass

                self.assertFalse(server.poll(timeout=1000))

                client.close()
                server.close()
            finally:
                ctx.term()
        finally:
            shutil.rmtree(wrong_dir, ignore_errors=True)


class TestLocalhostBinding(CustomTestCase):
    """Verify that get_zmq_socket_on_host and get_zmq_socket bind to
    localhost by default, not to all interfaces (CVE-2026-3059/3060)."""

    def test_get_zmq_socket_on_host_defaults_to_localhost(self):
        ctx = zmq.Context()
        try:
            port, sock = get_zmq_socket_on_host(ctx, zmq.PULL)
            endpoint = sock.getsockopt_string(zmq.LAST_ENDPOINT)
            self.assertIn("127.0.0.1", endpoint)
            self.assertNotIn("0.0.0.0", endpoint)
            sock.close()
        finally:
            ctx.term()

    def test_get_zmq_socket_on_host_explicit_host(self):
        ctx = zmq.Context()
        try:
            port, sock = get_zmq_socket_on_host(ctx, zmq.PULL, host="127.0.0.1")
            endpoint = sock.getsockopt_string(zmq.LAST_ENDPOINT)
            self.assertIn("127.0.0.1", endpoint)
            sock.close()
        finally:
            ctx.term()

    def test_get_zmq_socket_random_port_binds_localhost(self):
        ctx = zmq.Context()
        try:
            port, sock = get_zmq_socket(ctx, zmq.PULL)
            endpoint = sock.getsockopt_string(zmq.LAST_ENDPOINT)
            self.assertIn("127.0.0.1", endpoint)
            self.assertNotIn("0.0.0.0", endpoint)
            sock.close()
        finally:
            ctx.term()

    def test_localhost_socket_reachable_locally(self):
        ctx = zmq.Context()
        try:
            port, server = get_zmq_socket_on_host(ctx, zmq.PULL)
            client = ctx.socket(zmq.PUSH)
            client.connect(f"tcp://127.0.0.1:{port}")
            client.send(b"test-localhost")
            self.assertTrue(server.poll(timeout=2000))
            msg = server.recv()
            self.assertEqual(msg, b"test-localhost")
            client.close()
            server.close()
        finally:
            ctx.term()


def _import_scheduler_client():
    """Import scheduler_client directly from its file path, bypassing the
    heavy sglang.multimodal_gen package __init__ which pulls in imageio."""
    import importlib.util
    import sys

    mod_name = "sglang.multimodal_gen.runtime.scheduler_client"
    if mod_name in sys.modules:
        return sys.modules[mod_name]

    from types import ModuleType

    package_stubs = [
        "sglang.multimodal_gen",
        "sglang.multimodal_gen.runtime",
        "sglang.multimodal_gen.runtime.utils",
    ]
    for stub_name in package_stubs:
        if stub_name not in sys.modules:
            stub = ModuleType(stub_name)
            stub.__path__ = []
            stub.__package__ = stub_name
            sys.modules[stub_name] = stub

    logging_mod_name = "sglang.multimodal_gen.runtime.utils.logging_utils"
    if logging_mod_name not in sys.modules:
        log_stub = ModuleType(logging_mod_name)
        import logging as _logging

        log_stub.init_logger = lambda name: _logging.getLogger(name)
        sys.modules[logging_mod_name] = log_stub

    server_args_mod_name = "sglang.multimodal_gen.runtime.server_args"
    if server_args_mod_name not in sys.modules:
        sa_stub = ModuleType(server_args_mod_name)
        sa_stub.ServerArgs = type("ServerArgs", (), {})
        sys.modules[server_args_mod_name] = sa_stub

    sc_file = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "python",
        "sglang",
        "multimodal_gen",
        "runtime",
        "scheduler_client.py",
    )
    sc_file = os.path.abspath(sc_file)
    spec = importlib.util.spec_from_file_location(mod_name, sc_file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestSchedulerClientSecurity(CustomTestCase):
    """Verify that run_zeromq_broker binds to localhost and applies CURVE,
    and that SchedulerClient applies CURVE on its socket (CVE-2026-3059)."""

    @classmethod
    def setUpClass(cls):
        cls.sc_mod = _import_scheduler_client()

    def test_broker_binds_to_localhost(self):
        """run_zeromq_broker must bind to 127.0.0.1, not tcp://*."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock, patch

        mock_server_args = MagicMock()
        mock_server_args.broker_port = 0

        mock_socket = MagicMock()
        mock_socket.recv = AsyncMock(side_effect=asyncio.CancelledError)
        mock_socket.bind = MagicMock()

        mock_ctx = MagicMock()
        mock_ctx.socket.return_value = mock_socket

        with patch("zmq.asyncio.Context", return_value=mock_ctx), patch(
            "sglang.srt.utils.network.get_curve_config", return_value=None
        ):
            try:
                asyncio.run(
                    asyncio.wait_for(
                        self.sc_mod.run_zeromq_broker(mock_server_args), 0.1
                    )
                )
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

        bound_endpoint = mock_socket.bind.call_args[0][0]
        self.assertIn("127.0.0.1", bound_endpoint, "Broker must bind to localhost")
        self.assertNotIn("*", bound_endpoint, "Broker must not bind to tcp://*")

    @unittest.skipUnless(CURVE_AVAILABLE, "libzmq built without CURVE support")
    def test_broker_applies_curve_when_enabled(self):
        """run_zeromq_broker must call apply_curve_server when CURVE is configured."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock, patch

        tmp_dir = tempfile.mkdtemp()
        try:
            generate_certificates(tmp_dir)
            curve = CurveConfig.from_keys_dir(tmp_dir)

            mock_socket = MagicMock()
            mock_socket.recv = AsyncMock(side_effect=asyncio.CancelledError)

            mock_ctx = MagicMock()
            mock_ctx.socket.return_value = mock_socket

            mock_server_args = MagicMock()
            mock_server_args.broker_port = 0

            with patch("zmq.asyncio.Context", return_value=mock_ctx), patch(
                "sglang.srt.utils.network.get_curve_config", return_value=curve
            ):
                try:
                    asyncio.run(
                        asyncio.wait_for(
                            self.sc_mod.run_zeromq_broker(mock_server_args), 0.1
                        )
                    )
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass

            self.assertTrue(
                mock_socket.curve_server,
                "Broker socket must have curve_server=True",
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    @unittest.skipUnless(CURVE_AVAILABLE, "libzmq built without CURVE support")
    def test_broker_rejects_unauthenticated_client(self):
        """A real broker REP socket with CURVE should reject plain clients."""
        tmp_dir = tempfile.mkdtemp()
        try:
            generate_certificates(tmp_dir)
            curve = CurveConfig.from_keys_dir(tmp_dir)

            ctx = zmq.Context()
            server = ctx.socket(zmq.REP)
            apply_curve_server(server, curve)
            port = server.bind_to_random_port("tcp://127.0.0.1")

            plain_client = ctx.socket(zmq.REQ)
            plain_client.setsockopt(zmq.LINGER, 0)
            plain_client.setsockopt(zmq.SNDTIMEO, 500)
            plain_client.setsockopt(zmq.RCVTIMEO, 500)
            plain_client.connect(f"tcp://127.0.0.1:{port}")

            try:
                plain_client.send(b"attack-payload")
            except zmq.Again:
                pass

            self.assertFalse(
                server.poll(timeout=1000),
                "CURVE-protected server must reject plain client",
            )

            plain_client.close()
            server.close()
            ctx.term()
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    @unittest.skipUnless(CURVE_AVAILABLE, "libzmq built without CURVE support")
    def test_scheduler_client_uses_connect_with_curve(self):
        """SchedulerClient.initialize must use connect_with_curve (not manual boilerplate)."""
        from unittest.mock import MagicMock, patch

        mock_server_args = MagicMock()
        mock_server_args.scheduler_endpoint = "tcp://127.0.0.1:9999"

        client = self.sc_mod.SchedulerClient()
        with patch(
            "sglang.srt.utils.network.connect_with_curve"
        ) as mock_connect:
            client.initialize(mock_server_args)

        mock_connect.assert_called_once()
        call_args = mock_connect.call_args[0]
        self.assertEqual(
            call_args[1],
            "tcp://127.0.0.1:9999",
            "connect_with_curve must be called with the scheduler endpoint",
        )
        client.close()


@unittest.skipUnless(CURVE_AVAILABLE, "libzmq built without CURVE support")
class TestConnectWithCurve(CustomTestCase):
    """Verify connect_with_curve() applies CURVE for TCP and skips for IPC."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        generate_certificates(self.tmp_dir)
        self.curve = CurveConfig.from_keys_dir(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_tcp_applies_curve_and_connects(self):
        """connect_with_curve must apply CURVE client opts for TCP endpoints."""
        from unittest.mock import patch

        ctx = zmq.Context()
        try:
            server = ctx.socket(zmq.PULL)
            apply_curve_server(server, self.curve)
            port = server.bind_to_random_port("tcp://127.0.0.1")

            client = ctx.socket(zmq.PUSH)
            client.setsockopt(zmq.LINGER, 0)
            with patch(
                "sglang.srt.utils.network.get_curve_config", return_value=self.curve
            ):
                connect_with_curve(client, f"tcp://127.0.0.1:{port}")

            client.send(b"via-helper")
            self.assertTrue(server.poll(timeout=3000))
            msg = server.recv()
            self.assertEqual(msg, b"via-helper")

            client.close()
            server.close()
        finally:
            ctx.term()

    def test_ipc_skips_curve(self):
        """connect_with_curve must skip CURVE for IPC endpoints."""
        from unittest.mock import MagicMock, patch

        mock_socket = MagicMock()
        with patch(
            "sglang.srt.utils.network.get_curve_config", return_value=self.curve
        ), patch("sglang.srt.utils.network.apply_curve_client") as mock_apply:
            connect_with_curve(mock_socket, "ipc:///tmp/test.sock")

        mock_apply.assert_not_called()
        mock_socket.connect.assert_called_once_with("ipc:///tmp/test.sock")

    def test_explicit_curve_config(self):
        """connect_with_curve must use an explicitly provided CurveConfig."""
        from unittest.mock import patch

        ctx = zmq.Context()
        try:
            server = ctx.socket(zmq.PULL)
            apply_curve_server(server, self.curve)
            port = server.bind_to_random_port("tcp://127.0.0.1")

            client = ctx.socket(zmq.PUSH)
            client.setsockopt(zmq.LINGER, 0)
            connect_with_curve(
                client, f"tcp://127.0.0.1:{port}", curve=self.curve
            )

            client.send(b"explicit-curve")
            self.assertTrue(server.poll(timeout=3000))
            msg = server.recv()
            self.assertEqual(msg, b"explicit-curve")

            client.close()
            server.close()
        finally:
            ctx.term()


class TestMMSchedulerUsesSrtGetZmqSocket(CustomTestCase):
    """Verify the multimodal scheduler imports get_zmq_socket from
    srt/utils/network.py (the hardened version with CURVE support)."""

    def test_mm_scheduler_imports_srt_get_zmq_socket(self):
        """The multimodal scheduler must not use its own unhardened get_zmq_socket."""
        import importlib.util
        import sys
        from types import ModuleType

        mm_sched_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "python",
            "sglang",
            "multimodal_gen",
            "runtime",
            "managers",
            "scheduler.py",
        )
        mm_sched_path = os.path.abspath(mm_sched_path)
        with open(mm_sched_path) as f:
            source = f.read()

        self.assertIn(
            "from sglang.srt.utils.network import get_zmq_socket",
            source,
            "multimodal scheduler must import get_zmq_socket from srt.utils.network",
        )
        self.assertNotIn(
            "from sglang.multimodal_gen.runtime.utils.common import get_zmq_socket",
            source,
            "multimodal scheduler must NOT import get_zmq_socket from multimodal_gen",
        )


class TestSafePickleLoad(CustomTestCase):
    """Verify SafeUnpickler blocks RCE gadgets (CVE-2026-3989)."""

    def test_safe_load_normal_data(self):
        data = {"requests": [{"text": "hello", "id": 1}], "count": 42}
        buf = pickle.dumps(data)
        import io

        result = safe_pickle_load(io.BytesIO(buf))
        self.assertEqual(result, data)

    def test_safe_load_blocks_os_system(self):
        class Exploit:
            def __reduce__(self):
                import os

                return (os.system, ("echo pwned",))

        buf = pickle.dumps(Exploit())
        import io

        with self.assertRaises(RuntimeError):
            safe_pickle_load(io.BytesIO(buf))

    def test_safe_load_blocks_subprocess(self):
        class Exploit:
            def __reduce__(self):
                import subprocess

                return (subprocess.Popen, (["echo", "pwned"],))

        buf = pickle.dumps(Exploit())
        import io

        with self.assertRaises(RuntimeError):
            safe_pickle_load(io.BytesIO(buf))

    def test_safe_load_blocks_eval(self):
        class Exploit:
            def __reduce__(self):
                return (eval, ("__import__('os').system('echo pwned')",))

        buf = pickle.dumps(Exploit())
        import io

        with self.assertRaises(RuntimeError):
            safe_pickle_load(io.BytesIO(buf))


if __name__ == "__main__":
    unittest.main()
