import os
import pickle
import shutil
import tempfile
import time
import unittest

import zmq

from sglang.srt.utils.common import safe_pickle_load
from sglang.srt.utils.gen_zmq_keys import generate_certificates
from sglang.srt.utils.network import (
    CurveConfig,
    apply_curve_client,
    apply_curve_server,
    connect_with_curve,
    get_curve_config,
    get_zmq_socket,
    get_zmq_socket_on_host,
    set_curve_config,
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


def _reset_curve_state():
    """Reset global CurveConfig cache so get_curve_config() re-evaluates."""
    import sglang.srt.utils.network as net_mod

    saved = (net_mod._curve_config_cache, net_mod._curve_config_loaded)
    net_mod._curve_config_cache = None
    net_mod._curve_config_loaded = False
    return saved


def _restore_curve_state(saved):
    import sglang.srt.utils.network as net_mod

    net_mod._curve_config_cache, net_mod._curve_config_loaded = saved


@unittest.skipUnless(CURVE_AVAILABLE, "libzmq built without CURVE support")
class TestGetCurveConfigDisabled(CustomTestCase):
    def test_returns_none_when_no_zmq_curve_set(self):
        saved = _reset_curve_state()
        try:
            os.environ["SGLANG_NO_ZMQ_CURVE"] = "1"
            try:
                result = get_curve_config()
                self.assertIsNone(result)
            finally:
                os.environ.pop("SGLANG_NO_ZMQ_CURVE", None)
        finally:
            _restore_curve_state(saved)


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

    def test_get_zmq_socket_random_port_binds_all_interfaces(self):
        ctx = zmq.Context()
        try:
            port, sock = get_zmq_socket(ctx, zmq.PULL)
            endpoint = sock.getsockopt_string(zmq.LAST_ENDPOINT)
            self.assertIn("0.0.0.0", endpoint)
            sock.close()
        finally:
            ctx.term()

    def test_localhost_socket_reachable_locally(self):
        ctx = zmq.Context()
        try:
            port, server = get_zmq_socket_on_host(ctx, zmq.PULL)
            client = ctx.socket(zmq.PUSH)
            connect_with_curve(client, f"tcp://127.0.0.1:{port}")
            client.send(b"test-localhost")
            self.assertTrue(server.poll(timeout=3000))
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

        with patch("zmq.asyncio.Context", return_value=mock_ctx), patch.object(
            self.sc_mod, "get_curve_config", return_value=None
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

            with patch("zmq.asyncio.Context", return_value=mock_ctx), patch.object(
                self.sc_mod, "get_curve_config", return_value=curve
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
        with patch.object(self.sc_mod, "connect_with_curve") as mock_connect:
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

        ctx = zmq.Context()
        try:
            server = ctx.socket(zmq.PULL)
            apply_curve_server(server, self.curve)
            port = server.bind_to_random_port("tcp://127.0.0.1")

            client = ctx.socket(zmq.PUSH)
            client.setsockopt(zmq.LINGER, 0)
            connect_with_curve(client, f"tcp://127.0.0.1:{port}", curve=self.curve)

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


@unittest.skipUnless(CURVE_AVAILABLE, "libzmq built without CURVE support")
class TestCurveConfigGenerate(CustomTestCase):
    """Verify CurveConfig.generate() creates a valid in-memory keypair."""

    def test_generate_returns_valid_keypair(self):
        cfg = CurveConfig.generate()
        self.assertIsNotNone(cfg.public_key)
        self.assertIsNotNone(cfg.secret_key)
        self.assertEqual(len(cfg.public_key), 40)
        self.assertEqual(len(cfg.secret_key), 40)

    def test_generate_produces_unique_keys(self):
        a = CurveConfig.generate()
        b = CurveConfig.generate()
        self.assertNotEqual(a.public_key, b.public_key)
        self.assertNotEqual(a.secret_key, b.secret_key)


@unittest.skipUnless(CURVE_AVAILABLE, "libzmq built without CURVE support")
class TestCurveConfigFromRawEnv(CustomTestCase):
    """Verify CurveConfig.from_raw_env() reads Z85 keys from env vars."""

    def test_from_raw_env(self):
        cfg = CurveConfig.generate()
        os.environ["SGLANG_ZMQ_CURVE_PUBLIC_KEY"] = cfg.public_key.decode("ascii")
        os.environ["SGLANG_ZMQ_CURVE_SECRET_KEY"] = cfg.secret_key.decode("ascii")
        try:
            loaded = CurveConfig.from_raw_env()
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.public_key, cfg.public_key)
            self.assertEqual(loaded.secret_key, cfg.secret_key)
        finally:
            os.environ.pop("SGLANG_ZMQ_CURVE_PUBLIC_KEY", None)
            os.environ.pop("SGLANG_ZMQ_CURVE_SECRET_KEY", None)

    def test_from_raw_env_returns_none_when_unset(self):
        os.environ.pop("SGLANG_ZMQ_CURVE_PUBLIC_KEY", None)
        os.environ.pop("SGLANG_ZMQ_CURVE_SECRET_KEY", None)
        self.assertIsNone(CurveConfig.from_raw_env())


@unittest.skipUnless(CURVE_AVAILABLE, "libzmq built without CURVE support")
class TestAutoGenViaCurveConfig(CustomTestCase):
    """Verify get_curve_config() auto-generates when no env/flags are set."""

    def test_auto_generates_keypair(self):
        saved = _reset_curve_state()
        os.environ.pop("SGLANG_ZMQ_CURVE_KEYS_DIR", None)
        os.environ.pop("SGLANG_ZMQ_CURVE_PUBLIC_KEY", None)
        os.environ.pop("SGLANG_ZMQ_CURVE_SECRET_KEY", None)
        os.environ.pop("SGLANG_NO_ZMQ_CURVE", None)
        try:
            cfg = get_curve_config()
            self.assertIsNotNone(cfg)
            self.assertEqual(len(cfg.public_key), 40)
            self.assertEqual(len(cfg.secret_key), 40)
        finally:
            _restore_curve_state(saved)


@unittest.skipUnless(CURVE_AVAILABLE, "libzmq built without CURVE support")
class TestSetCurveConfig(CustomTestCase):
    """Verify set_curve_config() injects a config that get_curve_config() returns."""

    def test_set_and_get(self):
        saved = _reset_curve_state()
        try:
            injected = CurveConfig.generate()
            set_curve_config(injected)
            result = get_curve_config()
            self.assertIs(result, injected)
        finally:
            _restore_curve_state(saved)


@unittest.skipUnless(CURVE_AVAILABLE, "libzmq built without CURVE support")
class TestConnectWithCurveServerKey(CustomTestCase):
    """Verify connect_with_curve passes server_public_key correctly."""

    def test_explicit_server_public_key(self):
        from unittest.mock import MagicMock, patch

        server_cfg = CurveConfig.generate()
        client_cfg = CurveConfig.generate()

        mock_socket = MagicMock()
        with patch(
            "sglang.srt.utils.network.get_curve_config", return_value=client_cfg
        ):
            connect_with_curve(
                mock_socket,
                "tcp://127.0.0.1:9999",
                server_public_key=server_cfg.public_key,
            )

        self.assertEqual(mock_socket.curve_serverkey, server_cfg.public_key)
        self.assertEqual(mock_socket.curve_publickey, client_cfg.public_key)
        mock_socket.connect.assert_called_once_with("tcp://127.0.0.1:9999")

    def test_shared_key_fallback(self):
        from unittest.mock import MagicMock, patch

        cfg = CurveConfig.generate()
        mock_socket = MagicMock()
        with patch("sglang.srt.utils.network.get_curve_config", return_value=cfg):
            connect_with_curve(mock_socket, "tcp://127.0.0.1:9999")

        self.assertEqual(mock_socket.curve_serverkey, cfg.public_key)


@unittest.skipUnless(CURVE_AVAILABLE, "libzmq built without CURVE support")
class TestGetZmqSocketOnHostAppliesCurveOnLoopback(CustomTestCase):
    """Verify get_zmq_socket_on_host applies CURVE even on loopback addresses."""

    def test_loopback_gets_curve(self):
        saved = _reset_curve_state()
        os.environ.pop("SGLANG_NO_ZMQ_CURVE", None)
        os.environ.pop("SGLANG_ZMQ_CURVE_KEYS_DIR", None)
        os.environ.pop("SGLANG_ZMQ_CURVE_PUBLIC_KEY", None)
        os.environ.pop("SGLANG_ZMQ_CURVE_SECRET_KEY", None)
        try:
            ctx = zmq.Context()
            try:
                port, sock = get_zmq_socket_on_host(ctx, zmq.PULL, host="127.0.0.1")
                self.assertTrue(
                    sock.curve_server,
                    "CURVE server should be applied even on loopback",
                )
                sock.close()
            finally:
                ctx.term()
        finally:
            _restore_curve_state(saved)


@unittest.skipUnless(CURVE_AVAILABLE, "libzmq built without CURVE support")
class TestPerInstanceKeyExchange(CustomTestCase):
    """End-to-end: two instances with different keypairs communicate using
    proper asymmetric CURVE (server_public_key routing)."""

    def test_cross_instance_communication(self):
        server_cfg = CurveConfig.generate()
        client_cfg = CurveConfig.generate()

        ctx = zmq.Context()
        try:
            server = ctx.socket(zmq.PULL)
            apply_curve_server(server, server_cfg)
            port = server.bind_to_random_port("tcp://127.0.0.1")

            client = ctx.socket(zmq.PUSH)
            client.setsockopt(zmq.LINGER, 0)
            apply_curve_client(
                client, client_cfg, server_public_key=server_cfg.public_key
            )
            client.connect(f"tcp://127.0.0.1:{port}")

            client.send(b"cross-instance-msg")
            self.assertTrue(server.poll(timeout=3000))
            msg = server.recv()
            self.assertEqual(msg, b"cross-instance-msg")

            client.close()
            server.close()
        finally:
            ctx.term()


class TestDisaggregationMetadataRoundTrip(CustomTestCase):
    """Verify curve_public_key round-trips through PrefillRankInfo."""

    def test_prefill_rank_info_with_curve_key(self):
        import dataclasses

        from sglang.srt.disaggregation.common.conn import PrefillRankInfo

        info = PrefillRankInfo(
            rank_ip="192.168.1.1",
            rank_port=5555,
            curve_public_key="A" * 40,
        )
        as_dict = dataclasses.asdict(info)
        self.assertEqual(as_dict["curve_public_key"], "A" * 40)

        restored = PrefillRankInfo(**as_dict)
        self.assertEqual(restored.curve_public_key, "A" * 40)

    def test_prefill_rank_info_without_curve_key(self):
        from sglang.srt.disaggregation.common.conn import PrefillRankInfo

        info = PrefillRankInfo(rank_ip="10.0.0.1", rank_port=6666)
        self.assertIsNone(info.curve_public_key)


@unittest.skipUnless(CURVE_AVAILABLE, "libzmq built without CURVE support")
class TestBroadcastWorkerPortsKeyDistribution(CustomTestCase):
    """Verify _broadcast_worker_ports payload includes CURVE keys."""

    def test_broadcast_payload_structure(self):
        cfg = CurveConfig.generate()
        payload = {
            "worker_ports": [10000, 10001],
            "curve_public": cfg.public_key,
            "curve_secret": cfg.secret_key,
        }
        self.assertIn("curve_public", payload)
        self.assertIn("curve_secret", payload)
        self.assertEqual(payload["worker_ports"], [10000, 10001])

        received = CurveConfig(
            public_key=payload["curve_public"],
            secret_key=payload["curve_secret"],
        )
        self.assertEqual(received.public_key, cfg.public_key)
        self.assertEqual(received.secret_key, cfg.secret_key)

    def test_set_curve_config_from_broadcast(self):
        saved = _reset_curve_state()
        try:
            cfg = CurveConfig.generate()
            set_curve_config(cfg)
            result = get_curve_config()
            self.assertEqual(result.public_key, cfg.public_key)
            self.assertEqual(result.secret_key, cfg.secret_key)
        finally:
            _restore_curve_state(saved)


@unittest.skipUnless(CURVE_AVAILABLE, "libzmq built without CURVE support")
class TestDisaggBootstrapCurveKeyRoundTrip(CustomTestCase):
    """Integration: spin up a real CommonKVBootstrapServer, register a prefill
    worker with curve_public_key, and verify the decode-side GET returns it."""

    def test_bootstrap_put_get_curve_key(self):
        import requests as http_requests

        from sglang.srt.utils.network import get_open_port

        port = get_open_port()
        from sglang.srt.disaggregation.common.conn import CommonKVBootstrapServer

        server = CommonKVBootstrapServer(host="127.0.0.1", port=port)
        time.sleep(0.5)  # let aiohttp start

        try:
            base_url = f"http://127.0.0.1:{port}"

            test_curve_key = "A" * 40
            payload = {
                "attn_tp_size": 1,
                "attn_tp_rank": 0,
                "attn_cp_size": 1,
                "attn_cp_rank": 0,
                "attn_dp_size": 1,
                "attn_dp_rank": 0,
                "pp_size": 1,
                "pp_rank": 0,
                "system_dp_size": 1,
                "system_dp_rank": 0,
                "rank_ip": "192.168.1.100",
                "rank_port": 5555,
                "page_size": 16,
                "kv_cache_dtype": "auto",
                "load_balance_method": "round_robin",
                "curve_public_key": test_curve_key,
            }
            resp = http_requests.put(f"{base_url}/route", json=payload, timeout=5)
            self.assertEqual(resp.status_code, 200)

            get_resp = http_requests.get(
                f"{base_url}/route",
                params={
                    "prefill_dp_rank": "0",
                    "prefill_cp_rank": "0",
                    "target_tp_rank": "0",
                    "target_pp_rank": "0",
                },
                timeout=5,
            )
            self.assertEqual(get_resp.status_code, 200)
            data = get_resp.json()
            self.assertEqual(data["rank_ip"], "192.168.1.100")
            self.assertEqual(data["rank_port"], 5555)
            self.assertEqual(
                data["curve_public_key"],
                test_curve_key,
                "curve_public_key must survive bootstrap PUT → GET round-trip",
            )
        finally:
            server.close()

    def test_bootstrap_put_get_without_curve_key(self):
        """Backward compat: registrations without curve_public_key still work."""
        import requests as http_requests

        from sglang.srt.utils.network import get_open_port

        port = get_open_port()
        from sglang.srt.disaggregation.common.conn import CommonKVBootstrapServer

        server = CommonKVBootstrapServer(host="127.0.0.1", port=port)
        time.sleep(0.5)

        try:
            base_url = f"http://127.0.0.1:{port}"
            payload = {
                "attn_tp_size": 1,
                "attn_tp_rank": 0,
                "attn_cp_size": 1,
                "attn_cp_rank": 0,
                "attn_dp_size": 1,
                "attn_dp_rank": 0,
                "pp_size": 1,
                "pp_rank": 0,
                "system_dp_size": 1,
                "system_dp_rank": 0,
                "rank_ip": "10.0.0.1",
                "rank_port": 6666,
                "page_size": 16,
                "kv_cache_dtype": "auto",
            }
            resp = http_requests.put(f"{base_url}/route", json=payload, timeout=5)
            self.assertEqual(resp.status_code, 200)

            get_resp = http_requests.get(
                f"{base_url}/route",
                params={
                    "prefill_dp_rank": "0",
                    "prefill_cp_rank": "0",
                    "target_tp_rank": "0",
                    "target_pp_rank": "0",
                },
                timeout=5,
            )
            self.assertEqual(get_resp.status_code, 200)
            data = get_resp.json()
            self.assertIsNone(data.get("curve_public_key"))
        finally:
            server.close()


class TestTransferInfoFromZmq(CustomTestCase):
    """Verify TransferInfo.from_zmq parses the curve_public_key frame."""

    def test_with_curve_public_key(self):
        import numpy as np

        from sglang.srt.disaggregation.mooncake.conn import TransferInfo

        kv_indices = np.array([1, 2, 3], dtype=np.int32)
        curve_key = "B" * 40
        msg = [
            b"42",  # room
            b"192.168.1.1",  # endpoint
            b"5555",  # dst_port
            b"session-abc",  # mooncake_session_id
            kv_indices.tobytes(),  # dst_kv_indices
            b"7",  # dst_aux_index
            b"",  # dst_state_indices (empty)
            b"2",  # required_dst_info_num
            curve_key.encode("ascii"),  # curve_public_key
        ]
        info = TransferInfo.from_zmq(msg)
        self.assertEqual(info.room, 42)
        self.assertEqual(info.endpoint, "192.168.1.1")
        self.assertEqual(info.dst_port, 5555)
        self.assertEqual(info.curve_public_key, curve_key)
        self.assertFalse(info.is_dummy)

    def test_without_curve_public_key(self):
        import numpy as np

        from sglang.srt.disaggregation.mooncake.conn import TransferInfo

        kv_indices = np.array([10], dtype=np.int32)
        msg = [
            b"1",
            b"10.0.0.2",
            b"6000",
            b"session-xyz",
            kv_indices.tobytes(),
            b"0",
            b"",
            b"1",
        ]
        info = TransferInfo.from_zmq(msg)
        self.assertIsNone(info.curve_public_key)

    def test_dummy_transfer_info(self):
        from sglang.srt.disaggregation.mooncake.conn import TransferInfo

        msg = [
            b"99",
            b"10.0.0.3",
            b"7000",
            b"session-dummy",
            b"",  # empty kv_indices → dummy
            b"",  # empty aux_index → dummy
            b"",
            b"1",
            b"C" * 40,  # curve key still present
        ]
        info = TransferInfo.from_zmq(msg)
        self.assertTrue(info.is_dummy)
        self.assertEqual(info.curve_public_key, "C" * 40)


class TestKVArgsRegisterInfoFromZmq(CustomTestCase):
    """Verify KVArgsRegisterInfo.from_zmq parses the curve_public_key frame."""

    def test_with_curve_public_key(self):
        import struct

        from sglang.srt.disaggregation.mooncake.conn import KVArgsRegisterInfo

        curve_key = "D" * 40
        msg = [
            b"None",  # room
            b"10.0.0.5",  # endpoint
            b"8000",  # dst_port
            b"session-reg",  # mooncake_session_id
            struct.pack("Q", 0xDEAD),  # dst_kv_ptrs
            struct.pack("Q", 0xBEEF),  # dst_aux_ptrs
            struct.pack("Q", 0xCAFE),  # dst_state_data_ptrs
            b"0",  # dst_tp_rank
            b"1",  # dst_attn_tp_size
            b"128",  # dst_kv_item_len
            b"",  # dst_state_item_lens
            b"",  # dst_state_dim_per_tensor
            curve_key.encode("ascii"),  # curve_public_key
        ]
        info = KVArgsRegisterInfo.from_zmq(msg)
        self.assertEqual(info.room, "None")
        self.assertEqual(info.endpoint, "10.0.0.5")
        self.assertEqual(info.curve_public_key, curve_key)

    def test_without_curve_public_key(self):
        import struct

        from sglang.srt.disaggregation.mooncake.conn import KVArgsRegisterInfo

        msg = [
            b"None",
            b"10.0.0.6",
            b"9000",
            b"session-noreg",
            struct.pack("Q", 0x1234),
            struct.pack("Q", 0x5678),
            struct.pack("Q", 0x9ABC),
            b"0",
            b"2",
            b"256",
            b"",
            b"",
        ]
        info = KVArgsRegisterInfo.from_zmq(msg)
        self.assertIsNone(info.curve_public_key)


@unittest.skipUnless(CURVE_AVAILABLE, "libzmq built without CURVE support")
class TestDisaggZmqCurveHandshake(CustomTestCase):
    """Integration: simulate prefill PULL + decode PUSH with different CURVE
    keypairs using server_public_key routing (asymmetric CURVE)."""

    def test_prefill_decode_zmq_with_curve(self):
        prefill_cfg = CurveConfig.generate()
        decode_cfg = CurveConfig.generate()

        ctx = zmq.Context()
        try:
            prefill_pull = ctx.socket(zmq.PULL)
            apply_curve_server(prefill_pull, prefill_cfg)
            port = prefill_pull.bind_to_random_port("tcp://127.0.0.1")

            decode_push = ctx.socket(zmq.PUSH)
            decode_push.setsockopt(zmq.LINGER, 0)
            apply_curve_client(
                decode_push, decode_cfg, server_public_key=prefill_cfg.public_key
            )
            decode_push.connect(f"tcp://127.0.0.1:{port}")

            decode_push.send_multipart([b"42", b"test-kv-data"])
            self.assertTrue(prefill_pull.poll(timeout=3000))
            msg = prefill_pull.recv_multipart()
            self.assertEqual(msg, [b"42", b"test-kv-data"])

            decode_push.close()
            prefill_pull.close()
        finally:
            ctx.term()

    def test_connect_with_curve_server_public_key(self):
        """connect_with_curve with explicit server_public_key works across
        different instance keypairs."""
        prefill_cfg = CurveConfig.generate()
        decode_cfg = CurveConfig.generate()

        saved = _reset_curve_state()
        try:
            set_curve_config(decode_cfg)

            ctx = zmq.Context()
            try:
                server = ctx.socket(zmq.PULL)
                apply_curve_server(server, prefill_cfg)
                port = server.bind_to_random_port("tcp://127.0.0.1")

                client = ctx.socket(zmq.PUSH)
                client.setsockopt(zmq.LINGER, 0)
                connect_with_curve(
                    client,
                    f"tcp://127.0.0.1:{port}",
                    server_public_key=prefill_cfg.public_key,
                )

                client.send(b"cross-key-msg")
                self.assertTrue(server.poll(timeout=3000))
                self.assertEqual(server.recv(), b"cross-key-msg")

                client.close()
                server.close()
            finally:
                ctx.term()
        finally:
            _restore_curve_state(saved)


if __name__ == "__main__":
    unittest.main()
