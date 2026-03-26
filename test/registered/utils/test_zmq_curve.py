import os
import shutil
import tempfile
import threading
import time
import unittest

import zmq

from sglang.srt.utils.gen_zmq_keys import generate_certificates
from sglang.srt.utils.network import (
    CurveConfig,
    apply_curve_client,
    apply_curve_server,
    get_curve_config,
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


if __name__ == "__main__":
    unittest.main()
