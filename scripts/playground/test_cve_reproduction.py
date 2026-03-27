#!/usr/bin/env python3
"""
CVE reproduction / fix-verification tests for SGLang.

CVE 1 (CVSS 8.4 High):
    SafeUnpickler bypass via builtins.__import__ + builtins.getattr.
    Tests whether the allow-listed builtins can be chained to achieve RCE.

CVE 2 (CVSS 9.8 Critical):
    Unauthenticated raw pickle.loads() on LAN-facing ZMQ sockets.
    Tests whether an unauthenticated attacker on the same LAN can inject a
    malicious pickle payload into:
      - shm_broadcast remote reader (XPUB)
      - encode_receiver PULL socket

Usage:
    uv run python scripts/playground/test_cve_reproduction.py
"""
import io
import os
import pickle
import sys
import time

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROOF_FILES = [
    "/tmp/sglang_poc_rce",
    "/tmp/sglang_poc_getattr",
    "/tmp/sglang_poc_import",
    "/tmp/sglang_poc_os_popen",
    "/tmp/sglang_unauth_rce",
]

passed = []
failed = []


def _cleanup():
    for f in PROOF_FILES:
        if os.path.exists(f):
            os.remove(f)


def _result(name: str, ok: bool, msg: str = ""):
    tag = "[PASS]" if ok else "[FAIL]"
    print(f"  {tag} {name}" + (f": {msg}" if msg else ""))
    (passed if ok else failed).append(name)


def _expect_blocked(name: str, fn, *args, **kwargs):
    """Expect fn() to raise RuntimeError (i.e. the payload is blocked)."""
    try:
        fn(*args, **kwargs)
        _result(name, False, "payload was NOT blocked — vulnerability still present")
    except RuntimeError as e:
        _result(name, True, f"correctly blocked: {e}")
    except Exception as e:
        _result(name, False, f"unexpected exception type {type(e).__name__}: {e}")


def _expect_rce_fails(name: str, proof_file: str, fn, *args, **kwargs):
    """Expect that fn() does NOT create proof_file (RCE blocked)."""
    if os.path.exists(proof_file):
        os.remove(proof_file)
    try:
        fn(*args, **kwargs)
    except Exception:
        pass
    if os.path.exists(proof_file):
        _result(name, False, f"proof file {proof_file} was created — RCE succeeded")
    else:
        _result(name, True, "proof file not created — RCE blocked")


# ---------------------------------------------------------------------------
# Import the real SafeUnpickler from the repo
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python"))

try:
    from sglang.srt.utils.common import SafeUnpickler  # type: ignore

    def safe_unpickle(data: bytes):
        return SafeUnpickler(io.BytesIO(data)).load()

    SAFE_UNPICKLER_AVAILABLE = True
except Exception as e:
    print(f"[WARN] Could not import SafeUnpickler from sglang: {e}")
    print("       Falling back to inline copy (tests the algorithm, not the live code)")
    SAFE_UNPICKLER_AVAILABLE = False

    class SafeUnpickler(pickle.Unpickler):
        """Inline copy from python/sglang/srt/utils/common.py (current state)."""

        ALLOWED_MODULE_PREFIXES = {
            "builtins.",
            "collections.",
            "copyreg.",
            "functools.",
            "itertools.",
            "operator.",
            "types.",
            "weakref.",
            "torch.",
            "torch._tensor.",
            "torch.storage.",
            "torch.nn.parameter.",
            "torch.autograd.function.",
            "torch.distributed.",
            "torch.distributed._shard.",
            "torch.distributed._composable.",
            "torch._C._distributed_c10d.",
            "torch._C._distributed_fsdp.",
            "torch.distributed.optim.",
            "multiprocessing.resource_sharer.",
            "multiprocessing.reduction.",
            "pickletools.",
            "peft.",
            "transformers.",
            "huggingface_hub.",
            "sglang.srt.weight_sync.tensor_bucket.",
            "sglang.srt.model_executor.model_runner.",
            "sglang.srt.layers.",
            "sglang.srt.utils.",
            "torch_npu.",
        }
        DENY_CLASSES = {
            ("builtins", "eval"),
            ("builtins", "exec"),
            ("builtins", "compile"),
            ("os", "system"),
            ("subprocess", "Popen"),
            ("subprocess", "run"),
            ("codecs", "decode"),
            ("types", "CodeType"),
            ("types", "FunctionType"),
        }

        def find_class(self, module, name):
            if (module, name) in self.DENY_CLASSES:
                raise RuntimeError(
                    f"Blocked unsafe class loading ({module}.{name})"
                )
            if any(
                (module + ".").startswith(prefix)
                for prefix in self.ALLOWED_MODULE_PREFIXES
            ):
                return super().find_class(module, name)
            raise RuntimeError(
                f"Blocked unsafe class loading ({module}.{name})"
            )

    def safe_unpickle(data: bytes):
        return SafeUnpickler(io.BytesIO(data)).load()


# ---------------------------------------------------------------------------
# CVE 1 — SafeUnpickler bypass tests
# ---------------------------------------------------------------------------

print()
print("=" * 72)
print("CVE 1: SafeUnpickler bypass via builtins.__import__ / builtins.getattr")
print("=" * 72)
src = "(from repo)" if SAFE_UNPICKLER_AVAILABLE else "(inline fallback)"
print(f"Testing SafeUnpickler {src}")
print()

# PoC 1: __import__ passes allowlist — returns live module object
print("PoC 1 — builtins.__import__ passes allowlist")
_expect_blocked(
    "PoC1: __import__('os') blocked",
    safe_unpickle,
    b"cbuiltins\n__import__\n(S'os'\ntR.",
)

# PoC 2: builtins.getattr passes allowlist, executes os.getcwd()
print("\nPoC 2 — builtins.getattr chains __import__ to call os.getcwd()")
_expect_blocked(
    "PoC2: getattr(__import__('os'), 'getcwd')() blocked",
    safe_unpickle,
    (
        b"cbuiltins\ngetattr\n(cbuiltins\n__import__\n(S'os'\ntR"
        b"S'getcwd'\ntR(tR."
    ),
)

# PoC 3: Full RCE — os.popen creates proof file
print("\nPoC 3 — Full RCE via getattr + __import__ + os.popen")
_expect_rce_fails(
    "PoC3: RCE via os.popen blocked",
    "/tmp/sglang_poc_rce",
    safe_unpickle,
    (
        b"cbuiltins\ngetattr\n(cbuiltins\n__import__\n(S'os'\ntR"
        b"S'popen'\ntR(S'touch /tmp/sglang_poc_rce'\ntR."
    ),
)

# PoC 4: DENY_CLASSES bypass — os.system via getattr (os.system is in DENY_CLASSES)
print("\nPoC 4 — DENY_CLASSES bypass: reaching os.system via getattr")
_expect_rce_fails(
    "PoC4: DENY_CLASSES bypass (os.system via getattr) blocked",
    "/tmp/sglang_poc_getattr",
    safe_unpickle,
    (
        b"cbuiltins\ngetattr\n(cbuiltins\n__import__\n(S'os'\ntR"
        b"S'system'\ntR(S'touch /tmp/sglang_poc_getattr'\ntR."
    ),
)

# PoC 5: functools.partial gadget — returns reusable callable to os.popen
print("\nPoC 5 — functools.partial gadget creates persistent os.popen callable")
_expect_blocked(
    "PoC5: functools.partial(getattr(os, 'popen')) blocked",
    safe_unpickle,
    (
        b"cfunctools\npartial\n(cbuiltins\ngetattr\n(cbuiltins\n__import__\n"
        b"(S'os'\ntRS'popen'\ntRtR."
    ),
)

# Control tests — these MUST be blocked (denylist working correctly)
print("\nControl tests — direct GLOBAL opcode references (must be blocked)")
for mod, name in [("os", "system"), ("builtins", "eval"), ("subprocess", "Popen")]:
    payload = f"c{mod}\n{name}\n(S'test'\ntR.".encode()
    _expect_blocked(f"Control: direct {mod}.{name} blocked", safe_unpickle, payload)

# ---------------------------------------------------------------------------
# CVE 2 — Unauthenticated ZMQ pickle injection tests
# ---------------------------------------------------------------------------

print()
print("=" * 72)
print("CVE 2: Unauthenticated ZMQ pickle injection (CurveZMQ fix verification)")
print("=" * 72)
print()

try:
    import zmq
    from sglang.srt.utils.network import (
        CurveConfig,
        apply_curve_client,
        apply_curve_server,
        get_curve_config,
    )

    HAS_CURVE = zmq.has("curve")
except ImportError as e:
    print(f"[SKIP] zmq / sglang not available: {e}")
    HAS_CURVE = False

RCE_PAYLOAD = (
    b"cbuiltins\ngetattr\n"
    b"(cbuiltins\n__import__\n"
    b"(S'os'\ntR"
    b"S'system'\n"
    b"tR"
    b"(S'touch /tmp/sglang_unauth_rce'\n"
    b"tR."
)


def _test_curve_xpub_blocks_unauthenticated():
    """
    Spin up a CURVE-protected XPUB socket (mimicking shm_broadcast remote
    reader endpoint) and verify an unauthenticated SUB client cannot inject
    a malicious pickle payload.

    Without CURVE an adversary can PUB straight to the XPUB and the data
    arrives at the receiver.  With CURVE the ZMQ handshake fails and no data
    is delivered.
    """
    if not HAS_CURVE:
        _result("CVE2-Endpoint1 (shm_broadcast XPUB)", False, "CURVE not available in this libzmq build")
        return

    proof = "/tmp/sglang_unauth_rce"
    if os.path.exists(proof):
        os.remove(proof)

    server_curve = CurveConfig.generate()

    ctx = zmq.Context()

    # --- Server side: XPUB with CURVE (like shm_broadcast) ---
    server_sock = ctx.socket(zmq.XPUB)
    server_sock.setsockopt(zmq.XPUB_VERBOSE, True)
    apply_curve_server(server_sock, server_curve)
    port = server_sock.bind_to_random_port("tcp://127.0.0.1")

    # --- Attacker side: raw PUB, NO CURVE keys ---
    attacker_sock = ctx.socket(zmq.PUB)
    attacker_sock.connect(f"tcp://127.0.0.1:{port}")

    # Allow sockets to settle; an unauthenticated client's handshake will be
    # silently rejected by CURVE — it never progresses to the messaging phase.
    time.sleep(0.5)
    attacker_sock.send(RCE_PAYLOAD)
    time.sleep(0.3)

    # Try to receive — the XPUB server side should see nothing (no subscription
    # from the attacker because CURVE rejected it).
    server_sock.setsockopt(zmq.RCVTIMEO, 300)
    try:
        msg = server_sock.recv()
        # If we get here the subscription message arrived — meaning the
        # attacker's connection was accepted, which is the vulnerability.
        _result(
            "CVE2-Endpoint1 (shm_broadcast XPUB)",
            False,
            f"XPUB received data from unauthenticated client (CURVE not protecting): {msg[:40]!r}",
        )
    except zmq.Again:
        _result(
            "CVE2-Endpoint1 (shm_broadcast XPUB)",
            True,
            "CURVE rejected unauthenticated connection — no data delivered to XPUB",
        )
    finally:
        attacker_sock.close(linger=0)
        server_sock.close(linger=0)
        ctx.term()

    if os.path.exists(proof):
        _result("CVE2-Endpoint1 proof file", False, "RCE proof file created!")
    else:
        _result("CVE2-Endpoint1 proof file", True, "no proof file created")


def _test_curve_pull_blocks_unauthenticated():
    """
    Spin up a CURVE-protected PULL socket (mimicking encode_receiver endpoint)
    and verify an unauthenticated PUSH client cannot inject data.
    """
    if not HAS_CURVE:
        _result("CVE2-Endpoint2 (encode_receiver PULL)", False, "CURVE not available in this libzmq build")
        return

    proof = "/tmp/sglang_unauth_rce"
    if os.path.exists(proof):
        os.remove(proof)

    server_curve = CurveConfig.generate()

    ctx = zmq.Context()

    # --- Server: PULL with CURVE (like encode_receiver) ---
    server_sock = ctx.socket(zmq.PULL)
    apply_curve_server(server_sock, server_curve)
    port = server_sock.bind_to_random_port("tcp://127.0.0.1")

    # --- Attacker: raw PUSH, no CURVE keys ---
    attacker_sock = ctx.socket(zmq.PUSH)
    attacker_sock.connect(f"tcp://127.0.0.1:{port}")

    time.sleep(0.4)
    attacker_sock.send(RCE_PAYLOAD)
    time.sleep(0.3)

    server_sock.setsockopt(zmq.RCVTIMEO, 400)
    try:
        msg = server_sock.recv()
        # Received payload from unauthenticated sender — vulnerability present
        _result(
            "CVE2-Endpoint2 (encode_receiver PULL)",
            False,
            f"PULL received data from unauthenticated client: {msg[:40]!r}",
        )
    except zmq.Again:
        _result(
            "CVE2-Endpoint2 (encode_receiver PULL)",
            True,
            "CURVE rejected unauthenticated PUSH — no data delivered to PULL",
        )
    finally:
        attacker_sock.close(linger=0)
        server_sock.close(linger=0)
        ctx.term()


def _test_authenticated_client_still_works():
    """
    Sanity check: a properly authenticated PUSH client CAN send to a
    CURVE-protected PULL (i.e. CURVE doesn't break legitimate traffic).
    """
    if not HAS_CURVE:
        _result("CVE2-Sanity (authenticated client works)", False, "CURVE not available")
        return

    server_curve = CurveConfig.generate()

    ctx = zmq.Context()
    server_sock = ctx.socket(zmq.PULL)
    apply_curve_server(server_sock, server_curve)
    port = server_sock.bind_to_random_port("tcp://127.0.0.1")

    # Legitimate client uses the same keypair (shared-key model used internally)
    client_sock = ctx.socket(zmq.PUSH)
    apply_curve_client(client_sock, server_curve, server_curve.public_key)
    client_sock.connect(f"tcp://127.0.0.1:{port}")

    time.sleep(0.3)
    client_sock.send(b"hello from legitimate client")
    time.sleep(0.2)

    server_sock.setsockopt(zmq.RCVTIMEO, 500)
    try:
        msg = server_sock.recv()
        _result(
            "CVE2-Sanity (authenticated client works)",
            msg == b"hello from legitimate client",
            f"received: {msg!r}",
        )
    except zmq.Again:
        _result(
            "CVE2-Sanity (authenticated client works)",
            False,
            "legitimate client message not received",
        )
    finally:
        client_sock.close(linger=0)
        server_sock.close(linger=0)
        ctx.term()


if HAS_CURVE:
    print("Testing Endpoint 1 (shm_broadcast XPUB — remote reader):")
    _test_curve_xpub_blocks_unauthenticated()

    print("\nTesting Endpoint 2 (encode_receiver PULL):")
    _test_curve_pull_blocks_unauthenticated()

    print("\nSanity check (legitimate authenticated client):")
    _test_authenticated_client_still_works()
else:
    _result("CVE2-Endpoint1", False, "libzmq built without CURVE — all endpoints unprotected")
    _result("CVE2-Endpoint2", False, "libzmq built without CURVE — all endpoints unprotected")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

_cleanup()

print()
print("=" * 72)
print(f"Results: {len(passed)} passed, {len(failed)} failed")
if passed:
    print(f"  PASSED: {', '.join(passed)}")
if failed:
    print(f"  FAILED: {', '.join(failed)}")
print("=" * 72)

sys.exit(1 if failed else 0)
