from setuptools import setup

try:
    from setuptools_rust import RustExtension

    rust_extensions = [
        RustExtension(
            "sglang.srt.sgl_scheduler",
            path="../rust/sgl-scheduler/Cargo.toml",
            binding="pyo3",
            optional=True,  # Don't fail if Rust toolchain missing
        )
    ]
except ImportError:
    # setuptools-rust not installed; skip Rust extension
    rust_extensions = []

setup(
    rust_extensions=rust_extensions,
)
