"""
This unittest is introduced in #22360, preventing duplicate transformer safetensors variants being loaded together
"""

import json
import sys
import tempfile
import types
import unittest
from builtins import __import__ as _real_import
from types import SimpleNamespace
from unittest.mock import patch

import torch

partial_json_parser = types.ModuleType("partial_json_parser")
partial_json_parser_core = types.ModuleType("partial_json_parser.core")
partial_json_parser_exceptions = types.ModuleType("partial_json_parser.core.exceptions")
partial_json_parser_options = types.ModuleType("partial_json_parser.core.options")


class _MalformedJSON(Exception):
    pass


class _Allow:
    STR = 1
    OBJ = 2
    ARR = 4
    ALL = STR | OBJ | ARR


def _loads(input_str, _flags=None):
    return json.loads(input_str)


partial_json_parser_exceptions.MalformedJSON = _MalformedJSON
partial_json_parser_options.Allow = _Allow
partial_json_parser.loads = _loads
sys.modules.setdefault("partial_json_parser", partial_json_parser)
sys.modules.setdefault("partial_json_parser.core", partial_json_parser_core)
sys.modules.setdefault(
    "partial_json_parser.core.exceptions", partial_json_parser_exceptions
)
sys.modules.setdefault("partial_json_parser.core.options", partial_json_parser_options)

from sglang.multimodal_gen.configs.models.dits.wanvideo import WanVideoConfig
from sglang.multimodal_gen.runtime.layers.linear import UnquantizedLinearMethod
from sglang.multimodal_gen.runtime.layers.quantization import modelopt_quant
from sglang.multimodal_gen.runtime.layers.quantization.configs.nunchaku_config import (
    NunchakuConfig,
)
from sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    _prepare_nvfp4_weight_bytes,
)
from sglang.multimodal_gen.runtime.loader.transformer_load_utils import (
    _filter_duplicate_precision_variant_safetensors,
    _Flux2Nvfp4FallbackAdapter,
    resolve_transformer_quant_load_spec,
    resolve_transformer_safetensors_to_load,
)
from sglang.multimodal_gen.runtime.models.dits.flux import FluxSingleTransformerBlock
from sglang.multimodal_gen.runtime.platforms.cuda import CudaPlatformBase
from sglang.multimodal_gen.runtime.utils.quantization_utils import (
    build_nvfp4_config_from_safetensors_list,
)
from sglang.multimodal_gen.tools.build_modelopt_nvfp4_transformer import (
    _updated_quant_config,
)


class _FakeFluxTransformer:
    pass


class _FakeQuantConfig:
    @classmethod
    def get_name(cls):
        return "modelopt_fp4"


class _FakeSafeTensorSlice:
    def __init__(self, shape):
        self._shape = shape

    def get_shape(self):
        return self._shape


class _FakeSafeTensorFile:
    def __init__(self, tensors):
        self._tensors = tensors

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def keys(self):
        return list(self._tensors.keys())

    def get_slice(self, key):
        return _FakeSafeTensorSlice(tuple(self._tensors[key].shape))

    def get_tensor(self, key):
        return self._tensors[key]


class TestTransformerQuantHelpers(unittest.TestCase):
    def _patch_nvfp4_kernel_imports(
        self,
        fake_nvfp4_module,
        *,
        missing_flashinfer: bool = False,
    ):
        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "sgl_kernel":
                raise ImportError("missing sgl_kernel")
            if missing_flashinfer and name == "flashinfer":
                raise ImportError("missing flashinfer")
            if name == "sglang.jit_kernel.nvfp4":
                return fake_nvfp4_module
            return _real_import(name, globals, locals, fromlist, level)

        return patch("builtins.__import__", side_effect=fake_import)

    def _reset_modelopt_fp4_runtime_state(self):
        modelopt_quant._FORCE_CUTLASS_FP4_GEMM = False
        modelopt_quant._FLASHINFER_FP4_GEMM_BACKEND_OVERRIDE = None
        modelopt_quant._get_fp4_gemm_op.cache_clear()
        modelopt_quant._get_cutlass_fp4_gemm_op.cache_clear()

    def _make_server_args(self, **overrides):
        defaults = dict(
            transformer_weights_path=None,
            pipeline_config=SimpleNamespace(
                dit_precision="bf16",
                dit_config=SimpleNamespace(
                    arch_config=SimpleNamespace(param_names_mapping={})
                ),
            ),
            nunchaku_config=None,
            tp_size=1,
            dit_cpu_offload=False,
            text_encoder_cpu_offload=False,
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_resolve_transformer_safetensors_to_load_uses_single_override_file(self):
        with tempfile.NamedTemporaryFile(suffix=".safetensors") as f:
            server_args = self._make_server_args(transformer_weights_path=f.name)
            resolved = resolve_transformer_safetensors_to_load(
                server_args, "/unused/component/path"
            )

        self.assertEqual(resolved, [f.name])

    @patch(
        "sglang.multimodal_gen.runtime.loader.transformer_load_utils.maybe_download_model",
        side_effect=lambda path, **kw: path,
    )
    def test_resolve_transformer_safetensors_to_load_prefers_mixed_export(
        self, _mock_download
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            mixed = f"{tmpdir}/flux2-dev-nvfp4-mixed.safetensors"
            full = f"{tmpdir}/flux2-dev-nvfp4.safetensors"
            open(mixed, "a").close()
            open(full, "a").close()

            server_args = self._make_server_args(transformer_weights_path=tmpdir)
            resolved = resolve_transformer_safetensors_to_load(
                server_args, "/unused/component/path"
            )

        self.assertEqual(resolved, [mixed])

    def test_filter_transformer_precision_variants_prefers_canonical_file(self):
        files = [
            "/tmp/transformer/diffusion_pytorch_model.fp16.safetensors",
            "/tmp/transformer/diffusion_pytorch_model.safetensors",
            "/tmp/transformer/other.safetensors",
        ]

        resolved = _filter_duplicate_precision_variant_safetensors(files)

        self.assertEqual(
            resolved,
            [
                "/tmp/transformer/diffusion_pytorch_model.safetensors",
                "/tmp/transformer/other.safetensors",
            ],
        )

    def test_filter_transformer_precision_variants_keeps_precision_only_family(self):
        files = [
            "/tmp/transformer/diffusion_pytorch_model.bf16.safetensors",
            "/tmp/transformer/diffusion_pytorch_model.fp16.safetensors",
        ]

        resolved = _filter_duplicate_precision_variant_safetensors(files)

        self.assertEqual(resolved, files)

    @patch(
        "sglang.multimodal_gen.runtime.loader.transformer_load_utils.build_nvfp4_config_from_safetensors_list",
        return_value=None,
    )
    @patch(
        "sglang.multimodal_gen.runtime.loader.transformer_load_utils.maybe_download_model"
    )
    @patch(
        "sglang.multimodal_gen.runtime.loader.transformer_load_utils.get_quant_config_from_safetensors_metadata",
        return_value=None,
    )
    @patch(
        "sglang.multimodal_gen.runtime.loader.transformer_load_utils.get_metadata_from_safetensors_file"
    )
    @patch(
        "sglang.multimodal_gen.runtime.loader.transformer_load_utils.maybe_download_model",
        side_effect=lambda path, **kw: path,
    )
    def test_resolve_transformer_quant_load_spec_keeps_nunchaku_hook(
        self,
        _mock_download,
        mock_metadata,
        _mock_quant_metadata,
        mock_maybe_download,
        _mock_nvfp4,
    ):
        mock_maybe_download.side_effect = AssertionError(
            "local safetensors path should not trigger maybe_download_model"
        )
        mock_metadata.return_value = {
            "config": json.dumps({"_class_name": _FakeFluxTransformer.__name__})
        }
        with tempfile.NamedTemporaryFile(suffix=".safetensors") as f:
            nunchaku_config = NunchakuConfig(transformer_weights_path=f.name)
            server_args = self._make_server_args(
                transformer_weights_path=nunchaku_config.transformer_weights_path,
                nunchaku_config=nunchaku_config,
            )

            spec = resolve_transformer_quant_load_spec(
                hf_config={},
                server_args=server_args,
                safetensors_list=[nunchaku_config.transformer_weights_path],
                component_model_path="/unused/component/path",
                model_cls=_FakeFluxTransformer,
                cls_name=_FakeFluxTransformer.__name__,
            )

        self.assertIsNone(spec.quant_config)
        self.assertIs(spec.nunchaku_config, nunchaku_config)
        self.assertIsNone(spec.param_dtype)
        self.assertEqual(len(spec.post_load_hooks), 1)
        self.assertIs(nunchaku_config.model_cls, _FakeFluxTransformer)
        mock_maybe_download.assert_not_called()

    def test_flux2_mixed_nvfp4_fallback_disables_conflicting_offloads(self):
        server_args = self._make_server_args(
            transformer_weights_path="/tmp/flux2-dev-nvfp4-mixed.safetensors",
            tp_size=2,
            dit_cpu_offload=True,
            text_encoder_cpu_offload=True,
        )

        _Flux2Nvfp4FallbackAdapter._maybe_adjust_flux2_nvfp4_fallback_defaults(
            cls_name="Flux2Transformer2DModel",
            server_args=server_args,
            quant_config=_FakeQuantConfig(),
        )

        self.assertFalse(server_args.dit_cpu_offload)
        self.assertFalse(server_args.text_encoder_cpu_offload)

    def test_prepare_nvfp4_weight_bytes_swaps_nibbles(self):
        weight = torch.tensor([[0xAB, 0x10]], dtype=torch.uint8)

        prepared = _prepare_nvfp4_weight_bytes(weight, swap_weight_nibbles=True)

        self.assertEqual(prepared.tolist(), [[0xBA, 0x01]])

    def test_prepare_nvfp4_weight_bytes_can_skip_nibble_swap(self):
        weight = torch.tensor([[0xAB, 0x10]], dtype=torch.uint8)

        prepared = _prepare_nvfp4_weight_bytes(weight, swap_weight_nibbles=False)

        self.assertEqual(prepared.tolist(), [[0xAB, 0x10]])

    def test_cutlass_fp4_gemm_loader_falls_back_to_jit_kernel(self):
        fake_nvfp4_module = types.ModuleType("sglang.jit_kernel.nvfp4")
        fake_cutlass_fp4_gemm = object()
        fake_nvfp4_module.cutlass_scaled_fp4_mm = fake_cutlass_fp4_gemm
        modelopt_quant._get_cutlass_fp4_gemm_op.cache_clear()
        try:
            with self._patch_nvfp4_kernel_imports(fake_nvfp4_module):
                loaded = modelopt_quant._get_cutlass_fp4_gemm_op()
        finally:
            modelopt_quant._get_cutlass_fp4_gemm_op.cache_clear()

        self.assertIs(loaded, fake_cutlass_fp4_gemm)

    def test_cuda_platform_fp4_ops_fall_back_to_jit_kernel(self):
        fake_nvfp4_module = types.ModuleType("sglang.jit_kernel.nvfp4")
        fake_fp4_quantize = object()
        fake_cutlass_fp4_gemm = object()
        fake_nvfp4_module.scaled_fp4_quant = fake_fp4_quantize
        fake_nvfp4_module.cutlass_scaled_fp4_mm = fake_cutlass_fp4_gemm
        CudaPlatformBase.get_modelopt_fp4_quantize_op.cache_clear()
        CudaPlatformBase.get_modelopt_fp4_gemm_op.cache_clear()
        try:
            with self._patch_nvfp4_kernel_imports(
                fake_nvfp4_module, missing_flashinfer=True
            ):
                fp4_quantize = CudaPlatformBase.get_modelopt_fp4_quantize_op()
                fp4_gemm, backend = CudaPlatformBase.get_modelopt_fp4_gemm_op()
        finally:
            CudaPlatformBase.get_modelopt_fp4_quantize_op.cache_clear()
            CudaPlatformBase.get_modelopt_fp4_gemm_op.cache_clear()

        self.assertIs(fp4_quantize, fake_fp4_quantize)
        self.assertIs(fp4_gemm, fake_cutlass_fp4_gemm)
        self.assertIsNone(backend)

    def test_get_fp4_gemm_op_uses_runtime_flashinfer_backend_override(self):
        fake_fp4_gemm = object()
        self._reset_modelopt_fp4_runtime_state()
        try:
            with patch.object(
                modelopt_quant.current_platform,
                "get_modelopt_fp4_gemm_op",
                return_value=(fake_fp4_gemm, "cudnn"),
            ):
                modelopt_quant._override_flashinfer_fp4_runtime_backend(
                    "cutlass", "unit test"
                )
                fp4_gemm, backend = modelopt_quant._get_fp4_gemm_op()
        finally:
            self._reset_modelopt_fp4_runtime_state()

        self.assertIs(fp4_gemm, fake_fp4_gemm)
        self.assertEqual(backend, "cutlass")

    def test_modelopt_fp4_apply_retries_with_flashinfer_cutlass_backend(self):
        quant_config = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=[],
        )
        method = modelopt_quant.ModelOptFp4LinearMethod(quant_config)
        layer = SimpleNamespace(
            input_scale_inv=torch.tensor(1.0, dtype=torch.float32),
            weight=torch.zeros((4, 4), dtype=torch.uint8),
            weight_scale_interleaved=torch.ones(
                (4, 1), dtype=torch.float8_e4m3fn
            ),
            alpha=torch.tensor(1.0, dtype=torch.float32),
            output_size_per_partition=4,
        )
        backend_calls = []

        def fake_fp4_quantize(x, _input_scale_inv):
            return (
                torch.zeros((x.shape[0], 4), dtype=torch.uint8),
                torch.ones((x.shape[0], 1), dtype=torch.float8_e4m3fn),
            )

        def fake_fp4_gemm(*gemm_args, backend=None):
            backend_calls.append(backend)
            if backend == "cudnn":
                raise RuntimeError(
                    "Multiple libcudart libraries found: libcudart.so.12 and libcudart.so.13"
                )
            output_dtype = gemm_args[5]
            return torch.zeros(
                (gemm_args[0].shape[0], layer.output_size_per_partition),
                dtype=output_dtype,
            )

        self._reset_modelopt_fp4_runtime_state()
        runtime_backend_override = None
        try:
            with patch.object(
                modelopt_quant, "_get_fp4_quantize_op", return_value=fake_fp4_quantize
            ), patch.object(
                modelopt_quant, "_get_fp4_gemm_op", return_value=(fake_fp4_gemm, "cudnn")
            ), patch.object(
                modelopt_quant,
                "_get_cutlass_fp4_gemm_op",
                side_effect=AssertionError("JIT CUTLASS fallback should not be used"),
            ), patch.object(
                modelopt_quant,
                "pad_nvfp4_activation_for_cutlass",
                side_effect=lambda x, _padding_cols: x,
            ), patch.object(
                modelopt_quant,
                "slice_nvfp4_output",
                side_effect=lambda out, _output_size: out,
            ):
                out = method.apply(
                    layer,
                    torch.ones((1, 8), dtype=torch.bfloat16),
                )
                runtime_backend_override = (
                    modelopt_quant._FLASHINFER_FP4_GEMM_BACKEND_OVERRIDE
                )
        finally:
            self._reset_modelopt_fp4_runtime_state()

        self.assertEqual(backend_calls, ["cudnn", "cutlass"])
        self.assertEqual(runtime_backend_override, "cutlass")
        self.assertEqual(out.shape, (1, 4))

    def test_modelopt_fp4_config_reads_swap_weight_nibbles_from_flat_config(self):
        config = ModelOptFp4Config.from_config(
            {
                "quant_algo": "NVFP4",
                "group_size": 16,
                "ignore": [],
                "swap_weight_nibbles": False,
            }
        )

        self.assertFalse(config.swap_weight_nibbles)

    def test_modelopt_fp4_config_reads_swap_weight_nibbles_from_nested_config(self):
        config = ModelOptFp4Config.from_config(
            {
                "quantization": {
                    "quant_algo": "NVFP4",
                    "exclude_modules": [],
                    "swap_weight_nibbles": False,
                },
                "config_groups": {"default": {"weights": {"group_size": 16}}},
            }
        )

        self.assertFalse(config.swap_weight_nibbles)

    def test_builder_adds_diffusers_quant_type_for_nvfp4(self):
        updated = _updated_quant_config(
            {
                "quantization_config": {
                    "quant_method": "modelopt",
                    "quant_algo": "NVFP4",
                    "ignore": [],
                }
            },
            fallback_patterns=["single_transformer_blocks.*.proj_mlp*"],
            swap_weight_nibbles=False,
        )

        self.assertEqual(updated["quantization_config"]["quant_type"], "NVFP4")
        self.assertEqual(
            updated["quantization_config"]["ignore"],
            ["single_transformer_blocks.*.proj_mlp*"],
        )

    @patch(
        "sglang.multimodal_gen.runtime.utils.quantization_utils.get_metadata_from_safetensors_file",
        return_value={"_quantization_metadata": '{"quant_algo": "NVFP4"}'},
    )
    @patch("sglang.multimodal_gen.runtime.utils.quantization_utils.safe_open")
    def test_build_nvfp4_config_maps_bfl_fallback_modules_to_custom_prefixes(
        self, mock_safe_open, _mock_metadata
    ):
        mock_safe_open.return_value = _FakeSafeTensorFile(
            {
                "single_blocks.0.linear1.weight": torch.empty(
                    55296, 3072, dtype=torch.uint8
                ),
                "single_blocks.0.linear1.weight_scale": torch.empty(
                    55296, 384, dtype=torch.float8_e4m3fn
                ),
                "final_layer.linear.weight": torch.empty(
                    128, 6144, dtype=torch.bfloat16
                ),
            }
        )

        config = build_nvfp4_config_from_safetensors_list(
            ["fake.safetensors"],
            {
                r"^single_blocks\.(\d+)\.linear1\.(.*)$": (
                    r"single_transformer_blocks.\1.attn.to_qkv_mlp_proj.\2"
                ),
                r"^final_layer\.linear\.(.*)$": r"proj_out.\1",
            },
        )

        self.assertIsNotNone(config)
        self.assertIn("proj_out", config.exclude_modules)

    @patch(
        "sglang.multimodal_gen.runtime.utils.quantization_utils.get_metadata_from_safetensors_file",
        return_value={"_quantization_metadata": '{"quant_algo": "NVFP4"}'},
    )
    @patch("sglang.multimodal_gen.runtime.utils.quantization_utils.safe_open")
    def test_build_nvfp4_config_keeps_raw_bfl_prefixes_for_mlp_wrappers(
        self, mock_safe_open, _mock_metadata
    ):
        mock_safe_open.return_value = _FakeSafeTensorFile(
            {
                "blocks.0.to_q.weight": torch.empty(5120, 2560, dtype=torch.uint8),
                "blocks.0.to_q.weight_scale": torch.empty(
                    5120, 320, dtype=torch.float8_e4m3fn
                ),
                "blocks.0.ffn.net.0.proj.weight": torch.empty(
                    13824, 5120, dtype=torch.bfloat16
                ),
                "blocks.0.ffn.net.2.weight": torch.empty(
                    5120, 13824, dtype=torch.bfloat16
                ),
            }
        )

        config = build_nvfp4_config_from_safetensors_list(
            ["fake.safetensors"],
            WanVideoConfig().param_names_mapping,
            WanVideoConfig().reverse_param_names_mapping,
        )

        self.assertIsNotNone(config)
        self.assertIn("blocks.0.ffn.net.0.proj", config.exclude_modules)
        self.assertIn("blocks.0.ffn.net.2", config.exclude_modules)

    @patch("sglang.multimodal_gen.runtime.layers.linear.get_group_rank", return_value=0)
    @patch("sglang.multimodal_gen.runtime.layers.linear.get_group_size", return_value=1)
    @patch(
        "sglang.multimodal_gen.runtime.layers.linear.get_tp_group", return_value=None
    )
    @patch(
        "sglang.multimodal_gen.runtime.layers.attention.layer.get_ring_parallel_world_size",
        return_value=1,
    )
    @patch(
        "sglang.multimodal_gen.runtime.layers.attention.selector.get_global_server_args",
        return_value=SimpleNamespace(attention_backend=None),
    )
    def test_flux_single_transformer_block_modelopt_excludes_use_full_prefix(
        self,
        _mock_server_args,
        _mock_ring_world_size,
        _mock_tp_group,
        _mock_group_size,
        _mock_group_rank,
    ):
        quant_config = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=[
                "single_transformer_blocks.*.proj_mlp*",
                "single_transformer_blocks.*.proj_out*",
                "single_transformer_blocks.*.attn.to_q",
            ],
        )

        block = FluxSingleTransformerBlock(
            dim=64,
            num_attention_heads=4,
            attention_head_dim=16,
            mlp_ratio=2.0,
            quant_config=quant_config,
            prefix="single_transformer_blocks.0",
        )

        self.assertEqual(block.proj_mlp.prefix, "single_transformer_blocks.0.proj_mlp")
        self.assertEqual(block.proj_out.prefix, "single_transformer_blocks.0.proj_out")
        self.assertEqual(
            block.attn.to_q.prefix, "single_transformer_blocks.0.attn.to_q"
        )
        self.assertIsInstance(block.proj_mlp.quant_method, UnquantizedLinearMethod)
        self.assertIsInstance(block.proj_out.quant_method, UnquantizedLinearMethod)
        self.assertIsInstance(block.attn.to_q.quant_method, UnquantizedLinearMethod)


if __name__ == "__main__":
    unittest.main()
