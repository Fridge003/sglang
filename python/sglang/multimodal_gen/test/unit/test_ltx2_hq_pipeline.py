from types import SimpleNamespace
from unittest.mock import patch

from sglang.multimodal_gen.configs.sample.ltx_2 import LTX23SamplingParams
from sglang.multimodal_gen.runtime.pipelines.ltx_2_pipeline import (
    _add_ltx2_stage1_generation_stages,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.ltx_2_denoising import (
    LTX2DenoisingStage,
    LTX23_TWO_STAGE_HQ_STAGE1_GUIDER_DEFAULTS,
)
from sglang.multimodal_gen.runtime.server_args import (
    is_ltx2_two_stage_pipeline_name,
)


def test_is_ltx2_two_stage_pipeline_name_accepts_hq_variant():
    assert is_ltx2_two_stage_pipeline_name("LTX2TwoStagePipeline")
    assert is_ltx2_two_stage_pipeline_name("LTX2TwoStageHQPipeline")
    assert not is_ltx2_two_stage_pipeline_name("LTX2Pipeline")


def test_ltx23_sampling_params_only_emits_explicit_stage1_guider_overrides():
    params = LTX23SamplingParams()

    extra = params.build_request_extra()

    assert "ltx2_stage1_guider_params" not in extra


def test_ltx23_sampling_params_copies_stage1_block_lists():
    video_blocks = [7, 9]
    audio_blocks = [3]
    params = LTX23SamplingParams(
        video_stg_blocks=video_blocks,
        audio_stg_blocks=audio_blocks,
    )

    extra = params.build_request_extra()

    assert extra["ltx2_stage1_guider_params"]["video_stg_blocks"] == [7, 9]
    assert extra["ltx2_stage1_guider_params"]["audio_stg_blocks"] == [3]

    video_blocks.append(11)
    audio_blocks.append(5)

    assert extra["ltx2_stage1_guider_params"]["video_stg_blocks"] == [7, 9]
    assert extra["ltx2_stage1_guider_params"]["audio_stg_blocks"] == [3]


def test_ltx2_hq_stage1_guider_defaults_merge_request_overrides():
    stage = object.__new__(LTX2DenoisingStage)
    server_args = SimpleNamespace(pipeline_class_name="LTX2TwoStageHQPipeline")

    defaults = stage._get_ltx2_stage1_guider_params(
        SimpleNamespace(extra={}),
        server_args,
        "stage1",
    )
    assert defaults == LTX23_TWO_STAGE_HQ_STAGE1_GUIDER_DEFAULTS
    assert defaults is not LTX23_TWO_STAGE_HQ_STAGE1_GUIDER_DEFAULTS

    merged = stage._get_ltx2_stage1_guider_params(
        SimpleNamespace(
            extra={
                "ltx2_stage1_guider_params": {
                    "video_stg_blocks": [17],
                    "audio_modality_scale": 2.5,
                }
            }
        ),
        server_args,
        "stage1",
    )
    assert merged["video_stg_blocks"] == [17]
    assert merged["audio_modality_scale"] == 2.5
    assert merged["video_rescale_scale"] == 0.45
    assert merged["audio_cfg_scale"] == 7.0


class _FakePipeline:
    def __init__(self):
        self.single_stages = []
        self.bulk_stages = []
        self.prepare_extra_kwargs = None

    def add_stage(self, stage):
        self.single_stages.append(stage)

    def add_standard_timestep_preparation_stage(self, prepare_extra_kwargs):
        self.prepare_extra_kwargs = prepare_extra_kwargs

    def add_stages(self, stages):
        self.bulk_stages.extend(stages)

    def get_module(self, name):
        return f"module:{name}"


def test_add_ltx2_stage1_generation_stages_threads_res2s_sampler():
    pipeline = _FakePipeline()
    denoising_ctor_calls = []

    def _record_ctor(name):
        def _ctor(*args, **kwargs):
            if name == "denoising":
                denoising_ctor_calls.append(kwargs)
            return SimpleNamespace(name=name, kwargs=kwargs)

        return _ctor

    with (
        patch(
            "sglang.multimodal_gen.runtime.pipelines.ltx_2_pipeline.LTX2SigmaPreparationStage",
            side_effect=_record_ctor("sigma"),
        ),
        patch(
            "sglang.multimodal_gen.runtime.pipelines.ltx_2_pipeline.LTX2AVLatentPreparationStage",
            side_effect=_record_ctor("latent_prep"),
        ),
        patch(
            "sglang.multimodal_gen.runtime.pipelines.ltx_2_pipeline.LTX2ImageEncodingStage",
            side_effect=_record_ctor("image_encode"),
        ),
        patch(
            "sglang.multimodal_gen.runtime.pipelines.ltx_2_pipeline.LTX2AVDenoisingStage",
            side_effect=_record_ctor("denoising"),
        ),
    ):
        _add_ltx2_stage1_generation_stages(
            pipeline,
            denoising_sampler_name="res2s",
        )

    assert pipeline.prepare_extra_kwargs is not None
    assert len(denoising_ctor_calls) == 1
    assert denoising_ctor_calls[0]["sampler_name"] == "res2s"
