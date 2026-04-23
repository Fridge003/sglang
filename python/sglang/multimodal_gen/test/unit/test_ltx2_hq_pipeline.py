from types import SimpleNamespace
from unittest.mock import patch

from sglang.multimodal_gen.configs.pipeline_configs.ltx_2 import LTX2PipelineConfig
from sglang.multimodal_gen.configs.sample.ltx_2 import (
    LTX23HQSamplingParams,
    LTX23SamplingParams,
)
from sglang.multimodal_gen.registry import get_pipeline_config_classes
from sglang.multimodal_gen.runtime.pipelines.ltx_2_pipeline import (
    LTX2TwoStageHQPipeline,
    LTX2TwoStagePipeline,
    _add_ltx2_stage1_generation_stages,
    build_official_ltx2_sigmas,
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


def test_build_official_ltx2_sigmas_handles_single_step_without_nan():
    sigmas = build_official_ltx2_sigmas(1, number_of_tokens=1024)

    assert len(sigmas) == 1
    assert sigmas[0] == 1.0


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


def test_ltx23_hq_sampling_params_defaults_match_official_hq_parser():
    params = LTX23HQSamplingParams()
    extra = params.build_request_extra()

    assert params.height == 1088
    assert params.width == 1920
    assert params.num_inference_steps == 15
    assert params.distilled_lora_strength_stage_1 == 0.25
    assert params.distilled_lora_strength_stage_2 == 0.5
    assert params.video_rescale_scale == 0.45
    assert params.audio_rescale_scale == 1.0
    assert params.video_stg_blocks == []
    assert params.audio_stg_blocks == []
    assert extra["ltx2_distilled_lora_strength_stage_1"] == 0.25
    assert extra["ltx2_distilled_lora_strength_stage_2"] == 0.5


def test_ltx2_two_stage_pipeline_reads_request_lora_strength_overrides():
    pipeline = object.__new__(LTX2TwoStageHQPipeline)

    assert (
        pipeline._get_stage_distilled_lora_strength(
            "stage1",
            SimpleNamespace(extra={"ltx2_distilled_lora_strength_stage_1": 0.3}),
        )
        == 0.3
    )
    assert (
        pipeline._get_stage_distilled_lora_strength(
            "stage2",
            SimpleNamespace(extra={"ltx2_distilled_lora_strength_stage_2": 0.6}),
        )
        == 0.6
    )


def test_ltx23_hq_pipeline_registers_pipeline_specific_sampling_params():
    config_classes = get_pipeline_config_classes("LTX2TwoStageHQPipeline")

    assert config_classes is not None
    pipeline_config_cls, sampling_params_cls = config_classes
    assert pipeline_config_cls is LTX2PipelineConfig
    assert sampling_params_cls is LTX23HQSamplingParams


def test_ltx2_hq_and_ti2v_use_split_guided_passes():
    hq_batch = SimpleNamespace(ltx2_num_image_tokens=0)
    ti2v_batch = SimpleNamespace(ltx2_num_image_tokens=128)
    server_args = SimpleNamespace(
        pipeline_class_name="LTX2TwoStageHQPipeline",
        num_gpus=1,
        tp_size=1,
    )
    non_hq_server_args = SimpleNamespace(
        pipeline_class_name="LTX2TwoStagePipeline",
        num_gpus=1,
        tp_size=1,
    )

    assert LTX2DenoisingStage._should_use_split_two_stage_guided_passes(
        hq_batch,
        server_args,
    )
    assert LTX2DenoisingStage._should_use_split_two_stage_guided_passes(
        ti2v_batch,
        server_args,
    )
    assert not LTX2DenoisingStage._should_use_split_two_stage_guided_passes(
        hq_batch,
        non_hq_server_args,
    )


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


class _FakeTwoStagePipeline:
    def __init__(self, stage1_sampler_name: str, stage2_sampler_name: str):
        self.STAGE_1_DENOISING_SAMPLER_NAME = stage1_sampler_name
        self.STAGE_2_DENOISING_SAMPLER_NAME = stage2_sampler_name
        self.single_stages = []
        self.bulk_stages = []

    def add_stage(self, stage):
        self.single_stages.append(stage)

    def add_stages(self, stages):
        self.bulk_stages.extend(stages)

    def get_module(self, name):
        return f"module:{name}"


def test_ltx2_two_stage_pipeline_threads_stage2_sampler_by_variant():
    refinement_ctor_calls = []

    def _record_ctor(name):
        def _ctor(*args, **kwargs):
            if name == "refinement":
                refinement_ctor_calls.append(kwargs)
            return SimpleNamespace(name=name, kwargs=kwargs)

        return _ctor

    with (
        patch(
            "sglang.multimodal_gen.runtime.pipelines.ltx_2_pipeline._add_ltx2_front_stages"
        ),
        patch(
            "sglang.multimodal_gen.runtime.pipelines.ltx_2_pipeline._add_ltx2_stage1_generation_stages"
        ),
        patch(
            "sglang.multimodal_gen.runtime.pipelines.ltx_2_pipeline._add_ltx2_decoding_stage"
        ),
        patch(
            "sglang.multimodal_gen.runtime.pipelines.ltx_2_pipeline.LTX2HalveResolutionStage",
            side_effect=_record_ctor("halve"),
        ),
        patch(
            "sglang.multimodal_gen.runtime.pipelines.ltx_2_pipeline.LTX2LoRASwitchStage",
            side_effect=_record_ctor("lora_switch"),
        ),
        patch(
            "sglang.multimodal_gen.runtime.pipelines.ltx_2_pipeline.LTX2UpsampleStage",
            side_effect=_record_ctor("upsample"),
        ),
        patch(
            "sglang.multimodal_gen.runtime.pipelines.ltx_2_pipeline.LTX2ImageEncodingStage",
            side_effect=_record_ctor("image_encode"),
        ),
        patch(
            "sglang.multimodal_gen.runtime.pipelines.ltx_2_pipeline.LTX2RefinementStage",
            side_effect=_record_ctor("refinement"),
        ),
    ):
        LTX2TwoStagePipeline.create_pipeline_stages(
            _FakeTwoStagePipeline(
                stage1_sampler_name=LTX2TwoStagePipeline.STAGE_1_DENOISING_SAMPLER_NAME,
                stage2_sampler_name=LTX2TwoStagePipeline.STAGE_2_DENOISING_SAMPLER_NAME,
            ),
            server_args=None,
        )
        LTX2TwoStageHQPipeline.create_pipeline_stages(
            _FakeTwoStagePipeline(
                stage1_sampler_name=LTX2TwoStageHQPipeline.STAGE_1_DENOISING_SAMPLER_NAME,
                stage2_sampler_name=LTX2TwoStageHQPipeline.STAGE_2_DENOISING_SAMPLER_NAME,
            ),
            server_args=None,
        )

    assert refinement_ctor_calls[0]["sampler_name"] == "euler"
    assert refinement_ctor_calls[1]["sampler_name"] == "res2s"
