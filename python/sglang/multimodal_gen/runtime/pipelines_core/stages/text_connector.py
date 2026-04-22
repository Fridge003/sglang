import torch

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.probe_utils import dump_probe_payload


class LTX2TextConnectorStage(PipelineStage):
    """
    Stage for applying LTX-2 Text Connectors to split/transform text embeddings
    into video and audio contexts.
    """

    def __init__(self, connectors):
        super().__init__()
        self.connectors = connectors

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        # Input: batch.prompt_embeds (from Gemma, [B, S, D])
        # Output: batch.prompt_embeds (Video Context), batch.audio_prompt_embeds (Audio Context)

        prompt_embeds = batch.prompt_embeds
        prompt_attention_mask = batch.prompt_attention_mask
        neg_prompt_embeds = batch.negative_prompt_embeds
        neg_prompt_attention_mask = batch.negative_attention_mask

        if isinstance(prompt_embeds, list):
            prompt_embeds = prompt_embeds[0] if len(prompt_embeds) > 0 else None

        if isinstance(prompt_attention_mask, list):
            prompt_attention_mask = (
                prompt_attention_mask[0] if len(prompt_attention_mask) > 0 else None
            )

        if isinstance(neg_prompt_embeds, list):
            neg_prompt_embeds = (
                neg_prompt_embeds[0] if len(neg_prompt_embeds) > 0 else None
            )

        if isinstance(neg_prompt_attention_mask, list):
            neg_prompt_attention_mask = (
                neg_prompt_attention_mask[0]
                if len(neg_prompt_attention_mask) > 0
                else None
            )

        # Handle CFG: Concatenate negative and positive inputs
        if batch.do_classifier_free_guidance:
            dump_probe_payload(
                batch,
                "text_connector/input_cfg",
                {
                    "positive_prompt_embeds": prompt_embeds,
                    "negative_prompt_embeds": neg_prompt_embeds,
                    "positive_attention_mask": prompt_attention_mask,
                    "negative_attention_mask": neg_prompt_attention_mask,
                },
            )

            # Concatenate: [Negative, Positive]
            prompt_embeds = torch.cat([neg_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat(
                [neg_prompt_attention_mask, prompt_attention_mask], dim=0
            )

        # Prepare additive mask for connectors (as per Diffusers implementation)
        dtype = prompt_embeds.dtype

        additive_attention_mask = (prompt_attention_mask.to(torch.int64) - 1).to(
            dtype
        ) * torch.finfo(dtype).max

        should_dump_feature_extractor_outputs = bool(envs.SGLANG_DIFFUSION_PROBE_DIR)
        if should_dump_feature_extractor_outputs:
            (
                connector_video_hidden_states,
                connector_audio_hidden_states,
                connector_attention_mask,
            ) = self.connectors.prepare_connector_inputs(
                prompt_embeds, additive_attention_mask, additive_mask=True
            )
            if batch.do_classifier_free_guidance:
                neg_video_hidden_states, pos_video_hidden_states = (
                    connector_video_hidden_states.chunk(2, dim=0)
                )
                neg_audio_hidden_states, pos_audio_hidden_states = (
                    connector_audio_hidden_states.chunk(2, dim=0)
                )
                neg_attention_mask, pos_attention_mask = connector_attention_mask.chunk(
                    2, dim=0
                )
                dump_probe_payload(
                    batch,
                    "text_connector/feature_extractor_output_split",
                    {
                        "positive_video_hidden_states": pos_video_hidden_states,
                        "negative_video_hidden_states": neg_video_hidden_states,
                        "positive_audio_hidden_states": pos_audio_hidden_states,
                        "negative_audio_hidden_states": neg_audio_hidden_states,
                        "positive_attention_mask": pos_attention_mask,
                        "negative_attention_mask": neg_attention_mask,
                    },
                )
            else:
                dump_probe_payload(
                    batch,
                    "text_connector/feature_extractor_output_positive",
                    {
                        "video_hidden_states": connector_video_hidden_states,
                        "audio_hidden_states": connector_audio_hidden_states,
                        "attention_mask": connector_attention_mask,
                    },
                )

        # Call connectors
        # Expects: prompt_embeds, attention_mask, additive_mask=True
        with set_forward_context(current_timestep=None, attn_metadata=None):
            if should_dump_feature_extractor_outputs:
                connector_prompt_embeds, connector_audio_prompt_embeds, connector_mask = (
                    self.connectors.forward_from_prepared_inputs(
                        video_hidden_states=connector_video_hidden_states,
                        audio_hidden_states=connector_audio_hidden_states,
                        attention_mask=connector_attention_mask,
                    )
                )
            else:
                connector_prompt_embeds, connector_audio_prompt_embeds, connector_mask = (
                    self.connectors(
                        prompt_embeds, additive_attention_mask, additive_mask=True
                    )
                )

        dump_probe_payload(
            batch,
            "text_connector/output_combined",
            {
                "connector_prompt_embeds": connector_prompt_embeds,
                "connector_audio_prompt_embeds": connector_audio_prompt_embeds,
                "connector_mask": connector_mask,
            },
        )

        # Split results if CFG was enabled
        if batch.do_classifier_free_guidance:
            neg_embeds, pos_embeds = connector_prompt_embeds.chunk(2, dim=0)
            neg_audio_embeds, pos_audio_embeds = connector_audio_prompt_embeds.chunk(
                2, dim=0
            )
            neg_mask, pos_mask = connector_mask.chunk(2, dim=0)

            dump_probe_payload(
                batch,
                "text_connector/output_split",
                {
                    "positive_prompt_embeds": pos_embeds,
                    "negative_prompt_embeds": neg_embeds,
                    "positive_audio_prompt_embeds": pos_audio_embeds,
                    "negative_audio_prompt_embeds": neg_audio_embeds,
                    "positive_attention_mask": pos_mask,
                    "negative_attention_mask": neg_mask,
                },
            )

            batch.prompt_embeds = [pos_embeds]
            batch.audio_prompt_embeds = [pos_audio_embeds]
            batch.prompt_attention_mask = pos_mask

            batch.negative_prompt_embeds = [neg_embeds]
            batch.negative_audio_prompt_embeds = [neg_audio_embeds]
            batch.negative_attention_mask = neg_mask

            inject_path = envs.SGLANG_DIFFUSION_LTX2_INJECT_CONNECTOR_OUTPUT
            if inject_path:
                print(
                    f"[INJECT_CONNECTOR] loading official dump from {inject_path}",
                    flush=True,
                )
                inj = torch.load(str(inject_path), map_location="cpu")
                print(
                    f"[INJECT_CONNECTOR] pre-inject shapes: "
                    f"pos={tuple(pos_embeds.shape)} neg={tuple(neg_embeds.shape)}; "
                    f"dump pos={tuple(inj['positive_prompt_embeds'].shape)} "
                    f"dump pos_audio={tuple(inj['positive_audio_prompt_embeds'].shape)}",
                    flush=True,
                )
                tgt_device = pos_embeds.device
                tgt_dtype = pos_embeds.dtype
                batch.prompt_embeds = [
                    inj["positive_prompt_embeds"].to(device=tgt_device, dtype=tgt_dtype)
                ]
                batch.audio_prompt_embeds = [
                    inj["positive_audio_prompt_embeds"].to(
                        device=tgt_device, dtype=tgt_dtype
                    )
                ]
                batch.prompt_attention_mask = inj["positive_attention_mask"].to(
                    device=tgt_device
                )
                batch.negative_prompt_embeds = [
                    inj["negative_prompt_embeds"].to(device=tgt_device, dtype=tgt_dtype)
                ]
                batch.negative_audio_prompt_embeds = [
                    inj["negative_audio_prompt_embeds"].to(
                        device=tgt_device, dtype=tgt_dtype
                    )
                ]
                batch.negative_attention_mask = inj["negative_attention_mask"].to(
                    device=tgt_device
                )
        else:
            dump_probe_payload(
                batch,
                "text_connector/output_positive",
                {
                    "prompt_embeds": connector_prompt_embeds,
                    "audio_prompt_embeds": connector_audio_prompt_embeds,
                    "attention_mask": connector_mask,
                },
            )
            # Update positive fields
            batch.prompt_embeds = [connector_prompt_embeds]
            batch.audio_prompt_embeds = [connector_audio_prompt_embeds]
            batch.prompt_attention_mask = connector_mask

        return batch
